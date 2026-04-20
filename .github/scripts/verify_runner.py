#!/usr/bin/env python3
"""Verify-PR runner: submit benchmark workloads to SaFE training pods.

For each changed script (model), creates a single SaFE workload (pod)
that iterates over all (tp, conc, isl, osl) points, running both
baseline (PR base ref) and optimized (PR head ref) benchmarks inside
the pod. Results are written to NFS and read back by the CI runner.

Usage (called from verify-pr.yml):
    python3 .github/scripts/verify_runner.py \
        --script dsr1_fp8_mi355x.sh \
        --runs-json '[{"tp":8,"conc":4,"isl":1024,"osl":1024}]' \
        --base-sha abc123 --head-sha def456 \
        --image lmsysorg/sglang:v0.5.9-rocm700-mi35x \
        --model-path /hyperloom/models/DeepSeek-R1-0528 \
        --nfs-result-dir /shared_nfs/lss/verify-pr/42/dsr1_fp8_mi355x \
        --output-dir verify-results
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from textwrap import dedent

from safe_client import SaFEClient

log = logging.getLogger("verify-runner")

REPO_CLONE_URL = "https://github.com/{repo}.git"


def build_entrypoint(
    *,
    script: str,
    runs: list[dict],
    base_sha: str,
    head_sha: str,
    image: str,
    model_path: str,
    models_root: str,
    random_range_ratio: str,
    nfs_result_dir: str,
    repo_full_name: str,
) -> str:
    """Build the shell script that runs inside the SaFE training pod.

    The pod has GPU + NFS access. It clones the repo, checks out both
    base and head refs, runs each benchmark point, and writes results
    to NFS for the CI runner to collect.
    """
    runs_json_str = json.dumps(runs)
    repo_url = REPO_CLONE_URL.format(repo=repo_full_name)

    return dedent(f"""\
        #!/bin/bash
        set -uo pipefail

        echo "=== SaFE verify-pr pod started at $(date -u) ==="
        echo "Script: {script}"
        echo "Points: {len(runs)}"
        echo "Base: {base_sha[:10]}"
        echo "Head: {head_sha[:10]}"

        # Install jq if missing
        if ! command -v jq >/dev/null 2>&1; then
            apt-get update -qq && apt-get install -y -qq jq git || true
        fi
        if ! command -v git >/dev/null 2>&1; then
            apt-get update -qq && apt-get install -y -qq git || true
        fi

        NFS_DIR="{nfs_result_dir}"
        mkdir -p "$NFS_DIR"
        SUMMARY="$NFS_DIR/summary-{script.replace('.sh', '')}.json"
        echo '[]' > "$SUMMARY"

        # Clone repo (full history for checkout of base/head refs)
        REPO_DIR="/tmp/inferencex-repo"
        rm -rf "$REPO_DIR"
        git clone --quiet {repo_url} "$REPO_DIR"
        cd "$REPO_DIR"

        RUNS_JSON='{runs_json_str}'
        NPTS=$(echo "$RUNS_JSON" | jq 'length')
        echo "Will run $NPTS points (x2 for baseline+optimized) for {script}"

        run_one() {{
            local mode=$1 ref=$2 tp=$3 conc=$4 isl=$5 osl=$6
            local mml=$((isl + osl + 200))
            local tag="${{mode}}-tp${{tp}}-conc${{conc}}-${{isl}}_${{osl}}"

            echo "--- Run $tag @ $ref ---"
            local work="/tmp/work_${{tag}}"
            rm -rf "$work" && mkdir -p "$work"

            cd "$REPO_DIR"
            git --work-tree="$work" checkout "$ref" -- benchmarks utils runners 2>/dev/null || \\
                git --work-tree="$work" checkout "$ref" -- benchmarks utils

            # Patch hf download to no-op (model is local on NFS)
            sed -i 's|^[[:space:]]*hf download.*|true # patched: model is local|' \\
                "$work/benchmarks/single_node/{script}"

            local result_file="verify-${{tag}}-{script.replace('.sh', '')}"
            set +e
            cd "$work"

            MODEL="{model_path}" \\
            TP="$tp" CONC="$conc" \\
            ISL="$isl" OSL="$osl" MAX_MODEL_LEN="$mml" \\
            RANDOM_RANGE_RATIO="{random_range_ratio}" \\
            RESULT_FILENAME="$result_file" \\
            RUN_EVAL=false EVAL_ONLY=false EXP_NAME=verify \\
                bash benchmarks/single_node/{script}
            local rc=$?
            set -e

            local result_json="$work/${{result_file}}.json"
            local throughput="null"
            if [[ -f "$result_json" ]]; then
                throughput=$(jq -r '.output_throughput // .total_token_throughput // .request_throughput // empty' "$result_json" || true)
            fi

            # Copy artifacts to NFS
            cp -f "$work/server.log" "$NFS_DIR/${{tag}}.server.log" 2>/dev/null || true
            [[ -f "$result_json" ]] && cp -f "$result_json" "$NFS_DIR/${{result_file}}.json"

            # Append to summary
            jq --arg s "{script}" --arg m "$mode" --arg t "${{throughput:-null}}" \\
               --arg tp "$tp" --arg c "$conc" --arg i "$isl" --arg o "$osl" --arg rc "$rc" \\
               '. += [{{script:$s, mode:$m, throughput:($t|tonumber? // null),
                       tp:($tp|tonumber), conc:($c|tonumber),
                       isl:($i|tonumber), osl:($o|tonumber),
                       exit_code:($rc|tonumber)}}]' \\
               "$SUMMARY" > "$SUMMARY.tmp" && mv "$SUMMARY.tmp" "$SUMMARY"

            echo "--- Done $tag (rc=$rc, throughput=$throughput) ---"
        }}

        for i in $(seq 0 $((NPTS - 1))); do
            tp=$(echo "$RUNS_JSON" | jq -r ".[$i].tp")
            conc=$(echo "$RUNS_JSON" | jq -r ".[$i].conc")
            isl=$(echo "$RUNS_JSON" | jq -r ".[$i].isl")
            osl=$(echo "$RUNS_JSON" | jq -r ".[$i].osl")
            echo "=== point $((i+1))/$NPTS: tp=$tp conc=$conc isl=$isl osl=$osl ==="
            run_one baseline  "{base_sha}" "$tp" "$conc" "$isl" "$osl"
            run_one optimized "{head_sha}" "$tp" "$conc" "$isl" "$osl"
        done

        echo ""
        echo "=== Summary for {script} ==="
        cat "$SUMMARY"
        echo ""
        echo "=== SaFE verify-pr pod finished at $(date -u) ==="
    """)


def run(args: argparse.Namespace) -> int:
    """Main: create SaFE workload, poll, collect results."""
    log.info("Script: %s | Points: %d | Image: %s", args.script, len(args.runs), args.image)

    client = SaFEClient(
        api_key=args.safe_api_key,
        base_url=args.safe_api_base,
        verify_ssl=not args.no_verify_ssl,
    )

    nfs_result_dir = args.nfs_result_dir
    os.makedirs(nfs_result_dir, exist_ok=True)

    entrypoint = build_entrypoint(
        script=args.script,
        runs=args.runs,
        base_sha=args.base_sha,
        head_sha=args.head_sha,
        image=args.image,
        model_path=args.model_path,
        models_root=args.models_root,
        random_range_ratio=args.random_range_ratio,
        nfs_result_dir=nfs_result_dir,
        repo_full_name=args.repo,
    )

    script_stem = args.script.replace(".sh", "")
    workload_name = f"verify-{script_stem}-{args.head_sha[:8]}"

    log.info("Creating SaFE workload: %s", workload_name)
    workload = client.create_workload(
        workspace_id=args.safe_workspace_id,
        name=workload_name,
        image=args.image,
        command=entrypoint,
        gpu_count=args.gpu_count,
        cpu=args.cpu,
        memory=args.memory,
        env_vars={
            "MODELS_ROOT": args.models_root,
        },
    )

    workload_id = workload.get("id") or workload.get("workloadId", "")
    if not workload_id:
        log.error("No workload ID returned: %s", workload)
        return 1

    log.info("Workload submitted: %s", workload_id)

    status = client.poll_until_done(
        workload_id,
        timeout=args.timeout,
        interval=30,
        heartbeat_interval=300,
    )
    log.info("Workload %s finished: %s", workload_id, status)

    # Collect results from NFS -> local output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_name = f"summary-{script_stem}.json"
    nfs_summary = Path(nfs_result_dir) / summary_name
    local_summary = output_dir / summary_name

    if nfs_summary.exists():
        import shutil
        shutil.copy2(nfs_summary, local_summary)
        log.info("Summary copied: %s -> %s", nfs_summary, local_summary)

        for f in Path(nfs_result_dir).glob("*.json"):
            if f.name != summary_name:
                shutil.copy2(f, output_dir / f.name)
        for f in Path(nfs_result_dir).glob("*.server.log"):
            shutil.copy2(f, output_dir / f.name)
    else:
        log.warning("Summary not found at %s", nfs_summary)
        local_summary.write_text("[]")

    if status != "completed":
        log.error("Workload did not complete successfully: %s", status)
        return 1

    return 0


def main():
    parser = argparse.ArgumentParser(description="Verify-PR: SaFE workload runner")
    parser.add_argument("--script", required=True, help="Benchmark script name (e.g. dsr1_fp8_mi355x.sh)")
    parser.add_argument("--runs-json", required=True, help="JSON array of {tp,conc,isl,osl} points")
    parser.add_argument("--base-sha", required=True, help="PR base commit SHA")
    parser.add_argument("--head-sha", required=True, help="PR head commit SHA")
    parser.add_argument("--image", required=True, help="Docker image for the benchmark")
    parser.add_argument("--model-path", required=True, help="Local NFS model path")
    parser.add_argument("--models-root", default="/hyperloom/models", help="NFS models mount root")
    parser.add_argument("--random-range-ratio", default="0.8")
    parser.add_argument("--repo", required=True, help="GitHub repo full name (owner/repo)")

    parser.add_argument("--nfs-result-dir", required=True, help="NFS directory for pod to write results")
    parser.add_argument("--output-dir", default="verify-results", help="Local output dir for CI artifacts")

    parser.add_argument("--safe-api-base", default="https://oci-slc.primus-safe.amd.com")
    parser.add_argument("--safe-api-key", default=os.environ.get("SAFE_API_KEY", ""))
    parser.add_argument("--safe-workspace-id", default=os.environ.get("SAFE_WORKSPACE_ID", ""))
    parser.add_argument("--no-verify-ssl", action="store_true", default=False)

    parser.add_argument("--gpu-count", type=int, default=8)
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--memory", default="128Gi")
    parser.add_argument("--timeout", type=int, default=7200, help="Max seconds to wait for workload")

    args = parser.parse_args()
    args.runs = json.loads(args.runs_json)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sys.exit(run(args))


if __name__ == "__main__":
    main()
