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
import re
import sys
import time
from pathlib import Path
from textwrap import dedent

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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

        NFS_DIR="{nfs_result_dir}"
        mkdir -p "$NFS_DIR"
        POD_LOG="$NFS_DIR/pod.log"
        exec > >(stdbuf -oL tee -a "$POD_LOG") 2>&1

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

        SUMMARY="$NFS_DIR/summary-{script.replace('.sh', '')}.json"
        echo '[]' > "$SUMMARY"

        # Clone repo with full history; fetch upstream if base SHA missing (fork PR)
        REPO_DIR="/tmp/inferencex-repo"
        rm -rf "$REPO_DIR"
        if ! git clone --quiet {repo_url} "$REPO_DIR"; then
            echo "FATAL: git clone failed" >&2
            exit 1
        fi
        cd "$REPO_DIR"
        # Fetch base SHA from upstream if not in fork history
        if ! git cat-file -e "{base_sha}" 2>/dev/null; then
            echo "Base SHA not in clone, fetching from upstream..."
            git fetch --quiet origin "{base_sha}" 2>/dev/null || \
            git fetch --quiet https://github.com/SemiAnalysisAI/InferenceX.git "{base_sha}" 2>/dev/null || true
        fi
        if ! git cat-file -e "{base_sha}" 2>/dev/null; then
            echo "FATAL: base SHA {base_sha[:10]} not found" >&2
            exit 1
        fi
        if ! git cat-file -e "{head_sha}" 2>/dev/null; then
            echo "FATAL: head SHA {head_sha[:10]} not found" >&2
            exit 1
        fi

        RUNS_JSON='{runs_json_str}'
        NPTS=$(echo "$RUNS_JSON" | jq 'length')
        echo "Will run $NPTS points (x2 for baseline+optimized) for {script}"

        run_one() {{
            local mode=$1 ref=$2 tp=$3 conc=$4 isl=$5 osl=$6 ep=${{7:-1}}
            local mml=$((isl + osl + 200))
            local tag="${{mode}}-tp${{tp}}-conc${{conc}}-${{isl}}_${{osl}}"

            echo "--- Run $tag @ $ref ---"
            # Graceful stop (SIGTERM) then force kill (SIGKILL) as fallback
            pkill -f "sglang.launch_server" 2>/dev/null || true
            pkill -f "sglang.srt" 2>/dev/null || true
            pkill -f "vllm serve" 2>/dev/null || true
            pkill -f "vllm.entrypoints" 2>/dev/null || true
            pkill -f "benchmark_serving" 2>/dev/null || true
            pkill -f "ray::" 2>/dev/null || true
            sleep 8
            # Force kill anything still alive
            pkill -9 -f "sglang" 2>/dev/null || true
            pkill -9 -f "vllm" 2>/dev/null || true
            sleep 2
            fuser -k 8888/tcp 2>/dev/null || true
            # Clean up leaked shared memory
            rm -f /dev/shm/vllm_* /dev/shm/sglang_* 2>/dev/null || true
            rm -rf /workspace && mkdir -p /workspace

            cd "$REPO_DIR"
            git --work-tree=/workspace checkout "$ref" -- benchmarks utils runners 2>/dev/null || \\
                git --work-tree=/workspace checkout "$ref" -- benchmarks utils

            # Patch hf download to no-op (model is local on NFS)
            sed -i 's|^[[:space:]]*hf download.*|true # patched: model is local|' \\
                "/workspace/benchmarks/single_node/{script}"

            local result_file="verify-${{tag}}-{script.replace('.sh', '')}"
            set +e
            cd /workspace

            MODEL="{model_path}" \\
            TP="$tp" CONC="$conc" EP_SIZE="$ep" \\
            ISL="$isl" OSL="$osl" MAX_MODEL_LEN="$mml" \\
            RANDOM_RANGE_RATIO="{random_range_ratio}" \\
            RESULT_FILENAME="$result_file" \\
            RUN_EVAL=false EVAL_ONLY=false EXP_NAME=verify \\
                bash benchmarks/single_node/{script}
            local rc=$?
            set -e

            local result_json="/workspace/${{result_file}}.json"
            local output_tp="null" total_tp="null" request_tp="null"
            if [[ -f "$result_json" ]]; then
                output_tp=$(jq -r '.output_throughput // empty' "$result_json" 2>/dev/null || echo "null")
                total_tp=$(jq -r '.total_token_throughput // empty' "$result_json" 2>/dev/null || echo "null")
                request_tp=$(jq -r '.request_throughput // empty' "$result_json" 2>/dev/null || echo "null")
            fi
            # Convert to per-GPU (÷ TP, matching official process_result.py)
            local otpg="null" tpg="null" itpg="null"
            if [[ "$output_tp" != "null" && -n "$output_tp" ]]; then
                otpg=$(echo "$output_tp $tp" | awk '{{printf "%.4f", $1/$2}}')
            fi
            if [[ "$total_tp" != "null" && -n "$total_tp" ]]; then
                tpg=$(echo "$total_tp $tp" | awk '{{printf "%.4f", $1/$2}}')
            fi
            if [[ "$total_tp" != "null" && "$output_tp" != "null" && -n "$total_tp" && -n "$output_tp" ]]; then
                itpg=$(echo "$total_tp $output_tp $tp" | awk '{{printf "%.4f", ($1-$2)/$3}}')
            fi

            # Copy artifacts to NFS
            cp -f /workspace/server.log "$NFS_DIR/${{tag}}.server.log" 2>/dev/null || true
            [[ -f "$result_json" ]] && cp -f "$result_json" "$NFS_DIR/${{result_file}}.json"

            # Append to summary (per-GPU metrics aligned with official Dashboard)
            jq --arg s "{script}" --arg m "$mode" \\
               --arg otpg "$otpg" --arg tpg "$tpg" --arg itpg "$itpg" \\
               --arg tp "$tp" --arg c "$conc" --arg i "$isl" --arg o "$osl" --arg rc "$rc" \\
               '. += [{{script:$s, mode:$m,
                       output_tput_per_gpu:($otpg|tonumber? // null),
                       tput_per_gpu:($tpg|tonumber? // null),
                       input_tput_per_gpu:($itpg|tonumber? // null),
                       tp:($tp|tonumber), conc:($c|tonumber),
                       isl:($i|tonumber), osl:($o|tonumber),
                       exit_code:($rc|tonumber)}}]' \\
               "$SUMMARY" > "$SUMMARY.tmp" && mv "$SUMMARY.tmp" "$SUMMARY"

            echo "--- Done $tag (rc=$rc, output_tput_per_gpu=$otpg, tput_per_gpu=$tpg) ---"
        }}

        for i in $(seq 0 $((NPTS - 1))); do
            tp=$(echo "$RUNS_JSON" | jq -r ".[$i].tp")
            conc=$(echo "$RUNS_JSON" | jq -r ".[$i].conc")
            isl=$(echo "$RUNS_JSON" | jq -r ".[$i].isl")
            osl=$(echo "$RUNS_JSON" | jq -r ".[$i].osl")
            ep=$(echo "$RUNS_JSON" | jq -r ".[$i].ep // 1")
            echo "=== point $((i+1))/$NPTS: tp=$tp conc=$conc isl=$isl osl=$osl ep=$ep ==="
            run_one baseline  "{base_sha}" "$tp" "$conc" "$isl" "$osl" "$ep"
            run_one optimized "{head_sha}" "$tp" "$conc" "$isl" "$osl" "$ep"
        done

        # ── GSM8K accuracy eval (once per script, using the first point's TP/EP) ──
        # Uses EVAL_ONLY=true so the benchmark script:
        #   1. Calls setup_eval_context → sets correct EVAL_MAX_MODEL_LEN for server
        #   2. Starts server with eval-appropriate context length
        #   3. Skips run_benchmark_serving (no throughput run)
        #   4. Runs run_eval (GSM8K via lm-eval) only
        run_eval_once() {{
            local mode=$1 ref=$2 tp=$3 ep=$4
            local tag="eval-${{mode}}"

            echo "--- Eval $tag @ $ref (tp=$tp ep=$ep) ---"
            pkill -f "sglang.launch_server" 2>/dev/null || true
            pkill -f "sglang.srt" 2>/dev/null || true
            pkill -f "vllm serve" 2>/dev/null || true
            pkill -f "vllm.entrypoints" 2>/dev/null || true
            pkill -f "ray::" 2>/dev/null || true
            sleep 8
            pkill -9 -f "sglang" 2>/dev/null || true
            pkill -9 -f "vllm" 2>/dev/null || true
            sleep 2
            fuser -k 8888/tcp 2>/dev/null || true
            rm -f /dev/shm/vllm_* /dev/shm/sglang_* 2>/dev/null || true
            rm -rf /workspace && mkdir -p /workspace

            cd "$REPO_DIR"
            git --work-tree=/workspace checkout "$ref" -- benchmarks utils runners 2>/dev/null || \\
                git --work-tree=/workspace checkout "$ref" -- benchmarks utils

            sed -i 's|^[[:space:]]*hf download.*|true # patched: model is local|' \\
                "/workspace/benchmarks/single_node/{script}"

            local result_file="verify-${{tag}}-{script.replace('.sh', '')}"
            set +e
            cd /workspace

            MODEL="{model_path}" \\
            TP="$tp" CONC=32 EP_SIZE="$ep" \\
            ISL=1024 OSL=8192 MAX_MODEL_LEN=16384 \\
            RANDOM_RANGE_RATIO="{random_range_ratio}" \\
            RESULT_FILENAME="$result_file" \\
            RUN_EVAL=true EVAL_ONLY=true EXP_NAME=verify-eval \\
                bash benchmarks/single_node/{script}
            local rc=$?
            set -e

            # Extract GSM8K score from lm-eval results
            local eval_score="null"
            local eval_json
            eval_json=$(find /workspace -path "*/results*.json" -name "results*.json" 2>/dev/null | head -1)
            if [[ -n "$eval_json" && -f "$eval_json" ]]; then
                eval_score=$(python3 -c "
import json, sys
try:
    d = json.load(open('$eval_json'))
    r = d.get('results', {{}})
    for task, metrics in r.items():
        for k in ['exact_match,strict-match', 'exact_match,flexible-extract', 'acc,none']:
            if k in metrics:
                print(f'{{metrics[k]:.4f}}')
                sys.exit(0)
    print('null')
except Exception:
    print('null')
" 2>/dev/null || echo "null")
                cp -f "$eval_json" "$NFS_DIR/${{tag}}-results.json" 2>/dev/null || true
            fi

            # Append eval result to summary
            jq --arg s "{script}" --arg m "$mode" --arg es "$eval_score" --arg rc "$rc" \\
               '. += [{{script:$s, mode:($m + "-eval"),
                       eval_score:($es|tonumber? // null),
                       exit_code:($rc|tonumber)}}]' \\
               "$SUMMARY" > "$SUMMARY.tmp" && mv "$SUMMARY.tmp" "$SUMMARY"

            echo "--- Done $tag (rc=$rc, eval_score=$eval_score) ---"
        }}

        EVAL_TP=$(echo "$RUNS_JSON" | jq -r '.[0].tp')
        EVAL_EP=$(echo "$RUNS_JSON" | jq -r '.[0].ep // 1')
        echo ""
        echo "=== GSM8K eval phase (tp=$EVAL_TP ep=$EVAL_EP) ==="
        run_eval_once baseline  "{base_sha}" "$EVAL_TP" "$EVAL_EP"
        run_eval_once optimized "{head_sha}" "$EVAL_TP" "$EVAL_EP"

        echo ""
        echo "=== Summary for {script} ==="
        cat "$SUMMARY"
        echo ""
        echo "=== SaFE verify-pr pod finished at $(date -u) ==="
    """)


def run(args: argparse.Namespace) -> int:
    """Main: create SaFE workload, poll, collect results."""
    if not args.safe_api_base:
        log.error("SAFE_API_BASE not set — configure it as a GitHub secret or pass --safe-api-base")
        return 1

    log.info("Script: %s | Points: %d | Image: %s", args.script, len(args.runs), args.image)

    client = SaFEClient(
        api_key=args.safe_api_key,
        base_url=args.safe_api_base,
        verify_ssl=not args.no_verify_ssl,
    )

    nfs_result_dir = args.nfs_result_dir
    import shutil
    if os.path.exists(nfs_result_dir):
        shutil.rmtree(nfs_result_dir, ignore_errors=True)
        # NFS may leave .nfs* stale handles; retry after short delay
        if os.path.exists(nfs_result_dir):
            time.sleep(2)
            shutil.rmtree(nfs_result_dir, ignore_errors=True)
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
    safe_name = re.sub(r"[^a-z0-9-]", "-", script_stem.lower()).strip("-")
    workload_name = f"vpr-{safe_name}-{args.head_sha[:8]}"[:44]

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

    # Poll with live log tailing from NFS
    pod_log_path = Path(nfs_result_dir) / "pod.log"
    log_offset = 0
    start = time.time()
    status = "running"

    while True:
        elapsed = time.time() - start
        if elapsed > args.timeout:
            log.warning("Timeout after %ds, stopping workload", args.timeout)
            client.stop_workload(workload_id)
            status = "timeout"
            break

        try:
            wl_status, wl_phase = client.get_workload_status(workload_id)
        except Exception as e:
            log.warning("Poll error: %s", e)
            time.sleep(30)
            continue

        # Stream pod.log incremental content
        if pod_log_path.exists():
            try:
                with open(pod_log_path, "r", errors="replace") as f:
                    f.seek(log_offset)
                    new_content = f.read()
                    if new_content:
                        print(new_content, end="", flush=True)
                        log_offset += len(new_content)
            except Exception:
                pass

        if wl_phase in ("succeeded", "completed") or wl_status in ("succeeded", "completed"):
            log.info("Workload %s completed (%.0fs)", workload_id, elapsed)
            status = "completed"
            break
        if wl_phase == "failed" or wl_status == "failed":
            log.error("Workload %s failed (%.0fs)", workload_id, elapsed)
            status = "failed"
            break

        time.sleep(30)

    # Final flush of pod log
    if pod_log_path.exists():
        try:
            with open(pod_log_path, "r", errors="replace") as f:
                f.seek(log_offset)
                remaining = f.read()
                if remaining:
                    print(remaining, end="", flush=True)
        except Exception:
            pass

    log.info("Workload %s finished: %s", workload_id, status)

    # Collect results from NFS -> local output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_name = f"summary-{script_stem}.json"
    nfs_summary = Path(nfs_result_dir) / summary_name
    local_summary = output_dir / summary_name

    import shutil
    if nfs_summary.exists():
        shutil.copy2(nfs_summary, local_summary)
        log.info("Summary copied: %s -> %s", nfs_summary, local_summary)

        for f in Path(nfs_result_dir).glob("*.json"):
            if f.name != summary_name:
                shutil.copy2(f, output_dir / f.name)
        for f in Path(nfs_result_dir).glob("*.server.log"):
            shutil.copy2(f, output_dir / f.name)
    else:
        log.error("Summary not found at %s — pod likely crashed before any benchmark ran", nfs_summary)
        local_summary.write_text("[]")
        return 1

    # Validate results: separate throughput rows from eval rows
    import json as _json
    from collections import defaultdict
    results = _json.loads(local_summary.read_text())

    throughput_results = [r for r in results if not r.get("mode", "").endswith("-eval")]
    eval_results = [r for r in results if r.get("mode", "").endswith("-eval")]

    total = len(throughput_results)
    ok = sum(1 for r in throughput_results
             if r.get("output_tput_per_gpu") and r["output_tput_per_gpu"] > 0)
    fail = total - ok
    log.info("Throughput results: %d total, %d ok, %d failed", total, ok, fail)

    if total == 0:
        log.error("No benchmark results produced")
        return 1

    expected_points = len(args.runs) * 2  # baseline + optimized per point
    if total < expected_points:
        log.warning("Incomplete: %d/%d throughput results (pod may have been killed early)",
                     total, expected_points)

    # Check paired completeness (throughput only)
    buckets = defaultdict(dict)
    for r in throughput_results:
        key = (r["tp"], r["conc"], r["isl"], r["osl"])
        buckets[key][r["mode"]] = r

    paired_ok = 0
    paired_fail = 0
    for key, modes in buckets.items():
        b = modes.get("baseline", {})
        o = modes.get("optimized", {})
        bt = b.get("output_tput_per_gpu") if b else None
        ot = o.get("output_tput_per_gpu") if o else None
        if bt and bt > 0 and ot and ot > 0:
            paired_ok += 1
        else:
            paired_fail += 1
            missing = []
            if not (bt and bt > 0): missing.append("baseline")
            if not (ot and ot > 0): missing.append("optimized")
            log.warning("Incomplete pair tp=%s conc=%s isl=%s osl=%s: missing %s",
                        key[0], key[1], key[2], key[3], "+".join(missing))

    log.info("Paired results: %d complete / %d incomplete out of %d points",
             paired_ok, paired_fail, len(buckets))

    if paired_ok == 0:
        log.error("No complete baseline+optimized pairs — verify cannot proceed")
        return 1

    # Validate GSM8K eval results (hard requirement: both must produce a score)
    eval_modes = {r["mode"].replace("-eval", ""): r for r in eval_results}
    eval_ok = True
    for mode_name in ("baseline", "optimized"):
        er = eval_modes.get(mode_name)
        if not er:
            log.error("GSM8K eval missing for %s — no eval row in summary", mode_name)
            eval_ok = False
        elif er.get("eval_score") is None:
            log.error("GSM8K eval produced no score for %s (exit_code=%s)",
                       mode_name, er.get("exit_code"))
            eval_ok = False
        else:
            log.info("GSM8K eval %s: score=%.4f", mode_name, er["eval_score"])

    if not eval_ok:
        log.error("GSM8K eval did not complete successfully for both baseline and optimized")
        return 1

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

    parser.add_argument("--safe-api-base", default=os.environ.get("SAFE_API_BASE", ""),
                        help="SaFE API base URL (or set SAFE_API_BASE env var)")
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
