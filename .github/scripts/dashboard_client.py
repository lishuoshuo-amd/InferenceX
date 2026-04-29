"""Fetch official benchmark data from InferenceX Dashboard API.

Used by verify-pr report to compare our results against the official
MI300X baseline published at https://inferencex.semianalysis.com/inference.
"""

from __future__ import annotations

import gzip
import json
import logging
import sys
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError

log = logging.getLogger(__name__)

DASHBOARD_API = "https://inferencex.semianalysis.com/api/v1/benchmarks"


def fetch_official_reference(
    dashboard_model: str,
    hardware: str = "mi300x",
    precision: str | None = None,
    framework: str | None = None,
    timeout: int = 15,
) -> dict[tuple[int, int, int, int], dict[str, Any]]:
    """Fetch latest official benchmark data for a model+hardware combo.

    Args:
        precision: e.g. "fp8", "bf16", "fp4", "int4". Filters API results.
        framework: e.g. "sglang", "vllm". Filters API results.

    Returns:
        {(isl, osl, conc, decode_tp): {tput_per_gpu, output_tput_per_gpu,
         input_tput_per_gpu, date, precision, framework}} keyed by point.
        Empty dict on any failure (graceful degradation).
    """
    url = f"{DASHBOARD_API}?model={dashboard_model}"
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            # API may return gzip even without Accept-Encoding header
            if raw[:2] == b'\x1f\x8b':
                raw = gzip.decompress(raw)
            data = json.loads(raw.decode("utf-8"))
    except (URLError, OSError, json.JSONDecodeError, ValueError) as e:
        log.warning("Dashboard API unavailable (%s): %s", url, e)
        return {}

    if not isinstance(data, list):
        log.warning("Unexpected Dashboard API response type: %s", type(data))
        return {}

    # Filter by hardware, precision, framework
    entries = [d for d in data if d.get("hardware") == hardware]
    if precision:
        entries = [d for d in entries if d.get("precision") == precision]
    if framework:
        entries = [d for d in entries if d.get("framework") == framework]

    if not entries:
        log.warning("No %s data for model=%s precision=%s framework=%s",
                     hardware, dashboard_model, precision, framework)
        return {}

    latest_date = max(d["date"] for d in entries)
    latest = [d for d in entries if d["date"] == latest_date]

    result: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for d in latest:
        tp = d.get("decode_tp") or d.get("prefill_tp") or 0
        key = (d["isl"], d["osl"], d["conc"], tp)
        m = d.get("metrics", {})
        result[key] = {
            "tput_per_gpu": m.get("tput_per_gpu"),
            "output_tput_per_gpu": m.get("output_tput_per_gpu"),
            "input_tput_per_gpu": m.get("input_tput_per_gpu"),
            "date": latest_date,
            "decode_tp": tp,
            "precision": d.get("precision"),
            "framework": d.get("framework"),
        }

    log.info("Dashboard: %s/%s prec=%s fw=%s date=%s points=%d",
             dashboard_model, hardware, precision, framework,
             latest_date, len(result))
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = sys.argv[1] if len(sys.argv) > 1 else "DeepSeek-R1-0528"
    prec = sys.argv[2] if len(sys.argv) > 2 else None
    fw = sys.argv[3] if len(sys.argv) > 3 else None
    ref = fetch_official_reference(model, precision=prec, framework=fw)
    for k, v in sorted(ref.items()):
        print(f"  isl={k[0]} osl={k[1]} conc={k[2]} tp={k[3]}: "
              f"output_tput_per_gpu={v['output_tput_per_gpu']:.1f} "
              f"tput_per_gpu={v['tput_per_gpu']:.1f}")
