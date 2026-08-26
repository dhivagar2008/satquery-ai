import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests

BASE = "http://localhost:8001"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

CASES = [
    ("VQA_SINGLE", "Describe the land cover in this image",
     {"optical_file": "chennai_optical_t1.tif"}),
    ("SPATIAL_SEGMENTATION", "Highlight water bodies",
     {"optical_file": "chennai_optical_t1.tif"}),
    ("CHANGE_DETECTION", "Has urban area expanded between the two dates?",
     {"optical_file": "chennai_optical_t1.tif", "bitemporal_file": "chennai_optical_t2.tif"}),
    ("CROSS_MODAL_FUSION", "Use optical and SAR images together to identify built-up and water regions",
     {"optical_file": "chennai_optical_t2.tif", "sar_file": "chennai_sar.tif"}),
]


def main():
    health = requests.get(f"{BASE}/api/health", timeout=10).json()
    print(f"health: {health['status']} | llm: {health['llm_available']}")
    print("-" * 64)

    all_ok = True
    for expected, query, files in CASES:
        data = {"query": query}
        fh = {}
        for field, fname in files.items():
            fh[field] = open(RAW / fname, "rb")
        try:
            r = requests.post(f"{BASE}/api/query", data=data, files=fh, timeout=120)
            r.raise_for_status()
            d = r.json()
        finally:
            for f in fh.values():
                f.close()

        ok = d["intent"] == expected
        all_ok &= ok
        status = "PASS" if ok else f"FAIL (expected {expected})"
        print(f"[{status}] {expected}")
        print(f"  router : {d['router_source']} | engine: {d['engine']}")
        for m in d["metrics"]:
            print(f"  {m['label']:<16} {m['value']}")
        n_overlay = len(d.get("overlays_b64") or {})
        answer_len = len(d["answer_markdown"])
        print(f"  overlays: {n_overlay} | answer chars: {answer_len}")
        print("-" * 64)

    print("ALL INTENTS VERIFIED" if all_ok else "SOME CHECKS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
