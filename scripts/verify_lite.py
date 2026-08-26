import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8002"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

CASES = [
    ("VQA_SINGLE", "Describe the land cover in this image",
     {"optical_file": "chennai_optical_t1.tif"}),
    ("SPATIAL_SEGMENTATION", "Highlight water bodies",
     {"optical_file": "chennai_optical_t1.tif"}),
    ("CHANGE_DETECTION", "What changed between T1 and T2?",
     {"optical_file": "chennai_optical_t1.tif", "bitemporal_file": "chennai_optical_t2.tif"}),
    ("CROSS_MODAL_FUSION", "Use optical and SAR to identify built-up and water regions",
     {"optical_file": "chennai_optical_t2.tif", "sar_file": "chennai_sar.tif"}),
]


def main():
    h = requests.get(f"{BASE}/api/health", timeout=15).json()
    print(f"health: {h['status']} | engine: {h['engine']}")
    print("-" * 56)

    ok_all = True
    for expected, query, files in CASES:
        fh = {k: open(RAW / v, "rb") for k, v in files.items()}
        try:
            r = requests.post(f"{BASE}/api/query", data={"query": query},
                              files=fh, timeout=120)
            d = r.json()
        finally:
            for f in fh.values():
                f.close()
        if r.status_code != 200:
            print(f"[FAIL] {expected}: HTTP {r.status_code} {str(d)[:160]}")
            ok_all = False
            continue
        ok = d["intent"] == expected
        ok_all &= ok
        n_ov = len(d.get("overlays_b64") or {})
        print(f"[{'PASS' if ok else 'FAIL'}] {expected} | overlays={n_ov}")
        for m in d["metrics"]:
            print(f"   {m['label']:<16}{m['value']}")

    print("-" * 56)
    print("LITE API: ALL INTENTS VERIFIED" if ok_all else "LITE API: FAILURES PRESENT")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
