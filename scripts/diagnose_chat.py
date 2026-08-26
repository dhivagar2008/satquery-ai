import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def main():
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "views" / "chat.py"),
                           default_timeout=300)
    at.session_state["optical_t1_path"] = str(RAW / "chennai_optical_t1.tif")
    at.session_state["optical_t2_path"] = str(RAW / "chennai_optical_t2.tif")
    at.session_state["sar_path"] = str(RAW / "chennai_sar.tif")
    at.run()

    print("initial exceptions:", [str(e.value)[:200] for e in at.exception])
    print("chat_input count:", len(at.chat_input))
    print("button count:", len(at.button))

    ci = at.chat_input[0]
    ci.set_value("Highlight water bodies")
    at.run()
    print("after submit exceptions:")
    for e in at.exception:
        print("  EXC:", str(e.value)[:500])
        if e.stack_trace:
            trace = e.stack_trace.splitlines()
            for line in trace[-8:]:
                print("   ", line)
    msgs = at.session_state["messages"]
    print("messages in state:", [(m["role"]) for m in msgs])
    if msgs and msgs[-1]["role"] == "assistant":
        last = msgs[-1]
        print("assistant keys:", sorted(last.keys()))
        print("answer chars:", len(last.get("content", "")))
    sys.exit(1 if at.exception else 0)


if __name__ == "__main__":
    main()
