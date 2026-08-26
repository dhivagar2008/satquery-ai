import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest


def run_page(rel_path: str):
    path = str(Path(__file__).resolve().parents[1] / rel_path)
    print(f"--- {rel_path} ---")
    at = AppTest.from_file(path, default_timeout=120)
    at.session_state["satquery_user"] = {"name": "Test Analyst", "email": "t@example.com", "picture": ""}
    at.run()
    if at.exception:
        for exc in at.exception:
            print(f"EXCEPTION: {exc.value}")
        return False
    print("OK")
    return True


def main():
    results = {}
    results["home"] = run_page("views/home.py")
    results["gallery"] = run_page("views/gallery.py")
    results["chat"] = run_page("views/chat.py")
    results["studio3d"] = run_page("views/studio3d.py")
    results["about"] = run_page("views/about.py")

    print("=" * 50)
    failed = [k for k, v in results.items() if not v]
    print("FAILED PAGES:", failed if failed else "none — ALL PAGES RENDER CLEANLY")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
