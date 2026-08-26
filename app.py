from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from views.common import render_shell

render_shell()

page = st.navigation(
    [
        st.Page("views/home.py", title="Mission Control", icon="🏠", default=True),
        st.Page("views/chat.py", title="Chat Analysis", icon="💬"),
        st.Page("views/gallery.py", title="Dataset Gallery", icon="🛰️"),
        st.Page("views/studio3d.py", title="3D Studio", icon="🌐"),
        st.Page("views/about.py", title="About", icon="📐"),
    ],
    position="sidebar",
)

with st.sidebar:
    st.markdown("### 🛰️ SatQuery AI")
    st.caption("Vision-Language Assistant for ISRO Remote Sensing — SIH26167")

page.run()
