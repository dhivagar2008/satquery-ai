from __future__ import annotations

import base64
import json

import streamlit as st

import config

USER_KEY = "satquery_user"


def is_authenticated() -> bool:
    return bool(st.session_state.get(USER_KEY))


def current_user() -> dict:
    return st.session_state.get(USER_KEY) or {}


def _decode_id_token(id_token: str) -> dict:
    try:
        payload = id_token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload))
        return {
            "name": data.get("name", ""),
            "email": data.get("email", ""),
            "picture": data.get("picture", ""),
        }
    except Exception:
        return {}


def google_login_button() -> None:
    from streamlit_oauth import OAuth2Component

    oauth2 = OAuth2Component(
        config.GOOGLE_CLIENT_ID,
        config.GOOGLE_CLIENT_SECRET,
        authorize_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
        token_endpoint="https://oauth2.googleapis.com/token",
        refresh_token_endpoint="https://oauth2.googleapis.com/token",
        revoke_token_endpoint="https://oauth2.googleapis.com/revoke",
    )
    result = oauth2.authorize_button(
        "Continue with Google",
        redirect_uri=config.GOOGLE_REDIRECT_URI,
        scope="openid email profile",
        icon="https://www.google.com/favicon.ico",
        use_container_width=True,
        key="google_oauth",
    )
    if result and "id_token" in result:
        user = _decode_id_token(result["id_token"])
        if user:
            st.session_state[USER_KEY] = user
            st.rerun()


def render_login_page() -> None:
    st.markdown(
        """<style>
        .login-wrap {display:flex; justify-content:center; padding-top:6vh;}
        .login-card {
            background: radial-gradient(ellipse at top left, rgba(6,182,212,.15), transparent 55%),
                        linear-gradient(150deg,#0e1830,#0b1120 70%);
            border:1px solid #1e2a45; border-radius:20px;
            width:430px; max-width:94vw; padding:2.4rem 2.4rem 1.8rem;
            box-shadow:0 18px 50px rgba(0,0,0,.45);
            text-align:center;
        }
        .login-logo {font-size:3rem; line-height:1;}
        .login-title {font-size:1.9rem; font-weight:800; color:#f8fafc; letter-spacing:-.5px; margin:.4rem 0 .2rem;}
        .login-title span {color:#06b6d4;}
        .login-sub {color:#94a3b8; font-size:.9rem; margin-bottom:1.6rem;}
        .login-note {color:#64748b; font-size:.75rem; margin-top:1.4rem;}
        </style>""",
        unsafe_allow_html=True,
    )
    _, center = st.columns([1, 2])
    with center:
        st.markdown(
            """<div class="login-wrap"><div class="login-card">
              <div class="login-logo">&#128752;</div>
              <div class="login-title">SatQuery <span>AI</span></div>
              <div class="login-sub">Interactive Vision-Language Assistant for<br>Multimodal Remote Sensing — ISRO / SIH26167</div>
            </div></div>""",
            unsafe_allow_html=True,
        )

        if config.GOOGLE_OAUTH_ENABLED:
            google_login_button()
        else:
            st.info(
                "**Google sign-in not configured.** Add `GOOGLE_CLIENT_ID` and "
                "`GOOGLE_CLIENT_SECRET` to `.env` to enable it (see README → *Google OAuth setup*).",
                icon="🔐",
            )

        st.markdown("")
        if st.button("👤 Continue as Guest (demo mode)", use_container_width=True):
            st.session_state[USER_KEY] = {"name": "Guest Analyst", "email": "", "picture": ""}
            st.rerun()

        st.markdown(
            '<div class="login-note" style="text-align:center">'
            f"Built by <b style='color:#cbd5e1'>{config.APP_AUTHOR}</b> · {config.APP_TAGLINE}"
            "</div>",
            unsafe_allow_html=True,
        )


def enforce_login() -> None:
    if not is_authenticated():
        render_login_page()
        st.stop()
