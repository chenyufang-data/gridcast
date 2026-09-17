"""Streamlit frontend: a pure HTTP client of the backend API (imports nothing from app/).

Phase 0 is a status page that pings the backend. Phase 4 ports the six views (zone
cards, Forecast, DAM Schedule, Forecast vs Actual, Prices, Load) and the chat panel.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")
DATA_CREDIT = (
    "Data: NYISO public MIS archive (http://mis.nyiso.com/public/csv/), fetched at "
    "runtime and not redistributed. This site is not affiliated with or endorsed by NYISO."
)

st.set_page_config(page_title="gridcast", page_icon="⚡", layout="wide")
st.title("⚡ gridcast — NYISO day-ahead load forecasting")

try:
    resp = requests.get(f"{API_BASE}/health", timeout=5)
    resp.raise_for_status()
    st.success(f"Backend OK (version {resp.json().get('version', '?')})")
except requests.RequestException as exc:
    st.error(f"Backend unreachable at {API_BASE}: {exc}")

st.caption(DATA_CREDIT)
