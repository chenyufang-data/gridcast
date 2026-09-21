"""Streamlit frontend: six views over the backend API plus a chat guide.

Run (two terminals):
    uvicorn app.main:app                      # backend on :8000
    streamlit run frontend/app.py             # UI on :8501

Layout: the sidebar holds the zone and view selectors and the served-model badge; the
page is the view; the chat guide sits behind a floating bubble (bottom right) that opens
a dialog window with a greeting, the message history, option pills (never a model call)
and a text box (Gemini on Vertex AI when configured and within the visitor's daily
limit, else the keyword guide; once the limit is hit the box locks and the guide points
at the options). A navigation reply closes the window and opens the view. Every action
is a labelled button; primary buttons are the ones that write (forecast, retrain, save,
backfill). This file imports nothing from ``app/``: the API is the only data source.

Widget state: ``session_state.view / zone / target / start / end / granularity`` are the
source of truth. Each widget has its own key, is seeded from that state right before it
is created, and writes back through an ``on_change`` callback, so chat intents and
buttons only ever touch the state and rerun.
"""

from __future__ import annotations

import sys
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitInvalidLayoutContextError

# `streamlit run` puts only this file's folder on sys.path, not the repo root
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from frontend import api, charts, llm, router  # noqa: E402
from frontend.api import ApiError, BackendDown  # noqa: E402
from frontend.router import VIEW_LABELS, VIEWS, ZONE_LABELS, ZONES  # noqa: E402

st.set_page_config(page_title="gridcast", page_icon="⚡", layout="wide")

ET = "America/New_York"
DATA_CREDIT = (
    "Data: NYISO public MIS archive (mis.nyiso.com/public/csv), fetched at runtime and "
    "not redistributed; weather: Open-Meteo (CC BY 4.0). Not affiliated with or endorsed "
    "by NYISO."
)
NAV = ["overview", *VIEWS]
NAV_LABELS = {"overview": "Overview", **VIEW_LABELS}
NAV_ICONS = {
    "overview": ":material/dashboard:",
    "forecast": ":material/query_stats:",
    "schedule": ":material/receipt_long:",
    "compare": ":material/fact_check:",
    "prices": ":material/attach_money:",
    "load": ":material/show_chart:",
}
OPTIONS: list[tuple[str, dict[str, Any]]] = [
    ("📋 All zones", {"action": "list_zones"}),
    ("🔮 Forecast", {"action": "show_zone", "view": "forecast"}),
    ("🧾 DAM schedule", {"action": "show_zone", "view": "schedule"}),
    ("📊 Forecast vs actual", {"action": "show_zone", "view": "compare"}),
    ("💲 Prices", {"action": "show_zone", "view": "prices"}),
    ("⚡ Load", {"action": "show_zone", "view": "load"}),
    ("❓ Help", {"action": "help"}),
]
OPTION_MAP = dict(OPTIONS)
NAVIGATING = ("list_zones", "show_zone")
MAX_BACKFILL_DAYS = 31
WELCOME = (
    "👋 Hi, I'm the gridcast guide. Tell me where you want to go in plain English, for "
    "example *how accurate was the Long Island forecast last week* or *NYC prices "
    "yesterday*, and I'll open that view for you. You can also choose one of the options "
    "below."
)
LIMIT_TEXT = (
    "⏸️ You've hit the chat limit: {reason}. Choose one of the options below (they never "
    "use the model), or close this window and use the menu on the left. Typed questions "
    "open again tomorrow."
)
PLACEHOLDER = 'Ask or navigate: "NYC forecast", "West prices last week"…'
PLACEHOLDER_LOCKED = "Daily chat limit reached: choose an option above"
PRESETS = {"Yesterday": 1, "Last 7 days": 7, "Last 30 days": 30}
BUBBLE_CSS = """
<style>
div.st-key-chat_bubble {
  position: fixed !important; right: 1.75rem; bottom: 1.75rem; z-index: 100;
  width: auto !important;
}
div.st-key-chat_bubble button {
  width: 3.5rem; height: 3.5rem; min-height: 3.5rem; border-radius: 999px; padding: 0;
  box-shadow: 0 6px 20px rgba(11, 11, 11, 0.28);
}
div.st-key-chat_bubble button span { font-size: 1.6rem; }
</style>
"""


# ─── helpers ──────────────────────────────────────────────────────────────────────────
def fmt_pct(x: float | None, digits: int = 2) -> str:
    return "—" if x is None else f"{x:.{digits}f} %"


def fmt_usd(x: float | None) -> str:
    return "—" if x is None else f"${x:,.0f}"


def fmt_mw(x: float | None) -> str:
    return "—" if x is None else f"{x:,.0f} MW"


def fmt_day(d: date) -> str:
    return f"{d:%a %b %d, %Y}"


def md(text: str) -> str:
    """Escape dollar signs for markdown: Streamlit renders a pair of them as LaTeX."""
    return text.replace("$", "\\$")


def model_kind(identity: str | None) -> str:
    if not identity:
        return "—"
    return "TFT" if identity.startswith("tft") else "Trees"


def model_badge(identity: str | None) -> str:
    if not identity:
        return ":gray-badge[no forecast]"
    if identity.startswith("tft"):
        return f":blue-badge[:material/verified: TFT · {identity.split(':')[1][:8]}]"
    return ":orange-badge[:material/forest: Trees (fallback)]"


def delta_vs_iso(ours: float | None, iso: float | None, unit: str = "") -> str | None:
    if ours is None or iso is None:
        return None
    return f"{ours - iso:+,.2f}{unit} vs NYISO"


def to_et(ts: pd.Series) -> pd.Series:
    """UTC ISO strings -> naive ET timestamps for plotting."""
    return pd.to_datetime(ts, utc=True).dt.tz_convert(ET).dt.tz_localize(None)


@st.cache_data(ttl=10, show_spinner=False)
def zones_cached() -> list[dict[str, Any]]:
    return api.client().zones()


@st.cache_data(ttl=30, show_spinner=False)
def health_cached() -> dict[str, Any]:
    return api.client().health()


def today_et() -> date:
    return datetime.fromisoformat(health_cached()["now_et"]).date()


def user_key() -> str:
    """Visitor identity for the chat limit: forwarded IP, else the session."""
    try:
        forwarded = st.context.headers.get("X-Forwarded-For")
        ip = forwarded.split(",")[0].strip() if forwarded else st.context.ip_address
    except Exception:
        ip = None
    return ip or str(st.session_state.user_key)


# ─── state and navigation ─────────────────────────────────────────────────────────────
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("view", "overview")
    ss.setdefault("zone", "N.Y.C.")
    ss.setdefault("target", None)
    ss.setdefault("start", None)
    ss.setdefault("end", None)
    ss.setdefault("granularity", "hour")
    ss.setdefault("chat", [("assistant", WELCOME)])
    ss.setdefault("chat_open", st.query_params.get("guide") == "open")
    ss.setdefault("limit_told", False)
    ss.setdefault("user_key", uuid.uuid4().hex)
    ss.setdefault("flash", None)
    ss.setdefault("sched_nonce", 0)


def sync(widget_key: str, state_key: str) -> None:
    """on_change callback: a widget wrote; copy its value into the state."""
    st.session_state[state_key] = st.session_state[widget_key]


def seed(widget_key: str, value: Any) -> None:
    """Set a widget's value right before it is created (never after)."""
    st.session_state[widget_key] = value


def go(view: str, zone: str | None = None, **state: Any) -> None:
    """Set the navigation state; the caller reruns."""
    ss = st.session_state
    ss.view = view
    if zone:
        ss.zone = zone
    for key, value in state.items():
        if value is not None:
            ss[key] = value


def _parse(d: str | None) -> date | None:
    return date.fromisoformat(d) if d else None


def apply_intent(intent: dict[str, Any]) -> str:
    """Translate an intent into navigation state; returns the assistant's reply."""
    ss = st.session_state
    action = intent.get("action")
    if action == "list_zones":
        go("overview")
        return "Showing every zone. Open one from its card."
    if action == "show_zone":
        zone = intent.get("zone") or ss.zone
        view = intent.get("view") or "forecast"
        target = _parse(intent.get("target"))
        start, end = _parse(intent.get("start")), _parse(intent.get("end"))
        if start and end:
            ss.start, ss.end = start, end
        if target and view in ("forecast", "schedule"):
            ss.target = target
        go(view, zone, granularity=intent.get("granularity"))
        reply = f"Opening **{VIEW_LABELS[view]}** for **{ZONE_LABELS[zone]}**"
        if start and end and view in ("compare", "prices", "load"):
            reply += f", {start:%b %d} – {end:%b %d}"
        elif target and view in ("forecast", "schedule"):
            reply += f" for {target:%b %d}"
        return reply + "."
    return str(intent.get("message") or router.HELP_TEXT)


def _on_zone() -> None:
    st.session_state.zone = st.session_state.zone_pick
    if st.session_state.view == "overview":
        st.session_state.view = "forecast"


def _open_chat() -> None:
    st.session_state.chat_open = True


def _close_chat() -> None:
    st.session_state.chat_open = False


def _on_option() -> None:
    """An option pill in the chat window: navigate (and close) or answer (stay open)."""
    ss = st.session_state
    label = ss.option_pick
    ss.option_pick = None
    if not label:
        return
    intent = dict(OPTION_MAP[label])
    reply = apply_intent(intent)
    ss.chat.append(("user", label))
    ss.chat.append(("assistant", reply))
    if intent["action"] in NAVIGATING:
        ss.chat_open = False
        ss.flash = ("toast", reply)


def _on_preset(key: str, today: date) -> None:
    ss = st.session_state
    choice = ss[f"{key}_preset"]
    ss[f"{key}_preset"] = None
    if choice in PRESETS:
        days = PRESETS[choice]
        ss.start, ss.end = today - timedelta(days=days), today - timedelta(days=1)


# ─── sidebar ──────────────────────────────────────────────────────────────────────────
def render_sidebar(health: dict[str, Any]) -> None:
    ss = st.session_state
    with st.sidebar:
        st.markdown("## ⚡ gridcast")
        st.caption("NYISO day-ahead load forecasts · bids close 05:00 ET on D−1")
        seed("zone_pick", ss.zone)
        st.selectbox(
            "Zone",
            ZONES,
            key="zone_pick",
            on_change=_on_zone,
            format_func=lambda z: f"{z} · {ZONE_LABELS[z]}",
        )
        seed("view_pick", ss.view)
        st.radio(
            "View",
            NAV,
            key="view_pick",
            on_change=sync,
            args=("view_pick", "view"),
            format_func=lambda v: f"{NAV_ICONS[v]} {NAV_LABELS[v]}",
        )
        st.divider()
        tft = health["models"]["tft"]
        st.markdown("**Served model**")
        if tft.get("loaded") and not tft.get("stale"):
            st.badge("TFT (ONNX)", icon=":material/verified:", color="blue")
            st.caption(
                f"{tft['version']}  \nfit cutoff {tft['fit_cutoff'][:10]} · "
                f"{tft.get('age_days') or 0:.0f} days old"
            )
        else:
            st.badge("LightGBM trees", icon=":material/forest:", color="orange")
            st.caption(f"TFT bundle: {tft.get('error') or 'stale'}")
        data = health["data"]["load_slots"]
        st.caption(f"Load data through {(data.get('last') or '—')[:16]} UTC")
        st.divider()
        st.caption(DATA_CREDIT)


# ─── chat guide: bubble + dialog window ───────────────────────────────────────────────
def render_bubble() -> None:
    """The floating button (bottom right) that opens the guide."""
    st.markdown(BUBBLE_CSS, unsafe_allow_html=True)
    with st.container(key="chat_bubble"):
        st.button(
            ":material/forum:",
            key="chat_bubble_btn",
            type="primary",
            on_click=_open_chat,
            help="Ask the gridcast guide",
        )


@st.dialog("gridcast guide", width="medium", icon=":material/forum:", on_dismiss=_close_chat)
def chat_dialog(today: date) -> None:
    """The message window. Runs as a fragment: typing reruns only this function."""
    ss = st.session_state
    if not ss.chat_open:  # an option pill navigated: close the window, show the view
        st.rerun()
    key = user_key()
    status = llm.status()
    allowed, reason = True, None
    if status["provider"]:
        allowed, reason = llm.limiter.check(key)
    if allowed:
        ss.limit_told = False
    elif not ss.limit_told:
        ss.limit_told = True
        ss.chat.append(("assistant", LIMIT_TEXT.format(reason=reason)))
    with st.container(height=380, border=False):
        for role, content in ss.chat:
            st.chat_message(role).markdown(md(content))  # model replies quote dollars too
    st.pills(
        "Options",
        list(OPTION_MAP),
        key="option_pick",
        on_change=_on_option,
        label_visibility="collapsed",
    )
    if status["provider"]:
        left = llm.limiter.remaining(key)
        st.caption(
            f"{status['model']} · {left} of {status['per_user_per_day']} model replies left "
            "today · options never count"
        )
    else:
        st.caption("Keyword guide (no chat model configured): name a zone and a view.")
    text = st.chat_input(
        PLACEHOLDER_LOCKED if not allowed else PLACEHOLDER, key="chat_text", disabled=not allowed
    )
    if not text:
        return
    ss.chat.append(("user", text))
    with st.spinner("Thinking…"):
        result = llm.chat(text, ss.chat[:-1], ZONES, key, today)
    if result.note:
        ss.chat.append(("assistant", f"_({result.note})_"))
    reply = apply_intent(result.intent)
    ss.chat.append(("assistant", reply))
    if result.intent.get("action") in NAVIGATING:
        ss.chat_open = False
        ss.flash = ("toast", reply)
        st.rerun()
    try:  # an answer: redraw only the window (a fragment rerun, which only exists while
        st.rerun(scope="fragment")  # the frontend runs the fragment; a full run of the
    except StreamlitInvalidLayoutContextError:  # script, as in tests, takes the long way)
        st.rerun()


# ─── overview ─────────────────────────────────────────────────────────────────────────
def render_overview(health: dict[str, Any]) -> None:
    cards = zones_cached()
    tft = health["models"]["tft"]
    served = "TFT (ONNX)" if tft.get("loaded") and not tft.get("stale") else "LightGBM trees"
    st.markdown("## Zone overview")
    st.caption(
        f"Served model: **{served}** · next bid day {health['next_target']} · 7-day figures "
        "are hourly MAPE and settled dollars next to NYISO's own pre-close forecast."
    )
    scored = [c for c in cards if c["last_7d"]["mape_hour"] is not None and not c["is_total"]]
    ours = float(np.mean([c["last_7d"]["mape_hour"] for c in scored])) if scored else None
    isos = [c["last_7d"]["isolf_mape_hour"] for c in scored]
    isos = [v for v in isos if v is not None]
    iso = float(np.mean(isos)) if isos else None
    usd = sum(c["last_7d"]["imbalance_usd"] or 0 for c in scored) if scored else None
    k = st.columns(4)
    k[0].metric(
        "7-day MAPE, all zones", fmt_pct(ours), delta=delta_vs_iso(ours, iso), delta_color="inverse"
    )
    k[1].metric("NYISO pre-close, same days", fmt_pct(iso))
    k[2].metric("7-day imbalance cost", fmt_usd(usd))
    k[3].metric("Zones with alerts", sum(1 for c in cards if c["alert"]))

    for row in range(0, len(cards), 3):
        cols = st.columns(3)
        for col, card in zip(cols, cards[row : row + 3], strict=False):
            with col, st.container(border=True):
                st.markdown(f"**{card['zone']}** · {ZONE_LABELS.get(card['zone'], '')}")
                week, latest = card["last_7d"], card["latest_forecast"]
                m = st.columns(2)
                m[0].metric(
                    "7-day MAPE",
                    fmt_pct(week["mape_hour"]),
                    delta=delta_vs_iso(week["mape_hour"], week["isolf_mape_hour"]),
                    delta_color="inverse",
                )
                m[1].metric(
                    "7-day imbalance",
                    "n/a" if card["is_total"] else fmt_usd(week["imbalance_usd"]),
                )
                if latest:
                    badge = model_badge(latest["model"])
                    st.markdown(f"Latest forecast **{latest['target_date']}** {badge}")
                else:
                    st.caption("No forecast stored yet")
                for a in card["alerts"][:2]:
                    st.markdown(
                        f":red-badge[:material/warning: {a['kind']}] {a['target_date']}: "
                        f"{md(a['message'])}"
                    )
                if st.button(
                    f"Open {card['zone']}",
                    key=f"open_{card['zone']}",
                    type="primary",
                    icon=":material/arrow_forward:",
                    width="stretch",
                ):
                    go("forecast", card["zone"])
                    st.rerun()


# ─── forecast view ────────────────────────────────────────────────────────────────────
def _hourly(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Mean per ET hour of the MW columns (MW is a rate, so a mean, not a sum)."""
    out = df.groupby(df["ts"].dt.floor("h"))[cols].mean()
    return out.rename_axis("ts").reset_index()


def _forecast_frame(fc: dict[str, Any], granularity: str) -> pd.DataFrame:
    df = pd.DataFrame(fc["values"])
    df["ts"] = to_et(df["ts_utc"])
    cols = [
        c
        for c in ("predicted", "p10", "p90", "p_alpha", "actual", "isolf_pre", "isolf_post")
        if c in df
    ]
    return _hourly(df, cols) if granularity == "hour" else df[["ts", *cols]]


def _version_label(v: dict[str, Any]) -> str:
    """One legend/pill name per stored version: model kind, hash and time made."""
    return f"{model_kind(v['model'])} · {v['model_version'][:8]} · {v['created_at'][:16]} UTC"


def _version_role(v: dict[str, Any]) -> str:
    if v["primary"]:
        return "primary (served model)"
    if v.get("requested") == "auto":
        return "earlier served version"
    return f"manual ({v.get('requested') or 'unknown'})"


def _versions_table(versions: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "Version": _version_label(v),
            "Role": _version_role(v),
            "Hourly MAPE %": v.get("mape_hour"),
            "NYISO MAPE %": v.get("isolf_mape_hour"),
            "Imbalance $": v.get("imbalance_usd"),
            "α-bid $": v.get("imbalance_alpha_usd"),
            "Inside band %": v.get("band_coverage"),
        }
        for v in versions
    ]
    return pd.DataFrame(rows).round(2)


def _run_forecast(zone: str, target: date | None, model: str, label: str) -> None:
    with st.spinner(f"{label}…"):
        try:
            fc = api.client().create_forecast(zone, target, model)
        except ApiError as exc:
            st.session_state.flash = ("error", f"Forecast failed: {exc.detail}")
        else:
            why = f" ({fc['fallback_reason']})" if fc.get("fallback_reason") else ""
            role = (
                ""
                if fc.get("primary", True)
                else " as an extra version: the served model's forecast stays the day's primary"
            )
            st.session_state.flash = (
                "success",
                f"{'Stored' if fc['new'] else 'Already stored'}: {fc['target_date']} with "
                f"{model_kind(fc['model'])}{why}, version {fc['model_version']}{role}.",
            )
            st.session_state.target = date.fromisoformat(fc["target_date"])
    zones_cached.clear()
    st.rerun()


def _score_row(score: dict[str, Any]) -> None:
    m = st.columns(4)
    m[0].metric(
        "Hourly MAPE",
        fmt_pct(score["mape_hour"]),
        delta=delta_vs_iso(score["mape_hour"], score["isolf_mape_hour"]),
        delta_color="inverse",
        help=f"NYISO pre-close forecast on the same day: {fmt_pct(score['isolf_mape_hour'])}",
    )
    m[1].metric(
        "Imbalance cost (median bid)",
        fmt_usd(score["imbalance_usd"]),
        delta=delta_vs_iso(score["imbalance_usd"], score["isolf_imbalance_usd"], " $"),
        delta_color="inverse",
    )
    m[2].metric("Imbalance cost (α-bid)", fmt_usd(score["imbalance_alpha_usd"]))
    m[3].metric(
        "Inside P10–P90",
        fmt_pct(score["band_coverage"], 0),
        help="Share of 15-min slots whose actual load fell inside the band (nominal 80 %).",
    )


def render_forecast(zone: str, health: dict[str, Any]) -> None:
    ss = st.session_state
    st.markdown(f"## Forecast · {zone} <small>{ZONE_LABELS[zone]}</small>", unsafe_allow_html=True)
    stored = api.client().forecasts(zone)
    dates = [date.fromisoformat(f["target_date"]) for f in stored]
    tomorrow = date.fromisoformat(health["next_target"])
    options = sorted(set(dates) | {tomorrow}, reverse=True)
    if ss.target not in options:
        ss.target = tomorrow if tomorrow in dates or not dates else dates[0]
    top = st.columns([1.3, 1.2, 1.4, 1.6], vertical_alignment="bottom")
    seed("fc_target", ss.target)
    top[0].selectbox(
        "Target day", options, key="fc_target", on_change=sync, args=("fc_target", "target"),
        format_func=fmt_day,
    )  # fmt: skip
    seed("fc_gran", "slot" if ss.granularity == "slot" else "hour")
    top[1].segmented_control(
        "Resolution",
        ["hour", "slot"],
        key="fc_gran",
        on_change=sync,
        args=("fc_gran", "granularity"),
        format_func=lambda g: "Hourly" if g == "hour" else "15-min",
    )
    if top[2].button(
        f"Forecast {tomorrow:%b %d} (served model)",
        type="primary",
        icon=":material/rocket_launch:",
        width="stretch",
        help="Runs the served model (the TFT when its bundle is valid) for the next bid day "
        "and stores the result.",
    ):
        _run_forecast(zone, tomorrow, "auto", "Forecasting with the served model")
    if top[3].button(
        "Retrain trees for this day",
        icon=":material/model_training:",
        width="stretch",
        help="Trains LightGBM on the last 365 days as of this day's cutoff, live, and stores "
        "a new version (about a minute).",
    ):
        _run_forecast(zone, ss.target, "lgbm", "Training the trees")

    if ss.target not in dates:
        st.info(f"No forecast stored for {fmt_day(ss.target)} yet.", icon=":material/info:")
        if st.button(
            f"Forecast {ss.target:%b %d} now", type="primary", icon=":material/rocket_launch:"
        ):
            _run_forecast(zone, ss.target, "auto", "Forecasting")
        return
    fc = api.client().forecast(zone, ss.target)
    gran = "slot" if ss.granularity == "slot" else "hour"
    df = _forecast_frame(fc, gran)
    versions = fc.get("versions") or []
    others = [v for v in versions if not v["primary"]]
    overlays: list[tuple[str, pd.DataFrame, str]] = []
    if others:
        labels = {_version_label(v): v for v in others}
        chosen = st.pills(
            "Also show",
            list(labels),
            selection_mode="multi",
            default=list(labels),
            key=f"fc_versions_{zone}_{ss.target}_{len(versions)}",
            help="Other versions stored for this day (a live retrain, an older bundle). The "
            "blue line is the day's primary: the served model's forecast. Click a legend "
            "entry to hide a line.",
        )
        for i, (label, v) in enumerate(labels.items()):
            if label in (chosen or []):
                extra = api.client().forecast(zone, ss.target, version=v["model_version"])
                color = charts.OVERLAY_COLORS[i % len(charts.OVERLAY_COLORS)]
                overlays.append((label, _forecast_frame(extra, gran), color))
    st.plotly_chart(
        charts.forecast_figure(df, overlays=overlays), width="stretch", config=charts.CONFIG
    )
    more = f" · primary of {len(versions)} versions" if len(versions) > 1 else ""
    st.markdown(
        f"{model_badge(fc['model'])} version `{fc['model_version']}`{more} · made "
        f"{fc['created_at']} UTC as of the {fc['cutoff_utc'][:16]} UTC cutoff · "
        f"α = {fc['alpha']:.2f} · band scale ×{fc['band_scale_p10']:.2f} / "
        f"×{fc['band_scale_p90']:.2f} · peak {fmt_mw(df['predicted'].max())}"
    )
    if fc.get("score"):
        _score_row(fc["score"])
    else:
        st.caption("Not scored yet: actual load and prices arrive the next morning.")
    if len(versions) > 1:
        st.markdown("**Every version of this day** · scored on the same actuals")
        st.dataframe(_versions_table(versions), width="stretch", hide_index=True)
    table = df
    for label, frame, _ in overlays:
        table = table.merge(
            frame[["ts", "predicted"]].rename(columns={"predicted": f"predicted · {label}"}),
            on="ts",
            how="left",
        )
    with st.expander("Table view"):
        st.dataframe(table.round(1), width="stretch", hide_index=True)
    scores = api.client().scores(zone, limit=30)
    if scores:
        with st.expander(f"Accuracy history · last {len(scores)} scored days"):
            hist = pd.DataFrame(scores)
            hist["date"] = pd.to_datetime(hist["target_date"])
            hist = hist.sort_values("date")
            st.plotly_chart(charts.mape_history_figure(hist), width="stretch", config=charts.CONFIG)
            cols = ["target_date", "model", "mape_hour", "isolf_mape_hour", "imbalance_usd",
                    "isolf_imbalance_usd", "band_coverage"]  # fmt: skip
            st.dataframe(hist[cols].round(2), width="stretch", hide_index=True)


# ─── DAM schedule view ────────────────────────────────────────────────────────────────
def _hourly_from_forecast(fc: dict[str, Any]) -> pd.DataFrame:
    vals = pd.DataFrame(fc["values"])
    vals["ts"] = to_et(vals["ts_utc"])
    hours = _hourly(vals, ["predicted", "p10", "p90", "p_alpha"])
    hours["label"] = hours["ts"].dt.strftime("%H:%M")
    return hours


def _schedule_csv(zone: str, target: date, hourly: pd.DataFrame) -> bytes:
    out = pd.DataFrame(
        {
            "zone": zone,
            "target_date": target.isoformat(),
            "hour_et": hourly["label"],
            "forecast_mw": hourly["predicted"].round(2),
            "alpha_bid_mw": hourly["p_alpha"].round(2),
            "bid_mw": hourly["bid_mw"].round(2),
        }
    )
    return out.to_csv(index=False).encode("utf-8")


def _set_bids(values: list[float]) -> None:
    st.session_state.sched_bids = [round(float(v), 1) for v in values]
    st.session_state.sched_nonce += 1


def render_schedule(zone: str, health: dict[str, Any]) -> None:
    ss = st.session_state
    st.markdown(
        f"## DAM schedule · {zone} <small>{ZONE_LABELS[zone]}</small>", unsafe_allow_html=True
    )
    stored = api.client().forecasts(zone)
    if not stored:
        st.info("No forecast stored for this zone yet. Make one on the Forecast view first.",
                icon=":material/info:")  # fmt: skip
        if st.button("Go to Forecast", type="primary", icon=":material/arrow_forward:"):
            go("forecast", zone)
            st.rerun()
        return
    dates = [date.fromisoformat(f["target_date"]) for f in stored]
    if ss.target not in dates:
        ss.target = dates[0]
    top = st.columns([1.3, 2.7], vertical_alignment="bottom")
    seed("sched_date", ss.target)
    top[0].selectbox(
        "Bid day", dates, key="sched_date", on_change=sync, args=("sched_date", "target"),
        format_func=fmt_day,
    )  # fmt: skip
    fc = api.client().forecast(zone, ss.target)
    stats = api.client().alpha(zone, ss.target)
    top[1].caption(  # dollar signs are escaped: a pair of them would render as LaTeX
        f"One MW bid per hour, settled at RT − DA. Trailing 30 days: α = {stats['alpha']:.2f} "
        f"(short costs {stats['c_under_usd_per_mwh'] or 0:.2f} \\$/MWh, long costs "
        f"{stats['c_over_usd_per_mwh'] or 0:.2f} \\$/MWh). Forecast {model_badge(fc['model'])}"
    )
    hourly = _hourly_from_forecast(fc)
    sig = f"{zone}|{ss.target}|{fc['model_version']}"
    if ss.get("sched_sig") != sig:
        ss.sched_sig = sig
        _set_bids(hourly["p_alpha"].tolist())

    left, right = st.columns([1.1, 1.6], gap="medium")
    with left:
        st.markdown("**Start from**")
        b = st.columns(2)
        if b[0].button("α-bid", icon=":material/functions:", width="stretch",
                       help="The cost-aware quantile bid."):  # fmt: skip
            _set_bids(hourly["p_alpha"].tolist())
            st.rerun()
        if b[1].button("Model median", icon=":material/timeline:", width="stretch"):
            _set_bids(hourly["predicted"].tolist())
            st.rerun()
        pct_row = st.columns([1.4, 1.2], vertical_alignment="bottom")
        pct = pct_row[0].number_input(
            "Scale the model by (%)", min_value=-50.0, max_value=50.0, value=0.0, step=0.5,
            format="%.1f",
        )  # fmt: skip
        if pct_row[1].button("Apply", icon=":material/percent:", width="stretch"):
            _set_bids((hourly["predicted"] * (1 + pct / 100)).tolist())
            st.rerun()
        editor = pd.DataFrame(
            {
                "Hour (ET)": hourly["label"],
                "Forecast MW": hourly["predicted"].round(1),
                "Bid MW": ss.sched_bids,
            }
        )
        edited = st.data_editor(
            editor,
            hide_index=True,
            width="stretch",
            height=420,
            key=f"sched_editor_{sig}_{ss.sched_nonce}",
            column_config={
                "Hour (ET)": st.column_config.TextColumn(disabled=True),
                "Forecast MW": st.column_config.NumberColumn(disabled=True, format="%.1f"),
                "Bid MW": st.column_config.NumberColumn(
                    min_value=0.0, format="%.1f", help="Edit any hour"
                ),
            },
        )
        bids = edited["Bid MW"].astype(float).fillna(0.0).to_numpy()
        ss.sched_bids = bids.round(1).tolist()

    hourly["bid_mw"] = bids
    total_bid, total_fc = float(bids.sum()), float(hourly["predicted"].sum())
    dev_mwh = float(np.abs(bids - hourly["predicted"].to_numpy()).sum())
    spread = stats.get("mean_abs_spread_usd_per_mwh")
    outside = int(((bids < hourly["p10"]) | (bids > hourly["p90"])).sum())
    with right:
        m = st.columns(3)
        m[0].metric(
            "Total bid (MWh)",
            f"{total_bid:,.0f}",
            delta=f"{(total_bid / total_fc - 1) * 100:+.1f} % vs model" if total_fc else None,
        )
        m[1].metric(
            "$ at risk",
            fmt_usd(dev_mwh * spread) if spread is not None else "—",
            help="Your deviation from the model in MWh × the trailing mean |RT − DA|: what "
            "being wrong about the model would cost.",
        )
        m[2].metric("Hours outside P10–P90", outside)
        if outside:
            st.warning(
                f"{outside} hour(s) bid outside the model's 80 % band.", icon=":material/warning:"
            )
        st.plotly_chart(charts.schedule_figure(hourly), width="stretch", config=charts.CONFIG)
        note = st.text_input(
            "Note (optional)", key=f"sched_note_{sig}",
            placeholder="e.g. heat advisory, bid above the model in the afternoon",
        )  # fmt: skip
        if st.button("Save schedule", type="primary", icon=":material/save:", width="stretch"):
            try:
                res = api.client().create_schedule(
                    zone, ss.target, [float(v) for v in bids], note.strip() or None,
                    {"source": "editor", "scale_pct": pct},
                )  # fmt: skip
            except ApiError as exc:
                st.error(f"Could not save: {exc.detail}")
            else:
                ss.flash = (
                    "success",
                    md(
                        f"Saved schedule #{res['schedule_id']} for {res['target_date']}: "
                        f"{res['total_bid_mwh']:,.0f} MWh, $ at risk "
                        f"{fmt_usd(res['usd_at_risk'])}."
                    ),
                )
                st.rerun()

    try:
        saved = api.client().schedule(zone, ss.target)
    except ApiError:
        saved = None
    if saved:
        st.divider()
        st.markdown(f"#### Saved schedule #{saved['schedule_id']}")
        st.caption(
            f"saved {saved['created_at']} UTC on forecast version `{saved['model_version']}` · "
            f"{saved['total_bid_mwh']:,.0f} MWh ({saved['delta_pct']:+.1f} % vs model)"
            + (f" · note: {saved['note']}" if saved.get("note") else "")
        )
        sc, fsc = saved.get("score"), saved.get("forecast_score")
        if sc and sc.get("mape_hour") is not None:
            model_mape = fsc.get("mape_hour") if fsc else None
            better = model_mape is None or sc["mape_hour"] <= model_mape
            icon = ":material/check_circle:" if better else ":material/warning:"
            st.markdown(
                md(
                    f"{icon} Scored: your schedule {fmt_pct(sc['mape_hour'])} hourly MAPE, "
                    f"{fmt_usd(sc['imbalance_usd'])} imbalance · model median "
                    f"{fmt_pct(model_mape)}, {fmt_usd(fsc['imbalance_usd']) if fsc else '—'}"
                )
            )
        csv_frame = hourly.copy()
        csv_frame["bid_mw"] = pd.DataFrame(saved["hourly"])["bid_mw"].to_numpy()
        st.download_button(
            "Download CSV", _schedule_csv(zone, ss.target, csv_frame),
            file_name=f"schedule_{zone.replace(' ', '_')}_{ss.target}.csv", mime="text/csv",
            icon=":material/download:",
        )  # fmt: skip
    history = api.client().schedules(zone)
    if history:
        with st.expander(f"Schedule history · {len(history)}"):
            cols = ["schedule_id", "target_date", "created_at", "mape_hour", "model_mape_hour",
                    "imbalance_usd", "model_imbalance_usd", "note"]  # fmt: skip
            st.dataframe(pd.DataFrame(history)[cols].round(2), width="stretch", hide_index=True)


# ─── range views: compare, prices, load ───────────────────────────────────────────────
def range_controls(
    key: str, today: date, granularities: tuple[str, ...] | None
) -> tuple[date, date, str]:
    ss = st.session_state
    ss.end = ss.end or (today - timedelta(days=1))
    ss.start = ss.start or (ss.end - timedelta(days=6))
    c = st.columns([1, 1, 1.6, 1.4], vertical_alignment="bottom")
    seed(f"{key}_start", ss.start)
    c[0].date_input("From", key=f"{key}_start", on_change=sync, args=(f"{key}_start", "start"))
    seed(f"{key}_end", ss.end)
    c[1].date_input("To", key=f"{key}_end", on_change=sync, args=(f"{key}_end", "end"))
    c[2].pills(
        "Presets", list(PRESETS), key=f"{key}_preset", on_change=_on_preset, args=(key, today),
        label_visibility="collapsed",
    )  # fmt: skip
    gran = ss.granularity
    if granularities:
        gran = gran if gran in granularities else granularities[0]
        seed(f"{key}_gran", gran)
        c[3].segmented_control(
            "Resolution", list(granularities), key=f"{key}_gran", on_change=sync,
            args=(f"{key}_gran", "granularity"), format_func=str.title,
        )  # fmt: skip
    start, end = ss.start, ss.end
    if start > end:
        st.error("The start date is after the end date.")
        st.stop()
    return start, end, gran


def _backfill(zone: str, days: list[date]) -> None:
    prog = st.progress(0.0, text="Forecasting…")
    errors = []
    for i, d in enumerate(days):
        try:
            api.client().create_forecast(zone, d, "auto")
        except ApiError as exc:
            errors.append(f"{d}: {exc.detail}")
        prog.progress((i + 1) / len(days), text=f"Forecasting {d} ({i + 1}/{len(days)})")
    prog.empty()
    st.session_state.flash = (
        "warning" if errors else "success",
        "; ".join(errors)
        if errors
        else f"Forecast {len(days)} day(s) with the served model; scoring runs each morning.",
    )
    st.rerun()


def render_compare(zone: str, today: date) -> None:
    st.markdown(
        f"## Forecast vs actual · {zone} <small>{ZONE_LABELS[zone]}</small>", unsafe_allow_html=True
    )
    start, end, gran = range_controls("cmp", today, ("slot", "hour", "day"))
    data = api.client().compare(zone, start, end, gran)
    stored = {f["target_date"] for f in api.client().forecasts(zone)}
    missing = [
        d for d in pd.date_range(start, end).date if d.isoformat() not in stored and d < today
    ]
    if missing:
        mc = st.columns([3, 1.2], vertical_alignment="center")
        shown = ", ".join(d.strftime("%b %d") for d in missing[:6]) + (
            "…" if len(missing) > 6 else ""
        )
        mc[0].warning(
            f"{len(missing)} day(s) in this range have no stored forecast: {shown}",
            icon=":material/event_busy:",
        )
        if len(missing) <= MAX_BACKFILL_DAYS and mc[1].button(
            f"Backfill {len(missing)} day(s)",
            type="primary",
            icon=":material/history:",
            width="stretch",
            help="Forecasts each missing day as of its own cutoff with the served model.",
        ):
            _backfill(zone, missing)
    if not data["points"]:
        st.info("No forecasts in this range yet.", icon=":material/info:")
        return
    s = data["summary"]
    m = st.columns(4)
    m[0].metric(
        "Hourly MAPE",
        fmt_pct(s["mape_hour"]),
        delta=delta_vs_iso(s["mape_hour"], s["isolf_mape_hour"]),
        delta_color="inverse",
        help=f"NYISO pre-close forecast on the same days: {fmt_pct(s['isolf_mape_hour'])}",
    )
    m[1].metric(
        "Imbalance cost (median bid)",
        fmt_usd(s["imbalance_usd"]),
        delta=delta_vs_iso(s["imbalance_usd"], s["isolf_imbalance_usd"], " $"),
        delta_color="inverse",
    )
    m[2].metric("Imbalance cost (α-bid)", fmt_usd(s["imbalance_alpha_usd"]))
    m[3].metric("Days", s["days"])
    df = pd.DataFrame(data["points"])
    df["ts"] = to_et(df["ts_utc"])
    st.plotly_chart(charts.forecast_figure(df), width="stretch", config=charts.CONFIG)
    if df["imbalance_usd"].notna().any():
        st.markdown(
            "**Imbalance dollars per bucket** · cost above zero, gain below "
            "(median bid settled at RT − DA)"
        )
        st.plotly_chart(
            charts.dollars_figure(df, "imbalance_usd"),
            width="stretch",
            config=charts.CONFIG,
        )
    st.caption(
        f"Models in this range: {', '.join(s['models'])}. Hourly MAPE uses complete hours; "
        "dollars settle each 15-min slot."
    )
    with st.expander("Table view"):
        st.dataframe(df.drop(columns=["ts_utc"]).round(2), width="stretch", hide_index=True)


def render_prices(zone: str, today: date) -> None:
    st.markdown(f"## Prices · {zone} <small>{ZONE_LABELS[zone]}</small>", unsafe_allow_html=True)
    if zone == "NYCA":
        st.info("NYCA is the statewide total and has no zonal price. Pick a zone.",
                icon=":material/info:")  # fmt: skip
        return
    start, end, _ = range_controls("prc", today, None)
    stats = api.client().alpha(zone, today + timedelta(days=1))
    m = st.columns(4)
    m[0].metric(
        "α (bid quantile)",
        f"{stats['alpha']:.2f}",
        help="Newsvendor ratio of the trailing 30-day spread as of tomorrow's cutoff: above "
        "0.5 bids high.",
    )
    m[1].metric("Cost of being short", f"{stats['c_under_usd_per_mwh'] or 0:.2f} $/MWh")
    m[2].metric("Cost of being long", f"{stats['c_over_usd_per_mwh'] or 0:.2f} $/MWh")
    m[3].metric("Mean |RT − DA|", f"{stats['mean_abs_spread_usd_per_mwh'] or 0:.2f} $/MWh")
    data = api.client().prices(zone, start, end)
    if not data["points"]:
        st.info("No prices in this range.", icon=":material/info:")
        return
    df = pd.DataFrame(data["points"])
    df["ts"] = to_et(df["ts_utc"])
    st.plotly_chart(charts.prices_figure(df), width="stretch", config=charts.CONFIG)
    st.markdown(
        "**RT − DA spread** · positive hours punish under-bidding, negative hours punish "
        "over-bidding"
    )
    st.plotly_chart(charts.spread_figure(df), width="stretch", config=charts.CONFIG)
    with st.expander("Table view"):
        st.dataframe(df.drop(columns=["ts_utc"]).round(2), width="stretch", hide_index=True)


def render_load(zone: str, today: date) -> None:
    st.markdown(f"## Load · {zone} <small>{ZONE_LABELS[zone]}</small>", unsafe_allow_html=True)
    start, end, gran = range_controls("load", today, ("slot", "hour", "day"))
    data = api.client().load(zone, start, end, gran)
    if not data["points"]:
        st.info("No load data in this range.", icon=":material/info:")
        return
    df = pd.DataFrame(data["points"])
    df["ts"] = to_et(df["ts_utc"])
    m = st.columns(3)
    m[0].metric("Peak", fmt_mw(df["actual"].max()))
    m[1].metric("Average", fmt_mw(df["actual"].mean()))
    m[2].metric("Points", len(df))
    st.plotly_chart(charts.load_figure(df), width="stretch", config=charts.CONFIG)
    with st.expander("Table view"):
        st.dataframe(df.drop(columns=["ts_utc"]).round(1), width="stretch", hide_index=True)


# ─── page ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    init_state()
    try:
        health = health_cached()
    except BackendDown:
        st.error(
            f"The backend is not reachable at {api.client().base_url}. Start it with "
            "`uvicorn app.main:app`.",
            icon=":material/cloud_off:",
        )
        return
    render_sidebar(health)
    today = today_et()
    if st.session_state.chat_open:
        chat_dialog(today)
    flash = st.session_state.flash
    if flash:
        st.session_state.flash = None
        if flash[0] == "toast":
            st.toast(flash[1], icon=":material/forum:")
        else:
            getattr(st, flash[0])(flash[1])
    view, zone = st.session_state.view, st.session_state.zone
    try:
        if view == "overview":
            render_overview(health)
        elif view == "forecast":
            render_forecast(zone, health)
        elif view == "schedule":
            render_schedule(zone, health)
        elif view == "compare":
            render_compare(zone, today)
        elif view == "prices":
            render_prices(zone, today)
        else:
            render_load(zone, today)
    except BackendDown:
        st.error("The backend went away; retry in a moment.", icon=":material/cloud_off:")
    except ApiError as exc:
        st.error(f"API error {exc.status}: {exc.detail}", icon=":material/error:")
    st.caption(DATA_CREDIT)
    render_bubble()


main()
