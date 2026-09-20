"""Plotly figures for the views, on one validated palette.

Palette: the reference instance of the data-viz method (light mode; the app forces the
light theme). Series hues are assigned by entity and never change with what is on
screen: the model forecast is slot 1 (blue), the actual load slot 2 (orange), NYISO's
own forecast slot 3 (aqua), the visitor's bid slot 4 (yellow, dotted); dollars use the
diverging blue/red pair around zero; status colors are reserved for alerts. Lines are
2 px, area washes ~10 % opacity, gridlines hairline and solid, one y axis per chart,
one unified crosshair tooltip, and every chart ships a table twin in the app.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

FORECAST = "#2a78d6"  # slot 1 blue: our median forecast, the alpha-bid (dotted) and the band
ACTUAL = "#eb6834"  # slot 2 orange: the actual load
ISO = "#1baf7a"  # slot 3 aqua: NYISO's pre-close forecast
BID = "#eda100"  # slot 4 yellow: the visitor's DAM bid
DA_PRICE = "#2a78d6"
RT_PRICE = "#eb6834"
COST = "#e34948"  # diverging warm pole: money lost
GAIN = "#2a78d6"  # diverging cool pole: money gained
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BAND_FILL = "rgba(42,120,214,0.12)"
AREA_FILL = "rgba(235,104,52,0.10)"

CONFIG = {"displayModeBar": False, "responsive": True}


FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _layout(fig: go.Figure, ytitle: str, height: int = 400, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 8, "r": 8, "t": 36 if legend else 12, "b": 8},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": FONT, "color": INK_SECONDARY, "size": 12},
        hovermode="x unified",
        hoverlabel={"bgcolor": "#ffffff", "bordercolor": GRID, "font": {"color": INK}},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        showlegend=legend,
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=AXIS,
        tickfont={"color": MUTED},
        showspikes=True,
        spikemode="across",
        spikethickness=1,
        spikecolor=AXIS,
        spikedash="solid",
    )
    fig.update_yaxes(
        title=ytitle,
        gridcolor=GRID,
        gridwidth=1,
        zeroline=False,
        linecolor=AXIS,
        tickfont={"color": MUTED},
        tickformat=",.0f",
    )
    return fig


def _line(x: pd.Series, y: pd.Series, name: str, color: str, dash: str = "solid") -> go.Scatter:
    return go.Scatter(
        x=x,
        y=y,
        name=name,
        mode="lines",
        line={"color": color, "width": 2, "dash": dash, "shape": "linear"},
        connectgaps=False,
    )


def _band(x: pd.Series, lo: pd.Series, hi: pd.Series, name: str) -> list[go.Scatter]:
    return [
        go.Scatter(x=x, y=hi, mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"),
        go.Scatter(
            x=x,
            y=lo,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=BAND_FILL,
            name=name,
            hoverinfo="skip",
        ),
    ]


def forecast_figure(
    df: pd.DataFrame,
    *,
    x: str = "ts",
    show_actual: bool = True,
    show_iso: bool = True,
    show_alpha: bool = True,
    height: int = 420,
) -> go.Figure:
    """Median + P10-P90 band, alpha-bid, ISO forecast and actual load, one MW axis."""
    fig = go.Figure()
    if {"p10", "p90"} <= set(df.columns):
        fig.add_traces(_band(df[x], df["p10"], df["p90"], "P10–P90 band"))
    fig.add_trace(_line(df[x], df["predicted"], "Forecast (median)", FORECAST))
    if show_alpha and "p_alpha" in df.columns and not df["p_alpha"].equals(df["predicted"]):
        fig.add_trace(_line(df[x], df["p_alpha"], "α-bid", FORECAST, dash="dot"))
    if show_iso and "isolf_pre" in df.columns and df["isolf_pre"].notna().any():
        fig.add_trace(_line(df[x], df["isolf_pre"], "NYISO pre-close forecast", ISO))
    if show_actual and "actual" in df.columns and df["actual"].notna().any():
        fig.add_trace(_line(df[x], df["actual"], "Actual load", ACTUAL))
    return _layout(fig, "MW", height)


def dollars_figure(df: pd.DataFrame, col: str, *, x: str = "ts", height: int = 260) -> go.Figure:
    """Imbalance dollars per bucket: cost above zero (warm), gain below (cool)."""
    vals = df[col].to_numpy(dtype=float)
    colors = np.where(vals >= 0, COST, GAIN)
    fig = go.Figure(
        go.Bar(
            x=df[x],
            y=vals,
            name="Imbalance $ (median bid)",
            marker={"color": colors, "cornerradius": 4},
            hovertemplate="%{y:$,.0f}<extra></extra>",
        )
    )
    fig.update_layout(bargap=0.35)
    fig = _layout(fig, "$", height, legend=False)
    fig.update_yaxes(tickformat="$,.0f", zeroline=True, zerolinecolor=AXIS, zerolinewidth=1)
    return fig


def prices_figure(df: pd.DataFrame, *, x: str = "ts", height: int = 360) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(_line(df[x], df["p_da"], "Day-ahead LBMP", DA_PRICE))
    fig.add_trace(_line(df[x], df["p_rt"], "Real-time LBMP", RT_PRICE))
    fig = _layout(fig, "$/MWh", height)
    fig.update_yaxes(tickformat="$,.0f")
    return fig


def spread_figure(df: pd.DataFrame, *, x: str = "ts", height: int = 220) -> go.Figure:
    vals = df["spread"].to_numpy(dtype=float)
    fig = go.Figure(
        go.Bar(
            x=df[x],
            y=vals,
            name="RT − DA spread",
            marker={"color": np.where(vals >= 0, COST, GAIN), "cornerradius": 4},
            hovertemplate="%{y:$,.2f}/MWh<extra></extra>",
        )
    )
    fig.update_layout(bargap=0.35)
    fig = _layout(fig, "$/MWh", height, legend=False)
    fig.update_yaxes(tickformat="$,.0f", zeroline=True, zerolinecolor=AXIS, zerolinewidth=1)
    return fig


def load_figure(df: pd.DataFrame, *, x: str = "ts", height: int = 400) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=df[x],
            y=df["actual"],
            name="Actual load",
            mode="lines",
            line={"color": ACTUAL, "width": 2},
            fill="tozeroy",
            fillcolor=AREA_FILL,
        )
    )
    return _layout(fig, "MW", height, legend=False)


def schedule_figure(hourly: pd.DataFrame, height: int = 340) -> go.Figure:
    """Hourly model forecast with its band against the visitor's bid."""
    x = hourly["label"]
    fig = go.Figure()
    if {"p10", "p90"} <= set(hourly.columns):
        fig.add_traces(_band(x, hourly["p10"], hourly["p90"], "P10–P90 band"))
    fig.add_trace(_line(x, hourly["predicted"], "Forecast (median)", FORECAST))
    if "p_alpha" in hourly.columns:
        fig.add_trace(_line(x, hourly["p_alpha"], "α-bid", FORECAST, dash="dot"))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=hourly["bid_mw"],
            name="Your bid",
            mode="lines+markers",
            line={"color": BID, "width": 2, "shape": "hv"},
            marker={"size": 8, "color": BID, "line": {"color": SURFACE, "width": 2}},
        )
    )
    return _layout(fig, "MW", height)


def mape_history_figure(df: pd.DataFrame, height: int = 260) -> go.Figure:
    """Daily hourly MAPE of our forecast next to NYISO's, one axis."""
    fig = go.Figure()
    fig.add_trace(_line(df["date"], df["mape_hour"], "gridcast", FORECAST))
    if "isolf_mape_hour" in df.columns and df["isolf_mape_hour"].notna().any():
        fig.add_trace(_line(df["date"], df["isolf_mape_hour"], "NYISO pre-close", ISO))
    fig = _layout(fig, "hourly MAPE %", height)
    fig.update_yaxes(tickformat=".1f", rangemode="tozero")
    return fig
