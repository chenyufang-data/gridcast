"""Temporal Fusion Transformer (Lim et al., 2021) for day-ahead zonal load, in plain torch.

A *global* sequence model over every zone in the run, as opposed to the per-slot trees in
:mod:`models.tabular`. Protocol (``scripts/run_tft.py``):

- one fit per refit period on the ``window_days`` ending at the first cutoff of the
  period (training targets ``T <= D0 - 2``, every input of a sample lies before ``T``'s
  own cutoff), then one forecast per day whose **encoder stops at that day's cutoff**:
  the leakage rule holds per day, only the weights are up to ``refit_days`` old;
- encoder: the last ``enc_days`` of 15-min load before the cutoff plus the known inputs;
  decoder: the 43 h from the cutoff (D-1 05:00 ET) to the end of D, padded to
  ``DEC_LEN``; the loss covers every decoder slot with a known target, the forecast is
  read off the slots of D;
- known inputs: quarter-hour sin/cos, weekday embedding, US federal holiday flags, the
  hour-level weather forecasts of ``app.weather`` (2-day lead, issued before the cutoff)
  and the signed distance from the cutoff; static input: the zone;
- outputs: the ``QUANTILES`` of the load, scaled per zone by its mean over the training
  window; pinball loss.

Architecture as in the paper: variable selection networks with static context, gated
residual networks, LSTM encoder/decoder seeded by the static contexts, static
enrichment, interpretable multi-head attention (shared values, causal over the decoder),
position-wise GRN, gated skips with layer norm. ``torch`` is imported at module level
on purpose: this module is only imported by the TFT script and its tests.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from models.features import (
    SLOT,
    cutoff_for,
    day_grid,
    is_holiday,
    local_fields,
    local_midnight_utc,
    regularize,
    repair,
    training_targets,
)
from src.config import SLOTS_PER_HOUR

log = logging.getLogger(__name__)

SLOTS_PER_DAY = 24 * SLOTS_PER_HOUR
DEC_LEN = 19 * SLOTS_PER_HOUR + SLOTS_PER_DAY + SLOTS_PER_HOUR  # 05:00 D-1 .. end of a 25-h D
QUANTILES = (0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9)
WEATHER_COLS = ["temp_h", "app_h", "dew_h", "rh_h", "cloud_h", "wind_h", "rad_h", "temp_h_prev3"]
CALENDAR_COLS = ["tod_sin", "tod_cos", "is_holiday", "holiday_tomorrow", "rel_pos"]


# ---------------------------------------------------------------------------- data
@dataclass
class ZoneData:
    """One zone on a complete 15-min UTC grid that extends to the end of the last target."""

    zone: str
    index: pd.DatetimeIndex
    raw: np.ndarray  # coverage-masked load (NaN = missing), NaN beyond the data
    known: np.ndarray  # [T, n_known] float32: calendar + weather (rel_pos filled per sample)
    dow: np.ndarray  # [T] int64
    dates: np.ndarray  # [T] datetime64[D] local date of each slot
    tod: np.ndarray  # [T] int64 wall-clock quarter hour
    scale: float = 1.0
    known_cols: list[str] = field(default_factory=list)


def prepare_zone(
    zone: str, zone_slots: pd.DataFrame, hourly_feats: pd.DataFrame | None, last_target: date
) -> ZoneData:
    """Grid one zone's slots, attach the known inputs, extend the grid to `last_target`."""
    series = regularize(zone_slots)
    end = local_midnight_utc(last_target + timedelta(days=1))
    index = pd.date_range(series.index.min(), max(series.index.max(), end - SLOT), freq=SLOT)
    raw = series.reindex(index).to_numpy(dtype=float)
    fields = local_fields(pd.Series(index, index=index))
    tod = fields["tod"].to_numpy()
    k = pd.DataFrame(index=index)
    k["tod_sin"] = np.sin(2 * np.pi * tod / SLOTS_PER_DAY)
    k["tod_cos"] = np.cos(2 * np.pi * tod / SLOTS_PER_DAY)
    k["is_holiday"] = is_holiday(fields["date"]).to_numpy()
    k["holiday_tomorrow"] = is_holiday(fields["date"] + pd.Timedelta(days=1)).to_numpy()
    k["rel_pos"] = 0.0
    cols = list(CALENDAR_COLS)
    if hourly_feats is not None and not hourly_feats.empty:
        w = hourly_feats.set_index("hour_utc")
        present = [c for c in WEATHER_COLS if c in w.columns]
        wk = w[present].reindex(index.floor("h"))
        wk = wk.ffill(limit=8 * SLOTS_PER_HOUR).bfill(limit=8 * SLOTS_PER_HOUR)
        wk = wk.fillna(wk.mean())
        for c in present:
            k[c] = wk[c].to_numpy()
        cols += present
    return ZoneData(
        zone=zone,
        index=index,
        raw=raw,
        known=k[cols].to_numpy(dtype=np.float32),
        dow=fields["date"].dt.dayofweek.to_numpy(),
        dates=fields["date"].to_numpy().astype("datetime64[D]"),
        tod=tod.astype(np.int64),
        known_cols=cols,
    )


def aggregate_zone(name: str, members: list[ZoneData], weights: dict[str, float]) -> ZoneData:
    """A synthetic zone: the sum of `members` (loads add exactly), weather = load-weighted mean.

    Used as extra training series for the global model; never forecast itself.
    """
    first = members[0]
    w = np.array([weights.get(m.zone, 1.0) for m in members], dtype=float)
    w = w / w.sum()
    raw = np.zeros(len(first.index))
    known = first.known.copy()
    weather_j = [j for j, c in enumerate(first.known_cols) if c in WEATHER_COLS]
    known[:, weather_j] = 0.0
    for wi, m in zip(w, members, strict=True):
        pos = m.index.get_indexer(first.index)
        raw = raw + np.where(pos >= 0, m.raw[np.clip(pos, 0, None)], np.nan)
        known[:, weather_j] += wi * np.where(
            pos[:, None] >= 0, m.known[np.clip(pos, 0, None)][:, weather_j], np.nan
        )
    return ZoneData(
        zone=name,
        index=first.index,
        raw=raw,
        known=known.astype(np.float32),
        dow=first.dow,
        dates=first.dates,
        tod=first.tod,
        known_cols=list(first.known_cols),
    )


def block_bootstrap(
    zd: ZoneData, y: np.ndarray, rng: np.random.Generator, block: int = SLOTS_PER_DAY
) -> np.ndarray:
    """A bootstrap replica of `y`: trend + (weekday, tod) profile + block-resampled residual.

    Trend = centred 7-day mean; profile = mean of the detrended load per (weekday, tod);
    the residual is resampled in moving blocks of one day (Bergmeir et al., 2016). The
    NaN tail beyond the last observation is kept, so the replica obeys the same cutoff.
    """
    valid = np.flatnonzero(~np.isnan(y))
    if len(valid) == 0:
        return y.copy()
    n = int(valid.max()) + 1
    s = pd.Series(y[:n])
    trend = s.rolling(7 * SLOTS_PER_DAY, center=True, min_periods=SLOTS_PER_DAY).mean()
    trend = trend.ffill().bfill()
    detrended = s - trend
    key = zd.dow[:n] * SLOTS_PER_DAY + zd.tod[:n]
    profile = detrended.groupby(key).transform("mean")
    resid = np.nan_to_num((detrended - profile).to_numpy())
    n_blocks = -(-n // block)
    starts = rng.integers(0, max(n - block, 0) + 1, n_blocks)
    boot = np.concatenate([resid[st : st + block] for st in starts])[:n]
    out = np.full(len(y), np.nan)
    out[:n] = np.maximum((trend + profile).to_numpy() + boot, 1.0)
    out[:n][np.isnan(y[:n])] = np.nan
    return out


def series_asof(
    zd: ZoneData, cutoff: pd.Timestamp, tod_means: dict[int, float] | None = None
) -> tuple[np.ndarray, dict[int, float]]:
    """Repaired load on the zone grid, using only slots that end at or before `cutoff`.

    Returns ``(y, tod_means)`` with `y` NaN from the cutoff on, so a window that reads
    past the cutoff is visibly wrong instead of silently leaky.
    """
    n = int(zd.index.searchsorted(cutoff))  # slots strictly before the cutoff (start < cutoff)
    s = pd.Series(zd.raw[:n], index=zd.index[:n])
    repaired, means = repair(s, tod_means)
    y = np.full(len(zd.index), np.nan)
    y[:n] = repaired.to_numpy()
    return y, means


@dataclass
class Sample:
    zone_id: int
    enc: slice
    dec: slice
    cutoff_pos: int


def make_sample(zd: ZoneData, zone_id: int, target: date, enc_len: int) -> Sample | None:
    cutoff = cutoff_for(target)
    p0 = int(zd.index.searchsorted(cutoff))
    p1 = int(zd.index.searchsorted(local_midnight_utc(target + timedelta(days=1))))
    if p0 - enc_len < 0 or p1 - p0 > DEC_LEN or p1 > len(zd.index):
        return None
    return Sample(zone_id, slice(p0 - enc_len, p0), slice(p0, p1), p0)


def _tensors(
    zd: ZoneData, y: np.ndarray, s: Sample, enc_len: int, wmean: np.ndarray, wstd: np.ndarray
) -> dict[str, torch.Tensor]:
    """One sample as padded tensors (decoder padded to DEC_LEN, mask marks real slots)."""
    known = (zd.known - wmean) / wstd
    rel_col = zd.known_cols.index("rel_pos")
    kenc = known[s.enc].copy()
    kenc[:, rel_col] = np.arange(-enc_len, 0) / SLOTS_PER_DAY
    n_dec = s.dec.stop - s.dec.start
    kdec = np.zeros((DEC_LEN, known.shape[1]), dtype=np.float32)
    kdec[:n_dec] = known[s.dec]
    kdec[:n_dec, rel_col] = np.arange(n_dec) / SLOTS_PER_DAY
    yenc = np.nan_to_num(y[s.enc] / zd.scale, nan=1.0).astype(np.float32)
    ydec = np.zeros(DEC_LEN, dtype=np.float32)
    mask = np.zeros(DEC_LEN, dtype=np.float32)
    yd = y[s.dec] / zd.scale
    ok = ~np.isnan(yd)
    ydec[:n_dec][ok] = yd[ok]
    mask[:n_dec][ok] = 1.0
    dow_dec = np.zeros(DEC_LEN, dtype=np.int64)
    dow_dec[:n_dec] = zd.dow[s.dec]
    return {
        "x_enc": torch.from_numpy(np.concatenate([yenc[:, None], kenc], axis=1)),
        "dow_enc": torch.from_numpy(zd.dow[s.enc].astype(np.int64)),
        "x_dec": torch.from_numpy(kdec),
        "dow_dec": torch.from_numpy(dow_dec),
        "zone": torch.tensor(s.zone_id),
        "y": torch.from_numpy(ydec),
        "mask": torch.from_numpy(mask),
    }


def _collate(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


# ---------------------------------------------------------------------------- layers
class GLU(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.lin = nn.Linear(d, 2 * d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.lin(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GRN(nn.Module):
    """Gated residual network: ELU(W2 x + W3 c) -> W1 -> dropout -> GLU -> LayerNorm(skip + .)."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        ctx_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.skip: nn.Module = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.w2 = nn.Linear(in_dim, hidden)
        self.w3 = nn.Linear(ctx_dim, hidden, bias=False) if ctx_dim else None
        self.w1 = nn.Linear(hidden, out_dim)
        self.drop = nn.Dropout(dropout)
        self.glu = GLU(out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        h = self.w2(x)
        if c is not None and self.w3 is not None:
            hc = self.w3(c)
            h = h + (hc.unsqueeze(1) if x.dim() == 3 and hc.dim() == 2 else hc)
        h = self.w1(torch.nn.functional.elu(h))
        return self.norm(self.skip(x) + self.glu(self.drop(h)))


class VSN(nn.Module):
    """Variable selection: softmax weights over variables from a GRN on the flat embedding."""

    def __init__(self, n_vars: int, d: int, ctx: bool, dropout: float) -> None:
        super().__init__()
        self.flat = GRN(n_vars * d, d, n_vars, d if ctx else None, dropout)
        self.per = nn.ModuleList([GRN(d, d, d, None, dropout) for _ in range(n_vars)])

    def forward(
        self, e: torch.Tensor, c: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, n, d = e.shape
        w = torch.softmax(self.flat(e.reshape(b, t, n * d), c), dim=-1)  # [B,T,n]
        p = torch.stack([g(e[:, :, i]) for i, g in enumerate(self.per)], dim=-2)
        return (w.unsqueeze(-1) * p).sum(-2), w


class InterpretableMHA(nn.Module):
    """Multi-head attention with one shared value projection; heads are averaged."""

    def __init__(self, d: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.dh = d // heads
        self.q = nn.ModuleList([nn.Linear(d, self.dh) for _ in range(heads)])
        self.k = nn.ModuleList([nn.Linear(d, self.dh) for _ in range(heads)])
        self.v = nn.Linear(d, self.dh)
        self.out = nn.Linear(self.dh, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        v = self.v(kv)
        outs = []
        for wq, wk in zip(self.q, self.k, strict=True):
            scores = wq(q) @ wk(kv).transpose(1, 2) / math.sqrt(self.dh)
            scores = scores.masked_fill(~allowed, -1e9)
            outs.append(self.drop(torch.softmax(scores, dim=-1)) @ v)
        return self.out(torch.stack(outs, 0).mean(0))


class TFT(nn.Module):
    def __init__(
        self,
        n_zones: int,
        n_known: int,
        d: int = 48,
        heads: int = 4,
        dropout: float = 0.1,
        n_quantiles: int = len(QUANTILES),
    ) -> None:
        super().__init__()
        self.zone_emb = nn.Embedding(n_zones, d)
        self.dow_emb = nn.Embedding(7, d)
        self.y_lin = nn.Linear(1, d)
        self.known_lin = nn.ModuleList([nn.Linear(1, d) for _ in range(n_known)])
        self.vsn_static = VSN(1, d, False, dropout)
        self.vsn_enc = VSN(n_known + 2, d, True, dropout)
        self.vsn_dec = VSN(n_known + 1, d, True, dropout)
        self.ctx = nn.ModuleList([GRN(d, d, d, None, dropout) for _ in range(4)])
        self.lstm_enc = nn.LSTM(d, d, batch_first=True)
        self.lstm_dec = nn.LSTM(d, d, batch_first=True)
        self.gate_lstm = nn.Sequential(nn.Dropout(dropout), GLU(d))
        self.norm_lstm = nn.LayerNorm(d)
        self.enrich = GRN(d, d, d, d, dropout)
        self.attn = InterpretableMHA(d, heads, dropout)
        self.gate_attn = nn.Sequential(nn.Dropout(dropout), GLU(d))
        self.norm_attn = nn.LayerNorm(d)
        self.pos = GRN(d, d, d, None, dropout)
        self.gate_final = nn.Sequential(nn.Dropout(dropout), GLU(d))
        self.norm_final = nn.LayerNorm(d)
        self.out = nn.Linear(d, n_quantiles)

    def forward(self, b: dict[str, torch.Tensor]) -> torch.Tensor:
        x_enc, x_dec = b["x_enc"], b["x_dec"]
        te, td = x_enc.shape[1], x_dec.shape[1]
        s, _ = self.vsn_static(self.zone_emb(b["zone"])[:, None, None, :])
        s = s[:, 0]
        c_s, c_e, c_h, c_c = (g(s) for g in self.ctx)
        enc_vars = [self.y_lin(x_enc[..., :1])]
        enc_vars += [lin(x_enc[..., i + 1 : i + 2]) for i, lin in enumerate(self.known_lin)]
        enc_vars.append(self.dow_emb(b["dow_enc"]))
        dec_vars = [lin(x_dec[..., i : i + 1]) for i, lin in enumerate(self.known_lin)]
        dec_vars.append(self.dow_emb(b["dow_dec"]))
        v_enc, _ = self.vsn_enc(torch.stack(enc_vars, dim=-2), c_s)
        v_dec, _ = self.vsn_dec(torch.stack(dec_vars, dim=-2), c_s)
        state = (c_h.unsqueeze(0).contiguous(), c_c.unsqueeze(0).contiguous())
        o_enc, state = self.lstm_enc(v_enc, state)
        o_dec, _ = self.lstm_dec(v_dec, state)
        vsn_out = torch.cat([v_enc, v_dec], dim=1)
        temporal = self.norm_lstm(vsn_out + self.gate_lstm(torch.cat([o_enc, o_dec], dim=1)))
        enriched = self.enrich(temporal, c_e)
        allowed = torch.arange(te + td, device=x_enc.device)[None, :] <= (
            te + torch.arange(td, device=x_enc.device)[:, None]
        )
        q = enriched[:, te:]
        x = self.norm_attn(q + self.gate_attn(self.attn(q, enriched, allowed)))
        x = self.norm_final(temporal[:, te:] + self.gate_final(self.pos(x)))
        return self.out(x)


def pinball(
    pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, q: torch.Tensor
) -> torch.Tensor:
    diff = y.unsqueeze(-1) - pred
    loss = torch.maximum(q * diff, (q - 1) * diff).mean(-1)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------- forecaster
class TFTForecaster:
    """Fit once on many (zone, target day) windows; forecast any day from its own encoder."""

    def __init__(
        self,
        zones: list[str],
        *,
        enc_days: int = 7,
        hidden: int = 48,
        heads: int = 4,
        dropout: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 60,
        patience: int = 8,
        val_days: int = 14,
        quantiles: tuple[float, ...] = QUANTILES,
        device: str | None = None,
        seed: int = 0,
    ) -> None:
        self.zones = list(zones)
        self.zone_id = {z: i for i, z in enumerate(self.zones)}
        self.enc_len = enc_days * SLOTS_PER_DAY
        self.hidden, self.heads, self.dropout = hidden, heads, dropout
        self.lr, self.batch_size, self.epochs, self.patience = lr, batch_size, epochs, patience
        self.val_days = val_days
        self.quantiles = quantiles
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.model: TFT | None = None
        self.wmean: np.ndarray | None = None
        self.wstd: np.ndarray | None = None
        self.history: list[dict[str, float]] = []

    # --- training ------------------------------------------------------------------
    def fit(
        self,
        data: dict[str, tuple[ZoneData, np.ndarray]],
        targets: list[date],
        replicas: list[tuple[ZoneData, np.ndarray]] | None = None,
    ) -> TFTForecaster:
        """`data`: zone -> (ZoneData, load repaired as of the fit cutoff); `targets`: train days.

        `replicas` are extra (ZoneData, load) series of zones already in `data` (bootstrap
        copies): training windows only, never validation.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        first = next(iter(data.values()))[0]
        n_known = first.known.shape[1]
        # per-zone scale and pooled weather statistics over the training window
        rows = []
        for zd, y in data.values():
            lo = int(zd.index.searchsorted(local_midnight_utc(min(targets))))
            hi = int(zd.index.searchsorted(local_midnight_utc(max(targets) + timedelta(days=1))))
            zd.scale = float(np.nanmean(y[lo:hi])) or 1.0
            rows.append(zd.known[lo:hi])
        allk = np.concatenate(rows)
        self.wmean = allk.mean(0)
        self.wstd = np.where(allk.std(0) > 1e-6, allk.std(0), 1.0)
        for c in ("rel_pos",):
            j = first.known_cols.index(c)
            self.wmean[j], self.wstd[j] = 0.0, 1.0
        val_from = max(targets) - timedelta(days=self.val_days - 1)
        train_items: list[dict[str, torch.Tensor]] = []
        val_items: list[dict[str, torch.Tensor]] = []
        for zone, (zd, y) in data.items():
            for t in targets:
                s = make_sample(zd, self.zone_id[zone], t, self.enc_len)
                if s is None:
                    continue
                item = _tensors(zd, y, s, self.enc_len, self.wmean, self.wstd)
                (val_items if t >= val_from else train_items).append(item)
        for zd, y in replicas or []:
            for t in targets:
                if t >= val_from:
                    continue
                s = make_sample(zd, self.zone_id[zd.zone], t, self.enc_len)
                if s is not None:
                    train_items.append(_tensors(zd, y, s, self.enc_len, self.wmean, self.wstd))
        if not train_items or not val_items:
            raise ValueError("not enough windows to train the TFT")
        model = TFT(
            len(self.zones), n_known, self.hidden, self.heads, self.dropout, len(self.quantiles)
        )
        model.to(self.device)
        q = torch.tensor(self.quantiles, device=self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        val_batch = {k: v.to(self.device) for k, v in _collate(val_items).items()}
        best, best_state, bad = float("inf"), None, 0
        t0 = time.perf_counter()
        self.history = []
        rng = np.random.default_rng(self.seed)
        for epoch in range(self.epochs):
            model.train()
            order = rng.permutation(len(train_items))
            tot, nb = 0.0, 0
            for i in range(0, len(order), self.batch_size):
                batch = _collate([train_items[j] for j in order[i : i + self.batch_size]])
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss = pinball(model(batch), batch["y"], batch["mask"], q)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item()
                nb += 1
            model.eval()
            with torch.no_grad():
                vl = float(pinball(model(val_batch), val_batch["y"], val_batch["mask"], q))
            self.history.append({"epoch": epoch, "train": tot / max(nb, 1), "val": vl})
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.model = model
        log.info(
            "tft fit: %d train / %d val windows, %d zones, %d epochs, "
            "best val pinball %.4f, %.0fs on %s",
            len(train_items),
            len(val_items),
            len(data),
            len(self.history),
            best,
            time.perf_counter() - t0,
            self.device,
        )
        return self

    # --- inference -----------------------------------------------------------------
    def predict(self, zd: ZoneData, y_asof: np.ndarray, target: date) -> pd.DataFrame:
        """Quantile forecast of `target`'s slots; `y_asof` must be NaN from the cutoff on."""
        if self.model is None or self.wmean is None or self.wstd is None:
            raise RuntimeError("fit first")
        s = make_sample(zd, self.zone_id[zd.zone], target, self.enc_len)
        if s is None:
            raise ValueError(f"not enough history for {zd.zone} {target}")
        if not np.isnan(y_asof[s.dec]).all():
            raise ValueError("y_asof reaches past the cutoff")
        batch = {
            k: v.unsqueeze(0).to(self.device)
            for k, v in _tensors(zd, y_asof, s, self.enc_len, self.wmean, self.wstd).items()
        }
        with torch.no_grad():
            out = self.model(batch)[0].cpu().numpy() * zd.scale  # [DEC_LEN, Q]
        n_dec = s.dec.stop - s.dec.start
        out = np.sort(out[:n_dec], axis=1)  # monotone quantiles
        dates = zd.dates[s.dec]
        pick = dates == np.datetime64(target)
        grid = day_grid(target)
        if pick.sum() != len(grid):
            raise ValueError(
                f"{zd.zone} {target}: {pick.sum()} decoder slots for a {len(grid)}-slot day"
            )
        qcols = {f"q{int(round(q * 100))}": out[pick, i] for i, q in enumerate(self.quantiles)}
        return pd.concat([grid, pd.DataFrame(qcols)], axis=1)


def quantile_at(frame: pd.DataFrame, quantiles: tuple[float, ...], level: float) -> np.ndarray:
    """Linear interpolation of the ``q<pct>`` columns at `level` (clipped to the grid)."""
    cols = [f"q{int(round(q * 100))}" for q in quantiles]
    vals = frame[cols].to_numpy()
    return np.array([np.interp(level, quantiles, row) for row in vals])


def to_result_rows(frame: pd.DataFrame, quantiles: tuple[float, ...], alpha: float) -> pd.DataFrame:
    """``pred, p10, p90, p_alpha, alpha`` from the quantile columns; band kept around the median."""
    out = frame[["ts_utc", "date", "slot", "tod"]].copy()
    out["pred"] = quantile_at(frame, quantiles, 0.5)
    out["p10"] = np.minimum(quantile_at(frame, quantiles, 0.1), out["pred"])
    out["p90"] = np.maximum(quantile_at(frame, quantiles, 0.9), out["pred"])
    out["p_alpha"] = (
        out["pred"] if abs(alpha - 0.5) < 1e-9 else quantile_at(frame, quantiles, alpha)
    )
    out["alpha"] = float(alpha)
    return out


def periods(targets: list[date], refit_days: int) -> list[list[date]]:
    """Consecutive blocks of `refit_days` targets; each block is served by one fit."""
    return [targets[i : i + refit_days] for i in range(0, len(targets), refit_days)]


def fit_targets(first_target: date, window_days: int, available_dates: pd.Index) -> list[date]:
    """Training days for the fit that serves `first_target`: ``[D0-1-window, D0-2]`` with data."""
    return [d.date() for d in training_targets(first_target, window_days, available_dates)]


def zone_dates(zd: ZoneData) -> pd.Index:
    have = ~np.isnan(zd.raw)
    return pd.Index(pd.to_datetime(np.unique(zd.dates[have])))


def describe(f: TFTForecaster) -> dict[str, Any]:
    return {
        "enc_days": f.enc_len // SLOTS_PER_DAY,
        "hidden": f.hidden,
        "heads": f.heads,
        "dropout": f.dropout,
        "lr": f.lr,
        "batch": f.batch_size,
        "epochs_max": f.epochs,
        "patience": f.patience,
        "val_days": f.val_days,
        "quantiles": f.quantiles,
        "device": f.device,
    }
