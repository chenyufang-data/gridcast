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
on purpose: this module is only imported by the TFT scripts and their tests; the data
side lives in :mod:`models.tft_data` (torch-free, what the backend imports).
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from models.features import FEATURE_VERSION, local_midnight_utc
from models.tft_data import (
    CALENDAR_COLS,
    DEC_LEN,
    ONNX_FORMAT,
    ONNX_INPUTS,
    QUANTILES,
    SLOTS_PER_DAY,
    WEATHER_COLS,
    Sample,
    ZoneData,
    aggregate_zone,
    block_bootstrap,
    check_sample,
    fit_targets,
    make_sample,
    periods,
    prepare_zone,
    quantile_at,
    quantile_frame,
    sample_arrays,
    series_asof,
    sha256_of,
    to_result_rows,
    zone_dates,
)

__all__ = [
    "CALENDAR_COLS", "DEC_LEN", "QUANTILES", "SLOTS_PER_DAY", "WEATHER_COLS", "ONNX_FORMAT",
    "ONNX_INPUTS", "GLU", "GRN", "VSN", "InterpretableMHA", "TFT", "TFTForecaster", "Sample",
    "ZoneData", "aggregate_zone", "block_bootstrap", "check_sample", "describe", "fit_targets",
    "local_midnight_utc", "make_sample", "periods", "pinball", "prepare_zone", "quantile_at",
    "quantile_frame", "sample_arrays", "series_asof", "sha256_of", "to_result_rows",
    "zone_dates",
]  # fmt: skip

log = logging.getLogger(__name__)


def _tensors(
    zd: ZoneData, y: np.ndarray, s: Sample, enc_len: int, wmean: np.ndarray, wstd: np.ndarray
) -> dict[str, torch.Tensor]:
    """One sample as tensors (see :func:`models.tft_data.sample_arrays`)."""
    return {
        k: torch.from_numpy(v) for k, v in sample_arrays(zd, y, s, enc_len, wmean, wstd).items()
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
        self.scales: dict[str, float] = {}
        self._known_cols: list[str] = []
        self._examples: list[dict[str, torch.Tensor]] = []

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
            self.scales[zd.zone] = zd.scale
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
        self._known_cols = list(first.known_cols)
        self._examples = [{k: v.clone() for k, v in it.items()} for it in train_items[:4]]
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
        s = check_sample(zd, self.zone_id[zd.zone], y_asof, target, self.enc_len)
        arrays = sample_arrays(zd, y_asof, s, self.enc_len, self.wmean, self.wstd)
        batch = {k: torch.from_numpy(v).unsqueeze(0).to(self.device) for k, v in arrays.items()}
        with torch.no_grad():
            out = self.model(batch)[0].cpu().numpy()  # [DEC_LEN, Q], scaled
        return quantile_frame(zd, s, out * zd.scale, target, self.quantiles)

    # --- export --------------------------------------------------------------------
    def export(
        self, out_dir: Path, *, fit_cutoff: str, window_days: int, tolerance: float = 1e-4
    ) -> Path:
        """Write ``tft.onnx`` + ``tft.json`` (zones, constants, scales, hashes) to `out_dir`.

        The export pins eval mode (dropout off) and is verified against torch on the
        examples kept from training; a max abs difference above `tolerance` (on the
        scaled target, ~1.0 = the zone mean) raises instead of shipping a wrong file.
        """
        import onnxruntime as ort

        if self.model is None or self.wmean is None or self.wstd is None or not self._examples:
            raise RuntimeError("fit first")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model = self.model.to("cpu").eval()
        wrapped = _Exportable(model).eval()
        names = list(ONNX_INPUTS)
        example = _collate(self._examples[:2])
        args = tuple(example[k] for k in names)
        onnx_path = out_dir / "tft.onnx"
        torch.onnx.export(
            wrapped,
            args,
            str(onnx_path),
            input_names=names,
            output_names=["quantiles"],
            opset_version=17,
            dynamo=False,
            dynamic_axes={**{n: {0: "batch"} for n in names}, "quantiles": {0: "batch"}},
        )
        wrapped.eval()
        model.eval()
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        check = _collate(self._examples)
        with torch.no_grad():
            ref = wrapped(*[check[k] for k in names]).numpy()
        got = session.run(None, {k: check[k].numpy() for k in names})[0]
        diff = float(np.abs(got - ref).max())
        if diff > tolerance:
            onnx_path.unlink(missing_ok=True)
            raise RuntimeError(f"ONNX export differs from torch by {diff:.2e} (> {tolerance})")
        meta = {
            "format": ONNX_FORMAT,
            "feature_version": FEATURE_VERSION,
            "zones": self.zones,
            "scale": {z: float(v) for z, v in self.scales.items()},
            "enc_len": self.enc_len,
            "dec_len": DEC_LEN,
            "quantiles": list(self.quantiles),
            "known_cols": list(self._known_cols),
            "wmean": [float(v) for v in self.wmean],
            "wstd": [float(v) for v in self.wstd],
            "hidden": self.hidden,
            "heads": self.heads,
            "fit_cutoff": fit_cutoff,
            "window_days": window_days,
            "epochs": len(self.history),
            "val_pinball": min(h["val"] for h in self.history) if self.history else None,
            "torch_version": torch.__version__,
            "max_abs_diff_vs_torch": diff,
            "onnx_sha256": sha256_of(onnx_path),
        }
        (out_dir / "tft.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.model.to(self.device)
        log.info(
            "exported %s (%.2f MB, max |onnx - torch| %.1e) + tft.json",
            onnx_path,
            onnx_path.stat().st_size / 1e6,
            diff,
        )
        return out_dir


class _Exportable(nn.Module):
    """Positional-argument wrapper around :class:`TFT` for the ONNX exporter."""

    def __init__(self, model: TFT) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x_enc: torch.Tensor,
        dow_enc: torch.Tensor,
        x_dec: torch.Tensor,
        dow_dec: torch.Tensor,
        zone: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            {"x_enc": x_enc, "dow_enc": dow_enc, "x_dec": x_dec, "dow_dec": dow_dec, "zone": zone}
        )


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
