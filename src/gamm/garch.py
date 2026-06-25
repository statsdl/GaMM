from __future__ import annotations

import numpy as np
import pandas as pd


def fit_garch_volatility_features(
    returns: pd.Series,
    last_obs=None,
    include_figarch: bool = True,
) -> pd.DataFrame:
    """Fit GARCH-family volatility models and return conditional volatilities."""

    arch_model = _arch_model()
    clean = pd.Series(returns, index=returns.index, name="DailyReturns").dropna()
    specs = {
        "CV_GARCH": {"mean": "zero", "dist": "normal"},
        "CV_GJR_GARCH": {"mean": "zero", "p": 1, "o": 1, "q": 1, "dist": "normal"},
        "CV_ThGARCH": {"mean": "zero", "p": 1, "o": 1, "q": 1, "power": 1.0, "dist": "normal"},
        "CV_AVGARCH": {"mean": "zero", "dist": "normal", "power": 1.0},
    }
    if include_figarch:
        specs["CV_FIGARCH"] = {"mean": "zero", "dist": "normal", "vol": "FIGARCH"}

    out = {}
    for name, kwargs in specs.items():
        model = arch_model(clean, **kwargs)
        fit = model.fit(last_obs=last_obs, disp="off", options={"maxiter": 1000, "ftol": 1e-8})
        out[name] = fit.conditional_volatility.reindex(clean.index).to_numpy(dtype=float)
    return pd.DataFrame(out, index=clean.index).replace([np.inf, -np.inf], np.nan).ffill().bfill()


def _arch_model():
    try:
        from arch import arch_model
    except ImportError as exc:
        raise ImportError("arch is required for GARCH features. Install with `pip install gamm[arch]`.") from exc
    return arch_model
