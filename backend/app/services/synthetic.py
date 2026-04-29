"""Seeded log-return synthetic generator (Task 1.2 spec).

Distinct from `portfolio.data.make_universe`, which builds Σ analytically from a
factor model. This module instead *samples* daily log returns from a 2-factor
model, then computes the **sample** mean μ̂ and **sample** covariance Σ̂.
That is what the task spec requires:

    "Synthetic mode: seeded log-return generator → sample mean μ and covariance Σ."

Returns are constructed as
    r_{t,i} = β_i F^market_t + γ_i F^sector(i)_t + ε_{t,i}
with all factors and noises drawn N(0, σ_*^2 / sqrt(252)) so daily
parameters annualise sensibly (sqrt-of-time scaling). μ_i in the
generator is a small daily drift; we annualise the resulting sample
estimates by ×252 for μ̂ and ×252 for Σ̂.
"""

from __future__ import annotations

import numpy as np

from portfolio.data import AssetUniverse


_TRADING_DAYS = 252


def generate_log_returns_universe(
    n_assets: int,
    seed: int = 7,
    n_periods: int = 504,
    n_sectors: int | None = None,
    market_vol_ann: float = 0.18,
    sector_vol_ann: float = 0.12,
    idio_vol_ann_range: tuple[float, float] = (0.10, 0.30),
    daily_drift_range: tuple[float, float] = (0.0001, 0.0008),
) -> AssetUniverse:
    """Generate a synthetic log-return panel and return its sample-estimated
    AssetUniverse so downstream code (formulation, classical solvers) can
    consume it identically to the analytic-covariance path.
    """
    rng = np.random.default_rng(seed)

    if n_sectors is None:
        n_sectors = max(2, min(4, n_assets // 3 or 2))

    sectors = np.array([i % n_sectors for i in range(n_assets)])
    rng.shuffle(sectors)

    beta_market = rng.uniform(0.6, 1.4, size=n_assets)
    beta_sector = rng.uniform(0.5, 1.2, size=n_assets)
    idio_vol_ann = rng.uniform(*idio_vol_ann_range, size=n_assets)
    drift = rng.uniform(*daily_drift_range, size=n_assets)

    s_m = market_vol_ann / np.sqrt(_TRADING_DAYS)
    s_s = sector_vol_ann / np.sqrt(_TRADING_DAYS)
    s_eps = idio_vol_ann / np.sqrt(_TRADING_DAYS)

    f_market = rng.normal(0.0, s_m, size=n_periods)
    f_sectors = rng.normal(0.0, s_s, size=(n_periods, n_sectors))
    eps = rng.normal(0.0, 1.0, size=(n_periods, n_assets)) * s_eps

    R = (
        np.outer(f_market, beta_market)
        + f_sectors[:, sectors] * beta_sector
        + eps
        + drift
    )  # shape (T, N) of daily log returns

    mu_hat_daily = R.mean(axis=0)
    sigma_hat_daily = np.cov(R, rowvar=False, ddof=1)

    mu_ann = mu_hat_daily * _TRADING_DAYS
    sigma_ann = sigma_hat_daily * _TRADING_DAYS

    sigma_ann = 0.5 * (sigma_ann + sigma_ann.T)
    eigmin = np.linalg.eigvalsh(sigma_ann).min()
    if eigmin <= 0:
        sigma_ann = sigma_ann + (1e-8 - eigmin) * np.eye(n_assets)

    sector_names = [f"S{i}" for i in range(n_sectors)]
    asset_names = [f"A{i:02d}" for i in range(n_assets)]

    return AssetUniverse(
        n_assets=n_assets,
        mu=mu_ann,
        sigma=sigma_ann,
        sectors=sectors,
        sector_names=sector_names,
        asset_names=asset_names,
    )
