"""Synthetic asset universe.

We deliberately generate the data ourselves rather than depending on a market
data API:

* The challenge is about modelling and optimisation, not data sourcing.
* We need full control over the correlation structure so we can stress-test the
  diversification penalty and study scalability.

The model is a one-factor + sector-factor model -- a standard quant building
block which produces a positive-definite covariance matrix with realistic
sector-level correlation clusters.

Returns r_i = beta_i * F_market + gamma_{s(i)} * F_{s(i)} + epsilon_i

where F_market ~ N(0, sigma_m^2), F_s ~ N(0, sigma_s^2) and epsilon_i is
idiosyncratic noise.  Expected returns mu_i are sampled to be slightly
positive on average so the optimiser has to trade off return against risk
rather than just shorting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class AssetUniverse:
    """A small synthetic universe of N assets organised into S sectors."""

    n_assets: int
    mu: np.ndarray              # (N,) expected returns
    sigma: np.ndarray           # (N, N) covariance
    sectors: np.ndarray         # (N,) integer sector label
    sector_names: List[str] = field(default_factory=list)
    asset_names: List[str] = field(default_factory=list)

    def correlation(self) -> np.ndarray:
        d = np.sqrt(np.diag(self.sigma))
        return self.sigma / np.outer(d, d)

    def annual_vol(self) -> np.ndarray:
        return np.sqrt(np.diag(self.sigma))


def make_universe(
    n_assets: int = 12,
    n_sectors: int = 4,
    market_vol: float = 0.12,
    sector_vol: float = 0.08,
    idio_vol_range: tuple = (0.05, 0.20),
    mu_range: tuple = (0.02, 0.18),
    seed: int = 7,
) -> AssetUniverse:
    """Generate a positive-definite covariance via a 2-factor model.

    Parameters are annualised. The default produces equity-like volatilities
    (15-30%) with sector clustering visible in the correlation matrix.
    """
    rng = np.random.default_rng(seed)

    # Assign assets to sectors round-robin so sector sizes are balanced.
    sectors = np.array([i % n_sectors for i in range(n_assets)])
    rng.shuffle(sectors)

    # Factor loadings.
    beta_market = rng.uniform(0.6, 1.4, size=n_assets)
    beta_sector = rng.uniform(0.5, 1.2, size=n_assets)
    idio_vol = rng.uniform(*idio_vol_range, size=n_assets)

    # Construct loading matrix B (N x F) where F = 1 + n_sectors.
    B = np.zeros((n_assets, 1 + n_sectors))
    B[:, 0] = beta_market * market_vol
    for i in range(n_assets):
        B[i, 1 + sectors[i]] = beta_sector[i] * sector_vol

    factor_cov = np.eye(1 + n_sectors)  # factors are independent, unit-variance
    sigma = B @ factor_cov @ B.T + np.diag(idio_vol ** 2)

    # Symmetrise (numerical hygiene) and confirm PSD.
    sigma = 0.5 * (sigma + sigma.T)
    eigmin = np.linalg.eigvalsh(sigma).min()
    if eigmin <= 0:
        sigma += (1e-8 - eigmin) * np.eye(n_assets)

    mu = rng.uniform(*mu_range, size=n_assets)

    sector_names = [f"S{i}" for i in range(n_sectors)]
    asset_names = [f"A{i:02d}" for i in range(n_assets)]

    return AssetUniverse(
        n_assets=n_assets,
        mu=mu,
        sigma=sigma,
        sectors=sectors,
        sector_names=sector_names,
        asset_names=asset_names,
    )
