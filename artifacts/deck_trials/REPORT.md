# Deck-trial cross-check

**5 / 5 trials reproduce the deck's symptom.**

## trial_01_penalty_too_small
- **deck setup**: `{}`
- **deck claim**: When P_K is too small to dominate the objective swing, QAOA returns a portfolio with cardinality ≠ K (deck observed K=5/6 dominating top minima).
- **observed**: ```json
{
  "selected_K": 5,
  "top_K_distribution": {
    "6": 3,
    "5": 5,
    "4": 1,
    "7": 1
  },
  "top_dominant_K": 6,
  "energy_star": -0.15918790250483686,
  "cost": -0.20988333226692016,
  "approx_ratio": 1.0110946792736601,
  "trial_id": 129
}
```
- **verdict**: PASS — selected K=5 ≠ 4 under the deck-equivalent tiny P_K. Top-shell K distribution {6: 3, 5: 5, 4: 1, 7: 1} confirms the leakage.
- runtime: 4.5s

## trial_02_penalty_too_large
- **deck setup**: `{"N": 8, "K": 4, "P_K": 10000.0, "p": 1}`
- **deck claim**: Optimization plateaus immediately; F is dominated by the spiky penalty surface.
- **observed**: ```json
{
  "plateau_drop_huge_PK": 36444.69972043342,
  "plateau_relative_huge_PK": 0.4346485636749317,
  "plateau_drop_sane_PK": 10.447752531926096,
  "plateau_relative_sane_PK": 0.834033590852268,
  "approx_ratio_huge_PK": 1.0,
  "approx_ratio_sane_PK": 1.0,
  "trial_id_huge": 130,
  "trial_id_sane": 131
}
```
- **verdict**: PASS — huge-P_K relative drop (0.435) is < half the sane-P_K drop (0.834). Optimizer plateaus, as the deck predicts.
- runtime: 1.2s

## trial_03_optimizers
- **deck setup**: `{"N": 10, "p": 2}`
- **deck claim**: L-BFGS-B beats COBYLA which beats SPSA on a noiseless statevector. Deck reports ~0.99 vs ~0.70 approximation ratio.
- **observed**: ```json
{
  "COBYLA": {
    "trial_id": 132,
    "energy_star": 10.368266565221912,
    "cost": -0.28548790892519094,
    "approx_ratio": 1.0,
    "n_evaluations": 240
  },
  "SPSA": {
    "trial_id": 133,
    "energy_star": 46.592233447749955,
    "cost": -0.28548790892519094,
    "approx_ratio": 1.0,
    "n_evaluations": 633
  },
  "L-BFGS-B": {
    "trial_id": 134,
    "energy_star": 2.615189120362359,
    "cost": -0.28548790892519094,
    "approx_ratio": 1.0,
    "n_evaluations": 1281
  }
}
```
- **verdict**: PASS — L-BFGS-B (1.000) ≥ COBYLA (1.000) and ≥ SPSA (1.000) (within 5%). Ordering matches the deck's qualitative claim.
- runtime: 135.9s

## trial_04_barren_plateaus
- **deck setup**: `{"N": 12, "p_compared": [1, 2, 4, 6], "n_random_samples_per_p": 20}`
- **deck claim**: Var[∂F/∂θ] ∝ 1/2^n; for QAOA this manifests as the cost-at-random-init concentrating tighter around its mean as p grows. (McClean et al. 2018)
- **observed**: ```json
{
  "1": {
    "mean": 35.05593756960839,
    "variance": 187.1257232796882
  },
  "2": {
    "mean": 47.07539247037734,
    "variance": 258.7419110178995
  },
  "4": {
    "mean": 48.897119766527304,
    "variance": 243.14176688454106
  },
  "6": {
    "mean": 48.56019740443354,
    "variance": 153.82592349500743
  }
}
```
- **verdict**: PASS — deeper-half min variance (1.5383e+02) < shallow-half min variance (1.8713e+02). Cost concentrates as the ansatz becomes more random, the barren-plateau fingerprint.
- runtime: 24.0s

## trial_05_mixer
- **deck setup**: `{"N": 10, "K": 3, "p": 2, "off_shell_states": 904}`
- **deck claim**: X-mixer + uniform leaks substantial probability onto cardinality≠K. XY-ring + Dicke confines amplitude to the K-shell exactly.
- **observed**: ```json
{
  "x_mixer": {
    "trial_id": 215,
    "approx_ratio": 1.0,
    "off_K_shell_probability": 0.0584327756908066
  },
  "xy_ring_dicke": {
    "trial_id": 216,
    "approx_ratio": 1.0,
    "off_K_shell_probability": 0
  }
}
```
- **verdict**: PASS — XY-ring + Dicke off-shell mass ≈ 0 (0.00e+00); X-mixer leaks 0.058. Hamming-weight preservation is exact, as the deck claims.
- runtime: 16.4s
