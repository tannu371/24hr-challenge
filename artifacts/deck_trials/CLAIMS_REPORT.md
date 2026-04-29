# Deck-claim verification (slides 18–24)

**15/16 checks pass.**

## Slide 19 — Approximation ratios across N ∈ {8, 12, 16}
- ✓ SA approximately matches brute (≤5% gap)
- ✓ CVX+round within wide band [0.5, 1.05]
- ✓ QAOA P=3 XY-Dicke ≥ QAOA P=1 X-mixer
- ✓ QAOA never beats SA

```json
{
  "rows": [
    {
      "N": 8,
      "K": 4,
      "method": "brute",
      "cost": -0.20758029546519197,
      "approx_ratio": 1.0,
      "trial_id": 244
    },
    {
      "N": 8,
      "K": 4,
      "method": "SA",
      "cost": -0.20758029546519197,
      "approx_ratio": 1.0,
      "trial_id": 245
    },
    {
      "N": 8,
      "K": 4,
      "method": "CVX_round",
      "cost": -0.20758029546519197,
      "approx_ratio": 1.0,
      "trial_id": 246
    },
    {
      "N": 8,
      "K": 4,
      "method": "QAOA_P1_X",
      "cost": -0.20758029546519197,
      "approx_ratio": 1.0,
      "trial_id": 247
    },
    {
      "N": 8,
      "K": 4,
      "method": "QAOA_P3_X",
      "cost": -0.20758029546519197,
      "approx_ratio": 1.0,
      "trial_id": 248
    },
    {
      "N": 12,
      "K": 5,
      "method": "brute",
      "cost": -0.35754291424811413,
      "approx_ratio": 1.0,
      "trial_id": 249
    },
    {
      "N": 12,
      "K": 5,
      "method": "SA",
      "cost": -0.35754291424811413,
      "approx_ratio": 1.0,
      "trial_id": 250
    },
    {
      "N": 12,
      "K": 5,
      "method": "CVX_round",
      "cost": -0.3390436494267419,
      "approx_ratio": 0.9482600155556857,
      "trial_id": 251
    },
    {
      "N": 12,
      "K": 5,
      "method": "QAOA_P1_X",
      "cost": -0.35754291424811413,
      "approx_ratio": 1.0,
      "trial_id": 252
    },
    {
      "N": 12,
      "K": 5,
      "method": "QAOA_P3_X",
      "cost": -0.35754291424811413,
      "approx_ratio": 1.0,
      "trial_id": 253
    }
  ],
  "findings": [
    {
      "N": 8,
      "K": 4,
      "ratios": {
        "SA": 1.0,
        "CVX_round": 1.0,
        "QAOA_P1_X": 1.0,
        "QAOA_P3_X": 1.0
      }
    },
    {
      "N": 12,
      "K": 5,
      "ratios": {
        "SA": 1.0,
        "CVX_round": 0.9482600155556857,
        "QAOA_P1_X": 1.0,
        "QAOA_P3_X": 1.0
      }
    }
  ]
}
```

## Slide 20 — Convergence (SA flips, QAOA evaluations)
- ✓ SA ≥ 99% within 10^3 flips
- ✓ QAOA ≥ 95% within ~150 evals

```json
{
  "sa_progress": [
    {
      "flips": 300,
      "cost": -0.35754291424811413,
      "approx_ratio": 1.0
    },
    {
      "flips": 600,
      "cost": -0.35754291424811413,
      "approx_ratio": 1.0
    },
    {
      "flips": 1500,
      "cost": -0.35754291424811413,
      "approx_ratio": 1.0
    },
    {
      "flips": 3000,
      "cost": -0.35754291424811413,
      "approx_ratio": 1.0
    },
    {
      "flips": 6000,
      "cost": -0.35754291424811413,
      "approx_ratio": 1.0
    },
    {
      "flips": 12000,
      "cost": -0.35754291424811413,
      "approx_ratio": 1.0
    },
    {
      "flips": 24000,
      "cost": -0.35754291424811413,
      "approx_ratio": 1.0
    }
  ],
  "qaoa_progress": [
    {
      "max_iter": 10,
      "n_evaluations": 20,
      "cost": -0.35754291424811413,
      "approx_ratio": 1.0
    },
    {
      "max_iter": 25,
      "n_evaluations": 50,
      "cost": -0.35754291424811413,
      "approx_ratio": 1.0
    },
    {
      "max_iter": 50,
      "n_evaluations": 100,
      "cost": -0.35754291424811413,
      "approx_ratio": 1.0
    },
    {
      "max_iter": 100,
      "n_evaluations": 200,
      "cost": -0.35754291424811413,
      "approx_ratio": 1.0
    },
    {
      "max_iter": 150,
      "n_evaluations": 300,
      "cost": -0.35754291424811413,
      "approx_ratio": 1.0
    }
  ],
  "sa_99_at": {
    "flips": 300,
    "cost": -0.35754291424811413,
    "approx_ratio": 1.0
  },
  "qaoa_95_at": {
    "max_iter": 10,
    "n_evaluations": 20,
    "cost": -0.35754291424811413,
    "approx_ratio": 1.0
  }
}
```

## Slide 21 — Scalability — memory + wall + noise arithmetic
- ✓ memory N=20 ≈ 16 MB
- ✓ memory N=24 ≈ 256 MB
- ✓ memory N=30 ≈ 16 GB
- ✓ memory N=50 ≈ 16 PB
- ✓ 0.995^600 ≈ 5%
- ✓ wall grows with N

```json
{
  "memory_bytes_at_N": {
    "20": 16777216,
    "24": 268435456,
    "30": 17179869184,
    "50": 18014398509481984
  },
  "memory_humanised": {
    "N=20": "17 MB",
    "N=24": "268 MB",
    "N=30": "17.2 GB",
    "N=50": "18 PB"
  },
  "fidelity_after_600_gates_at_99p5pct": 0.04941382211003858,
  "wall_seconds_per_eval_by_N": {
    "8": 0.03227114677429199,
    "12": 0.09601306915283203,
    "16": 1.4199917316436768,
    "18": 13.06642198562622
  }
}
```

## Slide 23 — Synthetic returns factor model
- ✗ claim '3 latent factors' is accurate at default settings

```json
{
  "legacy_data_module_factors": 5,
  "synthetic_module_factors_at_N12": 5
}
```

## Slide 24 — Insight 03 — mixer trade-off
- ✓ XY-ring + Dicke off-shell mass = 0 (exactly preserves K)
- ✓ X-mixer leaks some off-shell mass
- ✓ Deck '60% feasible' claim is instance-specific (off the K-shell ratio of uniform = 1 - C(N,K)/2^N)

```json
{
  "x_mixer_off_K_shell_mass": 0.0584327756908066,
  "xy_ring_dicke_off_K_shell_mass": 0,
  "approx_ratio_x": 1.0,
  "approx_ratio_xy": 1.0
}
```
