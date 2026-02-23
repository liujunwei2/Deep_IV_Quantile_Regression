# Two-Stage Smoothed IV Quantile Regression via Conditional Diffusion (Experiments)

This repo contains runnable PyTorch code for four simulation designs used in our paper:
**Deep IV Quantile Regression via Conditional Diffusion**.

## Method overview (Algorithm 1)

Given i.i.d. data {(x_i, y_i, z_i)}_{i=1}^n and quantile level τ:

1) **Conditional diffusion** learns the conditional joint density p_{X,Y|Z}.
   We model W0 = (X, Y) and condition on Z. We train a score network b̂(t,w,z) ≈ ∇_w log p_t(w|z)
   using denoising score matching under an OU perturbation.

2) **Smoothed conditional moment + ERM**:
   Define a smoothed indicator I_h(v) approximating 1{v ≤ 0}.
   For each observed z_j, we sample {(X̂_{i,z_j}, Ŷ_{i,z_j})}_{i=1}^m ~ p̂_{X,Y|Z=z_j} and compute
   T̂_h f(z_j) = (1/m) Σ_i I_h( Ŷ_{i,z_j} - f(X̂_{i,z_j}) ).
   Then we estimate f by minimizing
   L̂(f) = (1/n) Σ_j ( T̂_h f(z_j) - τ )^2
   over a function class F (linear or MLP).

## Experiments

All experiments generate instruments Z, regressors X, and outcome Y with a calibrated error U^o such that
Q_τ(U^o) = 0 (so P(U^o ≤ 0) = τ). This enforces the IV-quantile restriction.

### 1) sel_iv1 (thick-tail + endogeneity)

Instrument:
Z ~ N(0,1).

Scale-mixture (bounded heavy tail):
S = σ^2 with prob π0, else S = 1.
(Ṽ1, Ṽ2) | S ~ N(0, S I2).
Let V1 = Ṽ1 and V2 = sqrt(1-ρ^2) Ṽ2 + ρ Ṽ1.

First stage:
X = γ2 + π Z + V2.

Structural error (endogeneity through V2):
U = V1 - β0 V2,
U^o = U - Q_τ(U).

Outcome:
Y = γ1 + β0 X + U^o.

True slope: β0.

### 2) sel_iv2 (heteroskedastic)

Z ~ N(0,1), X = γ2 + π Z + V2 (same V2 construction as above).

Let E ~ N(0,1) and E^o = E - Φ^{-1}(τ), so P(E^o ≤ 0) = τ.
Define heteroskedastic scale σ(V2) = a (1 + |V2|) and
U^o = σ(V2) E^o.

Outcome:
Y = γ1 + β0 X + U^o.

### 3) sel_iv3 (skewness + endogeneity)

Z ~ N(0,1), X = γ2 + π Z + V2.

Skewed error via Gaussian mixture:
E ~ ω N(μ2,1) + (1-ω) N(μ1,1).
Let E^o = E - Q_τ(E).

Inject endogeneity:
U_raw = E^o + λ V2,
U^o = U_raw - Q_τ(U_raw).

Outcome:
Y = γ1 + β0 X + U^o.

### 4) highdim_iib (high-dimensional, two endogenous regressors)

Z ∈ R^p, Z ~ N(0, I_p).
X_exo ∈ R^{d_exo}, X_exo ~ N(0, I_{d_exo}), independent of Z.

Two endogenous variables:
D1 = a1^T Z + η1,
D2 = a2^T Z + η2.

Generate (η1, η2, U) as a correlated vector (optionally with scale-mixture S),
then calibrate U^o = U - Q_τ(U).

Sparse exogenous effect:
θ ∈ R^{d_exo} with ||θ||_0 = s.

Outcome:
Y = α + β1 D1 + β2 D2 + X_exo^T θ + U^o.

Regressor vector:
X = (D1, D2, X_exo) ∈ R^{2 + d_exo}.

## How to run

Install dependencies:
```bash
pip install -r requirements.txt
