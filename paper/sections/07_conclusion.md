# 6. Conclusion and Future Directions

---

## 6.1 Summary of Contributions

This paper presented **Kinetopus Engine**, a parsimonious, White-Box Scientific Machine Learning (SciML) framework for financial time series analysis, grounded in the hypothesis that discrete, noisy market data — when observed at the correct temporal scale and projected onto the appropriate physical state space — exhibit geometrically coherent dynamics governed by discoverable mathematical laws. We demonstrated this hypothesis empirically, mathematically, and computationally across a six-asset, cross-class universe comprising 3,525 walk-forward evaluation iterations.

**Contribution 1: Empirical Validation of Physical Geometry in Financial Markets.**
The central claim of this work — that financial price dynamics contain a statistically significant directional structure beyond random walk — was confirmed with strong statistical evidence. Kinetopus achieved a directional Hit Ratio of 59.15% at the B30 horizon (~7.5 months), sustained over 705 valid walk-forward episodes, with a one-sided $z$-statistic of 4.942 ($p < 0.0001$) against the null hypothesis $H_0: \text{HR} = 0.50$. This result was replicated across four of six asset classes, with particularly pronounced physical coherence in momentum-driven markets: QQQ (71.6% HR), MSFT (70.3% HR), and BTC-USD (60.9% HR). The SINDy-discovered governing equations achieved a median in-sample coefficient of determination of $R^2 = 0.75$, confirming that the ODE formulation $\dot{r} = f(r, V)$ captures a meaningful portion of the local dynamic structure of the return process \cite{brunton2016discovering}. These results provide quantitative grounding for the thesis that markets are not uniformly random, but exhibit regime-dependent physical coherence.

**Contribution 2: A Five-Layer White-Box Pipeline with Sub-Millisecond Inference.**
The Kinetopus Engine integrates five mathematically principled processing stages: FFT spectral sensing, $C^2$ Spline topological manifold smoothing, CUSUM-guided regime slicing, SINDy sparse ODE discovery, and Euler-Maruyama stochastic differential equation integration. The entire inference pipeline executes on consumer-grade CPU hardware (16 GB RAM) with median latency below 1 millisecond per episode — approximately 1,525 times faster than the LSTM baseline (1,524.72 ms). Unlike deep learning architectures that produce opaque parameter vectors \cite{hochreiter1997long}, Kinetopus produces an explicit, inspectable, and falsifiable governing equation at each evaluation step: a scientific statement about the local physical dynamics of the market that can be audited, interpreted, and communicated.

**Contribution 3: Hydrodynamic Trajectory Stabilization as a Physically Motivated Correction.**
The ablation study (Section 5.6) demonstrated that the introduction of a multiplicative decay factor $\lambda = 0.99$ into the Euler-Maruyama integration — analogous to a hydrodynamic friction term in classical mechanics — resolves a structural instability inherent to forward ODE integration over long horizons. This correction reduced the mathematical failure rate from 12.17% to 1.40%, reduced the B60 median MAPE from 99.71% to 42.38% (−57.5%), and reduced the B60 median RMSE from 566.33 to 155.17 (−72.6%), while preserving the directional accuracy at B30 to within 0.07 percentage points. The critical finding is that this correction is not an empirical tuning trick — it is a consequence of a well-understood physical principle: dissipation prevents unbounded energy accumulation in an integrating dynamical system \cite{kloeden1992numerical}.

---

## 6.2 Open-Source Implementation, Interactive Calibration, and Telemetry

A complete, publicly accessible implementation of the Kinetopus Engine is deployed as an interactive application at:

> 🔗 **[https://huggingface.co/spaces/Juan778/KineTopus_Engine](https://huggingface.co/spaces/Juan778/KineTopus_Engine)**

The web interface enables practitioners and researchers to load any historical asset supported by the Yahoo Finance API (or upload custom CSV market data), configure the physical discovery parameters, and inspect the real-time telemetry generated across all five processing layers.

![Figure 6.1: Interactive application interface and physical parameter configuration panel on HuggingFace Spaces.](./figures/fig_app_control_panel.png)
*Figure 6.1: Interactive application control panel showing pre-configured baseline hyperparameters, asset selection, and the automated CUSUM Drift auto-tune utility.*

### 6.2.1 Parameter Calibration and the "Auto-Tune" Mechanism

The control panel (Figure 6.1) comes pre-configured with mathematically sound baseline defaults optimized for standard daily financial time series:
- **Historical Window & Frequency:** Default 5-year lookback (`5y`) sampled at daily resolution (`1d`).
- **Predictive Horizon ($t+N$):** Default forward integration window $N = 60$ candles.
- **SINDy Polynomial Complexity:** Default degree $d = 1$ (linear and cross-coupling friction terms $\Theta(r, V)$), with options up to degree 3 for higher-order nonlinearities.
- **Spline Manifold Tolerance:** Default noise isolation threshold $\tau = 0.0050$.
- **CUSUM Anomaly Threshold ($H$):** Default structural break tolerance $H = 5.00$.

The sole parameter requiring asset-specific calibration is the **CUSUM Drift ($k$)**, which governs the algorithm's sensitivity to background noise versus genuine regime shifts. The application incorporates a dedicated **"Auto-Tune CUSUM Drift"** engine: upon pressing this button, the system computes an empirical optimization over the asset's historical residual volatility distribution and recommends the optimal drift factor (e.g., $k = 1.71$ for `BTC-USD`). Users are guided to adjust the slider to this recommended setting, while also encouraged to freely explore alternative parameter combinations to test model sensitivity across diverse volatility regimes.

### 6.2.2 Physical-Predictive Telemetry Dashboard

Upon execution, the engine displays an interactive multi-panel telemetry dashboard that visualizes the physical state of the asset across time, phase space, and equation space:

![Figure 6.2: Physical-Predictive Telemetry Dashboard generated for BTC-USD (5-year historical window).](./figures/fig_telemetry_dashboard.png)
*Figure 6.2: Physical-predictive telemetry dashboard showing: (Top) raw price noise, piecewise $C^2$ Spline inertia, deterministic ODE trajectory, and Monte Carlo stochastic cone; (Bottom-Left) Phase-Space Attractor radar; (Bottom-Right) CUSUM cumulative tension and regime shift triggers.*

1. **Inertia Manifold and Stochastic Uncertainty Cone (Figure 6.2, Top Panel):**
   - *Raw Discrete Price (Grey points):* High-frequency market noise.
   - *Piecewise $C^2$ Spline Manifold (Cyan and Orange curves):* Smooth continuous trajectory $P(t)$ partitioned by vertical dashed lines marking structural regime breaks detected by CUSUM.
   - *Deterministic Physical Projection (Cyan dashed curve):* Multi-step forward trajectory obtained by integrating the discovered active ODE $\dot{r} = f(r, V)$ forward in time.
   - *Monte Carlo Stochastic Cone (Purple shaded fan):* 1,000 Euler-Maruyama paths bounded by the 10th and 90th percentiles ($\text{P10}-\text{P90}$), quantifying the epistemic uncertainty derived from the residual diffusion term $\sigma_{\text{res}} \mathrm{d}W_t$.

2. **Phase-Space Attractor Radar (Figure 6.2, Bottom-Left Panel):**
   - Depicts the two-dimensional dynamical orbital trajectory in phase space ($\dot{r}$ vs. $\ddot{r}$ or velocity vs. acceleration), tracing the geometric convergence toward the current active regime's equilibrium attractor (yellow dot). This provides an instant visual diagnostic of market state: closed orbital loops indicate oscillatory range-bound regimes, while open divergent spirals identify breakout momentum.

3. **CUSUM Cumulative Tension Detector (Figure 6.2, Bottom-Right Panel):**
   - Displays real-time cumulative positive ($S^+$) and negative ($S^-$) shock tension. When cumulative pressure exceeds the horizontal anomaly boundary ($H = 5.0$, white dashed line), the algorithm triggers a structural break (yellow 'x'), isolating past momentum and initiating a fresh physical discovery window for the new active regime.

![Figure 6.3: Discovered SINDy Momentum ODEs and closed-form general solution of the dynamical attractor.](./figures/fig_sindy_equations.png)
*Figure 6.3: Explicit coupled governing ODEs discovered by SINDy for the active BTC-USD regime, alongside real-time pipeline performance telemetry and the continuous analytical attractor solution.*

4. **Momentum Governing Equations and System Telemetry (Figure 6.3):**
   - The engine renders the exact coupled first-order ordinary differential equations discovered by SINDy for the active market regime:
     $$\frac{\mathrm{d}r}{\mathrm{d}t} = -0.00045 + 0.00338\,r - 0.00042\,V$$
     $$\frac{\mathrm{d}V}{\mathrm{d}t} = -0.00079 + 0.02781\,r - 0.00078\,V$$
   - *Performance Telemetry:* Displays exact mathematical quality metrics in real time: Spline MSE ($0.000748$), CUSUM regime detection latency ($2.69\,\text{ms}$), SINDy fitting accuracy ($R^2 = 0.9581$ across 6 non-zero polynomial coefficients), and residual noise standard deviation ($\sigma_r = 0.02458$).
   - *Continuous General Solution:* Presents the analytical closed-form representation of the trajectory attractor $r(t)$, demonstrating that the discovered dynamics follow a damped, harmonically oscillating exponential envelope:
     $$r(t) = -0.14985 e^{0.0013 t} \sin(\omega t) - 0.00783 e^{0.0013 t} \cos(\omega t) - 0.00212 \sin^2(\omega t) - 0.00212 \cos^2(\omega t)$$
     where $\omega = 0.0027118$.

---

## 6.3 Epistemological Implications

The results of this work carry implications that extend beyond the benchmarked metrics and into the epistemological foundations of financial modelling.

The Efficient Market Hypothesis in its weak form \cite{malkiel1970efficient} predicts that historical price information cannot be used to generate sustained above-random directional accuracy. The Kinetopus results reject this prediction for momentum-coherent asset classes at $p < 0.0001$ across three evaluation horizons (B10, B30, B60). We do not claim that this invalidates the EMH globally — the GLD case (HR = 44.0% at B30) demonstrates that markets governed by macroeconomic fundamentals and exogenous policy shocks exhibit no recoverable physical geometry in the state space $(r, V)$. Rather, the evidence suggests a refinement: **informational efficiency is regime-dependent**, and markets dominated by inertial, momentum-driven participant behaviour exhibit locally coherent geometric dynamics that are neither random nor unpredictable by an appropriately designed physical model.

The deeper epistemological claim of this work is methodological: the distinction between *discovering a governing equation* and *fitting a statistical correlation*. When SINDy identifies the ODE $\dot{r} = f(r, V)$, it produces a causal, mechanistic statement about the local dynamics of the system — a statement that can be falsified, interpreted, and placed in dialogue with physical theory \cite{brunton2016discovering, raissi2019physics}. This is categorically different from the coefficients of an LSTM or the parameters of an ARIMA model, which are functional approximations of observed correlations without mechanistic content. The shift from correlation-based modelling to equation-based discovery is the central methodological contribution of this work, and it reflects a broader movement in scientific computing toward *interpretable, physics-grounded machine learning* \cite{chen2018neural}.

The relationship between financial markets and physical dynamics is not merely analogical — it is inherently reciprocal. Collective market participants, trading at high frequency on technical signals, create emergent inertial dynamics in the price-volume space that mirror physical systems: damped oscillators, phase-space attractors, bifurcation events. Studying this geometry does not only improve financial forecasting; it provides empirical evidence about the conditions under which complex adaptive systems exhibit physical laws. **Financial markets may thus serve as a large-scale, observable laboratory for physical law discovery** — an observation that we propose as a research agenda for the broader SciML community.

---

## 6.4 Limitations and Boundary Conditions

We report the following limitations with scientific objectivity, as they define the boundary conditions of the current model's validity.

**L1 — Regime Inapplicability (Fundamentalist Markets).** The GLD case (HR = 44.0% at B30, below the random-walk threshold) demonstrates that the physical state space $(r, V)$ is insufficient for assets whose price formation is governed by macroeconomic regime variables — Federal Reserve interest rate policy, inflation dynamics, geopolitical risk premia. These are discontinuous, exogenous shocks that violate the assumption of trajectory continuity on which SINDy discovery depends. The current model should not be applied to fundamentalist assets without first enriching the state vector with relevant macroeconomic observables.

**L2 — Long-Horizon Magnitude Error Accumulation.** At B60 (~15 months), Kinetopus median MAPE (42.38%) significantly exceeds LSTM (26.07%) and ARIMA (24.44%). Forward Euler integration accumulates a discretization error of $O(H \cdot \Delta t)$ over 300 steps, which compounds into trajectory divergence at long horizons despite the damping correction. This limits the model's utility for magnitude-sensitive applications (e.g., option pricing, VaR estimation) at multi-month horizons.

**L3 — Idealized Economic Evaluation.** The Long/Short Profit% figures reported in Section 5.4 assume zero transaction costs, perfect execution at close price, and static position sizing. These are standard idealizations for academic model evaluation \cite{fischer2018deep}, but real-world deployment would require accounting for bid-ask spread, market impact, and position risk constraints. The reported Profit% figures represent a theoretical upper bound on economic utility.

**L4 — Universe Scope.** The benchmark covers six assets across four classes. The generalization of these results to broader cross-sectional universes (fixed income, FX, commodity baskets, international equities) requires additional validation and may reveal asset classes with intermediate levels of physical coherence not captured by the current binary regime classification (inertial vs. fundamentalist).

---

## 6.5 Future Research Directions

The empirical and theoretical findings of this work open five concrete research directions, ordered by proximity to the core results.

**FW1 — Adaptive ODE Solvers for Long-Horizon Trajectory Fidelity.**
The primary driver of long-horizon magnitude error in Kinetopus is the $O(\Delta t^2)$ truncation error of the first-order Euler integrator. Replacing forward Euler with an adaptive Runge-Kutta solver (RK4 or Dormand-Prince RK45) would reduce the local truncation error to $O(\Delta t^5)$, potentially closing the MAPE gap between Kinetopus and LSTM at B60 without requiring additional model parameters. This extension would also enable automatic step-size control in regions of high ODE stiffness, improving robustness during extreme volatility episodes.

**FW2 — Analytical Criteria for ODE Parameter Selection.**
The current pipeline selects the polynomial library degree and the STLSQ sparsity threshold $\gamma_{\text{STLSQ}}$ empirically. A mathematically rigorous extension would derive selection criteria from first principles: using Ljung-Box residual whiteness tests and Lyapunov stability analysis to certify that the discovered equation is (a) dynamically stable (no divergent fixed points in the relevant state range) and (b) informationally exhaustive (residuals are uncorrelated Gaussian noise). These criteria would eliminate the need for empirical hyperparameter search and make the discovery process fully automatic and theoretically justified — moving the pipeline from data-driven optimization toward principled physical identification.

**FW3 — State Enrichment for Fundamentalist Regimes.**
The GLD limitation motivates a direct extension: augmenting the state vector $(r, V)$ with macroeconomic observables $\mathbf{m}_t \in \{\Delta\text{FedFunds}_t, \text{CPI momentum}_t, \text{VIX}_t, \text{DXY}_t\}$. SINDy applied to the enriched state space $(r, V, \mathbf{m})$ could potentially recover governing equations for assets currently outside the model's regime boundary. This would generalize the physical discovery paradigm from purely technical (inertial) to fundamentalist and hybrid markets, substantially expanding the practical applicability of the framework.

**FW4 — Multi-Asset Physical Law Discovery (Cross-Sectional SINDy).**
An open scientific question is whether assets within the same class (e.g., all large-cap tech equities, or all crypto assets) share a common governing equation structure — a class-level physical law — or whether each asset requires an independent ODE. A cross-sectional SINDy extension would pool data across assets to identify both shared and asset-specific equation terms, enabling transfer of physical knowledge across instruments and providing a basis for a theoretical physics of asset classes.

**FW5 — The Reciprocal Physics-Finance Paradigm.**
The long-term scientific vision of this work is bidirectional: physical laws help understand market dynamics, and the observed geometry of market dynamics can, in turn, illuminate physical principles operating in complex adaptive systems. Future work will explore whether the ODE structures discovered by SINDy across large asset universes exhibit universal scaling laws, bifurcation signatures, or attractor topologies that correspond to known physical phenomena — effectively using financial markets as an empirical laboratory for studying the physics of collective human behaviour. This reciprocal paradigm — *finance as physics, physics as finance* — represents the deepest contribution this line of research can make to both scientific communities.

---

*The Kinetopus Engine is openly accessible at [https://huggingface.co/spaces/Juan778/KineTopus_Engine](https://huggingface.co/spaces/Juan778/KineTopus_Engine). All benchmark data, evaluation scripts, and model code are available in the accompanying repository to support full reproducibility of the results reported in this paper.*

---
