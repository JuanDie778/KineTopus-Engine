# 1. Introduction

Financial time series forecasting remains one of the most challenging problems in applied mathematics, quantitative finance, and signal processing. Traditionally, asset price dynamics have been modeled under the paradigm of the Efficient Market Hypothesis (EMH) and continuous-time Brownian motion \cite{malkiel1970efficient}, treating short-term price fluctuations as unforecastable stochastic noise. Classical econometric models—such as AutoRegressive Integrated Moving Average (ARIMA) and Generalized Autoregressive Conditional Heteroskedasticity (GARCH)—rely heavily on linear stationarity assumptions. Consequently, their predictive distributions rapidly collapse toward the unconditional historical mean after a few time steps ($3-5$ steps), discarding non-linear directional momentum \cite{box1970time}.

In recent years, the quantitative finance community has pivoted toward non-parametric Machine Learning (ML) and Deep Learning (DL) architectures, including Long Short-Term Memory (LSTM) networks \cite{hochreiter1997long, fischer2018deep}, Temporal Convolutional Networks (TCNs), Transformer-based models \cite{vaswani2017attention, zhou2021informer}, and Physics-Informed Neural Networks (PINNs) \cite{raissi2019physics}. Although these high-capacity black-box approaches excel at fitting complex historical correlations, they present major operational and theoretical bottlenecks:
1. **High Computational Complexity:** Deep neural network architectures require intensive GPU infrastructure, raising the cost per forecast and complicating lightweight deployment.
2. **Opacity and Lack of Interpretability:** Deep learning models function as opaque function approximators, outputting point predictions without revealing the underlying physical laws or differential equations driving system inertia.
3. **Susceptibility to Structural Regime Shifts:** Out-of-sample generalization deteriorates significantly when market dynamics undergo abrupt structural shifts, leading to catastrophic overfitting and unseen model drift.

### 1.1 Motivation: Justifying a Continuous Physical Perspective

At first glance, financial market data appear inherently discrete, noisy, and stochastic. However, when observed through an appropriate continuous lens, price-volume dynamics often display localized geometric trajectories—resembling damped harmonic oscillations, phase-space attractors, and exponential decay regimes. 

While multiple analytical perspectives exist, a fundamental research question arises: **Can a localized continuous physical perspective be mathematically justified and empirically validated over discrete financial series?**

To answer this question, we introduce the **Kinetopus Engine**, a parsimonious Scientific Machine Learning (SciML) framework designed to discover governing equations from noisy time series under low computational complexity. Rather than fitting empirical parameters to discrete, fragmented price bars, Kinetopus lifts discrete time series $X_t \in \mathbb{R}^d$ into a continuous, $C^2$-differentiable topological manifold $X(t)$. By evaluating analytical derivatives $\dot{X}(t)$, the system leverages Sparse Identification of Nonlinear Dynamics (SINDy) \cite{brunton2016discovering, champion2019data} to extract explicit Ordinary Differential Equations (ODEs) governing local dynamic momentum.

### 1.2 White-Box Stochastic Differential Equation (SDE) Formulation

We formalize local price evolution through a generalized Stochastic Differential Equation (SDE) defined over log-returns:

$$ \mathrm{d}r_t = f_{\text{SINDy}}(r_t, V_t) \, \mathrm{d}t + \sigma_{\text{res}} \, \mathrm{d}W_t $$

where $r_t = \ln(P_t / P_{t-1})$ denotes the log-return at discrete time $t$, $V_t$ represents $Z$-score normalized transactional volume acting as an external forcing parameter, $f_{\text{SINDy}}$ is the parsimonious nonlinear dynamic function (drift term) identified via sparse regression over a polynomial library $\Theta(r, V)$—capturing crucial cross-friction interaction terms such as $r_t \cdot V_t$—$\sigma_{\text{res}}$ measures local residual volatility, and $W_t$ represents a standard Wiener process \cite{kloeden1992numerical}.

To prevent historical contamination across structural market breaks, Kinetopus employs online Cumulative Sum (CUSUM) regime slicing \cite{page1954continuous}. The pipeline does not fit a single global equation across the entire time series; instead, CUSUM partitions historical momentum into distinct, stationary regimes, identifying localized SINDy differential equations for each segment and projecting forward *only the active regime*. Furthermore, residual errors are subjected to Ljung-Box hypothesis testing \cite{ljung1978measure} to verify that $f_{\text{SINDy}}$ exhausts all deterministic signals, collapsing non-modeled dynamics into uncorrelated Gaussian white noise.

### 1.3 Key Contributions

The main contributions of this work are summarized as follows:

* **A Unified Continuous-to-Physical SciML Pipeline:** We propose a 5-layer computational framework integrating Fast Fourier Transform (FFT) spectral sensing, $C^2$ Spline topological manifold smoothing, CUSUM-guided regime slicing, SINDy sparse ODE discovery, and Monte Carlo Euler-Maruyama SDE integration.
* **Empirical Multi-Asset Validation (3,525 Iterations):** Through rigorous walk-forward expanding-window evaluation across a heterogeneous 6-asset universe spanning cryptocurrencies (`BTC-USD`, `ETH-USD`), equities (`MSFT`), broad-market indices (`QQQ`), financial sector ETFs (`XLF`), and commodities (`GLD`), we demonstrate a statistically significant aggregate directional Hit Ratio of **59.15% at 150 daily candles** ($z = 4.942, p < 0.0001$), peaking at **71.57% on QQQ** and **70.30% on MSFT**.
* **Hydrodynamic Trajectory Stabilization:** We introduce a physically motivated hydrodynamic friction decay parameter ($\lambda = 0.99$) into forward ODE integration, resolving unbounded energy accumulation over long horizons. In controlled ablation, this term improves mathematical validity from **87.83% to 98.60%** and reduces long-horizon (300-candle) median MAPE by **57.5%** (from 99.71% to 42.38%) while preserving 100% of the mid-horizon directional accuracy.
* **Sub-Millisecond Computational Efficiency:** We show that explicit nonlinear equations can be discovered and integrated on consumer-grade CPU hardware in **<1 ms per episode**—approximately **1,525× faster than deep neural network baselines** (LSTM at 1,524.72 ms)—eliminating DataFrame and GPU overhead in execution hot loops.
* **Phase-Space Telemetry & Open Interactive Deployment:** We introduce an interactive Phase-Space Attractor topology ($\dot{r}_t$ vs. $\ddot{r}_t$) and real-time physical telemetry dashboard. We deploy the complete, functional implementation openly on Hugging Face Spaces (`https://huggingface.co/spaces/Juan778/KineTopus_Engine`) to ensure full academic transparency and reproducible experimentation.
