# Abstract

Financial time series analysis has long been trapped in a dichotomy between linear, oversimplified statistical models (e.g., ARIMA) and opaque, computationally expensive deep learning black boxes (e.g., LSTMs, Transformers). This paper introduces **Kinetopus Engine**, a parsimonious, White-Box Scientific Machine Learning (SciML) framework that challenges the assumption of pure stochastic randomness by constructing a mathematically justified continuous physical perspective over discrete financial data.

Operating under low computational complexity on consumer-grade CPU hardware, the pipeline integrates five core stages:
1. Spectral sensing via Fast Fourier Transform (FFT) to detect dominant cyclical frequencies.
2. Topological manifold smoothing via $C^2$ Splines to reconstruct continuous price-volume trajectories.
3. Online Cumulative Sum (CUSUM) control mechanisms for real-time structural regime-shift slicing.
4. Sparse Identification of Nonlinear Dynamics (SINDy) to discover explicit, parsimonious ordinary differential equations $\mathrm{d}r/\mathrm{d}t = f(r, V)$.
5. Probabilistic Monte Carlo trajectory projection via Euler-Maruyama stochastic differential equations (SDEs) stabilized by a hydrodynamic friction damping factor ($\lambda = 0.99$).

Empirical evaluation through an extensive walk-forward benchmark encompassing 3,525 evaluation iterations across a heterogeneous six-asset universe (`BTC-USD`, `ETH-USD`, `MSFT`, `QQQ`, `XLF`, `GLD`) demonstrates a sustained aggregate directional Hit Ratio of **59.15%** at a 150-candle horizon (~7.5 months), achieving strong statistical significance against a random-walk null hypothesis ($z = 4.942, p < 0.0001$) and peaking at **71.57%** on broad-market index equities (`QQQ`) and **70.30%** on technology equities (`MSFT`). Furthermore, the physical engine operates at sub-millisecond CPU latency (<1 ms per episode, ~1,525× faster than an LSTM baseline) with a **98.60%** mathematical validity rate, delivering full phase-space interpretability and generating explicit, inspectable governing equations. A complete, interactive deployment is made openly accessible on Hugging Face Spaces.
