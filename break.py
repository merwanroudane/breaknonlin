import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.signal import find_peaks
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Structural Breaks & Nonlinearity", layout="wide", page_icon="📊")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #ff7f0e;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #2ca02c;
        padding-bottom: 0.5rem;
    }
    .info-box {
        background-color: #e6f3ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">Structural Breaks & Nonlinearity in Time Series</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Introduction to Advanced Topics in Time Series Analysis<br>by Dr. Merwan Roudane</div>',
    unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("📚 Navigation")
section = st.sidebar.radio(
    "Choose a section:",
    ["Introduction", "Structural Breaks", "Nonlinearity Concepts",
     "Detection Methods", "Simulations & Examples", "Advanced Preparation",
     "Interactive Tools", "Summary & References"]
)

# ============================================================================
# SECTION 1: INTRODUCTION
# ============================================================================
if section == "Introduction":
    st.markdown('<div class="section-header">1. Introduction to Structural Breaks and Nonlinearity</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 🎯 Learning Objectives")
        st.markdown("""
        - Understand structural breaks and their economic implications
        - Recognize nonlinear patterns in time series
        - Learn detection and testing methodologies
        - Prepare for advanced models (Threshold, Markov-Switching)
        - Apply practical simulation techniques
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 📖 Course Prerequisites")
        st.markdown("""
        - Basic time series analysis (AR, MA, ARMA)
        - Linear regression fundamentals
        - Probability and statistics
        - Python programming basics
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Why Study Structural Breaks and Nonlinearity?")

    st.markdown("""
    Traditional linear time series models assume **parameter stability** and **linear relationships**. 
    However, real-world data often violate these assumptions:
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("**📉 Economic Crises**")
        st.markdown("Financial crises cause sudden regime changes in economic relationships")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("**🏛️ Policy Changes**")
        st.markdown("Monetary/fiscal policy shifts alter economic dynamics")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("**🔄 Regime Switching**")
        st.markdown("Variables exhibit different behavior in different states")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Mathematical Foundation")

    st.markdown("#### Linear Model (Baseline)")
    st.latex(r"y_t = \beta_0 + \beta_1 x_t + \epsilon_t, \quad \epsilon_t \sim N(0, \sigma^2)")

    st.markdown("#### Model with Structural Break")
    st.latex(r"""
    y_t = \begin{cases}
    \beta_0^{(1)} + \beta_1^{(1)} x_t + \epsilon_t & \text{if } t \leq T_b \\
    \beta_0^{(2)} + \beta_1^{(2)} x_t + \epsilon_t & \text{if } t > T_b
    \end{cases}
    """)

    st.markdown("where $T_b$ is the break point.")

    st.markdown("#### Nonlinear Model (General Form)")
    st.latex(r"y_t = f(x_t, \theta) + \epsilon_t")

    st.markdown("where $f(\cdot)$ is a nonlinear function.")

    # Simple illustration
    st.markdown("### Visual Illustration")

    np.random.seed(42)
    t = np.arange(200)

    # Linear series
    y_linear = 2 + 0.5 * t + np.random.normal(0, 5, 200)

    # Series with break
    y_break = np.concatenate([
        2 + 0.5 * t[:100] + np.random.normal(0, 5, 100),
        -10 + 0.8 * t[100:] + np.random.normal(0, 5, 100)
    ])

    # Nonlinear series
    y_nonlinear = 50 + 20 * np.sin(t / 10) + np.random.normal(0, 3, 200)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Linear Series", "Structural Break", "Nonlinear Series")
    )

    fig.add_trace(go.Scatter(x=t, y=y_linear, mode='lines', name='Linear',
                             line=dict(color='#1f77b4')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=y_break, mode='lines', name='Break',
                             line=dict(color='#ff7f0e')), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=y_nonlinear, mode='lines', name='Nonlinear',
                             line=dict(color='#2ca02c')), row=1, col=3)

    fig.add_vline(x=100, line_dash="dash", line_color="red", row=1, col=2,
                  annotation_text="Break Point")

    fig.update_layout(height=400, showlegend=False, title_text="Three Types of Time Series Behavior")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# SECTION 2: STRUCTURAL BREAKS
# ============================================================================
elif section == "Structural Breaks":
    st.markdown('<div class="section-header">2. Structural Breaks in Time Series</div>', unsafe_allow_html=True)

    st.markdown("### 2.1 Definition and Types")

    st.markdown("""
    A **structural break** occurs when the parameters of a time series model change at one or more points in time.
    This reflects fundamental changes in the data-generating process.
    """)

    st.markdown("#### Types of Structural Breaks")

    tab1, tab2, tab3, tab4 = st.tabs(["Level Shift", "Trend Change", "Variance Change", "Combined"])

    with tab1:
        st.markdown("**Level Shift**: Change in the intercept")
        st.latex(r"""
        y_t = \begin{cases}
        \mu_1 + \epsilon_t & t \leq T_b \\
        \mu_2 + \epsilon_t & t > T_b
        \end{cases}
        """)

        t = np.arange(200)
        y_level = np.concatenate([
            10 + np.random.normal(0, 2, 100),
            25 + np.random.normal(0, 2, 100)
        ])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y_level, mode='lines', name='Series',
                                 line=dict(color='#1f77b4')))
        fig.add_vline(x=100, line_dash="dash", line_color="red", annotation_text="Level Shift")
        fig.add_hline(y=10, line_dash="dot", line_color="green", annotation_text="μ₁=10")
        fig.add_hline(y=25, line_dash="dot", line_color="orange", annotation_text="μ₂=25")
        fig.update_layout(title="Level Shift Example", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("**Trend Change**: Change in the slope")
        st.latex(r"""
        y_t = \begin{cases}
        \beta_0 + \beta_1^{(1)} t + \epsilon_t & t \leq T_b \\
        \beta_0 + \beta_1^{(2)} t + \epsilon_t & t > T_b
        \end{cases}
        """)

        y_trend = np.concatenate([
            0.2 * t[:100] + np.random.normal(0, 2, 100),
            20 - 0.1 * t[100:] + np.random.normal(0, 2, 100)
        ])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y_trend, mode='lines', name='Series',
                                 line=dict(color='#ff7f0e')))
        fig.add_vline(x=100, line_dash="dash", line_color="red", annotation_text="Trend Change")
        fig.update_layout(title="Trend Change Example", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("**Variance Change**: Change in volatility")
        st.latex(r"""
        y_t = \mu + \epsilon_t, \quad \epsilon_t \sim \begin{cases}
        N(0, \sigma_1^2) & t \leq T_b \\
        N(0, \sigma_2^2) & t > T_b
        \end{cases}
        """)

        y_var = np.concatenate([
            10 + np.random.normal(0, 1, 100),
            10 + np.random.normal(0, 5, 100)
        ])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y_var, mode='lines', name='Series',
                                 line=dict(color='#2ca02c')))
        fig.add_vline(x=100, line_dash="dash", line_color="red", annotation_text="Variance Change")
        fig.update_layout(title="Variance Change Example", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("**Combined Break**: Multiple parameters change simultaneously")

        y_combined = np.concatenate([
            10 + 0.1 * t[:100] + np.random.normal(0, 1, 100),
            30 - 0.05 * t[100:] + np.random.normal(0, 3, 100)
        ])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y_combined, mode='lines', name='Series',
                                 line=dict(color='#d62728')))
        fig.add_vline(x=100, line_dash="dash", line_color="red",
                      annotation_text="Combined Break (Level+Trend+Variance)")
        fig.update_layout(title="Combined Structural Break Example", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 2.2 Economic Examples")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("#### 📈 Real-World Examples")
        st.markdown("""
        1. **2008 Financial Crisis**: Sharp break in GDP growth, unemployment
        2. **Oil Price Shocks**: 1973, 1979 oil crises
        3. **COVID-19 Pandemic**: 2020 economic disruption
        4. **Policy Regime Changes**: Volcker disinflation (1979-1982)
        5. **Brexit**: UK economic indicators post-2016
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Implications")
        st.markdown("""
        - **Forecasting**: Breaks reduce forecast accuracy
        - **Model Selection**: Need to account for instability
        - **Policy Analysis**: Pre/post-break periods differ
        - **Testing**: Standard tests may be invalid
        - **Estimation**: Biased parameters if ignored
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 2.3 Chow Test for Known Break Point")

    st.markdown("""
    The **Chow test** tests whether coefficients differ across two sub-samples when the break point is known.
    """)

    st.markdown("#### Null Hypothesis:")
    st.latex(r"H_0: \beta^{(1)} = \beta^{(2)} \text{ (no structural break)}")

    st.markdown("#### Test Statistic:")
    st.latex(r"""
    F = \frac{(RSS_r - RSS_1 - RSS_2) / k}{(RSS_1 + RSS_2) / (n - 2k)} \sim F(k, n-2k)
    """)

    st.markdown("""
    where:
    - $RSS_r$: Residual sum of squares from restricted model (full sample)
    - $RSS_1$: RSS from first sub-sample
    - $RSS_2$: RSS from second sub-sample
    - $k$: number of parameters
    - $n$: total sample size
    """)

    # Chow test simulation
    st.markdown("#### Interactive Chow Test Simulation")

    col1, col2 = st.columns(2)
    with col1:
        break_magnitude = st.slider("Break Magnitude (slope change)", 0.0, 2.0, 0.5, 0.1)
    with col2:
        noise_level = st.slider("Noise Level", 1, 10, 3, 1)

    np.random.seed(42)
    n = 200
    break_point = 100
    t = np.arange(n)

    # Generate data with break
    x = np.random.normal(0, 1, n)
    y1 = 5 + 0.5 * x[:break_point] + np.random.normal(0, noise_level, break_point)
    y2 = 5 + (0.5 + break_magnitude) * x[break_point:] + np.random.normal(0, noise_level, n - break_point)
    y = np.concatenate([y1, y2])

    # Estimate models
    from numpy.linalg import lstsq

    X = np.column_stack([np.ones(n), x])
    X1 = X[:break_point]
    X2 = X[break_point:]

    # Full sample
    beta_full = lstsq(X, y, rcond=None)[0]
    y_pred_full = X @ beta_full
    rss_r = np.sum((y - y_pred_full) ** 2)

    # Sub-samples
    beta1 = lstsq(X1, y[:break_point], rcond=None)[0]
    beta2 = lstsq(X2, y[break_point:], rcond=None)[0]

    y_pred1 = X1 @ beta1
    y_pred2 = X2 @ beta2

    rss_1 = np.sum((y[:break_point] - y_pred1) ** 2)
    rss_2 = np.sum((y[break_point:] - y_pred2) ** 2)

    # Chow statistic
    k = 2
    f_stat = ((rss_r - rss_1 - rss_2) / k) / ((rss_1 + rss_2) / (n - 2 * k))
    f_critical = stats.f.ppf(0.95, k, n - 2 * k)
    p_value = 1 - stats.f.cdf(f_stat, k, n - 2 * k)

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode='markers', name='Data',
                             marker=dict(size=4, color='lightblue')))
    fig.add_trace(go.Scatter(x=t[:break_point], y=y_pred1, mode='lines',
                             name='Regime 1', line=dict(color='green', width=3)))
    fig.add_trace(go.Scatter(x=t[break_point:], y=y_pred2, mode='lines',
                             name='Regime 2', line=dict(color='red', width=3)))
    fig.add_vline(x=break_point, line_dash="dash", line_color="black",
                  annotation_text="Known Break Point")

    fig.update_layout(title="Chow Test Illustration", height=400)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("F-Statistic", f"{f_stat:.4f}")
    col2.metric("Critical Value (5%)", f"{f_critical:.4f}")
    col3.metric("P-Value", f"{p_value:.4f}")

    if p_value < 0.05:
        st.success("✅ **Reject H₀**: Significant structural break detected!")
    else:
        st.info("❌ **Fail to reject H₀**: No significant structural break")

    st.markdown(f"""
    **Interpretation:**
    - Regime 1 slope: {beta1[1]:.4f}
    - Regime 2 slope: {beta2[1]:.4f}
    - Slope difference: {beta2[1] - beta1[1]:.4f}
    """)

# ============================================================================
# SECTION 3: NONLINEARITY CONCEPTS
# ============================================================================
elif section == "Nonlinearity Concepts":
    st.markdown('<div class="section-header">3. Nonlinearity in Time Series</div>', unsafe_allow_html=True)

    st.markdown("### 3.1 What is Nonlinearity?")

    st.markdown("""
    A time series is **nonlinear** if its behavior cannot be adequately described by a linear model.
    Nonlinearity can manifest in various forms:
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("#### Linear Model")
        st.latex(r"y_t = \phi_1 y_{t-1} + \phi_2 y_{t-2} + \epsilon_t")
        st.markdown("""
        **Characteristics:**
        - Constant parameters
        - Symmetric shocks
        - Linear impulse responses
        - Gaussian innovations sufficient
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("#### Nonlinear Model")
        st.latex(r"y_t = f(y_{t-1}, y_{t-2}, \ldots, \epsilon_t)")
        st.markdown("""
        **Characteristics:**
        - State-dependent dynamics
        - Asymmetric responses
        - Regime-dependent behavior
        - Complex dynamics (cycles, chaos)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 3.2 Types of Nonlinearity")

    tabs = st.tabs(["Threshold", "Smooth Transition", "Markov-Switching", "GARCH", "Bilinear"])

    with tabs[0]:
        st.markdown("#### Threshold Autoregressive (TAR) Model")
        st.latex(r"""
        y_t = \begin{cases}
        \phi_1^{(1)} y_{t-1} + \epsilon_t & \text{if } y_{t-d} \leq \tau \\
        \phi_1^{(2)} y_{t-1} + \epsilon_t & \text{if } y_{t-d} > \tau
        \end{cases}
        """)

        st.markdown("""
        - **Threshold variable**: $y_{t-d}$ (delay $d$)
        - **Threshold value**: $\\tau$
        - **Regimes**: Two (or more) distinct regimes
        - **Transition**: Abrupt/discrete
        """)

        # TAR simulation
        np.random.seed(42)
        n = 300
        y = np.zeros(n)
        y[0] = 0
        tau = 0
        phi1_low = 0.5
        phi1_high = -0.3

        for t in range(1, n):
            if y[t - 1] <= tau:
                y[t] = phi1_low * y[t - 1] + np.random.normal(0, 0.5)
            else:
                y[t] = phi1_high * y[t - 1] + np.random.normal(0, 0.5)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(n), y=y, mode='lines', name='TAR Series',
                                 line=dict(color='#1f77b4')))
        fig.add_hline(y=tau, line_dash="dash", line_color="red", annotation_text="Threshold τ=0")
        fig.update_layout(title="Threshold Autoregressive Model Example", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.markdown("#### Smooth Transition AR (STAR) Model")
        st.latex(r"""
        y_t = (\phi_1^{(1)} y_{t-1}) \cdot (1 - G(s_t)) + (\phi_1^{(2)} y_{t-1}) \cdot G(s_t) + \epsilon_t
        """)
        st.latex(r"G(s_t) = \frac{1}{1 + \exp(-\gamma(s_t - c))}")

        st.markdown("""
        - **Transition function**: $G(s_t)$ (logistic)
        - **Transition variable**: $s_t$ (e.g., $y_{t-d}$)
        - **Smoothness**: $\\gamma$ (speed of transition)
        - **Threshold**: $c$
        """)

        # STAR simulation
        t_arr = np.arange(300)
        s = np.linspace(-3, 3, 300)
        gamma = 2
        c = 0
        G = 1 / (1 + np.exp(-gamma * (s - c)))

        y_star = 0.8 * s * (1 - G) + (-0.3) * s * G + np.random.normal(0, 0.3, 300)

        fig = make_subplots(rows=2, cols=1, subplot_titles=("Transition Function G(s)", "STAR Series"))

        fig.add_trace(go.Scatter(x=s, y=G, mode='lines', name='G(s)',
                                 line=dict(color='#ff7f0e', width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t_arr, y=y_star, mode='lines', name='STAR',
                                 line=dict(color='#2ca02c')), row=2, col=1)

        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.markdown("#### Markov-Switching Model")
        st.latex(r"""
        y_t = \mu_{S_t} + \phi_{S_t} y_{t-1} + \sigma_{S_t} \epsilon_t
        """)
        st.latex(r"P(S_t = j | S_{t-1} = i) = p_{ij}")

        st.markdown("""
        - **State variable**: $S_t \\in \\{1, 2, \\ldots, M\\}$ (latent/unobserved)
        - **Transition probabilities**: $p_{ij}$
        - **State-dependent parameters**: $\\mu_{S_t}, \\phi_{S_t}, \\sigma_{S_t}$
        - **Stochastic switching**: Probabilistic regime changes
        """)

        # MS simulation
        np.random.seed(42)
        n = 300
        states = np.zeros(n, dtype=int)
        y_ms = np.zeros(n)

        p11 = 0.95  # Prob of staying in state 1
        p22 = 0.90  # Prob of staying in state 2

        states[0] = 0
        y_ms[0] = 0

        for t in range(1, n):
            if states[t - 1] == 0:
                states[t] = 0 if np.random.rand() < p11 else 1
            else:
                states[t] = 1 if np.random.rand() < p22 else 0

            if states[t] == 0:
                y_ms[t] = 2 + 0.5 * y_ms[t - 1] + np.random.normal(0, 0.5)
            else:
                y_ms[t] = -1 + 0.3 * y_ms[t - 1] + np.random.normal(0, 1.5)

        fig = make_subplots(rows=2, cols=1, subplot_titles=("Regime States", "MS Series"),
                            row_heights=[0.3, 0.7])

        fig.add_trace(go.Scatter(x=np.arange(n), y=states, mode='lines', name='State',
                                 line=dict(color='purple', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=np.arange(n), y=y_ms, mode='lines', name='MS Series',
                                 line=dict(color='#d62728')), row=2, col=1)

        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.markdown("#### GARCH: Conditional Heteroskedasticity")
        st.latex(r"y_t = \sigma_t \epsilon_t, \quad \epsilon_t \sim N(0,1)")
        st.latex(r"\sigma_t^2 = \omega + \alpha y_{t-1}^2 + \beta \sigma_{t-1}^2")

        st.markdown("""
        - **Nonlinearity in variance**: Volatility clustering
        - **Applications**: Financial returns
        - **Features**: Large shocks followed by large shocks
        """)

        # GARCH simulation
        np.random.seed(42)
        n = 500
        omega, alpha, beta = 0.1, 0.15, 0.80

        sigma2 = np.zeros(n)
        y_garch = np.zeros(n)
        sigma2[0] = omega / (1 - alpha - beta)

        for t in range(n):
            epsilon = np.random.normal(0, 1)
            y_garch[t] = np.sqrt(sigma2[t]) * epsilon
            if t < n - 1:
                sigma2[t + 1] = omega + alpha * y_garch[t] ** 2 + beta * sigma2[t]

        fig = make_subplots(rows=2, cols=1, subplot_titles=("GARCH Returns", "Conditional Volatility"))

        fig.add_trace(go.Scatter(x=np.arange(n), y=y_garch, mode='lines', name='Returns',
                                 line=dict(color='#1f77b4')), row=1, col=1)
        fig.add_trace(go.Scatter(x=np.arange(n), y=np.sqrt(sigma2), mode='lines', name='σₜ',
                                 line=dict(color='#ff7f0e', width=2)), row=2, col=1)

        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        st.markdown("#### Bilinear Model")
        st.latex(r"y_t = \phi y_{t-1} + \beta y_{t-1} \epsilon_{t-1} + \epsilon_t")

        st.markdown("""
        - **Interaction term**: $y_{t-1} \\epsilon_{t-1}$
        - **Nonlinear in mean**: Shocks affect dynamics
        - **Applications**: Economic interactions
        """)

        # Bilinear simulation
        np.random.seed(42)
        n = 300
        phi = 0.3
        beta = 0.5

        y_bil = np.zeros(n)
        eps = np.random.normal(0, 1, n)

        for t in range(1, n):
            y_bil[t] = phi * y_bil[t - 1] + beta * y_bil[t - 1] * eps[t - 1] + eps[t]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(n), y=y_bil, mode='lines', name='Bilinear',
                                 line=dict(color='#9467bd')))
        fig.update_layout(title="Bilinear Model Example", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 3.3 Why Does Nonlinearity Matter?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("**📊 Better Fit**")
        st.markdown("Captures asymmetries and regime changes in data")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("**🎯 Improved Forecasts**")
        st.markdown("State-dependent forecasts can outperform linear models")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("**🔬 Economic Insight**")
        st.markdown("Reflects real-world complexities (recessions vs expansions)")
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# SECTION 4: DETECTION METHODS
# ============================================================================
elif section == "Detection Methods":
    st.markdown('<div class="section-header">4. Detection and Testing Methods</div>', unsafe_allow_html=True)

    st.markdown("### 4.1 Tests for Structural Breaks")

    st.markdown("#### 4.1.1 Chow Test (Known Break Date)")
    st.markdown("Already covered - use when break date is known *a priori*")

    st.markdown("#### 4.1.2 Quandt Likelihood Ratio (QLR) Test")

    st.markdown("""
    When the break date is **unknown**, we test all possible break points:
    """)

    st.latex(r"QLR = \max_{T_b \in [\pi T, (1-\pi)T]} F_{Chow}(T_b)")

    st.markdown("""
    - Search over possible break dates (typically middle 70% of sample: $\\pi = 0.15$)
    - Choose the break date with maximum F-statistic
    - Critical values differ from standard Chow test (use Andrews, 1993)
    """)

    # QLR simulation
    st.markdown("#### Interactive QLR Test")

    col1, col2 = st.columns(2)
    with col1:
        true_break = st.slider("True Break Point", 50, 150, 100, 10)
    with col2:
        break_size = st.slider("Break Size", 0.0, 3.0, 1.0, 0.2)

    np.random.seed(42)
    n = 200
    t = np.arange(n)
    x = np.random.normal(0, 1, n)

    y = np.concatenate([
        5 + 0.5 * x[:true_break] + np.random.normal(0, 2, true_break),
        5 + (0.5 + break_size) * x[true_break:] + np.random.normal(0, 2, n - true_break)
    ])

    # Calculate F-stats for all possible breaks
    trim = 0.15
    possible_breaks = range(int(n * trim), int(n * (1 - trim)))
    f_stats = []

    X = np.column_stack([np.ones(n), x])
    beta_full = np.linalg.lstsq(X, y, rcond=None)[0]
    rss_r = np.sum((y - X @ beta_full) ** 2)

    for bp in possible_breaks:
        X1 = X[:bp]
        X2 = X[bp:]

        beta1 = np.linalg.lstsq(X1, y[:bp], rcond=None)[0]
        beta2 = np.linalg.lstsq(X2, y[bp:], rcond=None)[0]

        rss1 = np.sum((y[:bp] - X1 @ beta1) ** 2)
        rss2 = np.sum((y[bp:] - X2 @ beta2) ** 2)

        k = 2
        f = ((rss_r - rss1 - rss2) / k) / ((rss1 + rss2) / (n - 2 * k))
        f_stats.append(f)

    estimated_break = list(possible_breaks)[np.argmax(f_stats)]
    max_f = max(f_stats)

    # Plot F-statistics
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("F-Statistics Across Break Points", "Data with Estimated Break"))

    fig.add_trace(go.Scatter(x=list(possible_breaks), y=f_stats, mode='lines', name='F-stat',
                             line=dict(color='#1f77b4', width=2)), row=1, col=1)
    fig.add_vline(x=estimated_break, line_dash="dash", line_color="red", row=1, col=1,
                  annotation_text=f"Estimated: {estimated_break}")
    fig.add_vline(x=true_break, line_dash="dot", line_color="green", row=1, col=1,
                  annotation_text=f"True: {true_break}")

    fig.add_trace(go.Scatter(x=t, y=y, mode='markers', name='Data',
                             marker=dict(size=4, color='lightblue')), row=2, col=1)
    fig.add_vline(x=estimated_break, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_vline(x=true_break, line_dash="dot", line_color="green", row=2, col=1)

    fig.update_layout(height=700, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("True Break", true_break)
    col2.metric("Estimated Break", estimated_break)
    col3.metric("Max F-Statistic", f"{max_f:.2f}")

    st.markdown("#### 4.1.3 CUSUM Test")

    st.markdown("""
    **Cumulative Sum (CUSUM)** test detects parameter instability by examining recursive residuals:
    """)

    st.latex(r"CUSUM_t = \sum_{j=k+1}^{t} \frac{w_j}{s}, \quad t = k+1, \ldots, T")

    st.markdown("where $w_j$ are recursive residuals and $s$ is their standard deviation.")

    # CUSUM calculation
    n = 200
    recursive_resids = np.concatenate([
        np.random.normal(0, 1, 100),
        np.random.normal(1.5, 1, 100)  # Shift in mean
    ])

    cusum = np.cumsum(recursive_resids) / np.std(recursive_resids)

    # 5% significance bounds
    a = 0.948  # For 5% level
    upper_bound = a * np.sqrt(n) + 2 * a * (np.arange(n) - 0) / np.sqrt(n)
    lower_bound = -upper_bound

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(n), y=cusum, mode='lines', name='CUSUM',
                             line=dict(color='#1f77b4', width=2)))
    fig.add_trace(go.Scatter(x=np.arange(n), y=upper_bound, mode='lines', name='Upper 5%',
                             line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=np.arange(n), y=lower_bound, mode='lines', name='Lower 5%',
                             line=dict(color='red', dash='dash')))

    fig.update_layout(title="CUSUM Test for Parameter Stability", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Interpretation**: If CUSUM crosses the boundaries, reject parameter stability.
    """)

    st.markdown("### 4.2 Tests for Nonlinearity")

    st.markdown("#### 4.2.1 BDS Test")

    st.markdown("""
    The **BDS test** (Brock-Dechert-Scheinkman) tests for independence and identical distribution,
    detecting nonlinear dependence that linear models miss.
    """)

    st.latex(r"BDS = \frac{\sqrt{T}(C_m(\epsilon) - C_1(\epsilon)^m)}{\sigma_m(\epsilon)}")

    st.markdown("""
    - $C_m(\\epsilon)$: Correlation integral at embedding dimension $m$
    - Under $H_0$ (iid): BDS $\\sim N(0,1)$
    - Rejects for nonlinear structure (chaos, GARCH, etc.)
    """)

    st.markdown("#### 4.2.2 Teräsvirta Test for Threshold/STAR")

    st.markdown("""
    Tests linearity against threshold or smooth transition alternatives using auxiliary regression:
    """)

    st.latex(r"y_t = \beta_0 + \beta_1 y_{t-1} + \beta_2 y_{t-1}^2 + \beta_3 y_{t-1}^3 + \epsilon_t")

    st.markdown("""
    - $H_0$: $\\beta_2 = \\beta_3 = 0$ (linearity)
    - $H_1$: At least one nonzero (nonlinearity)
    - Use F-test or LM test
    """)

    st.markdown("#### 4.2.3 Tsay Test")

    st.markdown("""
    Tests for threshold nonlinearity by arranging data based on threshold variable:
    """)

    st.latex(r"H_0: E[y_t | y_{t-d} \text{ small}] = E[y_t | y_{t-d} \text{ large}]")

    st.markdown("### 4.3 Visual Diagnostics")

    tab1, tab2, tab3 = st.tabs(["Recursive Residuals", "ACF of Residuals²", "Phase Diagram"])

    with tab1:
        st.markdown("**Recursive residuals** should be stable over time if model is correctly specified.")

        t_diag = np.arange(200)
        rec_res = np.concatenate([
            np.random.normal(0, 1, 100),
            np.random.normal(0, 2.5, 100)
        ])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_diag, y=rec_res, mode='markers', name='Recursive Residuals',
                                 marker=dict(size=5, color='#1f77b4')))
        fig.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="±2 SD")
        fig.add_hline(y=-2, line_dash="dash", line_color="red")
        fig.update_layout(title="Recursive Residuals Plot", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("**ACF of squared residuals** detects GARCH effects and nonlinear dependence.")

        from statsmodels.tsa.stattools import acf

        # Generate GARCH-like residuals
        eps = np.random.normal(0, 1, 500)
        sigma2 = np.zeros(500)
        sigma2[0] = 1

        for t in range(1, 500):
            sigma2[t] = 0.1 + 0.15 * eps[t - 1] ** 2 + 0.80 * sigma2[t - 1]
            eps[t] = np.sqrt(sigma2[t]) * eps[t]

        acf_vals = acf(eps ** 2, nlags=20)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=np.arange(len(acf_vals)), y=acf_vals, name='ACF',
                             marker_color='#2ca02c'))
        fig.add_hline(y=1.96 / np.sqrt(500), line_dash="dash", line_color="red")
        fig.add_hline(y=-1.96 / np.sqrt(500), line_dash="dash", line_color="red")
        fig.update_layout(title="ACF of Squared Residuals (GARCH detection)", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("**Phase diagram** ($y_t$ vs $y_{t-1}$) reveals nonlinear patterns.")

        # Generate TAR series
        n = 500
        y_phase = np.zeros(n)
        for t in range(1, n):
            if y_phase[t - 1] < 0:
                y_phase[t] = 0.7 * y_phase[t - 1] + np.random.normal(0, 0.5)
            else:
                y_phase[t] = -0.4 * y_phase[t - 1] + np.random.normal(0, 0.5)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_phase[:-1], y=y_phase[1:], mode='markers',
                                 marker=dict(size=4, color='#d62728', opacity=0.5)))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(title="Phase Diagram (yₜ vs yₜ₋₁)",
                          xaxis_title="yₜ₋₁", yaxis_title="yₜ", height=500)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# SECTION 5: SIMULATIONS & EXAMPLES
# ============================================================================
elif section == "Simulations & Examples":
    st.markdown('<div class="section-header">5. Simulations and Practical Examples</div>', unsafe_allow_html=True)

    st.markdown("""
    Simulations help us understand the behavior of structural breaks and nonlinear models 
    under controlled conditions. This is crucial for:
    - Understanding model properties
    - Validating detection methods
    - Building intuition for real data analysis
    """)

    sim_type = st.selectbox(
        "Select Simulation Type:",
        ["Structural Break Simulation", "TAR Model", "STAR Model",
         "Markov-Switching", "GARCH Model", "Comparison Study"]
    )

    if sim_type == "Structural Break Simulation":
        st.markdown("### Structural Break Simulation")

        col1, col2, col3 = st.columns(3)
        with col1:
            n_obs = st.slider("Sample Size", 100, 500, 200, 50)
        with col2:
            break_type = st.selectbox("Break Type", ["Level", "Trend", "Variance", "Combined"])
        with col3:
            noise = st.slider("Noise Level", 0.5, 5.0, 2.0, 0.5)

        np.random.seed(42)
        t = np.arange(n_obs)
        break_point = n_obs // 2

        if break_type == "Level":
            y = np.concatenate([
                10 + np.random.normal(0, noise, break_point),
                20 + np.random.normal(0, noise, n_obs - break_point)
            ])
        elif break_type == "Trend":
            y = np.concatenate([
                0.2 * t[:break_point] + np.random.normal(0, noise, break_point),
                20 - 0.1 * t[break_point:] + np.random.normal(0, noise, n_obs - break_point)
            ])
        elif break_type == "Variance":
            y = np.concatenate([
                10 + np.random.normal(0, noise, break_point),
                10 + np.random.normal(0, noise * 3, n_obs - break_point)
            ])
        else:  # Combined
            y = np.concatenate([
                10 + 0.1 * t[:break_point] + np.random.normal(0, noise, break_point),
                25 - 0.05 * t[break_point:] + np.random.normal(0, noise * 2, n_obs - break_point)
            ])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y, mode='lines+markers', name='Series',
                                 line=dict(color='#1f77b4'), marker=dict(size=4)))
        fig.add_vline(x=break_point, line_dash="dash", line_color="red",
                      annotation_text=f"Break at t={break_point}")

        # Add regime means
        mean1 = np.mean(y[:break_point])
        mean2 = np.mean(y[break_point:])

        fig.add_hline(y=mean1, line_dash="dot", line_color="green",
                      annotation_text=f"Pre-break mean: {mean1:.2f}")
        fig.add_hline(y=mean2, line_dash="dot", line_color="orange",
                      annotation_text=f"Post-break mean: {mean2:.2f}")

        fig.update_layout(title=f"{break_type} Shift Simulation", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pre-break Mean", f"{mean1:.3f}")
        col2.metric("Post-break Mean", f"{mean2:.3f}")
        col3.metric("Pre-break Std", f"{np.std(y[:break_point]):.3f}")
        col4.metric("Post-break Std", f"{np.std(y[break_point:]):.3f}")

    elif sim_type == "TAR Model":
        st.markdown("### Threshold Autoregressive (TAR) Model")

        st.latex(r"""
        y_t = \begin{cases}
        \phi_1^{(L)} y_{t-1} + \phi_2^{(L)} y_{t-2} + \epsilon_t & \text{if } y_{t-d} \leq \tau \\
        \phi_1^{(H)} y_{t-1} + \phi_2^{(H)} y_{t-2} + \epsilon_t & \text{if } y_{t-d} > \tau
        \end{cases}
        """)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            phi1_low = st.slider("φ₁ (Low regime)", -1.0, 1.0, 0.6, 0.1)
        with col2:
            phi1_high = st.slider("φ₁ (High regime)", -1.0, 1.0, -0.3, 0.1)
        with col3:
            threshold = st.slider("Threshold τ", -2.0, 2.0, 0.0, 0.5)
        with col4:
            n_tar = st.slider("Sample Size", 200, 1000, 500, 100)

        np.random.seed(42)
        y_tar = np.zeros(n_tar)
        y_tar[0] = 0
        regimes = np.zeros(n_tar)

        for t in range(1, n_tar):
            if y_tar[t - 1] <= threshold:
                y_tar[t] = phi1_low * y_tar[t - 1] + np.random.normal(0, 1)
                regimes[t] = 0
            else:
                y_tar[t] = phi1_high * y_tar[t - 1] + np.random.normal(0, 1)
                regimes[t] = 1

        fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                            subplot_titles=("TAR Series", "Regime Indicator"))

        fig.add_trace(go.Scatter(x=np.arange(n_tar), y=y_tar, mode='lines', name='TAR',
                                 line=dict(color='#1f77b4')), row=1, col=1)
        fig.add_hline(y=threshold, line_dash="dash", line_color="red", row=1, col=1,
                      annotation_text=f"Threshold={threshold}")

        fig.add_trace(go.Scatter(x=np.arange(n_tar), y=regimes, mode='lines', name='Regime',
                                 line=dict(color='purple'), fill='tozeroy'), row=2, col=1)

        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Regime statistics
        low_regime_obs = np.sum(regimes == 0)
        high_regime_obs = np.sum(regimes == 1)

        col1, col2, col3 = st.columns(3)
        col1.metric("Low Regime %", f"{100 * low_regime_obs / n_tar:.1f}%")
        col2.metric("High Regime %", f"{100 * high_regime_obs / n_tar:.1f}%")
        col3.metric("Threshold Crossings", f"{np.sum(np.diff(regimes) != 0)}")

        # Phase diagram
        st.markdown("#### Phase Diagram")
        fig = go.Figure()

        colors = ['blue' if r == 0 else 'red' for r in regimes[:-1]]

        fig.add_trace(go.Scatter(x=y_tar[:-1], y=y_tar[1:], mode='markers',
                                 marker=dict(size=4, color=colors, opacity=0.6),
                                 name='Phase'))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=threshold, line_dash="dash", line_color="red")

        fig.update_layout(title="Phase Diagram (colored by regime)",
                          xaxis_title="yₜ₋₁", yaxis_title="yₜ", height=500)
        st.plotly_chart(fig, use_container_width=True)

    elif sim_type == "STAR Model":
        st.markdown("### Smooth Transition AR (STAR) Model")

        st.latex(r"y_t = (\phi_1^{(1)} y_{t-1})(1 - G(y_{t-1})) + (\phi_1^{(2)} y_{t-1})G(y_{t-1}) + \epsilon_t")
        st.latex(r"G(s) = \frac{1}{1 + \exp(-\gamma(s - c))}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            phi_1 = st.slider("φ₁ (Regime 1)", -1.0, 1.0, 0.8, 0.1)
        with col2:
            phi_2 = st.slider("φ₂ (Regime 2)", -1.0, 1.0, -0.5, 0.1)
        with col3:
            gamma = st.slider("γ (smoothness)", 0.1, 10.0, 2.0, 0.5)
        with col4:
            c = st.slider("c (threshold)", -2.0, 2.0, 0.0, 0.5)

        np.random.seed(42)
        n_star = 500
        y_star = np.zeros(n_star)
        y_star[0] = 0
        G_vals = np.zeros(n_star)

        for t in range(1, n_star):
            G = 1 / (1 + np.exp(-gamma * (y_star[t - 1] - c)))
            G_vals[t] = G
            y_star[t] = phi_1 * y_star[t - 1] * (1 - G) + phi_2 * y_star[t - 1] * G + np.random.normal(0, 0.5)

        fig = make_subplots(rows=3, cols=1, row_heights=[0.5, 0.25, 0.25],
                            subplot_titles=("STAR Series", "Transition Function G(yₜ₋₁)", "Effective φₜ"))

        fig.add_trace(go.Scatter(x=np.arange(n_star), y=y_star, mode='lines', name='STAR',
                                 line=dict(color='#2ca02c')), row=1, col=1)

        fig.add_trace(go.Scatter(x=np.arange(n_star), y=G_vals, mode='lines', name='G',
                                 line=dict(color='orange')), row=2, col=1)

        phi_eff = phi_1 * (1 - G_vals) + phi_2 * G_vals
        fig.add_trace(go.Scatter(x=np.arange(n_star), y=phi_eff, mode='lines', name='φₑff',
                                 line=dict(color='purple')), row=3, col=1)

        fig.update_layout(height=700, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Transition function plot
        st.markdown("#### Transition Function Shape")
        s_range = np.linspace(-3, 3, 100)
        G_range = 1 / (1 + np.exp(-gamma * (s_range - c)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s_range, y=G_range, mode='lines', name='G(s)',
                                 line=dict(color='#ff7f0e', width=3)))
        fig.add_vline(x=c, line_dash="dash", line_color="red", annotation_text=f"c={c}")
        fig.update_layout(title=f"Transition Function (γ={gamma}, c={c})",
                          xaxis_title="s", yaxis_title="G(s)", height=400)
        st.plotly_chart(fig, use_container_width=True)

    elif sim_type == "Markov-Switching":
        st.markdown("### Markov-Switching Model")

        st.latex(r"y_t = \mu_{S_t} + \phi_{S_t} y_{t-1} + \sigma_{S_t} \epsilon_t")

        col1, col2, col3 = st.columns(3)
        with col1:
            mu1 = st.slider("μ₁ (State 1)", -5.0, 5.0, 2.0, 0.5)
            phi1 = st.slider("φ₁ (State 1)", -1.0, 1.0, 0.5, 0.1)
            sigma1 = st.slider("σ₁ (State 1)", 0.1, 3.0, 0.5, 0.1)
        with col2:
            mu2 = st.slider("μ₂ (State 2)", -5.0, 5.0, -1.0, 0.5)
            phi2 = st.slider("φ₂ (State 2)", -1.0, 1.0, 0.3, 0.1)
            sigma2 = st.slider("σ₂ (State 2)", 0.1, 3.0, 1.5, 0.1)
        with col3:
            p11 = st.slider("P(1→1)", 0.5, 0.99, 0.95, 0.01)
            p22 = st.slider("P(2→2)", 0.5, 0.99, 0.90, 0.01)

        np.random.seed(42)
        n_ms = 500
        states = np.zeros(n_ms, dtype=int)
        y_ms = np.zeros(n_ms)

        states[0] = 0
        y_ms[0] = mu1

        for t in range(1, n_ms):
            if states[t - 1] == 0:
                states[t] = 0 if np.random.rand() < p11 else 1
            else:
                states[t] = 1 if np.random.rand() < p22 else 0

            if states[t] == 0:
                y_ms[t] = mu1 + phi1 * y_ms[t - 1] + sigma1 * np.random.normal()
            else:
                y_ms[t] = mu2 + phi2 * y_ms[t - 1] + sigma2 * np.random.normal()

        fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                            subplot_titles=("MS Series", "Hidden States"))

        fig.add_trace(go.Scatter(x=np.arange(n_ms), y=y_ms, mode='lines', name='MS',
                                 line=dict(color='#d62728')), row=1, col=1)

        fig.add_trace(go.Scatter(x=np.arange(n_ms), y=states, mode='lines', name='State',
                                 line=dict(color='purple'), fill='tozeroy'), row=2, col=1)

        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Transition matrix
        st.markdown("#### Transition Matrix")
        p12 = 1 - p11
        p21 = 1 - p22

        trans_matrix = pd.DataFrame(
            [[p11, p12], [p21, p22]],
            index=['State 1', 'State 2'],
            columns=['→ State 1', '→ State 2']
        )

        st.dataframe(trans_matrix.style.format("{:.3f}"))

        # Ergodic probabilities
        pi1 = p21 / (p12 + p21)
        pi2 = p12 / (p12 + p21)

        col1, col2, col3 = st.columns(3)
        col1.metric("Ergodic π₁", f"{pi1:.3f}")
        col2.metric("Ergodic π₂", f"{pi2:.3f}")
        col3.metric("Expected Duration State 1", f"{1 / p12:.1f}")

    elif sim_type == "GARCH Model":
        st.markdown("### GARCH(1,1) Model")

        st.latex(r"y_t = \mu + \sigma_t \epsilon_t, \quad \epsilon_t \sim N(0,1)")
        st.latex(r"\sigma_t^2 = \omega + \alpha y_{t-1}^2 + \beta \sigma_{t-1}^2")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            omega = st.slider("ω", 0.01, 1.0, 0.1, 0.01)
        with col2:
            alpha = st.slider("α", 0.01, 0.3, 0.15, 0.01)
        with col3:
            beta = st.slider("β", 0.5, 0.95, 0.80, 0.01)
        with col4:
            n_garch = st.slider("Sample Size", 200, 2000, 1000, 100)

        if alpha + beta >= 1:
            st.error("⚠️ Stationarity condition violated: α + β must be < 1")
        else:
            np.random.seed(42)
            sigma2 = np.zeros(n_garch)
            y_garch = np.zeros(n_garch)

            sigma2[0] = omega / (1 - alpha - beta)

            for t in range(n_garch):
                eps = np.random.normal(0, 1)
                y_garch[t] = np.sqrt(sigma2[t]) * eps
                if t < n_garch - 1:
                    sigma2[t + 1] = omega + alpha * y_garch[t] ** 2 + beta * sigma2[t]

            fig = make_subplots(rows=3, cols=1, row_heights=[0.4, 0.3, 0.3],
                                subplot_titles=("Returns", "Conditional Volatility σₜ", "Squared Returns"))

            fig.add_trace(go.Scatter(x=np.arange(n_garch), y=y_garch, mode='lines', name='Returns',
                                     line=dict(color='#1f77b4')), row=1, col=1)

            fig.add_trace(go.Scatter(x=np.arange(n_garch), y=np.sqrt(sigma2), mode='lines', name='σₜ',
                                     line=dict(color='#ff7f0e', width=2)), row=2, col=1)

            fig.add_trace(go.Scatter(x=np.arange(n_garch), y=y_garch ** 2, mode='lines', name='y²',
                                     line=dict(color='#2ca02c')), row=3, col=1)

            fig.update_layout(height=800, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean Return", f"{np.mean(y_garch):.4f}")
            col2.metric("Std Return", f"{np.std(y_garch):.4f}")
            col3.metric("Unconditional σ²", f"{omega / (1 - alpha - beta):.4f}")
            col4.metric("Persistence", f"{alpha + beta:.4f}")

    else:  # Comparison Study
        st.markdown("### Model Comparison Study")

        st.markdown("""
        Compare different model types on the same dataset to understand their behavior.
        """)

        np.random.seed(42)
        n = 300
        t = np.arange(n)

        # Generate base series with structural break
        y_true = np.concatenate([
            5 + 0.1 * t[:150] + np.random.normal(0, 1, 150),
            10 - 0.05 * t[150:] + np.random.normal(0, 2, 150)
        ])

        # Fit different models
        # 1. Linear (ignores break)
        X = np.column_stack([np.ones(n), t])
        beta_linear = np.linalg.lstsq(X, y_true, rcond=None)[0]
        y_linear = X @ beta_linear

        # 2. Two-regime (correct specification)
        X1 = X[:150]
        X2 = X[150:]
        beta1 = np.linalg.lstsq(X1, y_true[:150], rcond=None)[0]
        beta2 = np.linalg.lstsq(X2, y_true[150:], rcond=None)[0]
        y_break = np.concatenate([X1 @ beta1, X2 @ beta2])

        # 3. Polynomial (misspecification)
        X_poly = np.column_stack([np.ones(n), t, t ** 2, t ** 3])
        beta_poly = np.linalg.lstsq(X_poly, y_true, rcond=None)[0]
        y_poly = X_poly @ beta_poly

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=t, y=y_true, mode='markers', name='True Data',
                                 marker=dict(size=4, color='lightgray')))
        fig.add_trace(go.Scatter(x=t, y=y_linear, mode='lines', name='Linear Model',
                                 line=dict(color='red', width=2, dash='dash')))
        fig.add_trace(go.Scatter(x=t, y=y_break, mode='lines', name='Structural Break Model',
                                 line=dict(color='green', width=2)))
        fig.add_trace(go.Scatter(x=t, y=y_poly, mode='lines', name='Polynomial Model',
                                 line=dict(color='blue', width=2, dash='dot')))

        fig.add_vline(x=150, line_dash="dash", line_color="black", annotation_text="True Break")

        fig.update_layout(title="Model Comparison", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Residual analysis
        res_linear = y_true - y_linear
        res_break = y_true - y_break
        res_poly = y_true - y_poly

        fig = make_subplots(rows=1, cols=3, subplot_titles=("Linear", "Structural Break", "Polynomial"))

        fig.add_trace(go.Scatter(x=t, y=res_linear, mode='markers', name='Linear',
                                 marker=dict(size=3, color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=res_break, mode='markers', name='Break',
                                 marker=dict(size=3, color='green')), row=1, col=2)
        fig.add_trace(go.Scatter(x=t, y=res_poly, mode='markers', name='Poly',
                                 marker=dict(size=3, color='blue')), row=1, col=3)

        fig.update_layout(height=400, showlegend=False, title_text="Residuals Comparison")
        st.plotly_chart(fig, use_container_width=True)

        # Performance metrics
        st.markdown("#### Model Performance")

        mse_linear = np.mean(res_linear ** 2)
        mse_break = np.mean(res_break ** 2)
        mse_poly = np.mean(res_poly ** 2)

        mae_linear = np.mean(np.abs(res_linear))
        mae_break = np.mean(np.abs(res_break))
        mae_poly = np.mean(np.abs(res_poly))

        metrics_df = pd.DataFrame({
            'Model': ['Linear', 'Structural Break', 'Polynomial'],
            'MSE': [mse_linear, mse_break, mse_poly],
            'MAE': [mae_linear, mae_break, mae_poly],
            'Std(Residuals)': [np.std(res_linear), np.std(res_break), np.std(res_poly)]
        })

        st.dataframe(metrics_df.style.highlight_min(subset=['MSE', 'MAE', 'Std(Residuals)'], color='lightgreen'))

# ============================================================================
# SECTION 6: ADVANCED PREPARATION
# ============================================================================
elif section == "Advanced Preparation":
    st.markdown('<div class="section-header">6. Preparation for Advanced Topics</div>', unsafe_allow_html=True)

    st.markdown("""
    This section bridges foundational concepts to advanced models you'll encounter in later lectures.
    """)

    adv_topic = st.selectbox(
        "Select Advanced Topic:",
        ["Threshold Models (TAR/SETAR)", "Markov-Switching Models",
         "Smooth Transition Models (STAR)", "Time-Varying Parameters", "Model Selection & Diagnostics"]
    )

    if adv_topic == "Threshold Models (TAR/SETAR)":
        st.markdown("### Threshold Autoregressive Models")

        st.markdown("""
        **Self-Exciting TAR (SETAR)** uses lagged values as the threshold variable:
        """)

        st.latex(r"""
        y_t = \begin{cases}
        \phi_0^{(1)} + \sum_{i=1}^{p_1} \phi_i^{(1)} y_{t-i} + \epsilon_t^{(1)} & \text{if } y_{t-d} \leq \tau \\
        \phi_0^{(2)} + \sum_{i=1}^{p_2} \phi_i^{(2)} y_{t-i} + \epsilon_t^{(2)} & \text{if } y_{t-d} > \tau
        \end{cases}
        """)

        st.markdown("#### Key Considerations")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**Identification Issues**")
            st.markdown("""
            1. **Threshold value** $\\tau$: Unknown, must be estimated
            2. **Delay parameter** $d$: Typically $1 \\leq d \\leq 4$
            3. **Order** $p_1, p_2$: May differ across regimes
            4. **Number of regimes**: Can be 2, 3, or more
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**Estimation Methods**")
            st.markdown("""
            1. **Grid search**: Try all possible $\\tau$ values
            2. **Sequential testing**: Start with linear, test for threshold
            3. **Information criteria**: AIC/BIC for model selection
            4. **Bootstrap**: For inference on $\\tau$
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("#### Hansen Test for Threshold Effect")

        st.latex(r"LR = T(\tilde{S}_T - \hat{S}_T)")

        st.markdown("""
        where:
        - $\\tilde{S}_T$: Sum of squared residuals under linear model
        - $\\hat{S}_T$: SSR under threshold model
        - $T$: Sample size
        - **Bootstrap p-values** required (non-standard distribution)
        """)

        st.markdown("#### Forecasting with TAR")

        st.markdown("""
        Multi-step forecasts are complex because future regimes are uncertain:
        """)

        st.latex(
            r"E[y_{t+h} | \mathcal{F}_t] = E[y_{t+h} | y_{t+h-d} \leq \tau] P(y_{t+h-d} \leq \tau) + E[y_{t+h} | y_{t+h-d} > \tau] P(y_{t+h-d} > \tau)")

        st.markdown("Requires simulation or numerical integration!")

        st.markdown("#### Applications")

        examples = pd.DataFrame({
            'Field': ['Macroeconomics', 'Finance', 'Energy', 'Environmental'],
            'Application': [
                'GDP growth (recession vs expansion)',
                'Stock returns (bull vs bear markets)',
                'Electricity prices (peak vs off-peak)',
                'Temperature dynamics (seasonal thresholds)'
            ],
            'Threshold Variable': [
                'GDP growth rate',
                'Market index level',
                'Demand level',
                'Temperature'
            ]
        })

        st.dataframe(examples, use_container_width=True)

    elif adv_topic == "Markov-Switching Models":
        st.markdown("### Markov-Switching Regression Models")

        st.markdown("""
        **Hamilton (1989)** introduced regime-switching models where states follow a Markov chain.
        """)

        st.markdown("#### General Framework")

        st.latex(r"y_t = \mu(S_t) + \sum_{i=1}^p \phi_i(S_t) y_{t-i} + \sigma(S_t) \epsilon_t")

        st.latex(r"P(S_t = j | S_{t-1} = i, S_{t-2}, \ldots) = P(S_t = j | S_{t-1} = i) = p_{ij}")

        st.markdown("#### Hamilton Filter")

        st.markdown("""
        The **Hamilton filter** computes filtered probabilities $P(S_t = j | \\mathcal{F}_t)$:
        """)

        st.latex(r"\xi_{t|t}(j) = P(S_t = j | y_1, \ldots, y_t)")

        st.markdown("**Algorithm:**")
        st.latex(r"""
        \begin{align}
        \text{Prediction:} \quad & \xi_{t|t-1}(j) = \sum_{i=1}^M p_{ij} \xi_{t-1|t-1}(i) \\
        \text{Update:} \quad & \xi_{t|t}(j) = \frac{f(y_t | S_t = j, \mathcal{F}_{t-1}) \xi_{t|t-1}(j)}{\sum_{k=1}^M f(y_t | S_t = k, \mathcal{F}_{t-1}) \xi_{t|t-1}(k)}
        \end{align}
        """)

        st.markdown("#### Smoothed Probabilities")

        st.markdown("""
        **Kim smoother** provides $P(S_t = j | y_1, \\ldots, y_T)$ using all data (backward recursion):
        """)

        st.latex(r"\xi_{t|T}(j) = \xi_{t|t}(j) \sum_{i=1}^M \frac{p_{ji}}{\xi_{t+1|t}(i)} \xi_{t+1|T}(i)")

        # Illustration
        st.markdown("#### Filtered vs Smoothed Probabilities")

        np.random.seed(42)
        n = 200

        # True states
        states_true = np.zeros(n, dtype=int)
        p11, p22 = 0.95, 0.90

        for t in range(1, n):
            if states_true[t - 1] == 0:
                states_true[t] = 0 if np.random.rand() < p11 else 1
            else:
                states_true[t] = 1 if np.random.rand() < p22 else 0

        # Simple filtering (approximate)
        filtered_prob = np.zeros(n)
        for t in range(n):
            if t < 20:
                filtered_prob[t] = 0.5
            else:
                filtered_prob[t] = 0.7 if states_true[t] == 0 else 0.3
                filtered_prob[t] += np.random.normal(0, 0.1)
                filtered_prob[t] = np.clip(filtered_prob[t], 0, 1)

        # Smoothed (more accurate)
        smoothed_prob = filtered_prob + 0.1 * (states_true - filtered_prob)
        smoothed_prob = np.clip(smoothed_prob, 0, 1)

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=np.arange(n), y=states_true, mode='lines', name='True State',
                                 line=dict(color='black', width=2)))
        fig.add_trace(go.Scatter(x=np.arange(n), y=filtered_prob, mode='lines', name='Filtered P(S=0)',
                                 line=dict(color='blue', dash='dash')))
        fig.add_trace(go.Scatter(x=np.arange(n), y=smoothed_prob, mode='lines', name='Smoothed P(S=0)',
                                 line=dict(color='red')))

        fig.update_layout(title="Filtered vs Smoothed State Probabilities", height=400,
                          yaxis_title="Probability / State")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Estimation")

        st.markdown("""
        **Maximum Likelihood Estimation** via EM algorithm:

        1. **E-step**: Compute $\\xi_{t|T}(j)$ using current parameter estimates
        2. **M-step**: Maximize expected log-likelihood
        3. **Iterate** until convergence

        **Challenges**:
        - Multiple local maxima
        - Requires good starting values
        - Label switching problem
        """)

        st.markdown("#### Model Extensions")

        extensions = pd.DataFrame({
            'Extension': [
                'Time-Varying Transition Probabilities',
                'Duration-Dependent Markov',
                'Multivariate MS Models',
                'MS-GARCH'
            ],
            'Key Feature': [
                'pᵢⱼ(t) = Λ(γ\'xₜ)',
                'P(duration > d) decreases with d',
                'Multiple series, common regimes',
                'Regime-switching volatility'
            ],
            'Application': [
                'Policy regime changes',
                'Business cycle dating',
                'International linkages',
                'Financial contagion'
            ]
        })

        st.dataframe(extensions, use_container_width=True)

    elif adv_topic == "Smooth Transition Models (STAR)":
        st.markdown("### Smooth Transition Autoregressive (STAR) Models")

        st.markdown("#### LSTAR vs ESTAR")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Logistic STAR (LSTAR)**")
            st.latex(r"G(s_t) = \frac{1}{1 + \exp(-\gamma(s_t - c))}")

            s = np.linspace(-3, 3, 100)
            G_lstar = 1 / (1 + np.exp(-2 * (s - 0)))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s, y=G_lstar, mode='lines', name='LSTAR',
                                     line=dict(color='blue', width=3)))
            fig.update_layout(title="LSTAR Transition", height=300)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Asymmetric**: Different behavior above/below threshold")

        with col2:
            st.markdown("**Exponential STAR (ESTAR)**")
            st.latex(r"G(s_t) = 1 - \exp(-\gamma(s_t - c)^2)")

            G_estar = 1 - np.exp(-0.5 * (s - 0) ** 2)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s, y=G_estar, mode='lines', name='ESTAR',
                                     line=dict(color='red', width=3)))
            fig.update_layout(title="ESTAR Transition", height=300)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Symmetric**: Similar at extremes, different at center")

        st.markdown("#### Linearity Testing (Teräsvirta Procedure)")

        st.markdown("""
        1. **Test linearity** against STAR
        2. **If rejected**, determine LSTAR vs ESTAR
        3. **Estimate** chosen model
        4. **Diagnostic checking**
        """)

        st.markdown("**Auxiliary regression:**")
        st.latex(r"y_t = \beta_0 + \beta_1 y_{t-1} + \beta_2 y_{t-1}^2 + \beta_3 y_{t-1}^3 + \text{error}")

        st.markdown("""
        - Test $H_0: \\beta_2 = \\beta_3 = 0$ for linearity
        - If $|t_{\\beta_3}| > |t_{\\beta_2}|$, choose LSTAR; else ESTAR
        """)

        st.markdown("#### Interpretation of Parameters")

        param_table = pd.DataFrame({
            'Parameter': ['γ (gamma)', 'c', 's_t', 'φ⁽¹⁾', 'φ⁽²⁾'],
            'Meaning': [
                'Speed of transition',
                'Location of transition',
                'Transition variable',
                'Parameters in regime 1 (G≈0)',
                'Parameters in regime 2 (G≈1)'
            ],
            'Typical Values': [
                '0.1 (slow) to 10+ (sharp)',
                'Sample median/mean',
                'yₜ₋ₐ or external variable',
                'Estimated',
                'Estimated'
            ]
        })

        st.dataframe(param_table, use_container_width=True)

        st.markdown("#### Applications and Examples")

        st.markdown("""
        **LSTAR**: Asymmetric business cycles
        - Expansion regime (low $s_t$): Persistent, low volatility
        - Contraction regime (high $s_t$): Sharp, high volatility

        **ESTAR**: Purchasing Power Parity (PPP)
        - Small deviations: Random walk (weak arbitrage)
        - Large deviations: Mean reversion (strong arbitrage)
        """)

    elif adv_topic == "Time-Varying Parameters":
        st.markdown("### Time-Varying Parameter (TVP) Models")

        st.markdown("""
        Unlike structural breaks (discrete changes), TVP models allow **continuous parameter evolution**.
        """)

        st.markdown("#### State-Space Representation")

        st.latex(r"""
        \begin{align}
        y_t &= Z_t \alpha_t + \epsilon_t \quad \text{(Observation equation)} \\
        \alpha_t &= T_t \alpha_{t-1} + \eta_t \quad \text{(State equation)}
        \end{align}
        """)

        st.markdown("where $\\alpha_t$ contains time-varying parameters.")

        st.markdown("#### Random Walk Parameters")

        st.latex(r"\beta_t = \beta_{t-1} + u_t, \quad u_t \sim N(0, Q)")

        # Simulation
        st.markdown("#### TVP Simulation Example")

        np.random.seed(42)
        n = 300
        beta_t = np.zeros(n)
        beta_t[0] = 0.5

        for t in range(1, n):
            beta_t[t] = beta_t[t - 1] + np.random.normal(0, 0.02)

        x = np.random.normal(0, 1, n)
        y_tvp = beta_t * x + np.random.normal(0, 0.5, n)

        fig = make_subplots(rows=2, cols=1, subplot_titles=("Time-Varying Coefficient βₜ", "Observations"))

        fig.add_trace(go.Scatter(x=np.arange(n), y=beta_t, mode='lines', name='βₜ',
                                 line=dict(color='purple', width=2)), row=1, col=1)

        fig.add_trace(go.Scatter(x=np.arange(n), y=y_tvp, mode='markers', name='yₜ',
                                 marker=dict(size=3, color='lightblue')), row=2, col=1)

        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Kalman Filter")

        st.markdown("""
        Optimal recursive estimation of $\\alpha_t$:

        **Prediction:**
        """)
        st.latex(r"""
        \begin{align}
        \alpha_{t|t-1} &= T_t \alpha_{t-1|t-1} \\
        P_{t|t-1} &= T_t P_{t-1|t-1} T_t' + Q_t
        \end{align}
        """)

        st.markdown("**Update:**")
        st.latex(r"""
        \begin{align}
        v_t &= y_t - Z_t \alpha_{t|t-1} \\
        K_t &= P_{t|t-1} Z_t' F_t^{-1} \\
        \alpha_{t|t} &= \alpha_{t|t-1} + K_t v_t \\
        P_{t|t} &= P_{t|t-1} - K_t F_t K_t'
        \end{align}
        """)

        st.markdown("where $F_t = Z_t P_{t|t-1} Z_t' + H_t$ is forecast error variance.")

        st.markdown("#### Comparison: Structural Breaks vs TVP")

        comp_df = pd.DataFrame({
            'Aspect': ['Parameter Change', 'Number of Regimes', 'Transition', 'Estimation', 'Forecasting'],
            'Structural Breaks': ['Discrete jumps', 'Finite (2-3)', 'Instantaneous', 'OLS per regime',
                                  'Regime-dependent'],
            'TVP Models': ['Continuous evolution', 'Infinite (continuum)', 'Gradual', 'Kalman filter', 'Adaptive']
        })

        st.dataframe(comp_df, use_container_width=True)

    else:  # Model Selection & Diagnostics
        st.markdown("### Model Selection and Diagnostic Checking")

        st.markdown("#### Information Criteria")

        st.markdown("**Akaike Information Criterion (AIC):**")
        st.latex(r"AIC = -2\log L + 2k")

        st.markdown("**Bayesian Information Criterion (BIC):**")
        st.latex(r"BIC = -2\log L + k \log T")

        st.markdown("""
        - **AIC**: Asymptotically efficient (minimizes forecast error)
        - **BIC**: Consistent (selects true model as $T \\to \\infty$)
        - **Rule**: Choose model with **lowest** AIC/BIC
        """)

        # Example
        st.markdown("#### Example: Comparing Models")

        models_comparison = pd.DataFrame({
            'Model': ['Linear AR(2)', 'TAR(2)', 'LSTAR', 'MS-AR(2)'],
            'Parameters (k)': [3, 6, 7, 8],
            'Log-Likelihood': [-250.3, -235.1, -232.8, -230.5],
            'AIC': [506.6, 482.2, 479.6, 477.0],
            'BIC': [517.2, 503.4, 503.9, 506.1]
        })

        models_comparison['AIC_Rank'] = models_comparison['AIC'].rank()
        models_comparison['BIC_Rank'] = models_comparison['BIC'].rank()

        st.dataframe(models_comparison.style.highlight_min(subset=['AIC', 'BIC'], color='lightgreen'))

        st.markdown("**Conclusion**: MS-AR(2) has lowest AIC, but TAR(2) has lowest BIC.")

        st.markdown("#### Diagnostic Tests")

        st.markdown("**1. Residual Autocorrelation**")
        st.latex(r"Q(m) = T(T+2) \sum_{k=1}^m \frac{\hat{\rho}_k^2}{T-k} \sim \chi^2(m-p)")

        st.markdown("**2. ARCH Effects (LM Test)**")
        st.latex(r"LM = T \cdot R^2 \sim \chi^2(q)")

        st.markdown("from auxiliary regression:")
        st.latex(
            r"\hat{\epsilon}_t^2 = \gamma_0 + \gamma_1 \hat{\epsilon}_{t-1}^2 + \cdots + \gamma_q \hat{\epsilon}_{t-q}^2 + \text{error}")

        st.markdown("**3. Normality (Jarque-Bera)**")
        st.latex(r"JB = \frac{T}{6}\left(S^2 + \frac{(K-3)^2}{4}\right) \sim \chi^2(2)")

        st.markdown("where $S$ = skewness, $K$ = kurtosis.")

        st.markdown("**4. Parameter Stability (CUSUM/CUSUM-SQ)**")

        st.markdown("""
        - CUSUM: Detects level shifts
        - CUSUM-SQ: Detects variance changes
        """)

        st.markdown("#### Cross-Validation for Forecast Evaluation")

        st.markdown("""
        **Rolling Window:**
        1. Estimate on $[1, t]$
        2. Forecast $t+1, \\ldots, t+h$
        3. Roll window forward
        4. Compute RMSE, MAE

        **Expanding Window:**
        - Always start from $t=1$, expand end point
        """)

        # Illustration
        fig = go.Figure()

        for i in range(5):
            fig.add_shape(type="rect", x0=i * 20, x1=i * 20 + 50, y0=i, y1=i + 0.8,
                          fillcolor="lightblue", opacity=0.5, line=dict(color="blue"))
            fig.add_shape(type="rect", x0=i * 20 + 50, x1=i * 20 + 60, y0=i, y1=i + 0.8,
                          fillcolor="lightcoral", opacity=0.5, line=dict(color="red"))

        fig.update_layout(title="Rolling Window Cross-Validation",
                          xaxis_title="Time", yaxis_title="Fold",
                          showlegend=False, height=400,
                          yaxis=dict(showticklabels=False))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Practical Recommendations")

        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("""
        1. **Start simple**: Linear AR model baseline
        2. **Test for breaks**: Chow, QLR, CUSUM
        3. **Test for nonlinearity**: BDS, Teräsvirta
        4. **Compare models**: AIC/BIC, forecast performance
        5. **Diagnostic checking**: Residuals should be white noise
        6. **Economic interpretation**: Parameters must make sense
        7. **Out-of-sample validation**: Ultimate test
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# SECTION 7: INTERACTIVE TOOLS
# ============================================================================
elif section == "Interactive Tools":
    st.markdown('<div class="section-header">7. Interactive Learning Tools</div>', unsafe_allow_html=True)

    tool = st.selectbox(
        "Select Interactive Tool:",
        ["Break Point Detector", "TAR Model Builder", "Transition Function Explorer",
         "Regime Probability Visualizer", "Forecast Comparison"]
    )

    if tool == "Break Point Detector":
        st.markdown("### Interactive Break Point Detection")

        st.markdown("""
        Upload your own data or use simulated data to detect structural breaks.
        """)

        data_source = st.radio("Data Source:", ["Simulate", "Upload CSV"])

        if data_source == "Simulate":
            col1, col2, col3 = st.columns(3)
            with col1:
                n = st.slider("Sample Size", 100, 500, 200, 50)
            with col2:
                true_break_pct = st.slider("Break Location (%)", 20, 80, 50, 5)
            with col3:
                noise = st.slider("Noise", 0.5, 5.0, 2.0, 0.5)

            true_break = int(n * true_break_pct / 100)

            np.random.seed(42)
            t = np.arange(n)
            y = np.concatenate([
                10 + 0.1 * t[:true_break] + np.random.normal(0, noise, true_break),
                25 - 0.05 * t[true_break:] + np.random.normal(0, noise, n - true_break)
            ])

            df = pd.DataFrame({'Time': t, 'Value': y})
        else:
            uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.write("Data Preview:", df.head())
            else:
                st.info("Please upload a CSV file")
                st.stop()

        if st.button("Detect Breaks"):
            y = df['Value'].values
            n = len(y)

            # QLR test
            trim = 0.15
            possible_breaks = range(int(n * trim), int(n * (1 - trim)))
            f_stats = []

            X = np.column_stack([np.ones(n), np.arange(n)])
            beta_full = np.linalg.lstsq(X, y, rcond=None)[0]
            rss_r = np.sum((y - X @ beta_full) ** 2)

            for bp in possible_breaks:
                X1 = X[:bp]
                X2 = X[bp:]

                beta1 = np.linalg.lstsq(X1, y[:bp], rcond=None)[0]
                beta2 = np.linalg.lstsq(X2, y[bp:], rcond=None)[0]

                rss1 = np.sum((y[:bp] - X1 @ beta1) ** 2)
                rss2 = np.sum((y[bp:] - X2 @ beta2) ** 2)

                k = 2
                f = ((rss_r - rss1 - rss2) / k) / ((rss1 + rss2) / (n - 2 * k))
                f_stats.append(f)

            estimated_break = list(possible_breaks)[np.argmax(f_stats)]
            max_f = max(f_stats)

            # Plot
            fig = make_subplots(rows=2, cols=1,
                                subplot_titles=("Data with Detected Break", "F-Statistics"))

            fig.add_trace(go.Scatter(x=df['Time'], y=y, mode='lines+markers', name='Data',
                                     line=dict(color='#1f77b4'), marker=dict(size=4)), row=1, col=1)
            fig.add_vline(x=estimated_break, line_dash="dash", line_color="red", row=1, col=1,
                          annotation_text=f"Detected: {estimated_break}")

            fig.add_trace(go.Scatter(x=list(possible_breaks), y=f_stats, mode='lines', name='F',
                                     line=dict(color='green', width=2)), row=2, col=1)
            fig.add_vline(x=estimated_break, line_dash="dash", line_color="red", row=2, col=1)

            fig.update_layout(height=700, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            st.success(f"✅ Break detected at observation {estimated_break} with F-statistic {max_f:.2f}")

    elif tool == "TAR Model Builder":
        st.markdown("### Build Your Own TAR Model")

        st.markdown("Design and simulate a custom TAR model.")

        n_regimes = st.radio("Number of Regimes:", [2, 3])

        if n_regimes == 2:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Regime 1 (Low)**")
                phi1_1 = st.slider("φ₁", -1.0, 1.0, 0.7, 0.1, key='phi1_1')
                phi2_1 = st.slider("φ₂", -1.0, 1.0, 0.0, 0.1, key='phi2_1')
                sigma_1 = st.slider("σ", 0.1, 3.0, 0.5, 0.1, key='sigma_1')

            with col2:
                st.markdown("**Regime 2 (High)**")
                phi1_2 = st.slider("φ₁", -1.0, 1.0, -0.3, 0.1, key='phi1_2')
                phi2_2 = st.slider("φ₂", -1.0, 1.0, 0.0, 0.1, key='phi2_2')
                sigma_2 = st.slider("σ", 0.1, 3.0, 1.5, 0.1, key='sigma_2')

            threshold = st.slider("Threshold τ", -3.0, 3.0, 0.0, 0.5)

            # Simulate
            np.random.seed(42)
            n = 500
            y = np.zeros(n)
            y[0], y[1] = 0, 0
            regimes = np.zeros(n)

            for t in range(2, n):
                if y[t - 1] <= threshold:
                    y[t] = phi1_1 * y[t - 1] + phi2_1 * y[t - 2] + np.random.normal(0, sigma_1)
                    regimes[t] = 0
                else:
                    y[t] = phi1_2 * y[t - 1] + phi2_2 * y[t - 2] + np.random.normal(0, sigma_2)
                    regimes[t] = 1

            fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3])

            fig.add_trace(go.Scatter(x=np.arange(n), y=y, mode='lines', name='TAR',
                                     line=dict(color='#1f77b4')), row=1, col=1)
            fig.add_hline(y=threshold, line_dash="dash", line_color="red", row=1, col=1)

            fig.add_trace(go.Scatter(x=np.arange(n), y=regimes, mode='lines', name='Regime',
                                     line=dict(color='purple'), fill='tozeroy'), row=2, col=1)

            fig.update_layout(height=600, showlegend=False, title_text="Your Custom TAR Model")
            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean", f"{np.mean(y):.3f}")
            col2.metric("Std Dev", f"{np.std(y):.3f}")
            col3.metric("Regime 1 %", f"{100 * np.sum(regimes == 0) / n:.1f}%")

        else:  # 3 regimes
            st.info("3-regime TAR: Exercise for advanced students!")

    elif tool == "Transition Function Explorer":
        st.markdown("### Explore Transition Functions")

        st.markdown("Compare different transition functions interactively.")

        col1, col2 = st.columns(2)

        with col1:
            gamma = st.slider("γ (smoothness)", 0.1, 10.0, 2.0, 0.5)
        with col2:
            c = st.slider("c (location)", -3.0, 3.0, 0.0, 0.5)

        s = np.linspace(-5, 5, 200)

        # Different transition functions
        G_logistic = 1 / (1 + np.exp(-gamma * (s - c)))
        G_exponential = 1 - np.exp(-gamma * (s - c) ** 2)

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=s, y=G_logistic, mode='lines', name='Logistic (LSTAR)',
                                 line=dict(color='blue', width=3)))
        fig.add_trace(go.Scatter(x=s, y=G_exponential, mode='lines', name='Exponential (ESTAR)',
                                 line=dict(color='red', width=3)))
        fig.add_vline(x=c, line_dash="dash", line_color="green", annotation_text=f"c={c}")

        fig.update_layout(title=f"Transition Functions (γ={gamma}, c={c})",
                          xaxis_title="Transition Variable s",
                          yaxis_title="G(s)",
                          height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Effective parameter
        st.markdown("#### Effective Parameter Evolution")

        phi_low = st.slider("φ (Regime 1)", -1.0, 1.0, 0.8, 0.1)
        phi_high = st.slider("φ (Regime 2)", -1.0, 1.0, -0.4, 0.1)

        phi_eff_logistic = phi_low * (1 - G_logistic) + phi_high * G_logistic
        phi_eff_exponential = phi_low * (1 - G_exponential) + phi_high * G_exponential

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=s, y=phi_eff_logistic, mode='lines', name='LSTAR φ(s)',
                                 line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=s, y=phi_eff_exponential, mode='lines', name='ESTAR φ(s)',
                                 line=dict(color='red', width=2)))
        fig.add_hline(y=phi_low, line_dash="dot", line_color="gray", annotation_text=f"φ₁={phi_low}")
        fig.add_hline(y=phi_high, line_dash="dot", line_color="gray", annotation_text=f"φ₂={phi_high}")

        fig.update_layout(title="Effective AR Coefficient",
                          xaxis_title="s", yaxis_title="φₑff(s)",
                          height=400)
        st.plotly_chart(fig, use_container_width=True)

    elif tool == "Regime Probability Visualizer":
        st.markdown("### Markov-Switching Regime Probabilities")

        st.markdown("Visualize how regime probabilities evolve over time.")

        col1, col2 = st.columns(2)

        with col1:
            p11 = st.slider("P(State 1 → State 1)", 0.5, 0.99, 0.95, 0.01)
            mu1 = st.slider("μ₁", -5.0, 5.0, 2.0, 0.5)
            sigma1 = st.slider("σ₁", 0.1, 3.0, 0.5, 0.1)

        with col2:
            p22 = st.slider("P(State 2 → State 2)", 0.5, 0.99, 0.90, 0.01)
            mu2 = st.slider("μ₂", -5.0, 5.0, -1.0, 0.5)
            sigma2 = st.slider("σ₂", 0.1, 3.0, 1.5, 0.1)

        # Simulate
        np.random.seed(42)
        n = 300
        states = np.zeros(n, dtype=int)
        y = np.zeros(n)

        states[0] = 0

        for t in range(n):
            if t > 0:
                if states[t - 1] == 0:
                    states[t] = 0 if np.random.rand() < p11 else 1
                else:
                    states[t] = 1 if np.random.rand() < p22 else 0

            if states[t] == 0:
                y[t] = mu1 + np.random.normal(0, sigma1)
            else:
                y[t] = mu2 + np.random.normal(0, sigma2)

        # Simple filter (demonstration)
        prob_state1 = np.zeros(n)
        for t in range(n):
            prob_state1[t] = 0.7 if states[t] == 0 else 0.3
            prob_state1[t] += np.random.normal(0, 0.05)
            prob_state1[t] = np.clip(prob_state1[t], 0, 1)

        fig = make_subplots(rows=3, cols=1, row_heights=[0.4, 0.3, 0.3],
                            subplot_titles=("Observations", "True States", "Filtered P(State=1)"))

        fig.add_trace(go.Scatter(x=np.arange(n), y=y, mode='lines', name='y',
                                 line=dict(color='#1f77b4')), row=1, col=1)

        fig.add_trace(go.Scatter(x=np.arange(n), y=states, mode='lines', name='State',
                                 line=dict(color='black'), fill='tozeroy'), row=2, col=1)

        fig.add_trace(go.Scatter(x=np.arange(n), y=prob_state1, mode='lines', name='P(S=1)',
                                 line=dict(color='red', width=2)), row=3, col=1)
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=3, col=1)

        fig.update_layout(height=800, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Ergodic distribution
        p12 = 1 - p11
        p21 = 1 - p22
        pi1 = p21 / (p12 + p21)
        pi2 = 1 - pi1

        col1, col2 = st.columns(2)
        col1.metric("Ergodic π₁", f"{pi1:.3f}")
        col2.metric("Ergodic π₂", f"{pi2:.3f}")

    else:  # Forecast Comparison
        st.markdown("### Forecast Comparison Tool")

        st.markdown("Compare forecasts from different models.")

        # Generate data with break
        np.random.seed(42)
        n_train = 150
        n_test = 50
        n = n_train + n_test

        t = np.arange(n)
        y_true = np.concatenate([
            10 + 0.05 * t[:100] + np.random.normal(0, 1, 100),
            20 - 0.02 * t[100:] + np.random.normal(0, 1.5, n - 100)
        ])

        y_train = y_true[:n_train]
        y_test = y_true[n_train:]

        # Model 1: Linear on full training
        X_train = np.column_stack([np.ones(n_train), np.arange(n_train)])
        beta_linear = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

        X_test = np.column_stack([np.ones(n_test), np.arange(n_train, n)])
        forecast_linear = X_test @ beta_linear

        # Model 2: Use only recent data (last 50 obs)
        X_recent = X_train[-50:]
        y_recent = y_train[-50:]
        beta_recent = np.linalg.lstsq(X_recent, y_recent, rcond=None)[0]
        forecast_recent = X_test @ beta_recent

        # Model 3: Simple average of last 10
        forecast_avg = np.full(n_test, np.mean(y_train[-10:]))

        # Plot
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=t[:n_train], y=y_train, mode='lines', name='Training Data',
                                 line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=t[n_train:], y=y_test, mode='lines', name='Test Data (True)',
                                 line=dict(color='black', width=2)))

        fig.add_trace(go.Scatter(x=t[n_train:], y=forecast_linear, mode='lines', name='Linear Model',
                                 line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(x=t[n_train:], y=forecast_recent, mode='lines', name='Recent Data Only',
                                 line=dict(color='green', dash='dot')))
        fig.add_trace(go.Scatter(x=t[n_train:], y=forecast_avg, mode='lines', name='Simple Average',
                                 line=dict(color='orange', dash='dashdot')))

        fig.add_vline(x=n_train, line_dash="solid", line_color="gray", annotation_text="Forecast Start")

        fig.update_layout(title="Out-of-Sample Forecast Comparison", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Metrics
        rmse_linear = np.sqrt(np.mean((y_test - forecast_linear) ** 2))
        rmse_recent = np.sqrt(np.mean((y_test - forecast_recent) ** 2))
        rmse_avg = np.sqrt(np.mean((y_test - forecast_avg) ** 2))

        mae_linear = np.mean(np.abs(y_test - forecast_linear))
        mae_recent = np.mean(np.abs(y_test - forecast_recent))
        mae_avg = np.mean(np.abs(y_test - forecast_avg))

        metrics_comp = pd.DataFrame({
            'Model': ['Linear (Full)', 'Recent Data', 'Simple Avg'],
            'RMSE': [rmse_linear, rmse_recent, rmse_avg],
            'MAE': [mae_linear, mae_recent, mae_avg]
        })

        st.markdown("#### Forecast Performance")
        st.dataframe(metrics_comp.style.highlight_min(subset=['RMSE', 'MAE'], color='lightgreen'))

        st.info("""
        **Key Insight**: When structural breaks occur, models using only recent data 
        often outperform those using the full historical sample.
        """)

# ============================================================================
# SECTION 8: SUMMARY & REFERENCES
# ============================================================================
else:  # Summary & References
    st.markdown('<div class="section-header">8. Summary and Further Reading</div>', unsafe_allow_html=True)

    st.markdown("### Key Takeaways")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("#### Structural Breaks")
        st.markdown("""
        ✅ Discrete parameter changes at known/unknown dates  
        ✅ Detection: Chow test, QLR, CUSUM  
        ✅ Important for policy analysis and forecasting  
        ✅ Ignoring breaks leads to biased estimates  
        ✅ Multiple breaks possible in long time series  
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("#### Nonlinearity")
        st.markdown("""
        ✅ State-dependent dynamics, asymmetric responses  
        ✅ Types: TAR, STAR, Markov-Switching, GARCH  
        ✅ Detection: BDS, Teräsvirta, visual diagnostics  
        ✅ Better captures business cycles, crises  
        ✅ More complex estimation and interpretation  
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Roadmap to Advanced Topics")

    roadmap = """
    ```
    Structural Breaks & Nonlinearity (This Lecture)
              |
              ├─→ Threshold Models (Next)
              |   - TAR, SETAR estimation
              |   - Hansen threshold tests
              |   - Multi-regime models
              |
              ├─→ Smooth Transition Models
              |   - LSTAR, ESTAR
              |   - Linearity testing
              |   - Nonlinear impulse responses
              |
              ├─→ Markov-Switching Models
              |   - Hamilton filter
              |   - EM algorithm
              |   - Time-varying transition probabilities
              |
              ├─→ State-Space Models
              |   - Kalman filter
              |   - Time-varying parameters
              |   - Unobserved components
              |
              └─→ Machine Learning for Time Series
                  - Neural networks, LSTM
                  - Nonlinear forecasting
                  - Hybrid models
    ```
    """
    st.code(roadmap, language='text')

    st.markdown("### Essential References")

    st.markdown("#### Textbooks")

    books = pd.DataFrame({
        'Author': [
            'Hamilton (1994)',
            'Tsay (2010)',
            'Franses & van Dijk (2000)',
            'Lütkepohl & Krätzig (2004)',
            'Teräsvirta et al. (2010)'
        ],
        'Title': [
            'Time Series Analysis',
            'Analysis of Financial Time Series (3rd ed.)',
            'Nonlinear Time Series Models in Empirical Finance',
            'Applied Time Series Econometrics',
            'Modelling Nonlinear Economic Time Series'
        ],
        'Focus': [
            'Markov-Switching, state-space',
            'Financial applications, GARCH',
            'Threshold and smooth transition',
            'VAR, structural breaks',
            'Comprehensive nonlinear methods'
        ]
    })

    st.dataframe(books, use_container_width=True)

    st.markdown("#### Seminal Papers")

    papers = pd.DataFrame({
        'Author(s)': [
            'Chow (1960)',
            'Quandt (1960)',
            'Tong (1983)',
            'Hamilton (1989)',
            'Hansen (1997)',
            'Teräsvirta (1994)',
            'Bai & Perron (1998, 2003)'
        ],
        'Title': [
            'Tests of Equality Between Sets of Coefficients',
            'Tests of the Hypothesis that a Linear Regression Obeys Two Separate Regimes',
            'Threshold Models in Non-linear Time Series Analysis',
            'A New Approach to the Economic Analysis of Nonstationary Time Series',
            'Inference When a Nuisance Parameter Is Not Identified Under the Null',
            'Specification, Estimation, and Evaluation of Smooth Transition AR Models',
            'Estimating and Testing Linear Models with Multiple Structural Changes'
        ],
        'Journal': [
            'Econometrica',
            'JASA',
            'Springer Lecture Notes',
            'Econometrica',
            'Econometrica',
            'JASA',
            'Econometrica'
        ]
    })

    st.dataframe(papers, use_container_width=True)

    st.markdown("#### Software and Packages")

    software = pd.DataFrame({
        'Platform': ['R', 'R', 'Python', 'Python', 'Matlab', 'Stata'],
        'Package': ['tsDyn', 'MSwM', 'statsmodels', 'arch', 'Econometrics Toolbox', 'mswitch'],
        'Models': [
            'SETAR, LSTAR, ESTAR',
            'Markov-Switching',
            'ARIMA, state-space, structural breaks',
            'ARCH/GARCH family',
            'Various econometric models',
            'Markov-Switching'
        ]
    })

    st.dataframe(software, use_container_width=True)

    st.markdown("### Practice Exercises")

    with st.expander("📝 Exercise 1: Structural Break Detection"):
        st.markdown("""
        **Data**: Monthly US unemployment rate (1970-2020)

        **Tasks**:
        1. Plot the series and identify potential break points visually
        2. Apply Chow test at 2008 (financial crisis)
        3. Use QLR test to find unknown break point
        4. Estimate separate AR models pre/post break
        5. Compare in-sample and out-of-sample forecasts

        **Questions**:
        - How many significant breaks do you find?
        - Do forecasts improve when accounting for breaks?
        - What is the economic interpretation of parameter changes?
        """)

    with st.expander("📝 Exercise 2: TAR Model Estimation"):
        st.markdown("""
        **Data**: Quarterly real GDP growth

        **Tasks**:
        1. Test for linearity using Tsay test
        2. Estimate 2-regime SETAR model
        3. Identify threshold value and regimes
        4. Compute regime-specific impulse responses
        5. Forecast 4 quarters ahead

        **Questions**:
        - Do expansion and recession regimes have different dynamics?
        - Is the threshold economically meaningful?
        - How do forecasts differ from linear AR?
        """)
