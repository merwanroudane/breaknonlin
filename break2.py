import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Structural Breaks in Time Series", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 30px;
        margin-bottom: 15px;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 5px;
    }
    .author {
        font-size: 18px;
        color: #7f8c8d;
        text-align: center;
        font-style: italic;
        margin-bottom: 30px;
    }
    .definition-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 20px 0;
    }
    .theorem-box {
        background-color: #fff9e6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff7f0e;
        margin: 20px 0;
    }
    .important-note {
        background-color: #ffe6e6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #d62728;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">Structural Breaks in Economic Time Series</p>', unsafe_allow_html=True)
st.markdown('<p class="author">Dr. Merwan Roudane</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
section = st.sidebar.radio("Select Section:",
                           ["Introduction",
                            "Theoretical Foundation",
                            "Types of Structural Breaks",
                            "Break in Mean",
                            "Break in Variance",
                            "Simultaneous Breaks",
                            "Testing Procedures",
                            "Implications & Applications"])

# ===========================
# SECTION 1: INTRODUCTION
# ===========================
if section == "Introduction":
    st.markdown('<p class="sub-header">1. Introduction to Structural Breaks</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="definition-box">
    <h3>📌 Definition</h3>
    A <b>structural break</b> (or structural change) refers to an abrupt change in the parameters of a time series model 
    at some point in time. This change can affect the mean, variance, autocorrelation structure, or any combination 
    of these characteristics.
    </div>
    """, unsafe_allow_html=True)

    st.write("### Why Study Structural Breaks?")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Economic Relevance:**")
        st.write("- Policy regime changes")
        st.write("- Economic crises (2008 financial crisis)")
        st.write("- Technological innovations")
        st.write("- Regulatory changes")
        st.write("- Market structure transformations")

    with col2:
        st.write("**Statistical Consequences:**")
        st.write("- Biased parameter estimates")
        st.write("- Invalid inference")
        st.write("- Poor forecasting performance")
        st.write("- Misleading hypothesis tests")
        st.write("- Spurious regression results")

    st.write("### Mathematical Formulation")

    st.latex(r"""
    y_t = \begin{cases}
    \mu_1 + \varepsilon_t & \text{if } t \leq T_b \\
    \mu_2 + \varepsilon_t & \text{if } t > T_b
    \end{cases}
    """)

    st.write("where:")
    st.latex(r"\mu_1, \mu_2 \text{ are pre-break and post-break means}")
    st.latex(r"T_b \text{ is the break date (usually unknown)}")
    st.latex(r"\varepsilon_t \sim \text{i.i.d.}(0, \sigma^2)")

    st.write("### Simple Illustration")

    # Simulation parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        n = st.slider("Sample Size", 100, 500, 200)
    with col2:
        break_point = st.slider("Break Point (%)", 30, 70, 50)
    with col3:
        mu_change = st.slider("Mean Change", 0.5, 5.0, 2.0)

    # Generate data
    np.random.seed(42)
    T_b = int(n * break_point / 100)

    y = np.zeros(n)
    y[:T_b] = np.random.normal(0, 1, T_b)
    y[T_b:] = np.random.normal(mu_change, 1, n - T_b)

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(n), y=y, mode='lines', name='Time Series',
                             line=dict(color='#1f77b4', width=2)))
    fig.add_vline(x=T_b, line_dash="dash", line_color="red",
                  annotation_text=f"Break at t={T_b}", annotation_position="top")
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_hline(y=mu_change, line_dash="dot", line_color="gray", opacity=0.5)

    fig.update_layout(title="Simple Structural Break Example",
                      xaxis_title="Time", yaxis_title="Value",
                      height=400, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# ===========================
# SECTION 2: THEORETICAL FOUNDATION
# ===========================
elif section == "Theoretical Foundation":
    st.markdown('<p class="sub-header">2. Theoretical Foundation</p>', unsafe_allow_html=True)

    st.write("### 2.1 General Framework")

    st.write("Consider a time series process with potential structural instability:")

    st.latex(r"""
    y_t = x_t'\beta_t + \varepsilon_t, \quad t = 1, 2, \ldots, T
    """)

    st.write("where:")
    st.latex(r"x_t \text{ is a } k \times 1 \text{ vector of explanatory variables}")
    st.latex(r"\beta_t \text{ is a } k \times 1 \text{ parameter vector (potentially time-varying)}")
    st.latex(r"\varepsilon_t \text{ is the error term}")

    st.markdown("""
    <div class="definition-box">
    <h3>Structural Stability</h3>
    The model exhibits <b>structural stability</b> if:
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"\beta_t = \beta \quad \forall t \in [1, T]")

    st.write("### 2.2 Types of Parameter Variation")

    tab1, tab2, tab3 = st.tabs(["Abrupt Change", "Gradual Change", "Random Walk"])

    with tab1:
        st.write("**Discrete/Abrupt Break:**")
        st.latex(r"""
        \beta_t = \begin{cases}
        \beta_1 & \text{if } t \leq T_b \\
        \beta_2 & \text{if } t > T_b
        \end{cases}
        """)
        st.write("This is the classical structural break model.")

    with tab2:
        st.write("**Smooth Transition:**")
        st.latex(r"""
        \beta_t = \beta_1 + (\beta_2 - \beta_1) \cdot G(t; \gamma, c)
        """)
        st.write("where G(·) is a transition function (e.g., logistic).")

    with tab3:
        st.write("**Time-Varying Parameters:**")
        st.latex(r"""
        \beta_t = \beta_{t-1} + \eta_t, \quad \eta_t \sim N(0, Q)
        """)
        st.write("Used in state-space models and Kalman filtering.")

    st.write("### 2.3 Statistical Properties Under Breaks")

    st.markdown("""
    <div class="theorem-box">
    <h3>📊 Consequence of Ignoring Breaks</h3>
    When a structural break exists but is ignored in estimation:
    <br><br>
    1. <b>OLS estimators remain unbiased</b> (under standard assumptions)<br>
    2. <b>Standard errors are incorrect</b> → invalid inference<br>
    3. <b>Tests have non-standard distributions</b><br>
    4. <b>Forecasts are suboptimal</b>
    </div>
    """, unsafe_allow_html=True)

    st.write("### 2.4 The Break Date Problem")

    st.write("The break date $T_b$ is typically **unknown**, leading to:")

    st.latex(r"""
    \hat{T}_b = \arg\min_{1 < \tau < T} SSR(\tau)
    """)

    st.write("where SSR(τ) is the sum of squared residuals assuming a break at τ.")

    st.markdown("""
    <div class="important-note">
    <h3>⚠️ Important</h3>
    The search over possible break dates introduces <b>data-snooping bias</b>, which affects the distribution 
    of test statistics. This requires using modified critical values (e.g., from Andrews, 1993).
    </div>
    """, unsafe_allow_html=True)

# ===========================
# SECTION 3: TYPES OF STRUCTURAL BREAKS
# ===========================
elif section == "Types of Structural Breaks":
    st.markdown('<p class="sub-header">3. Types of Structural Breaks</p>', unsafe_allow_html=True)

    st.write("### Classification of Structural Breaks")

    st.write("#### 3.1 By Parameter Affected")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("**Level Break (Mean)**")
        st.latex(r"E[y_t] \text{ changes}")
        st.write("Common in:")
        st.write("- GDP trends")
        st.write("- Inflation rates")
        st.write("- Asset returns")

    with col2:
        st.info("**Volatility Break (Variance)**")
        st.latex(r"\text{Var}[y_t] \text{ changes}")
        st.write("Common in:")
        st.write("- Financial volatility")
        st.write("- Exchange rates")
        st.write("- Stock prices")

    with col3:
        st.info("**Structural Break (Both)**")
        st.latex(r"E[y_t] \text{ and } \text{Var}[y_t] \text{ change}")
        st.write("Common in:")
        st.write("- Crisis periods")
        st.write("- Regime shifts")
        st.write("- Policy changes")

    st.write("#### 3.2 By Number of Breaks")

    # Simulation for different break scenarios
    np.random.seed(123)
    n = 300

    # Single break
    y_single = np.concatenate([np.random.normal(0, 1, 150),
                               np.random.normal(3, 1, 150)])

    # Multiple breaks
    y_multiple = np.concatenate([np.random.normal(0, 1, 100),
                                 np.random.normal(3, 1, 100),
                                 np.random.normal(-2, 1, 100)])

    # Unknown number
    y_unknown = np.concatenate([np.random.normal(0, 1, 80),
                                np.random.normal(2, 1.5, 70),
                                np.random.normal(1, 1, 75),
                                np.random.normal(4, 2, 75)])

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("Single Break", "Multiple Breaks (Known)", "Multiple Breaks (Unknown #)"))

    fig.add_trace(go.Scatter(y=y_single, mode='lines', name='Single',
                             line=dict(color='#1f77b4')), row=1, col=1)
    fig.add_vline(x=150, line_dash="dash", line_color="red", row=1, col=1)

    fig.add_trace(go.Scatter(y=y_multiple, mode='lines', name='Multiple (2)',
                             line=dict(color='#ff7f0e')), row=1, col=2)
    fig.add_vline(x=100, line_dash="dash", line_color="red", row=1, col=2)
    fig.add_vline(x=200, line_dash="dash", line_color="red", row=1, col=2)

    fig.add_trace(go.Scatter(y=y_unknown, mode='lines', name='Unknown #',
                             line=dict(color='#2ca02c')), row=1, col=3)
    fig.add_vline(x=80, line_dash="dash", line_color="red", row=1, col=3)
    fig.add_vline(x=150, line_dash="dash", line_color="red", row=1, col=3)
    fig.add_vline(x=225, line_dash="dash", line_color="red", row=1, col=3)

    fig.update_layout(height=400, showlegend=False, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.write("#### 3.3 By Timing Knowledge")

    st.write("**Known Break Date:** (Exogenous)")
    st.latex(r"T_b \text{ is known a priori (e.g., policy change date)}")
    st.write("- Use Chow test")
    st.write("- Standard inference applies")

    st.write("**Unknown Break Date:** (Endogenous)")
    st.latex(r"T_b \text{ must be estimated from data}")
    st.write("- Use supremum tests (QLR, Andrews)")
    st.write("- Modified critical values needed")
    st.write("- Trimming parameter required")

# ===========================
# SECTION 4: BREAK IN MEAN
# ===========================
elif section == "Break in Mean":
    st.markdown('<p class="sub-header">4. Structural Break in Mean</p>', unsafe_allow_html=True)

    st.write("### 4.1 Model Specification")

    st.write("Consider a simple model with a break in mean only:")

    st.latex(r"""
    y_t = \mu_t + \varepsilon_t, \quad \varepsilon_t \sim \text{i.i.d.}(0, \sigma^2)
    """)

    st.write("where the mean follows:")

    st.latex(r"""
    \mu_t = \begin{cases}
    \mu_1 & \text{if } t \leq T_b \\
    \mu_2 & \text{if } t > T_b
    \end{cases}
    """)

    st.write("### 4.2 Estimation")

    st.markdown("""
    <div class="theorem-box">
    <h3>Estimation Procedure</h3>
    <b>Step 1:</b> For each potential break point τ ∈ [T·π, T·(1-π)], estimate:
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"""
    \hat{\mu}_1(\tau) = \frac{1}{\tau}\sum_{t=1}^{\tau} y_t, \quad 
    \hat{\mu}_2(\tau) = \frac{1}{T-\tau}\sum_{t=\tau+1}^{T} y_t
    """)

    st.latex(
        r"\text{Step 2: Calculate } SSR(\tau) = \sum_{t=1}^{\tau}(y_t - \hat{\mu}_1)^2 + \sum_{t=\tau+1}^{T}(y_t - \hat{\mu}_2)^2")

    st.latex(r"\text{Step 3: } \hat{T}_b = \arg\min_{\tau} SSR(\tau)")

    st.write("where π is a trimming parameter (typically π = 0.15).")

    st.write("### 4.3 Interactive Simulation")

    st.write("**Simulation Parameters:**")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        T = st.slider("Sample Size (T)", 100, 500, 200, key="mean_T")
    with col2:
        mu1 = st.slider("Pre-break Mean (μ₁)", -3.0, 3.0, 0.0, key="mu1")
    with col3:
        mu2 = st.slider("Post-break Mean (μ₂)", -3.0, 3.0, 2.0, key="mu2")
    with col4:
        sigma = st.slider("Std Dev (σ)", 0.5, 3.0, 1.0, key="sigma_mean")

    break_pct = st.slider("True Break Point (%)", 30, 70, 50, key="break_mean")

    # Generate data
    np.random.seed(42)
    T_b_true = int(T * break_pct / 100)

    y = np.zeros(T)
    y[:T_b_true] = mu1 + np.random.normal(0, sigma, T_b_true)
    y[T_b_true:] = mu2 + np.random.normal(0, sigma, T - T_b_true)

    # Estimate break point
    trimming = 0.15
    start = int(T * trimming)
    end = int(T * (1 - trimming))

    SSR = np.zeros(end - start)
    for i, tau in enumerate(range(start, end)):
        mu1_hat = np.mean(y[:tau])
        mu2_hat = np.mean(y[tau:])
        SSR[i] = np.sum((y[:tau] - mu1_hat) ** 2) + np.sum((y[tau:] - mu2_hat) ** 2)

    T_b_hat = start + np.argmin(SSR)

    # Create plots
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Time Series with Estimated Break", "SSR Function"),
                        vertical_spacing=0.15)

    # Time series plot
    fig.add_trace(go.Scatter(x=np.arange(T), y=y, mode='lines', name='Data',
                             line=dict(color='#1f77b4', width=2)), row=1, col=1)
    fig.add_vline(x=T_b_true, line_dash="dash", line_color="red",
                  annotation_text=f"True Break: {T_b_true}", row=1, col=1)
    fig.add_vline(x=T_b_hat, line_dash="dot", line_color="green",
                  annotation_text=f"Estimated: {T_b_hat}", row=1, col=1)

    # SSR plot
    fig.add_trace(go.Scatter(x=np.arange(start, end), y=SSR, mode='lines',
                             name='SSR', line=dict(color='#ff7f0e', width=2)), row=2, col=1)
    fig.add_vline(x=T_b_hat, line_dash="dash", line_color="green", row=2, col=1)

    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Potential Break Point", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Sum of Squared Residuals", row=2, col=1)

    fig.update_layout(height=700, showlegend=False, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Results
    st.write("### Estimation Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("True Break Point", T_b_true)
        st.metric("True μ₁", f"{mu1:.3f}")
    with col2:
        st.metric("Estimated Break Point", T_b_hat, delta=T_b_hat - T_b_true)
        st.metric("Estimated μ₁", f"{np.mean(y[:T_b_hat]):.3f}")
    with col3:
        st.metric("Estimation Error", T_b_hat - T_b_true)
        st.metric("Estimated μ₂", f"{np.mean(y[T_b_hat:]):.3f}")

    st.write("### 4.4 Statistical Properties")

    st.markdown("""
    <div class="theorem-box">
    <h3>Consistency of Break Date Estimator</h3>
    Under regularity conditions, as T → ∞:
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"\hat{T}_b - T_b = O_p(1)")

    st.write("This means the estimator is **super-consistent** (converges at rate T, not √T).")

# ===========================
# SECTION 5: BREAK IN VARIANCE
# ===========================
elif section == "Break in Variance":
    st.markdown('<p class="sub-header">5. Structural Break in Variance</p>', unsafe_allow_html=True)

    st.write("### 5.1 Model Specification")

    st.write("Consider a model with constant mean but changing variance:")

    st.latex(r"""
    y_t = \mu + \varepsilon_t, \quad \varepsilon_t \sim \text{i.i.d.}(0, \sigma_t^2)
    """)

    st.write("where the variance follows:")

    st.latex(r"""
    \sigma_t^2 = \begin{cases}
    \sigma_1^2 & \text{if } t \leq T_b \\
    \sigma_2^2 & \text{if } t > T_b
    \end{cases}
    """)

    st.write("### 5.2 Economic Relevance")

    st.write("**Volatility breaks are crucial in:**")
    st.write("- **Financial markets:** Great Moderation (1980s-2007)")
    st.write("- **Risk management:** VaR calculations")
    st.write("- **Option pricing:** Volatility is a key parameter")
    st.write("- **Monetary policy:** Inflation targeting effectiveness")

    st.write("### 5.3 Detection Methods")

    tab1, tab2 = st.tabs(["Likelihood Ratio Test", "ICSS Algorithm"])

    with tab1:
        st.write("**Likelihood Ratio Test for Variance Break:**")

        st.write("Under normality:")
        st.latex(r"""
        LR = T\left[\ln(\hat{\sigma}^2) - \frac{T_b}{T}\ln(\hat{\sigma}_1^2) - \frac{T-T_b}{T}\ln(\hat{\sigma}_2^2)\right]
        """)

        st.write("where:")
        st.latex(r"\hat{\sigma}^2 = \frac{1}{T}\sum_{t=1}^T (y_t - \bar{y})^2")
        st.latex(r"\hat{\sigma}_1^2 = \frac{1}{T_b}\sum_{t=1}^{T_b} (y_t - \bar{y}_1)^2")
        st.latex(r"\hat{\sigma}_2^2 = \frac{1}{T-T_b}\sum_{t=T_b+1}^{T} (y_t - \bar{y}_2)^2")

    with tab2:
        st.write("**Iterated Cumulative Sums of Squares (ICSS):**")
        st.write("Proposed by Inclan & Tiao (1994)")

        st.latex(r"""
        C_k = \frac{\sum_{t=1}^k e_t^2}{\sum_{t=1}^T e_t^2}, \quad k = 1, \ldots, T
        """)

        st.latex(r"""
        D_k = \frac{C_k - k/T}{\sqrt{T/2}}
        """)

        st.write("A break is detected if |Dₖ| exceeds a critical value.")

    st.write("### 5.4 Interactive Simulation")

    st.write("**Simulation Parameters:**")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        T_var = st.slider("Sample Size (T)", 200, 800, 400, key="var_T")
    with col2:
        sigma1 = st.slider("Pre-break Std (σ₁)", 0.5, 2.0, 1.0, key="sigma1")
    with col3:
        sigma2 = st.slider("Post-break Std (σ₂)", 0.5, 4.0, 2.5, key="sigma2")
    with col4:
        mean_val = st.slider("Mean (μ)", -2.0, 2.0, 0.0, key="mean_var")

    break_var_pct = st.slider("True Break Point (%)", 30, 70, 50, key="break_var")

    # Generate data
    np.random.seed(42)
    T_b_var_true = int(T_var * break_var_pct / 100)

    y_var = np.zeros(T_var)
    y_var[:T_b_var_true] = mean_val + np.random.normal(0, sigma1, T_b_var_true)
    y_var[T_b_var_true:] = mean_val + np.random.normal(0, sigma2, T_var - T_b_var_true)

    # Estimate variance break
    trimming = 0.15
    start_var = int(T_var * trimming)
    end_var = int(T_var * (1 - trimming))

    # Demean the data
    y_demean = y_var - np.mean(y_var)

    # Calculate likelihood for each potential break
    LR_vals = np.zeros(end_var - start_var)
    sigma_full = np.var(y_demean)

    for i, tau in enumerate(range(start_var, end_var)):
        sigma1_hat = np.var(y_demean[:tau])
        sigma2_hat = np.var(y_demean[tau:])

        if sigma1_hat > 0 and sigma2_hat > 0:
            LR_vals[i] = T_var * (np.log(sigma_full) -
                                  (tau / T_var) * np.log(sigma1_hat) -
                                  ((T_var - tau) / T_var) * np.log(sigma2_hat))

    T_b_var_hat = start_var + np.argmax(LR_vals)

    # Calculate rolling variance
    window = 40
    rolling_var = pd.Series(y_var).rolling(window=window).var()

    # Create plots
    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=("Time Series", "Rolling Variance", "Likelihood Ratio Statistic"),
                        vertical_spacing=0.1)

    # Time series
    fig.add_trace(go.Scatter(x=np.arange(T_var), y=y_var, mode='lines',
                             name='Data', line=dict(color='#1f77b4', width=1.5)), row=1, col=1)
    fig.add_vline(x=T_b_var_true, line_dash="dash", line_color="red",
                  annotation_text=f"True: {T_b_var_true}", row=1, col=1)
    fig.add_vline(x=T_b_var_hat, line_dash="dot", line_color="green",
                  annotation_text=f"Est: {T_b_var_hat}", row=1, col=1)

    # Rolling variance
    fig.add_trace(go.Scatter(x=np.arange(T_var), y=rolling_var, mode='lines',
                             name='Rolling Var', line=dict(color='#ff7f0e', width=2)), row=2, col=1)
    fig.add_vline(x=T_b_var_true, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_vline(x=T_b_var_hat, line_dash="dot", line_color="green", row=2, col=1)
    fig.add_hline(y=sigma1 ** 2, line_dash="dot", line_color="gray", row=2, col=1)
    fig.add_hline(y=sigma2 ** 2, line_dash="dot", line_color="gray", row=2, col=1)

    # Likelihood ratio
    fig.add_trace(go.Scatter(x=np.arange(start_var, end_var), y=LR_vals, mode='lines',
                             name='LR', line=dict(color='#2ca02c', width=2)), row=3, col=1)
    fig.add_vline(x=T_b_var_hat, line_dash="dash", line_color="green", row=3, col=1)

    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Variance", row=2, col=1)
    fig.update_yaxes(title_text="LR Statistic", row=3, col=1)

    fig.update_layout(height=900, showlegend=False, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Results
    st.write("### Estimation Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("True Break Point", T_b_var_true)
        st.metric("True σ₁²", f"{sigma1 ** 2:.3f}")
    with col2:
        st.metric("Estimated Break", T_b_var_hat, delta=T_b_var_hat - T_b_var_true)
        st.metric("Estimated σ₁²", f"{np.var(y_var[:T_b_var_hat]):.3f}")
    with col3:
        st.metric("Estimation Error", T_b_var_hat - T_b_var_true)
        st.metric("Estimated σ₂²", f"{np.var(y_var[T_b_var_hat:]):.3f}")

# ===========================
# SECTION 6: SIMULTANEOUS BREAKS
# ===========================
elif section == "Simultaneous Breaks":
    st.markdown('<p class="sub-header">6. Simultaneous Breaks in Mean and Variance</p>', unsafe_allow_html=True)

    st.write("### 6.1 Model Specification")

    st.write("The general case with breaks in both moments:")

    st.latex(r"""
    y_t = \mu_t + \varepsilon_t, \quad \varepsilon_t \sim \text{i.i.d.}(0, \sigma_t^2)
    """)

    st.latex(r"""
    \mu_t = \begin{cases}
    \mu_1 & \text{if } t \leq T_b \\
    \mu_2 & \text{if } t > T_b
    \end{cases}, \quad
    \sigma_t^2 = \begin{cases}
    \sigma_1^2 & \text{if } t \leq T_b \\
    \sigma_2^2 & \text{if } t > T_b
    \end{cases}
    """)

    st.markdown("""
    <div class="important-note">
    <h3>⚠️ Critical Issue</h3>
    When both mean and variance change simultaneously, we must account for <b>heteroskedasticity</b> in:
    <ul>
    <li>Parameter estimation</li>
    <li>Standard error calculation</li>
    <li>Test statistic construction</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.write("### 6.2 Joint Estimation")

    st.write("The log-likelihood function (assuming normality):")

    st.latex(r"""
    \ln L = -\frac{T}{2}\ln(2\pi) - \frac{1}{2}\sum_{t=1}^{T_b}\ln(\sigma_1^2) - \frac{1}{2\sigma_1^2}\sum_{t=1}^{T_b}(y_t-\mu_1)^2
    """)
    st.latex(r"""
    - \frac{1}{2}\sum_{t=T_b+1}^{T}\ln(\sigma_2^2) - \frac{1}{2\sigma_2^2}\sum_{t=T_b+1}^{T}(y_t-\mu_2)^2
    """)

    st.write("### 6.3 Interactive Simulation")

    st.write("**Simulation Parameters:**")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Pre-Break Regime:**")
        mu1_joint = st.slider("Mean μ₁", -3.0, 3.0, 0.0, key="mu1_joint")
        sigma1_joint = st.slider("Std σ₁", 0.5, 3.0, 1.0, key="sigma1_joint")

    with col2:
        st.write("**Post-Break Regime:**")
        mu2_joint = st.slider("Mean μ₂", -3.0, 3.0, 2.5, key="mu2_joint")
        sigma2_joint = st.slider("Std σ₂", 0.5, 3.0, 2.0, key="sigma2_joint")

    col1, col2 = st.columns(2)
    with col1:
        T_joint = st.slider("Sample Size", 200, 600, 400, key="T_joint")
    with col2:
        break_joint_pct = st.slider("True Break Point (%)", 30, 70, 50, key="break_joint")

    # Generate data
    np.random.seed(42)
    T_b_joint_true = int(T_joint * break_joint_pct / 100)

    y_joint = np.zeros(T_joint)
    y_joint[:T_b_joint_true] = mu1_joint + np.random.normal(0, sigma1_joint, T_b_joint_true)
    y_joint[T_b_joint_true:] = mu2_joint + np.random.normal(0, sigma2_joint, T_joint - T_b_joint_true)

    # Estimate break point by maximizing likelihood
    trimming = 0.15
    start_joint = int(T_joint * trimming)
    end_joint = int(T_joint * (1 - trimming))

    log_lik = np.zeros(end_joint - start_joint)

    for i, tau in enumerate(range(start_joint, end_joint)):
        # Pre-break estimates
        mu1_hat = np.mean(y_joint[:tau])
        sigma1_hat = np.std(y_joint[:tau])

        # Post-break estimates
        mu2_hat = np.mean(y_joint[tau:])
        sigma2_hat = np.std(y_joint[tau:])

        if sigma1_hat > 0 and sigma2_hat > 0:
            # Log-likelihood
            ll1 = -tau / 2 * np.log(2 * np.pi) - tau / 2 * np.log(sigma1_hat ** 2) - \
                  np.sum((y_joint[:tau] - mu1_hat) ** 2) / (2 * sigma1_hat ** 2)
            ll2 = -(T_joint - tau) / 2 * np.log(2 * np.pi) - (T_joint - tau) / 2 * np.log(sigma2_hat ** 2) - \
                  np.sum((y_joint[tau:] - mu2_hat) ** 2) / (2 * sigma2_hat ** 2)
            log_lik[i] = ll1 + ll2

    T_b_joint_hat = start_joint + np.argmax(log_lik)

    # Calculate statistics for both regimes
    mu1_est = np.mean(y_joint[:T_b_joint_hat])
    sigma1_est = np.std(y_joint[:T_b_joint_hat])
    mu2_est = np.mean(y_joint[T_b_joint_hat:])
    sigma2_est = np.std(y_joint[T_b_joint_hat:])

    # Create visualizations
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Time Series", "Log-Likelihood Function",
                                        "Distribution Pre-Break", "Distribution Post-Break"),
                        specs=[[{"colspan": 2}, None],
                               [{}, {}]],
                        vertical_spacing=0.15)

    # Time series
    fig.add_trace(go.Scatter(x=np.arange(T_joint), y=y_joint, mode='lines',
                             name='Data', line=dict(color='#1f77b4', width=1.5)), row=1, col=1)
    fig.add_vline(x=T_b_joint_true, line_dash="dash", line_color="red",
                  annotation_text=f"True: {T_b_joint_true}", row=1, col=1)
    fig.add_vline(x=T_b_joint_hat, line_dash="dot", line_color="green",
                  annotation_text=f"Est: {T_b_joint_hat}", row=1, col=1)

    # Add mean lines
    fig.add_hline(y=mu1_joint, line_dash="dot", line_color="gray", opacity=0.5, row=1, col=1)
    fig.add_hline(y=mu2_joint, line_dash="dot", line_color="gray", opacity=0.5, row=1, col=1)

    # Distributions
    x_range = np.linspace(min(y_joint) - 1, max(y_joint) + 1, 200)
    dist1_true = stats.norm.pdf(x_range, mu1_joint, sigma1_joint)
    dist1_est = stats.norm.pdf(x_range, mu1_est, sigma1_est)
    dist2_true = stats.norm.pdf(x_range, mu2_joint, sigma2_joint)
    dist2_est = stats.norm.pdf(x_range, mu2_est, sigma2_est)

    # Pre-break distribution
    fig.add_trace(go.Scatter(x=x_range, y=dist1_true, mode='lines', name='True',
                             line=dict(color='red', dash='dash')), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_range, y=dist1_est, mode='lines', name='Estimated',
                             line=dict(color='green')), row=2, col=1)

    # Post-break distribution
    fig.add_trace(go.Scatter(x=x_range, y=dist2_true, mode='lines', name='True',
                             line=dict(color='red', dash='dash'), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=x_range, y=dist2_est, mode='lines', name='Estimated',
                             line=dict(color='green'), showlegend=False), row=2, col=2)

    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Value", row=2, col=1)
    fig.update_xaxes(title_text="Value", row=2, col=2)
    fig.update_yaxes(title_text="y", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=2, col=1)
    fig.update_yaxes(title_text="Density", row=2, col=2)

    fig.update_layout(height=700, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Results table
    st.write("### Estimation Results")

    results_df = pd.DataFrame({
        'Parameter': ['Break Point', 'μ₁', 'σ₁', 'μ₂', 'σ₂'],
        'True Value': [T_b_joint_true, mu1_joint, sigma1_joint, mu2_joint, sigma2_joint],
        'Estimated': [T_b_joint_hat, mu1_est, sigma1_est, mu2_est, sigma2_est],
        'Error': [T_b_joint_hat - T_b_joint_true,
                  mu1_est - mu1_joint,
                  sigma1_est - sigma1_joint,
                  mu2_est - mu2_joint,
                  sigma2_est - sigma2_joint]
    })

    st.dataframe(results_df.style.format({'True Value': '{:.3f}',
                                          'Estimated': '{:.3f}',
                                          'Error': '{:.3f}'}),
                 hide_index=True, use_container_width=True)

# ===========================
# SECTION 7: TESTING PROCEDURES
# ===========================
elif section == "Testing Procedures":
    st.markdown('<p class="sub-header">7. Testing for Structural Breaks</p>', unsafe_allow_html=True)

    st.write("### 7.1 Chow Test (Known Break Date)")

    st.markdown("""
    <div class="definition-box">
    <h3>Chow Test</h3>
    When the break date is <b>known a priori</b>, we use the classical Chow test.
    </div>
    """, unsafe_allow_html=True)

    st.write("**Test Statistic:**")

    st.latex(r"""
    F = \frac{(SSR_r - SSR_u)/k}{SSR_u/(T-2k)} \sim F(k, T-2k)
    """)

    st.write("where:")
    st.latex(r"SSR_r = \text{restricted sum of squared residuals (no break)}")
    st.latex(r"SSR_u = \text{unrestricted sum of squared residuals (with break)}")
    st.latex(r"k = \text{number of parameters}")

    st.write("### 7.2 Quandt Likelihood Ratio (QLR) Test")

    st.write("**When break date is unknown:**")

    st.latex(r"""
    QLR = \sup_{\tau \in [\pi T, (1-\pi)T]} F(\tau)
    """)

    st.write("The QLR test takes the **supremum** (maximum) of the F-statistics over all potential break points.")

    st.markdown("""
    <div class="theorem-box">
    <h3>Critical Values</h3>
    The QLR test statistic does <b>not</b> follow a standard F-distribution due to the search over τ.
    Critical values depend on:
    <ul>
    <li>Number of parameters tested (k)</li>
    <li>Trimming parameter (π)</li>
    <li>Must use Andrews (1993) critical values</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.write("### 7.3 Interactive Testing Simulation")

    st.write("**Design the Data Generating Process:**")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Null Hypothesis (No Break):**")
        mu_null = st.slider("Mean under H₀", -2.0, 2.0, 0.0, key="mu_null")
        sigma_null = st.slider("Std under H₀", 0.5, 2.0, 1.0, key="sigma_null")

    with col2:
        st.write("**Alternative (With Break):**")
        has_break = st.checkbox("Include structural break", value=True)
        if has_break:
            mu_alt = st.slider("Post-break mean", -2.0, 3.0, 1.5, key="mu_alt")
            break_test_pct = st.slider("Break location (%)", 40, 60, 50, key="break_test")

    T_test = st.slider("Sample Size", 100, 400, 200, key="T_test")

    # Generate data
    np.random.seed(42)

    if has_break:
        T_b_test = int(T_test * break_test_pct / 100)
        y_test = np.concatenate([
            mu_null + np.random.normal(0, sigma_null, T_b_test),
            mu_alt + np.random.normal(0, sigma_null, T_test - T_b_test)
        ])
    else:
        y_test = mu_null + np.random.normal(0, sigma_null, T_test)
        T_b_test = T_test // 2

    # Calculate F-statistics for all potential break points
    trimming = 0.15
    start_test = int(T_test * trimming)
    end_test = int(T_test * (1 - trimming))

    F_stats = np.zeros(end_test - start_test)

    # Restricted SSR (no break)
    y_mean = np.mean(y_test)
    SSR_r = np.sum((y_test - y_mean) ** 2)

    for i, tau in enumerate(range(start_test, end_test)):
        # Unrestricted SSR (with break at tau)
        mu1_hat = np.mean(y_test[:tau])
        mu2_hat = np.mean(y_test[tau:])
        SSR_u = np.sum((y_test[:tau] - mu1_hat) ** 2) + np.sum((y_test[tau:] - mu2_hat) ** 2)

        # F-statistic (k=1 for mean only)
        k = 1
        F_stats[i] = ((SSR_r - SSR_u) / k) / (SSR_u / (T_test - 2 * k))

    QLR_stat = np.max(F_stats)
    QLR_location = start_test + np.argmax(F_stats)

    # Plot F-statistics
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Time Series Data", "F-Statistics Across Potential Break Points"),
                        vertical_spacing=0.15)

    # Time series
    fig.add_trace(go.Scatter(x=np.arange(T_test), y=y_test, mode='lines',
                             name='Data', line=dict(color='#1f77b4', width=2)), row=1, col=1)
    if has_break:
        fig.add_vline(x=T_b_test, line_dash="dash", line_color="red",
                      annotation_text=f"True Break: {T_b_test}", row=1, col=1)
    fig.add_vline(x=QLR_location, line_dash="dot", line_color="green",
                  annotation_text=f"QLR at: {QLR_location}", row=1, col=1)

    # F-statistics
    fig.add_trace(go.Scatter(x=np.arange(start_test, end_test), y=F_stats, mode='lines',
                             name='F-stat', line=dict(color='#ff7f0e', width=2)), row=2, col=1)

    # Critical value (approximate for k=1, π=0.15)
    critical_10 = 7.04
    critical_05 = 8.68
    critical_01 = 12.16

    fig.add_hline(y=critical_05, line_dash="dash", line_color="red",
                  annotation_text="5% critical value", row=2, col=1)
    fig.add_vline(x=QLR_location, line_dash="dot", line_color="green", row=2, col=1)

    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Potential Break Point", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="F-Statistic", row=2, col=1)

    fig.update_layout(height=700, showlegend=False, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Test results
    st.write("### Test Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("QLR Statistic", f"{QLR_stat:.3f}")
        st.metric("Break Location", QLR_location)

    with col2:
        st.write("**Critical Values:**")
        st.write(f"10%: {critical_10:.2f}")
        st.write(f"5%: {critical_05:.2f}")
        st.write(f"1%: {critical_01:.2f}")

    with col3:
        if QLR_stat > critical_05:
            st.success("✓ Reject H₀ at 5% level")
            st.write("Evidence of structural break")
        else:
            st.info("✗ Fail to reject H₀")
            st.write("No evidence of break")

    st.write("### 7.4 Multiple Break Tests")

    st.write("**Bai-Perron (1998, 2003) Sequential Testing:**")

    st.latex(r"\text{UDmax} = \max_{1 \leq m \leq M} QLR(m)")

    st.write("where QLR(m) is the test for m breaks vs. m-1 breaks.")

    st.write("**Procedure:**")
    st.write("1. Test 0 vs. 1 break")
    st.write("2. If rejected, test 1 vs. 2 breaks")
    st.write("3. Continue until fail to reject")

# ===========================
# SECTION 8: IMPLICATIONS & APPLICATIONS
# ===========================
elif section == "Implications & Applications":
    st.markdown('<p class="sub-header">8. Implications and Applications</p>', unsafe_allow_html=True)

    st.write("### 8.1 Consequences of Ignoring Breaks")

    st.write("**Impact on Forecasting:**")

    # Simulation comparing forecasts
    np.random.seed(42)
    T_fore = 150
    T_b_fore = 100
    h = 20  # forecast horizon

    mu1_fore, mu2_fore = 0, 3

    y_fore = np.concatenate([
        mu1_fore + np.random.normal(0, 1, T_b_fore),
        mu2_fore + np.random.normal(0, 1, T_fore - T_b_fore)
    ])

    # Model 1: Ignore break (use full sample mean)
    forecast_ignore = np.repeat(np.mean(y_fore), h)

    # Model 2: Account for break (use post-break mean)
    forecast_account = np.repeat(np.mean(y_fore[T_b_fore:]), h)

    # True future values
    y_future = mu2_fore + np.random.normal(0, 1, h)

    # Calculate errors
    MSE_ignore = np.mean((y_future - forecast_ignore) ** 2)
    MSE_account = np.mean((y_future - forecast_account) ** 2)

    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(x=np.arange(T_fore), y=y_fore, mode='lines',
                             name='Historical Data', line=dict(color='#1f77b4', width=2)))
    fig.add_vline(x=T_b_fore, line_dash="dash", line_color="red",
                  annotation_text="Structural Break")

    # Future realizations
    fig.add_trace(go.Scatter(x=np.arange(T_fore, T_fore + h), y=y_future, mode='markers+lines',
                             name='Actual Future', marker=dict(color='black', size=6),
                             line=dict(color='black', width=2)))

    # Forecasts
    fig.add_trace(go.Scatter(x=np.arange(T_fore, T_fore + h), y=forecast_ignore, mode='lines',
                             name='Forecast (Ignoring Break)', line=dict(color='red', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=np.arange(T_fore, T_fore + h), y=forecast_account, mode='lines',
                             name='Forecast (Accounting Break)', line=dict(color='green', width=2, dash='dot')))

    fig.update_layout(title="Forecasting Performance with and without Break Detection",
                      xaxis_title="Time", yaxis_title="Value",
                      height=500, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("MSE (Ignoring Break)", f"{MSE_ignore:.3f}")
    with col2:
        st.metric("MSE (Accounting Break)", f"{MSE_account:.3f}",
                  delta=f"{((MSE_account / MSE_ignore - 1) * 100):.1f}%", delta_color="inverse")

    st.write("### 8.2 Unit Root Testing with Breaks")

    st.markdown("""
    <div class="important-note">
    <h3>⚠️ Critical Issue: Perron (1989)</h3>
    Standard unit root tests (ADF, PP) have <b>low power</b> in the presence of structural breaks.
    A stationary series with a level shift can appear non-stationary.
    </div>
    """, unsafe_allow_html=True)

    st.write("**Illustration:**")

    # Generate trend-stationary with break
    np.random.seed(42)
    T_unit = 200
    T_b_unit = 100

    # Trend stationary with break
    t = np.arange(T_unit)
    y_ts_break = 0.05 * t + np.random.normal(0, 1, T_unit)
    y_ts_break[T_b_unit:] += 5  # level shift

    # Random walk (true unit root)
    y_rw = np.cumsum(np.random.normal(0, 1, T_unit))

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Trend-Stationary with Break", "Random Walk (True Unit Root)"))

    fig.add_trace(go.Scatter(x=t, y=y_ts_break, mode='lines', name='TS+Break',
                             line=dict(color='#1f77b4', width=2)), row=1, col=1)
    fig.add_vline(x=T_b_unit, line_dash="dash", line_color="red", row=1, col=1)

    fig.add_trace(go.Scatter(x=t, y=y_rw, mode='lines', name='Random Walk',
                             line=dict(color='#ff7f0e', width=2)), row=1, col=2)

    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=2)

    fig.update_layout(height=400, showlegend=False, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.write("**Solution: Perron-Type Tests**")

    st.write("Modified ADF test allowing for break:")

    st.latex(
        r"\Delta y_t = \mu + \beta t + \theta DU_t + \gamma DT_t + \alpha y_{t-1} + \sum_{i=1}^p \phi_i \Delta y_{t-i} + \varepsilon_t")

    st.write("where:")
    st.latex(
        r"DU_t = \begin{cases} 1 & \text{if } t > T_b \\ 0 & \text{otherwise} \end{cases} \quad \text{(level shift dummy)}")
    st.latex(
        r"DT_t = \begin{cases} t - T_b & \text{if } t > T_b \\ 0 & \text{otherwise} \end{cases} \quad \text{(trend shift dummy)}")

    st.write("### 8.3 Cointegration with Breaks")

    st.write("**Gregory-Hansen (1996) Test:**")

    st.write("Tests for cointegration allowing for regime shift in the cointegrating vector:")

    st.latex(r"""
    y_t = \mu_t + \alpha_t' x_t + u_t
    """)

    st.write("where μₜ and/or αₜ may change at unknown time T_b.")

    st.write("### 8.4 Practical Recommendations")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Pre-Estimation:**")
        st.write("✓ Plot your data")
        st.write("✓ Check for obvious breaks")
        st.write("✓ Consider economic events")
        st.write("✓ Test for multiple breaks")
        st.write("✓ Use recursive estimation")

    with col2:
        st.write("**Post-Estimation:**")
        st.write("✓ Test parameter stability")
        st.write("✓ Use robust standard errors")
        st.write("✓ Consider sub-sample analysis")
        st.write("✓ Out-of-sample validation")
        st.write("✓ Rolling window forecasts")

    st.write("### 8.5 Advanced Topics Preview")

    st.markdown("""
    <div class="definition-box">
    <h3>Extensions for Advanced Study</h3>
    <ul>
    <li><b>Markov-Switching Models:</b> Probabilistic regime changes</li>
    <li><b>Threshold Models (TAR, SETAR):</b> Endogenous regime switching</li>
    <li><b>Time-Varying Parameter Models:</b> Continuous parameter evolution</li>
    <li><b>Bayesian Change Point Detection:</b> Posterior probability of breaks</li>
    <li><b>Machine Learning Approaches:</b> Neural networks for break detection</li>
    <li><b>High-Frequency Data:</b> Intraday structural changes</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.write("### 8.6 Summary: Key Takeaways")

    st.write("**Theoretical Insights:**")
    st.write("1. Structural breaks violate parameter constancy assumption")
    st.write("2. Break date estimation is super-consistent but affects inference")
    st.write("3. Test statistics have non-standard distributions when break is unknown")

    st.write("**Practical Implications:**")
    st.write("1. Always test for structural stability")
    st.write("2. Use appropriate critical values (Andrews, Bai-Perron)")
    st.write("3. Consider multiple breaks")
    st.write("4. Account for breaks in forecasting")
    st.write("5. Be aware of interactions with unit roots and cointegration")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 20px;'>
<p><b>Structural Breaks in Economic Time Series</b></p>
<p>Lecture Material by Dr. Merwan Roudane</p>
<p>Foundation for Advanced Econometric Topics</p>
</div>
""", unsafe_allow_html=True)