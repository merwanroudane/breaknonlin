import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import jarque_bera, shapiro
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Nonlinearity in Economic Time Series",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stAlert {
        background-color: #e3f2fd;
        border-left: 5px solid #2196F3;
    }
    h1 {
        color: #1565C0;
        font-family: 'Arial', sans-serif;
    }
    h2 {
        color: #1976D2;
        font-family: 'Arial', sans-serif;
    }
    h3 {
        color: #1E88E5;
        font-family: 'Arial', sans-serif;
    }
    .highlight {
        background-color: #FFF3E0;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #FF9800;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Author
st.title("📊 Nonlinearity in Economic Time Series")
st.markdown("### *An Introduction to Advanced Econometric Analysis*")
st.markdown("**By Dr. Merwan Roudane**")
st.markdown("---")

# Sidebar Navigation
st.sidebar.title("📑 Navigation")
section = st.sidebar.radio(
    "Select Section:",
    [
        "1. Introduction",
        "2. Linear vs Nonlinear Models",
        "3. Sources of Nonlinearity",
        "4. Testing for Nonlinearity",
        "5. Threshold Models (TAR)",
        "6. Markov Switching Models",
        "7. Smooth Transition Models",
        "8. Simulations & Applications",
        "9. Conclusion & References"
    ]
)


# Helper Functions
def generate_linear_ts(n=500, ar_coef=0.7, noise_std=1):
    """Generate linear AR(1) process"""
    y = np.zeros(n)
    epsilon = np.random.normal(0, noise_std, n)
    y[0] = epsilon[0]
    for t in range(1, n):
        y[t] = ar_coef * y[t - 1] + epsilon[t]
    return y


def generate_tar_model(n=500, threshold=0, ar1_low=0.5, ar1_high=-0.3, noise_std=1):
    """Generate Threshold Autoregressive model"""
    y = np.zeros(n)
    epsilon = np.random.normal(0, noise_std, n)
    regime = np.zeros(n, dtype=int)
    y[0] = epsilon[0]

    for t in range(1, n):
        if y[t - 1] < threshold:
            y[t] = ar1_low * y[t - 1] + epsilon[t]
            regime[t] = 0
        else:
            y[t] = ar1_high * y[t - 1] + epsilon[t]
            regime[t] = 1

    return y, regime


def generate_markov_switching(n=500, ar_low=0.8, ar_high=-0.5, p11=0.95, p22=0.95, noise_std=1):
    """Generate Markov Switching AR model"""
    y = np.zeros(n)
    states = np.zeros(n, dtype=int)
    epsilon = np.random.normal(0, noise_std, n)

    # Transition probabilities
    states[0] = np.random.choice([0, 1])
    y[0] = epsilon[0]

    for t in range(1, n):
        # State transition
        if states[t - 1] == 0:
            states[t] = 0 if np.random.rand() < p11 else 1
        else:
            states[t] = 1 if np.random.rand() < p22 else 0

        # Generate observation
        if states[t] == 0:
            y[t] = ar_low * y[t - 1] + epsilon[t]
        else:
            y[t] = ar_high * y[t - 1] + epsilon[t]

    return y, states


def generate_star_model(n=500, c=0, gamma=2, ar_linear=0.7, ar_nonlinear=-0.5, noise_std=1):
    """Generate Smooth Transition AR model"""
    y = np.zeros(n)
    epsilon = np.random.normal(0, noise_std, n)
    transition = np.zeros(n)
    y[0] = epsilon[0]

    for t in range(1, n):
        # Logistic transition function
        G = 1 / (1 + np.exp(-gamma * (y[t - 1] - c)))
        transition[t] = G

        # STAR model
        y[t] = (ar_linear * (1 - G) + ar_nonlinear * G) * y[t - 1] + epsilon[t]

    return y, transition


def bds_test_simplified(data, m=2, epsilon_std=0.5):
    """Simplified BDS test for nonlinearity"""
    n = len(data)
    epsilon = epsilon_std * np.std(data)

    # Create embedding
    embedded = []
    for i in range(n - m + 1):
        embedded.append(data[i:i + m])
    embedded = np.array(embedded)

    # Calculate correlation integral
    def correlation_integral(eps, m_dim):
        count = 0
        n_points = len(embedded)
        for i in range(n_points):
            for j in range(i + 1, n_points):
                if np.max(np.abs(embedded[i] - embedded[j])) < eps:
                    count += 1
        return 2 * count / (n_points * (n_points - 1))

    C_m = correlation_integral(epsilon, m)
    C_1 = correlation_integral(epsilon, 1)

    # BDS statistic
    if C_1 > 0:
        bds_stat = np.sqrt(n) * (C_m - C_1 ** m)
        return bds_stat
    return 0


# Section 1: Introduction
if section == "1. Introduction":
    st.header("1. Introduction to Nonlinearity in Economics")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Why Study Nonlinearity?

        Economic time series often exhibit behaviors that cannot be adequately captured by linear models:

        **Key Characteristics:**
        - **Asymmetric responses** to positive and negative shocks
        - **Regime-dependent dynamics** (expansion vs. recession)
        - **Threshold effects** in policy variables
        - **State-dependent volatility** and persistence
        - **Complex cyclical patterns** beyond simple periodicity

        Traditional linear models assume:
        """)

        st.latex(r"y_t = \phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \phi_p y_{t-p} + \varepsilon_t")

        st.markdown("""
        This assumption may be too restrictive for many economic phenomena.
        """)

    with col2:
        st.info("""
        **Learning Objectives:**

        ✓ Understand sources of nonlinearity

        ✓ Test for nonlinear behavior

        ✓ Model threshold effects

        ✓ Apply Markov switching

        ✓ Forecast nonlinear series
        """)

    st.markdown("---")

    # Visual comparison
    st.subheader("📊 Linear vs Nonlinear Behavior: A First Look")

    np.random.seed(42)
    linear_data = generate_linear_ts(n=300, ar_coef=0.7)
    nonlinear_data, _ = generate_tar_model(n=300, threshold=0, ar1_low=0.8, ar1_high=-0.6)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Linear AR(1) Process", "Nonlinear TAR Process")
    )

    fig.add_trace(
        go.Scatter(y=linear_data, mode='lines', name='Linear', line=dict(color='#2196F3', width=2)),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(y=nonlinear_data, mode='lines', name='Nonlinear', line=dict(color='#FF5722', width=2)),
        row=1, col=2
    )

    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="Comparison of Time Series Behavior"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="highlight">
    <b>Observation:</b> The nonlinear series shows regime-dependent behavior with asymmetric 
    responses, while the linear series maintains consistent dynamics throughout.
    </div>
    """, unsafe_allow_html=True)

# Section 2: Linear vs Nonlinear Models
elif section == "2. Linear vs Nonlinear Models":
    st.header("2. Linear vs Nonlinear Models")

    st.markdown("""
    ### 2.1 Linear Time Series Models

    Linear models assume that the relationship between variables is constant over time and states.
    """)

    st.latex(r"\text{AR}(p): \quad y_t = c + \sum_{i=1}^{p} \phi_i y_{t-i} + \varepsilon_t")

    st.markdown("""
    **Properties of Linear Models:**
    - Constant parameters over time
    - Symmetric impulse responses
    - Gaussian distribution (if errors are Gaussian)
    - Superposition principle holds
    """)

    st.markdown("---")

    st.markdown("""
    ### 2.2 Nonlinear Time Series Models

    Nonlinear models allow for:
    - **Parameter variation** across states/regimes
    - **Asymmetric dynamics**
    - **State-dependent volatility**
    - **Complex transition mechanisms**
    """)

    # Interactive comparison
    st.subheader("🔬 Interactive Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Linear Model Parameters:**")
        ar_linear = st.slider("AR coefficient (φ)", -0.9, 0.9, 0.7, 0.1, key='linear_ar')
        noise_linear = st.slider("Noise std (σ)", 0.5, 2.0, 1.0, 0.1, key='linear_noise')

    with col2:
        st.markdown("**Nonlinear Model Parameters:**")
        ar_low = st.slider("AR coef (low regime)", -0.9, 0.9, 0.8, 0.1, key='nl_low')
        ar_high = st.slider("AR coef (high regime)", -0.9, 0.9, -0.5, 0.1, key='nl_high')

    # Generate data
    np.random.seed(42)
    y_linear = generate_linear_ts(n=400, ar_coef=ar_linear, noise_std=noise_linear)
    y_nonlinear, regime = generate_tar_model(n=400, threshold=0, ar1_low=ar_low, ar1_high=ar_high, noise_std=1)

    # Create comprehensive visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Linear Series", "Nonlinear Series (colored by regime)",
            "Linear ACF Pattern", "Nonlinear ACF Pattern"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Time series plots
    fig.add_trace(
        go.Scatter(y=y_linear, mode='lines', name='Linear', line=dict(color='#2196F3', width=1.5)),
        row=1, col=1
    )

    # Regime-colored nonlinear series
    colors = ['#4CAF50' if r == 0 else '#FF5722' for r in regime]
    fig.add_trace(
        go.Scatter(y=y_nonlinear, mode='markers', name='Nonlinear',
                   marker=dict(color=colors, size=4)),
        row=1, col=2
    )


    # ACF calculations
    def calculate_acf(data, nlags=20):
        acf_vals = [1.0]
        for lag in range(1, nlags + 1):
            c0 = np.var(data)
            c_lag = np.cov(data[:-lag], data[lag:])[0, 1]
            acf_vals.append(c_lag / c0)
        return acf_vals


    acf_linear = calculate_acf(y_linear)
    acf_nonlinear = calculate_acf(y_nonlinear)

    # ACF plots
    fig.add_trace(
        go.Bar(y=acf_linear, name='Linear ACF', marker_color='#2196F3'),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(y=acf_nonlinear, name='Nonlinear ACF', marker_color='#FF5722'),
        row=2, col=2
    )

    # Add confidence intervals
    ci = 1.96 / np.sqrt(len(y_linear))
    for col in [1, 2]:
        fig.add_hline(y=ci, line_dash="dash", line_color="gray", row=2, col=col)
        fig.add_hline(y=-ci, line_dash="dash", line_color="gray", row=2, col=col)

    fig.update_layout(height=700, showlegend=False)
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_xaxes(title_text="Lag", row=2, col=1)
    fig.update_xaxes(title_text="Lag", row=2, col=2)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=2)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="highlight">
    <b>Key Insights:</b>
    <ul>
    <li>The nonlinear series shows <b>regime-switching behavior</b> (green = low regime, red = high regime)</li>
    <li>ACF patterns may appear similar, but underlying dynamics differ fundamentally</li>
    <li>Nonlinearity is not always obvious from visual inspection alone</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Section 3: Sources of Nonlinearity
elif section == "3. Sources of Nonlinearity":
    st.header("3. Sources of Nonlinearity in Economics")

    st.markdown("""
    Nonlinearity in economic time series can arise from various sources. Understanding these 
    sources is crucial for proper model specification.
    """)

    # Tabs for different sources
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Structural Breaks",
        "🔄 Regime Switching",
        "📈 Asymmetric Adjustment",
        "⚡ Threshold Effects"
    ])

    with tab1:
        st.markdown("""
        ### Structural Breaks and Regime Changes

        Economic relationships may change due to:
        - **Policy regime changes** (monetary policy, regulation)
        - **Technological innovations**
        - **Financial crises**
        - **Structural economic reforms**
        """)

        st.latex(r"""
        y_t = \begin{cases}
        \phi_1 y_{t-1} + \varepsilon_t & \text{if } t \leq T^* \\
        \phi_2 y_{t-1} + \varepsilon_t & \text{if } t > T^*
        \end{cases}
        """)

        # Simulation
        np.random.seed(42)
        n = 300
        y = np.zeros(n)
        epsilon = np.random.normal(0, 1, n)
        T_star = 150

        y[0] = epsilon[0]
        for t in range(1, n):
            if t <= T_star:
                y[t] = 0.8 * y[t - 1] + epsilon[t]
            else:
                y[t] = -0.3 * y[t - 1] + epsilon[t]

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y[:T_star], mode='lines', name='Regime 1',
                                 line=dict(color='#4CAF50', width=2)))
        fig.add_trace(go.Scatter(y=y[T_star:], mode='lines', name='Regime 2',
                                 line=dict(color='#FF5722', width=2),
                                 x=np.arange(T_star, n)))
        fig.add_vline(x=T_star, line_dash="dash", line_color="black",
                      annotation_text="Structural Break")
        fig.update_layout(title="Structural Break at t=150", height=400,
                          xaxis_title="Time", yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("""
        ### Markov Switching Dynamics

        Regimes governed by an unobserved Markov chain:
        - **Business cycles** (expansion/recession)
        - **Market conditions** (bull/bear markets)
        - **Volatility regimes** (calm/turbulent)
        """)

        st.latex(r"""
        y_t = \mu_{S_t} + \phi_{S_t} y_{t-1} + \sigma_{S_t} \varepsilon_t
        """)

        st.latex(r"""
        P(S_t = j | S_{t-1} = i) = p_{ij}
        """)

        # Generate Markov Switching data
        np.random.seed(42)
        y_ms, states = generate_markov_switching(n=400, ar_low=0.7, ar_high=-0.4, p11=0.96, p22=0.96)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Time Series (colored by state)", "Hidden State"),
                            vertical_spacing=0.1)

        colors = ['#2196F3' if s == 0 else '#FF9800' for s in states]
        fig.add_trace(go.Scatter(y=y_ms, mode='markers', marker=dict(color=colors, size=4),
                                 name='Series'), row=1, col=1)
        fig.add_trace(go.Scatter(y=states, mode='lines', line=dict(color='black', width=2),
                                 name='State'), row=2, col=1)

        fig.update_layout(height=500, showlegend=False)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="State", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("""
        ### Asymmetric Adjustment Mechanisms

        Economic variables may respond differently to:
        - **Positive vs. negative shocks**
        - **Large vs. small shocks**
        - **Upward vs. downward movements**

        **Example:** Price rigidity - prices may increase faster than they decrease.
        """)

        st.latex(r"""
        \Delta y_t = \begin{cases}
        \alpha^+ \Delta x_t + \varepsilon_t & \text{if } \Delta x_t > 0 \\
        \alpha^- \Delta x_t + \varepsilon_t & \text{if } \Delta x_t \leq 0
        \end{cases}
        """)

        # Simulation of asymmetric response
        np.random.seed(42)
        x = np.random.normal(0, 1, 200)
        y_asym = np.zeros(200)

        for t in range(1, 200):
            if x[t] > 0:
                y_asym[t] = 0.8 * x[t]  # Strong positive response
            else:
                y_asym[t] = 0.3 * x[t]  # Weak negative response

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y_asym, mode='markers',
                                 marker=dict(color=x, colorscale='RdYlGn', size=6),
                                 name='Response'))
        fig.add_trace(go.Scatter(x=[-3, 3], y=[-3 * 0.8, 3 * 0.8], mode='lines',
                                 line=dict(dash='dash', color='gray'), name='Symmetric'))
        fig.update_layout(title="Asymmetric Response Function",
                          xaxis_title="Shock (Δx)", yaxis_title="Response (Δy)",
                          height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("""
        ### Threshold Effects

        Dynamics change when a variable crosses a threshold:
        - **Capacity constraints**
        - **Policy intervention thresholds**
        - **Transaction costs**
        - **Menu costs in price adjustment**
        """)

        st.latex(r"""
        y_t = \begin{cases}
        \phi_1 y_{t-1} + \varepsilon_t & \text{if } y_{t-d} \leq c \\
        \phi_2 y_{t-1} + \varepsilon_t & \text{if } y_{t-d} > c
        \end{cases}
        """)

        st.markdown("where $c$ is the threshold and $d$ is the delay parameter.")

        # Interactive threshold demonstration
        threshold_val = st.slider("Threshold value (c)", -2.0, 2.0, 0.0, 0.5)

        np.random.seed(42)
        y_tar, regime_tar = generate_tar_model(n=300, threshold=threshold_val,
                                               ar1_low=0.7, ar1_high=-0.5)

        fig = go.Figure()
        colors_regime = ['#4CAF50' if r == 0 else '#FF5722' for r in regime_tar]
        fig.add_trace(go.Scatter(y=y_tar, mode='markers',
                                 marker=dict(color=colors_regime, size=5),
                                 name='TAR Series'))
        fig.add_hline(y=threshold_val, line_dash="dash", line_color="black",
                      annotation_text=f"Threshold = {threshold_val}")
        fig.update_layout(title="Threshold Autoregressive (TAR) Process",
                          xaxis_title="Time", yaxis_title="Value", height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.info(f"""
        **Green points:** Below threshold (φ = 0.7, persistent)  
        **Red points:** Above threshold (φ = -0.5, mean-reverting)
        """)

# Section 4: Testing for Nonlinearity
elif section == "4. Testing for Nonlinearity":
    st.header("4. Testing for Nonlinearity")

    st.markdown("""
    Before specifying a nonlinear model, we need statistical evidence of nonlinearity.
    Several tests are available:
    """)

    test_type = st.selectbox(
        "Select Test to Explore:",
        ["Overview", "BDS Test", "Tsay Test", "McLeod-Li Test", "Ramsey RESET Test"]
    )

    if test_type == "Overview":
        st.markdown("""
        ### Common Tests for Nonlinearity

        | Test | Null Hypothesis | Advantages | Limitations |
        |------|-----------------|------------|-------------|
        | **BDS Test** | i.i.d. data | Detects general nonlinearity | Computationally intensive |
        | **Tsay Test** | Linear AR | Specific to AR processes | Requires correct lag order |
        | **McLeod-Li** | No ARCH effects | Simple to implement | Only for conditional heteroskedasticity |
        | **RESET Test** | Linear specification | Easy to compute | Low power in some cases |
        """)

        st.markdown("""
        ### General Testing Strategy

        1. **Visual inspection** of the time series
        2. **Estimate linear benchmark** model
        3. **Examine residuals** for patterns
        4. **Apply formal tests** for nonlinearity
        5. **Check for structural breaks**
        6. **Test for threshold effects**
        """)

    elif test_type == "BDS Test":
        st.markdown("""
        ### BDS Test (Brock, Dechert, Scheinkman, 1996)

        Tests for **i.i.d.** against general alternatives including nonlinearity, chaos, and non-stationarity.

        **Test Statistic:**
        """)

        st.latex(r"""
        BDS_{n,m}(\varepsilon) = \sqrt{n} \frac{C_{m,n}(\varepsilon) - C_{1,n}(\varepsilon)^m}{\sigma_{m,n}(\varepsilon)}
        """)

        st.markdown("""
        where:
        - $C_{m,n}(\varepsilon)$ is the correlation integral for embedding dimension $m$
        - $\varepsilon$ is the distance threshold
        - Under null hypothesis: $BDS \sim N(0,1)$ asymptotically
        """)

        st.markdown("---")
        st.subheader("📊 Interactive BDS Test Demonstration")

        col1, col2 = st.columns(2)
        with col1:
            data_choice = st.radio("Select Data Type:", ["Linear AR(1)", "Nonlinear TAR", "Custom"])

        with col2:
            if data_choice != "Custom":
                n_obs = st.slider("Number of observations", 200, 1000, 500, 100)

        # Generate or use data
        np.random.seed(42)
        if data_choice == "Linear AR(1)":
            test_data = generate_linear_ts(n=n_obs, ar_coef=0.7)
            st.success("Generated Linear AR(1) process with φ=0.7")
        elif data_choice == "Nonlinear TAR":
            test_data, _ = generate_tar_model(n=n_obs, threshold=0, ar1_low=0.8, ar1_high=-0.5)
            st.success("Generated Nonlinear TAR process")
        else:
            st.info("Upload your data or use default")
            test_data = generate_linear_ts(n=500, ar_coef=0.7)

        # Perform simplified BDS test
        embedding_dims = [2, 3, 4, 5]
        bds_stats = []

        for m in embedding_dims:
            bds_stat = bds_test_simplified(test_data, m=m, epsilon_std=0.5)
            bds_stats.append(bds_stat)

        # Visualization
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Time Series", "BDS Test Statistics"))

        fig.add_trace(go.Scatter(y=test_data, mode='lines', name='Data',
                                 line=dict(color='#2196F3', width=1.5)), row=1, col=1)

        fig.add_trace(go.Bar(x=[f'm={m}' for m in embedding_dims], y=bds_stats,
                             marker_color='#FF5722', name='BDS Stat'), row=1, col=2)

        # Critical values
        fig.add_hline(y=1.96, line_dash="dash", line_color="red", row=1, col=2,
                      annotation_text="5% critical value")
        fig.add_hline(y=-1.96, line_dash="dash", line_color="red", row=1, col=2)

        fig.update_layout(height=400, showlegend=False)
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Embedding Dimension", row=1, col=2)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="BDS Statistic", row=1, col=2)

        st.plotly_chart(fig, use_container_width=True)

        # Results interpretation
        st.markdown("### Test Results:")
        results_df = pd.DataFrame({
            'Embedding Dimension': embedding_dims,
            'BDS Statistic': [f"{stat:.3f}" for stat in bds_stats],
            'Reject H₀ (5%)': ['Yes' if abs(stat) > 1.96 else 'No' for stat in bds_stats]
        })
        st.dataframe(results_df, use_container_width=True)

        if any(abs(stat) > 1.96 for stat in bds_stats):
            st.error("**Conclusion:** Evidence of nonlinearity detected! Reject null hypothesis of i.i.d.")
        else:
            st.success("**Conclusion:** No strong evidence against i.i.d. hypothesis.")

    elif test_type == "Tsay Test":
        st.markdown("""
        ### Tsay Test for Nonlinearity

        Tests the null hypothesis of linearity in AR models against threshold nonlinearity.

        **Procedure:**
        1. Fit linear AR(p) model and obtain residuals
        2. Regress squared residuals on lagged values
        3. Test for significance of coefficients
        """)

        st.latex(r"""
        \hat{\varepsilon}_t^2 = \alpha_0 + \sum_{i=1}^{p} \alpha_i y_{t-i} + \sum_{i=1}^{p} \sum_{j=i}^{p} \beta_{ij} y_{t-i} y_{t-j} + u_t
        """)

        st.markdown("**Test statistic:** F-test for $H_0: \\beta_{ij} = 0$ for all $i,j$")

        # Simulation
        st.markdown("---")
        st.subheader("📊 Tsay Test Demonstration")

        np.random.seed(42)
        linear_series = generate_linear_ts(n=400, ar_coef=0.7)
        nonlinear_series, _ = generate_tar_model(n=400, threshold=0, ar1_low=0.8, ar1_high=-0.5)


        def tsay_test_visualization(y, title):
            # Fit AR(1)
            y_lag = y[:-1]
            y_current = y[1:]

            # Get residuals
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(y_lag.reshape(-1, 1), y_current)
            residuals = y_current - model.predict(y_lag.reshape(-1, 1))

            # Plot
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=(f"{title}", "Residuals vs Lagged Y"))

            fig.add_trace(go.Scatter(y=y, mode='lines', name='Series',
                                     line=dict(width=1.5)), row=1, col=1)

            fig.add_trace(go.Scatter(x=y_lag, y=residuals ** 2, mode='markers',
                                     marker=dict(size=4, opacity=0.6), name='ε²'),
                          row=1, col=2)

            fig.update_layout(height=350, showlegend=False)
            return fig


        st.markdown("**Linear Series:**")
        st.plotly_chart(tsay_test_visualization(linear_series, "Linear AR(1)"),
                        use_container_width=True)

        st.markdown("**Nonlinear Series:**")
        st.plotly_chart(tsay_test_visualization(nonlinear_series, "Nonlinear TAR"),
                        use_container_width=True)

        st.info("""
        **Interpretation:** 
        - If squared residuals show pattern with lagged Y → evidence of nonlinearity
        - Random scatter → consistent with linearity
        """)

    elif test_type == "McLeod-Li Test":
        st.markdown("""
        ### McLeod-Li Test for ARCH Effects

        Tests for conditional heteroskedasticity (ARCH/GARCH effects).

        **Procedure:** Apply Ljung-Box test to squared residuals
        """)

        st.latex(r"""
        Q(m) = n(n+2) \sum_{k=1}^{m} \frac{\hat{\rho}_k^2(\varepsilon^2)}{n-k} \sim \chi^2_m
        """)

        st.markdown("where $\\hat{\\rho}_k(\\varepsilon^2)$ is the ACF of squared residuals.")

        # Simulation
        st.markdown("---")
        st.subheader("📊 McLeod-Li Test Demonstration")

        # Generate ARCH process
        np.random.seed(42)
        n = 500
        arch_series = np.zeros(n)
        h = np.ones(n)  # conditional variance
        alpha0, alpha1 = 0.1, 0.7

        for t in range(1, n):
            h[t] = alpha0 + alpha1 * arch_series[t - 1] ** 2
            arch_series[t] = np.sqrt(h[t]) * np.random.normal()

        # Compare with constant variance
        const_var_series = np.random.normal(0, 1, n)


        def plot_mcleod_li(y, title):
            # Calculate ACF of squared series
            y_squared = y ** 2
            acf_vals = calculate_acf(y_squared, nlags=20)

            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=(title, "ACF of Squared Series"))

            fig.add_trace(go.Scatter(y=y, mode='lines', line=dict(width=1)),
                          row=1, col=1)
            fig.add_trace(go.Bar(y=acf_vals, marker_color='#FF5722'),
                          row=1, col=2)

            # Confidence bands
            ci = 1.96 / np.sqrt(len(y))
            fig.add_hline(y=ci, line_dash="dash", line_color="gray", row=1, col=2)
            fig.add_hline(y=-ci, line_dash="dash", line_color="gray", row=1, col=2)

            fig.update_layout(height=350, showlegend=False)
            return fig


        st.markdown("**Series with Constant Variance:**")
        st.plotly_chart(plot_mcleod_li(const_var_series, "Constant Variance"),
                        use_container_width=True)

        st.markdown("**Series with ARCH Effects:**")
        st.plotly_chart(plot_mcleod_li(arch_series, "ARCH(1) Process"),
                        use_container_width=True)

        st.success("""
        **Interpretation:** Significant ACF in squared series indicates time-varying volatility (ARCH effects)
        """)

    else:  # RESET Test
        st.markdown("""
        ### Ramsey RESET Test

        **R**egression **E**quation **S**pecification **E**rror **T**est

        Tests whether non-linear combinations of fitted values help explain the response variable.

        **Procedure:**
        1. Estimate original model: $y_t = \\beta' x_t + \\varepsilon_t$
        2. Obtain fitted values: $\\hat{y}_t$
        3. Estimate augmented model: $y_t = \\beta' x_t + \\gamma_1 \\hat{y}_t^2 + \\gamma_2 \\hat{y}_t^3 + u_t$
        4. Test $H_0: \\gamma_1 = \\gamma_2 = 0$
        """)

        st.latex(r"""
        F = \frac{(RSS_r - RSS_u)/q}{RSS_u/(n-k)} \sim F_{q, n-k}
        """)

        st.markdown("""
        where:
        - $RSS_r$ = residual sum of squares (restricted model)
        - $RSS_u$ = residual sum of squares (unrestricted model)
        - $q$ = number of restrictions
        - $n$ = sample size, $k$ = number of parameters
        """)

        st.info("""
        **Interpretation:**
        - Reject $H_0$ → Model misspecification, likely nonlinearity
        - Fail to reject → Linear specification adequate
        """)

# Section 5: Threshold Models
elif section == "5. Threshold Models (TAR)":
    st.header("5. Threshold Autoregressive (TAR) Models")

    st.markdown("""
    ### 5.1 Introduction to TAR Models

    **Threshold Autoregressive models** allow dynamics to change when a threshold variable 
    crosses a specified value.

    **General TAR Model:**
    """)

    st.latex(r"""
    y_t = \begin{cases}
    \phi_0^{(1)} + \sum_{i=1}^{p} \phi_i^{(1)} y_{t-i} + \varepsilon_t^{(1)} & \text{if } y_{t-d} \leq c \\
    \phi_0^{(2)} + \sum_{i=1}^{p} \phi_i^{(2)} y_{t-i} + \varepsilon_t^{(2)} & \text{if } y_{t-d} > c
    \end{cases}
    """)

    st.markdown("""
    **Key Parameters:**
    - $c$: threshold value
    - $d$: delay parameter
    - $p$: autoregressive order
    - $\phi^{(j)}$: regime-specific AR coefficients
    """)

    st.markdown("---")

    # Interactive TAR Model
    st.subheader("🎮 Interactive TAR Model Builder")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Lower Regime Parameters:**")
        phi_low = st.slider("φ₁ (Low)", -0.95, 0.95, 0.7, 0.05, key='phi_low')
        const_low = st.slider("Constant (Low)", -1.0, 1.0, 0.0, 0.1, key='const_low')

    with col2:
        st.markdown("**Upper Regime Parameters:**")
        phi_high = st.slider("φ₂ (High)", -0.95, 0.95, -0.5, 0.05, key='phi_high')
        const_high = st.slider("Constant (High)", -1.0, 1.0, 0.0, 0.1, key='const_high')

    with col3:
        st.markdown("**Model Specifications:**")
        threshold = st.slider("Threshold (c)", -2.0, 2.0, 0.0, 0.25, key='threshold')
        delay = st.slider("Delay (d)", 1, 5, 1, 1, key='delay')

    # Generate TAR series
    np.random.seed(42)
    n = 600
    y_tar = np.zeros(n)
    epsilon = np.random.normal(0, 1, n)
    regime_indicator = np.zeros(n, dtype=int)

    y_tar[0] = epsilon[0]
    for t in range(delay, n):
        if y_tar[t - delay] <= threshold:
            y_tar[t] = const_low + phi_low * y_tar[t - 1] + epsilon[t]
            regime_indicator[t] = 0
        else:
            y_tar[t] = const_high + phi_high * y_tar[t - 1] + epsilon[t]
            regime_indicator[t] = 1

    # Comprehensive visualization
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "TAR Time Series", "Regime Distribution",
            "Regime-Colored Series", "Transition Dynamics",
            "Histogram by Regime", "Phase Diagram"
        ),
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "histogram"}, {"type": "scatter"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )

    # 1. Time series
    colors = ['#4CAF50' if r == 0 else '#FF5722' for r in regime_indicator]
    fig.add_trace(go.Scatter(y=y_tar, mode='lines', line=dict(color='#2196F3', width=1),
                             name='TAR Series'), row=1, col=1)

    # 2. Regime distribution
    regime_counts = pd.Series(regime_indicator).value_counts().sort_index()
    fig.add_trace(go.Bar(x=['Low Regime', 'High Regime'], y=regime_counts.values,
                         marker_color=['#4CAF50', '#FF5722'], name='Count'),
                  row=1, col=2)

    # 3. Regime-colored series
    fig.add_trace(go.Scatter(y=y_tar, mode='markers',
                             marker=dict(color=colors, size=3),
                             name='By Regime'), row=2, col=1)
    fig.add_hline(y=threshold, line_dash="dash", line_color="black", row=2, col=1)

    # 4. Transition dynamics
    transitions = np.diff(regime_indicator)
    transition_points = np.where(transitions != 0)[0]
    fig.add_trace(go.Scatter(y=y_tar, mode='lines', line=dict(color='lightgray', width=1),
                             name='Series'), row=2, col=2)
    fig.add_trace(go.Scatter(x=transition_points, y=y_tar[transition_points],
                             mode='markers', marker=dict(color='red', size=8, symbol='star'),
                             name='Transitions'), row=2, col=2)

    # 5. Histogram by regime
    y_low = y_tar[regime_indicator == 0]
    y_high = y_tar[regime_indicator == 1]
    fig.add_trace(go.Histogram(x=y_low, marker_color='#4CAF50', opacity=0.7,
                               name='Low Regime', nbinsx=30), row=3, col=1)
    fig.add_trace(go.Histogram(x=y_high, marker_color='#FF5722', opacity=0.7,
                               name='High Regime', nbinsx=30), row=3, col=1)

    # 6. Phase diagram
    fig.add_trace(go.Scatter(x=y_tar[:-1], y=y_tar[1:], mode='markers',
                             marker=dict(color=colors[:-1], size=4),
                             name='Phase'), row=3, col=2)

    fig.update_layout(height=1000, showlegend=False, title_text="TAR Model Analysis")

    # Update axes
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=2)
    fig.update_xaxes(title_text="Value", row=3, col=1)
    fig.update_xaxes(title_text="y(t)", row=3, col=2)

    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=2)
    fig.update_yaxes(title_text="Frequency", row=3, col=1)
    fig.update_yaxes(title_text="y(t+1)", row=3, col=2)

    st.plotly_chart(fig, use_container_width=True)

    # Statistical summary
    st.markdown("---")
    st.subheader("📊 Statistical Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Observations", n)
        st.metric("Number of Transitions", len(transition_points))

    with col2:
        st.metric("Low Regime %", f"{100 * np.mean(regime_indicator == 0):.1f}%")
        st.metric("Mean (Low)", f"{np.mean(y_low):.3f}")

    with col3:
        st.metric("High Regime %", f"{100 * np.mean(regime_indicator == 1):.1f}%")
        st.metric("Mean (High)", f"{np.mean(y_high):.3f}")

    st.markdown("""
    <div class="highlight">
    <b>Key Properties of TAR Models:</b>
    <ul>
    <li><b>Regime Persistence:</b> Series tends to stay in one regime for extended periods</li>
    <li><b>Asymmetric Dynamics:</b> Different persistence in different regimes</li>
    <li><b>Threshold Identification:</b> Can be estimated via grid search or information criteria</li>
    <li><b>Applications:</b> Business cycles, interest rates, exchange rates, unemployment</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # Estimation procedure
    st.markdown("---")
    st.subheader("📐 Estimation Procedure")

    st.markdown("""
    **Step 1:** Specify delay parameter $d$ and AR order $p$

    **Step 2:** Grid search over potential threshold values:
    """)

    st.latex(r"""
    \hat{c} = \arg\min_{c} \sum_{t=1}^{T} \hat{\varepsilon}_t^2(c)
    """)

    st.markdown("""
    **Step 3:** For each $c$, estimate regime-specific parameters via OLS

    **Step 4:** Select $c$ that minimizes AIC or BIC:
    """)

    st.latex(r"""
    AIC = \ln(\hat{\sigma}^2) + \frac{2k}{T}, \quad BIC = \ln(\hat{\sigma}^2) + \frac{k \ln(T)}{T}
    """)

# Section 6: Markov Switching Models
elif section == "6. Markov Switching Models":
    st.header("6. Markov Switching Models")

    st.markdown("""
    ### 6.1 Introduction to Markov Switching

    **Markov Switching Models** (Hamilton, 1989) assume that model parameters are governed 
    by an unobserved state variable that follows a Markov chain.

    **Basic MS-AR Model:**
    """)

    st.latex(r"""
    y_t = \mu_{S_t} + \sum_{i=1}^{p} \phi_i(y_{t-i} - \mu_{S_{t-i}}) + \sigma_{S_t} \varepsilon_t
    """)

    st.markdown("""
    where $S_t \in \{1, 2, ..., M\}$ is the unobserved state at time $t$.

    **Markov Chain Transition Probabilities:**
    """)

    st.latex(r"""
    P(S_t = j | S_{t-1} = i) = p_{ij}, \quad \sum_{j=1}^{M} p_{ij} = 1
    """)

    st.markdown("**Transition Matrix (2-state case):**")

    st.latex(r"""
    P = \begin{pmatrix}
    p_{11} & 1-p_{11} \\
    1-p_{22} & p_{22}
    \end{pmatrix}
    """)

    st.markdown("---")

    # Interactive Markov Switching Model
    st.subheader("🎮 Interactive Markov Switching Model")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**State 1 (Expansion) Parameters:**")
        mu1 = st.slider("Mean (μ₁)", -1.0, 3.0, 1.0, 0.1, key='mu1')
        phi1 = st.slider("AR coef (φ₁)", -0.95, 0.95, 0.7, 0.05, key='phi1')
        sigma1 = st.slider("Volatility (σ₁)", 0.1, 2.0, 0.5, 0.1, key='sigma1')

    with col2:
        st.markdown("**State 2 (Recession) Parameters:**")
        mu2 = st.slider("Mean (μ₂)", -3.0, 1.0, -0.5, 0.1, key='mu2')
        phi2 = st.slider("AR coef (φ₂)", -0.95, 0.95, 0.3, 0.05, key='phi2')
        sigma2 = st.slider("Volatility (σ₂)", 0.1, 2.0, 1.5, 0.1, key='sigma2')

    st.markdown("**Transition Probabilities:**")
    col1, col2 = st.columns(2)
    with col1:
        p11 = st.slider("P(State 1 → State 1)", 0.5, 0.99, 0.95, 0.01, key='p11')
    with col2:
        p22 = st.slider("P(State 2 → State 2)", 0.5, 0.99, 0.90, 0.01, key='p22')

    # Generate MS data
    np.random.seed(42)
    n = 800
    y_ms = np.zeros(n)
    states = np.zeros(n, dtype=int)
    epsilon = np.random.normal(0, 1, n)

    states[0] = 0
    y_ms[0] = mu1 + epsilon[0] * sigma1

    for t in range(1, n):
        # State transition
        if states[t - 1] == 0:
            states[t] = 0 if np.random.rand() < p11 else 1
        else:
            states[t] = 1 if np.random.rand() < p22 else 0

        # Generate observation
        if states[t] == 0:
            y_ms[t] = mu1 + phi1 * (y_ms[t - 1] - mu1) + sigma1 * epsilon[t]
        else:
            y_ms[t] = mu2 + phi2 * (y_ms[t - 1] - mu2) + sigma2 * epsilon[t]

    # Comprehensive visualization
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Time Series with Regimes", "Hidden State Process",
            "State-Specific Distributions", "Regime Probabilities (Smoothed)",
            "Transition Matrix Heatmap", "Duration Analysis"
        ),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "histogram"}, {"type": "scatter"}],
               [{"type": "heatmap"}, {"type": "bar"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )

    # 1. Time series with regime coloring
    colors = ['#2196F3' if s == 0 else '#FF5722' for s in states]
    fig.add_trace(go.Scatter(y=y_ms, mode='markers',
                             marker=dict(color=colors, size=3),
                             name='MS Series'), row=1, col=1)

    # 2. Hidden state
    fig.add_trace(go.Scatter(y=states, mode='lines', line=dict(color='black', width=2),
                             fill='tozeroy', fillcolor='rgba(0,0,0,0.1)',
                             name='State'), row=1, col=2)

    # 3. State-specific distributions
    y_state0 = y_ms[states == 0]
    y_state1 = y_ms[states == 1]
    fig.add_trace(go.Histogram(x=y_state0, marker_color='#2196F3', opacity=0.7,
                               name='State 1', nbinsx=40), row=2, col=1)
    fig.add_trace(go.Histogram(x=y_state1, marker_color='#FF5722', opacity=0.7,
                               name='State 2', nbinsx=40), row=2, col=1)

    # 4. Smoothed probabilities (simplified)
    window = 20
    prob_state0 = pd.Series(states == 0).rolling(window).mean()
    fig.add_trace(go.Scatter(y=prob_state0, mode='lines',
                             line=dict(color='#2196F3', width=2),
                             name='P(State 1)'), row=2, col=2)
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=2, col=2)

    # 5. Transition matrix heatmap
    trans_matrix = np.array([[p11, 1 - p11], [1 - p22, p22]])
    fig.add_trace(go.Heatmap(z=trans_matrix, x=['State 1', 'State 2'],
                             y=['State 1', 'State 2'], colorscale='RdYlBu',
                             text=trans_matrix, texttemplate='%{text:.3f}',
                             showscale=False), row=3, col=1)

    # 6. Duration analysis
    durations_0 = []
    durations_1 = []
    current_duration = 1
    current_state = states[0]

    for t in range(1, n):
        if states[t] == current_state:
            current_duration += 1
        else:
            if current_state == 0:
                durations_0.append(current_duration)
            else:
                durations_1.append(current_duration)
            current_duration = 1
            current_state = states[t]

    fig.add_trace(go.Bar(x=['State 1', 'State 2'],
                         y=[np.mean(durations_0), np.mean(durations_1)],
                         marker_color=['#2196F3', '#FF5722'],
                         name='Avg Duration'), row=3, col=2)

    fig.update_layout(height=1000, showlegend=False,
                      title_text="Markov Switching Model Analysis")

    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    st.markdown("---")
    st.subheader("📊 Model Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("State 1 Frequency", f"{100 * np.mean(states == 0):.1f}%")
        st.metric("Mean (State 1)", f"{np.mean(y_state0):.3f}")

    with col2:
        st.metric("State 2 Frequency", f"{100 * np.mean(states == 1):.1f}%")
        st.metric("Mean (State 2)", f"{np.mean(y_state1):.3f}")

    with col3:
        st.metric("Avg Duration (State 1)", f"{np.mean(durations_0):.1f}")
        st.metric("Std (State 1)", f"{np.std(y_state0):.3f}")

    with col4:
        st.metric("Avg Duration (State 2)", f"{np.mean(durations_1):.1f}")
        st.metric("Std (State 2)", f"{np.std(y_state1):.3f}")

    # Expected duration formula
    st.markdown("---")
    st.subheader("📐 Expected Duration in Each State")

    st.markdown("The expected duration in state $i$ is given by:")

    st.latex(r"""
    E[D_i] = \frac{1}{1 - p_{ii}}
    """)

    expected_duration_1 = 1 / (1 - p11)
    expected_duration_2 = 1 / (1 - p22)

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Expected Duration (State 1):** {expected_duration_1:.2f} periods")
    with col2:
        st.info(f"**Expected Duration (State 2):** {expected_duration_2:.2f} periods")

    st.markdown("""
    <div class="highlight">
    <b>Key Applications:</b>
    <ul>
    <li><b>Business Cycles:</b> Expansion vs. recession regimes</li>
    <li><b>Financial Markets:</b> Bull vs. bear markets</li>
    <li><b>Monetary Policy:</b> Hawkish vs. dovish regimes</li>
    <li><b>Volatility:</b> Low vs. high volatility periods</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # Inference
    st.markdown("---")
    st.subheader("📊 Inference: Hamilton Filter")

    st.markdown("""
    **Filtering:** Compute $P(S_t = j | Y_t)$ where $Y_t = \{y_1, ..., y_t\}$

    **Prediction:**
    """)

    st.latex(r"""
    P(S_t = j | Y_{t-1}) = \sum_{i=1}^{M} P(S_t = j | S_{t-1} = i) P(S_{t-1} = i | Y_{t-1})
    """)

    st.markdown("**Update:**")

    st.latex(r"""
    P(S_t = j | Y_t) = \frac{f(y_t | S_t = j) P(S_t = j | Y_{t-1})}{\sum_{i=1}^{M} f(y_t | S_t = i) P(S_t = i | Y_{t-1})}
    """)

    st.markdown("""
    **Smoothing:** Compute $P(S_t = j | Y_T)$ using all available data (Kim's smoother)
    """)

# Section 7: Smooth Transition Models
elif section == "7. Smooth Transition Models":
    st.header("7. Smooth Transition Autoregressive (STAR) Models")

    st.markdown("""
    ### 7.1 Introduction to STAR Models

    Unlike TAR models with abrupt regime changes, **STAR models** feature smooth transitions 
    between regimes via a transition function.

    **General STAR Model:**
    """)

    st.latex(r"""
    y_t = (\phi_0^{(1)} + \phi_1^{(1)} y_{t-1})(1 - G(s_t; \gamma, c)) + 
          (\phi_0^{(2)} + \phi_1^{(2)} y_{t-1})G(s_t; \gamma, c) + \varepsilon_t
    """)

    st.markdown("""
    where $G(s_t; \\gamma, c)$ is the **transition function**.

    ### 7.2 Types of Transition Functions
    """)

    # Transition function selector
    transition_type = st.selectbox(
        "Select Transition Function:",
        ["Logistic (LSTAR)", "Exponential (ESTAR)", "Comparison"]
    )

    if transition_type == "Logistic (LSTAR)":
        st.markdown("""
        ### Logistic STAR (LSTAR)

        **Transition Function:**
        """)

        st.latex(r"""
        G(s_t; \gamma, c) = \frac{1}{1 + \exp(-\gamma(s_t - c))}, \quad \gamma > 0
        """)

        st.markdown("""
        **Properties:**
        - Bounded: $G \in [0, 1]$
        - Monotonically increasing in $s_t$
        - $\gamma$ controls transition speed
        - $c$ is the threshold/center
        - Asymmetric around $c$
        """)

        # Interactive LSTAR
        st.markdown("---")
        st.subheader("🎮 Interactive LSTAR Model")

        col1, col2 = st.columns(2)

        with col1:
            gamma = st.slider("Transition speed (γ)", 0.1, 10.0, 2.0, 0.1, key='gamma_lstar')
            c = st.slider("Threshold (c)", -2.0, 2.0, 0.0, 0.1, key='c_lstar')

        with col2:
            phi1_low = st.slider("φ₁ (Lower regime)", -0.9, 0.9, 0.7, 0.05, key='phi1_lstar')
            phi1_high = st.slider("φ₁ (Upper regime)", -0.9, 0.9, -0.5, 0.05, key='phi1_lstar2')

        # Generate LSTAR data
        np.random.seed(42)
        n = 600
        y_lstar = np.zeros(n)
        G_values = np.zeros(n)
        epsilon = np.random.normal(0, 1, n)

        y_lstar[0] = epsilon[0]
        for t in range(1, n):
            G = 1 / (1 + np.exp(-gamma * (y_lstar[t - 1] - c)))
            G_values[t] = G
            y_lstar[t] = (phi1_low * (1 - G) + phi1_high * G) * y_lstar[t - 1] + epsilon[t]

        # Visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "LSTAR Time Series", "Transition Function G(s)",
                "Time Series with Transition", "Effective AR Coefficient"
            ),
            vertical_spacing=0.15
        )

        # 1. Time series
        fig.add_trace(go.Scatter(y=y_lstar, mode='lines', line=dict(color='#2196F3', width=1.5),
                                 name='LSTAR'), row=1, col=1)

        # 2. Transition function
        s_range = np.linspace(-3, 3, 200)
        G_range = 1 / (1 + np.exp(-gamma * (s_range - c)))
        fig.add_trace(go.Scatter(x=s_range, y=G_range, mode='lines',
                                 line=dict(color='#FF5722', width=3), name='G(s)'),
                      row=1, col=2)
        fig.add_vline(x=c, line_dash="dash", line_color="black", row=1, col=2,
                      annotation_text=f"c={c}")

        # 3. Series colored by G
        colors_G = G_values
        fig.add_trace(go.Scatter(y=y_lstar, mode='markers',
                                 marker=dict(color=colors_G, colorscale='RdYlGn',
                                             size=4, showscale=True,
                                             colorbar=dict(title="G(s)", x=1.15)),
                                 name='Transition'), row=2, col=1)

        # 4. Effective AR coefficient
        effective_phi = phi1_low * (1 - G_values) + phi1_high * G_values
        fig.add_trace(go.Scatter(y=effective_phi, mode='lines',
                                 line=dict(color='purple', width=2),
                                 name='φ(t)'), row=2, col=2)
        fig.add_hline(y=phi1_low, line_dash="dash", line_color="green", row=2, col=2,
                      annotation_text="φ₁ (low)")
        fig.add_hline(y=phi1_high, line_dash="dash", line_color="red", row=2, col=2,
                      annotation_text="φ₁ (high)")

        fig.update_layout(height=700, showlegend=False)
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="s", row=1, col=2)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=2)

        st.plotly_chart(fig, use_container_width=True)

        st.info("""
        **Interpretation:** 
        - When $s_t \ll c$: $G ≈ 0$, model behaves like regime 1
        - When $s_t \gg c$: $G ≈ 1$, model behaves like regime 2
        - Larger $\gamma$ → faster transition
        """)

    elif transition_type == "Exponential (ESTAR)":
        st.markdown("""
        ### Exponential STAR (ESTAR)

        **Transition Function:**
        """)

        st.latex(r"""
        G(s_t; \gamma, c) = 1 - \exp(-\gamma(s_t - c)^2), \quad \gamma > 0
        """)

        st.markdown("""
        **Properties:**
        - Bounded: $G \in [0, 1]$
        - Symmetric around $c$
        - $G(c) = 0$ (linear regime at center)
        - $G(s) \\to 1$ as $|s - c| \\to \infty$
        - Useful for modeling symmetric deviations
        """)

        # Interactive ESTAR
        st.markdown("---")
        st.subheader("🎮 Interactive ESTAR Model")

        col1, col2 = st.columns(2)

        with col1:
            gamma_estar = st.slider("Transition speed (γ)", 0.1, 5.0, 1.0, 0.1, key='gamma_estar')
            c_estar = st.slider("Center (c)", -2.0, 2.0, 0.0, 0.1, key='c_estar')

        with col2:
            phi_center = st.slider("φ (center regime)", -0.9, 0.9, 0.8, 0.05, key='phi_center')
            phi_outer = st.slider("φ (outer regime)", -0.9, 0.9, -0.3, 0.05, key='phi_outer')

        # Generate ESTAR data
        np.random.seed(42)
        n = 600
        y_estar = np.zeros(n)
        G_estar = np.zeros(n)
        epsilon = np.random.normal(0, 1, n)

        y_estar[0] = epsilon[0]
        for t in range(1, n):
            G = 1 - np.exp(-gamma_estar * (y_estar[t - 1] - c_estar) ** 2)
            G_estar[t] = G
            y_estar[t] = (phi_center * (1 - G) + phi_outer * G) * y_estar[t - 1] + epsilon[t]

        # Visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "ESTAR Time Series", "Transition Function G(s)",
                "Phase Diagram", "Effective AR Coefficient"
            ),
            vertical_spacing=0.15
        )

        # 1. Time series
        fig.add_trace(go.Scatter(y=y_estar, mode='lines', line=dict(color='#2196F3', width=1.5)),
                      row=1, col=1)

        # 2. Transition function
        s_range = np.linspace(-3, 3, 200)
        G_range_estar = 1 - np.exp(-gamma_estar * (s_range - c_estar) ** 2)
        fig.add_trace(go.Scatter(x=s_range, y=G_range_estar, mode='lines',
                                 line=dict(color='#FF5722', width=3)), row=1, col=2)
        fig.add_vline(x=c_estar, line_dash="dash", line_color="black", row=1, col=2)

        # 3. Phase diagram
        fig.add_trace(go.Scatter(x=y_estar[:-1], y=y_estar[1:], mode='markers',
                                 marker=dict(color=G_estar[1:], colorscale='Viridis',
                                             size=4, showscale=True)), row=2, col=1)

        # 4. Effective coefficient
        effective_phi_estar = phi_center * (1 - G_estar) + phi_outer * G_estar
        fig.add_trace(go.Scatter(y=effective_phi_estar, mode='lines',
                                 line=dict(color='purple', width=2)), row=2, col=2)

        fig.update_layout(height=700, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.success("""
        **Key Feature:** ESTAR is symmetric - deviations above and below $c$ are treated similarly.
        Useful for modeling mean reversion with distance-dependent adjustment.
        """)

    else:  # Comparison
        st.markdown("### Comparison: LSTAR vs ESTAR")

        # Side-by-side comparison
        col1, col2 = st.columns(2)

        gamma_comp = st.slider("Transition speed (γ)", 0.5, 5.0, 2.0, 0.5, key='gamma_comp')

        s_vals = np.linspace(-3, 3, 300)
        G_lstar = 1 / (1 + np.exp(-gamma_comp * s_vals))
        G_estar = 1 - np.exp(-gamma_comp * s_vals ** 2)

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("LSTAR: G(s) = 1/(1+exp(-γs))",
                                            "ESTAR: G(s) = 1-exp(-γs²)"))

        fig.add_trace(go.Scatter(x=s_vals, y=G_lstar, mode='lines',
                                 line=dict(color='#2196F3', width=3), name='LSTAR'),
                      row=1, col=1)

        fig.add_trace(go.Scatter(x=s_vals, y=G_estar, mode='lines',
                                 line=dict(color='#FF5722', width=3), name='ESTAR'),
                      row=1, col=2)

        fig.update_layout(height=400, showlegend=False)
        fig.update_xaxes(title_text="s", row=1, col=1)
        fig.update_xaxes(title_text="s", row=1, col=2)
        fig.update_yaxes(title_text="G(s)", row=1, col=1)
        fig.update_yaxes(title_text="G(s)", row=1, col=2)

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        | Feature | LSTAR | ESTAR |
        |---------|-------|-------|
        | **Symmetry** | Asymmetric | Symmetric |
        | **Shape** | Monotonic S-curve | Bell-shaped derivative |
        | **At center** | G(0) = 0.5 | G(0) = 0 |
        | **Applications** | Directional effects | Mean reversion |
        | **Example** | Interest rate adjustments | Exchange rate bands |
        """)

# Section 8: Simulations & Applications
elif section == "8. Simulations & Applications":
    st.header("8. Simulations & Applications")

    st.markdown("""
    ### Understanding Nonlinear Models Through Simulation

    Simulations help us:
    - Understand model behavior under different parameter configurations
    - Assess forecasting performance
    - Evaluate estimation accuracy
    - Compare different model specifications
    """)

    app_type = st.selectbox(
        "Select Application:",
        ["Business Cycle Modeling", "Exchange Rate Dynamics", "Volatility Clustering",
         "Interest Rate Modeling", "Monte Carlo Comparison"]
    )

    if app_type == "Business Cycle Modeling":
        st.markdown("""
        ### Business Cycle Modeling with Markov Switching

        Economic output often exhibits different dynamics during **expansions** and **recessions**.
        """)

        st.markdown("**Model Specification:**")
        st.latex(r"""
        \Delta y_t = \mu_{S_t} + \phi_{S_t}(\Delta y_{t-1} - \mu_{S_{t-1}}) + \sigma_{S_t} \varepsilon_t
        """)

        # Parameters
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Expansion (State 1):**")
            mu_exp = st.slider("Growth rate", 0.0, 1.0, 0.5, 0.05, key='mu_exp')
            phi_exp = st.slider("Persistence", 0.0, 0.95, 0.6, 0.05, key='phi_exp')
            sigma_exp = st.slider("Volatility", 0.1, 1.0, 0.3, 0.05, key='sigma_exp')

        with col2:
            st.markdown("**Recession (State 2):**")
            mu_rec = st.slider("Growth rate", -1.0, 0.0, -0.3, 0.05, key='mu_rec')
            phi_rec = st.slider("Persistence", 0.0, 0.95, 0.3, 0.05, key='phi_rec')
            sigma_rec = st.slider("Volatility", 0.1, 2.0, 0.8, 0.1, key='sigma_rec')

        p11_bc = st.slider("P(Expansion → Expansion)", 0.8, 0.99, 0.95, 0.01, key='p11_bc')
        p22_bc = st.slider("P(Recession → Recession)", 0.7, 0.99, 0.85, 0.01, key='p22_bc')

        # Generate business cycle data
        np.random.seed(42)
        n = 500
        gdp_growth = np.zeros(n)
        bc_states = np.zeros(n, dtype=int)
        epsilon = np.random.normal(0, 1, n)

        bc_states[0] = 0  # Start in expansion
        gdp_growth[0] = mu_exp + sigma_exp * epsilon[0]

        for t in range(1, n):
            # State transition
            if bc_states[t - 1] == 0:
                bc_states[t] = 0 if np.random.rand() < p11_bc else 1
            else:
                bc_states[t] = 1 if np.random.rand() < p22_bc else 0

            # GDP growth
            if bc_states[t] == 0:
                gdp_growth[t] = mu_exp + phi_exp * (gdp_growth[t - 1] - mu_exp) + sigma_exp * epsilon[t]
            else:
                gdp_growth[t] = mu_rec + phi_rec * (gdp_growth[t - 1] - mu_rec) + sigma_rec * epsilon[t]

        # GDP level
        gdp_level = 100 * np.exp(np.cumsum(gdp_growth) / 100)

        # Visualization
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("GDP Level", "GDP Growth Rate", "Business Cycle States"),
            vertical_spacing=0.1,
            row_heights=[0.4, 0.3, 0.3]
        )

        # GDP level with shaded recessions
        colors_bc = ['rgba(76,175,80,0.3)' if s == 0 else 'rgba(244,67,54,0.3)' for s in bc_states]

        fig.add_trace(go.Scatter(y=gdp_level, mode='lines', line=dict(color='#1976D2', width=2),
                                 name='GDP Level', fill='tonexty'), row=1, col=1)

        # Add recession bars
        in_recession = False
        recession_start = 0
        for t in range(n):
            if bc_states[t] == 1 and not in_recession:
                recession_start = t
                in_recession = True
            elif bc_states[t] == 0 and in_recession:
                fig.add_vrect(x0=recession_start, x1=t, fillcolor="red", opacity=0.2,
                              layer="below", line_width=0, row=1, col=1)
                in_recession = False

        # GDP growth
        colors_growth = ['#4CAF50' if s == 0 else '#F44336' for s in bc_states]
        fig.add_trace(go.Scatter(y=gdp_growth, mode='markers',
                                 marker=dict(color=colors_growth, size=4),
                                 name='Growth'), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)

        # States
        fig.add_trace(go.Scatter(y=bc_states, mode='lines', line=dict(color='black', width=2),
                                 fill='tozeroy', name='State'), row=3, col=1)

        fig.update_layout(height=800, showlegend=False,
                          title_text="Business Cycle Simulation")
        fig.update_yaxes(title_text="GDP (Index)", row=1, col=1)
        fig.update_yaxes(title_text="Growth (%)", row=2, col=1)
        fig.update_yaxes(title_text="State", row=3, col=1)
        fig.update_xaxes(title_text="Quarter", row=3, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # Statistics
        col1, col2, col3 = st.columns(3)

        expansion_periods = np.sum(bc_states == 0)
        recession_periods = np.sum(bc_states == 1)

        with col1:
            st.metric("Expansion Quarters", expansion_periods)
            st.metric("Avg Growth (Expansion)", f"{np.mean(gdp_growth[bc_states == 0]):.2f}%")

        with col2:
            st.metric("Recession Quarters", recession_periods)
            st.metric("Avg Growth (Recession)", f"{np.mean(gdp_growth[bc_states == 1]):.2f}%")

        with col3:
            st.metric("Expansion/Recession Ratio", f"{expansion_periods / recession_periods:.2f}")
            st.metric("Volatility Ratio",
                      f"{np.std(gdp_growth[bc_states == 1]) / np.std(gdp_growth[bc_states == 0]):.2f}")

    elif app_type == "Exchange Rate Dynamics":
        st.markdown("""
        ### Exchange Rate Modeling with TAR

        Exchange rates may exhibit different dynamics depending on deviation from equilibrium,
        due to transaction costs and central bank intervention.
        """)

        st.markdown("**TAR Model for Exchange Rate:**")
        st.latex(r"""
        \Delta e_t = \begin{cases}
        \phi_1 (e_{t-1} - e^*) + \varepsilon_t & \text{if } |e_{t-1} - e^*| \leq \text{band} \\
        \phi_2 (e_{t-1} - e^*) + \varepsilon_t & \text{if } |e_{t-1} - e^*| > \text{band}
        \end{cases}
        """)

        # Parameters
        equilibrium = st.slider("Equilibrium rate (e*)", 90.0, 110.0, 100.0, 1.0, key='eq_rate')
        band_width = st.slider("Intervention band", 0.0, 10.0, 5.0, 0.5, key='band_width')
        phi_inside = st.slider("φ (inside band)", -0.2, 0.2, 0.05, 0.05, key='phi_inside')
        phi_outside = st.slider("φ (outside band)", -0.9, -0.1, -0.4, 0.05, key='phi_outside')

        # Generate exchange rate
        np.random.seed(42)
        n = 800
        exchange_rate = np.zeros(n)
        regime_fx = np.zeros(n, dtype=int)
        epsilon = np.random.normal(0, 0.5, n)

        exchange_rate[0] = equilibrium + np.random.normal(0, 2)

        for t in range(1, n):
            deviation = exchange_rate[t - 1] - equilibrium

            if abs(deviation) <= band_width:
                # Inside band - weak mean reversion
                exchange_rate[t] = exchange_rate[t - 1] + phi_inside * deviation + epsilon[t]
                regime_fx[t] = 0
            else:
                # Outside band - strong mean reversion
                exchange_rate[t] = exchange_rate[t - 1] + phi_outside * deviation + epsilon[t]
                regime_fx[t] = 1

        # Visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Exchange Rate with Intervention Bands",
                "Deviations from Equilibrium",
                "Speed of Adjustment",
                "Histogram of Exchange Rate"
            ),
            vertical_spacing=0.15,
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "histogram"}]]
        )

        # 1. Exchange rate with bands
        colors_fx = ['#4CAF50' if r == 0 else '#FF5722' for r in regime_fx]
        fig.add_trace(go.Scatter(y=exchange_rate, mode='markers',
                                 marker=dict(color=colors_fx, size=3),
                                 name='Rate'), row=1, col=1)
        fig.add_hline(y=equilibrium, line_dash="solid", line_color="blue",
                      annotation_text="Equilibrium", row=1, col=1)
        fig.add_hline(y=equilibrium + band_width, line_dash="dash", line_color="red",
                      annotation_text="+Band", row=1, col=1)
        fig.add_hline(y=equilibrium - band_width, line_dash="dash", line_color="red",
                      annotation_text="-Band", row=1, col=1)

        # 2. Deviations
        deviations = exchange_rate - equilibrium
        fig.add_trace(go.Scatter(y=deviations, mode='lines', line=dict(color='purple', width=1.5),
                                 name='Deviation'), row=1, col=2)
        fig.add_hline(y=band_width, line_dash="dash", line_color="red", row=1, col=2)
        fig.add_hline(y=-band_width, line_dash="dash", line_color="red", row=1, col=2)
        fig.add_hline(y=0, line_dash="solid", line_color="black", row=1, col=2)

        # 3. Adjustment speed
        adjustment_speed = np.where(regime_fx == 0, phi_inside, phi_outside)
        fig.add_trace(go.Scatter(y=adjustment_speed, mode='lines',
                                 line=dict(color='orange', width=2),
                                 name='φ(t)'), row=2, col=1)

        # 4. Histogram
        fig.add_trace(go.Histogram(x=exchange_rate, marker_color='#2196F3',
                                   nbinsx=40, name='Distribution'), row=2, col=2)
        fig.add_vline(x=equilibrium, line_dash="solid", line_color="red",
                      annotation_text="e*", row=2, col=2)

        fig.update_layout(height=700, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Mean Rate", f"{np.mean(exchange_rate):.2f}")
            st.metric("Std Dev", f"{np.std(exchange_rate):.2f}")

        with col2:
            st.metric("% Inside Band", f"{100 * np.mean(regime_fx == 0):.1f}%")
            st.metric("% Outside Band", f"{100 * np.mean(regime_fx == 1):.1f}%")

        with col3:
            st.metric("Max Deviation", f"{np.max(np.abs(deviations)):.2f}")
            st.metric("Avg |Deviation|", f"{np.mean(np.abs(deviations)):.2f}")

    elif app_type == "Volatility Clustering":
        st.markdown("""
        ### Volatility Clustering with MS-GARCH

        Financial returns exhibit **volatility clustering**: periods of high volatility 
        followed by high volatility, and vice versa.
        """)

        st.markdown("**Markov Switching GARCH Model:**")
        st.latex(r"""
        r_t = \mu_{S_t} + \varepsilon_t, \quad \varepsilon_t = \sigma_t z_t, \quad z_t \sim N(0,1)
        """)
        st.latex(r"""
        \sigma_t^2 = \omega_{S_t} + \alpha_{S_t} \varepsilon_{t-1}^2 + \beta_{S_t} \sigma_{t-1}^2
        """)

        # Parameters
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Low Volatility Regime:**")
            omega_low = st.slider("ω (low)", 0.01, 0.5, 0.05, 0.01, key='omega_low')
            alpha_low = st.slider("α (low)", 0.01, 0.3, 0.1, 0.01, key='alpha_low')
            beta_low = st.slider("β (low)", 0.5, 0.95, 0.8, 0.05, key='beta_low')

        with col2:
            st.markdown("**High Volatility Regime:**")
            omega_high = st.slider("ω (high)", 0.1, 2.0, 0.5, 0.1, key='omega_high')
            alpha_high = st.slider("α (high)", 0.05, 0.5, 0.2, 0.05, key='alpha_high')
            beta_high = st.slider("β (high)", 0.3, 0.9, 0.7, 0.05, key='beta_high')

        p11_vol = st.slider("P(Low → Low)", 0.85, 0.99, 0.95, 0.01, key='p11_vol')
        p22_vol = st.slider("P(High → High)", 0.85, 0.99, 0.92, 0.01, key='p22_vol')

        # Generate MS-GARCH data
        np.random.seed(42)
        n = 1000
        returns = np.zeros(n)
        volatility = np.zeros(n)
        vol_states = np.zeros(n, dtype=int)
        z = np.random.normal(0, 1, n)

        vol_states[0] = 0
        volatility[0] = np.sqrt(omega_low / (1 - alpha_low - beta_low))
        returns[0] = volatility[0] * z[0]

        for t in range(1, n):
            # State transition
            if vol_states[t - 1] == 0:
                vol_states[t] = 0 if np.random.rand() < p11_vol else 1
            else:
                vol_states[t] = 1 if np.random.rand() < p22_vol else 0

            # Volatility
            if vol_states[t] == 0:
                volatility[t] = np.sqrt(omega_low + alpha_low * returns[t - 1] ** 2 +
                                        beta_low * volatility[t - 1] ** 2)
            else:
                volatility[t] = np.sqrt(omega_high + alpha_high * returns[t - 1] ** 2 +
                                        beta_high * volatility[t - 1] ** 2)

            # Returns
            returns[t] = volatility[t] * z[t]

        # Visualization
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Returns", "Conditional Volatility",
                "Volatility States", "|Returns| vs Volatility",
                "Returns Distribution by Regime", "ACF of Squared Returns"
            ),
            vertical_spacing=0.12,
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )

        # 1. Returns
        colors_vol = ['#2196F3' if s == 0 else '#FF5722' for s in vol_states]
        fig.add_trace(go.Scatter(y=returns, mode='lines', line=dict(color='black', width=0.5),
                                 name='Returns'), row=1, col=1)

        # 2. Volatility
        fig.add_trace(go.Scatter(y=volatility, mode='lines',
                                 line=dict(color=colors_vol[0], width=2),
                                 name='σ(t)'), row=1, col=2)
        for t in range(1, n):
            if vol_states[t] != vol_states[t - 1]:
                fig.add_vline(x=t, line_dash="dot", line_color="gray",
                              opacity=0.3, row=1, col=2)

        # 3. States
        fig.add_trace(go.Scatter(y=vol_states, mode='lines',
                                 line=dict(color='black', width=2),
                                 fill='tozeroy'), row=2, col=1)

        # 4. Returns vs Volatility
        fig.add_trace(go.Scatter(x=volatility, y=np.abs(returns), mode='markers',
                                 marker=dict(color=colors_vol, size=4, opacity=0.5),
                                 name='|r| vs σ'), row=2, col=2)

        # 5. Distribution by regime
        returns_low = returns[vol_states == 0]
        returns_high = returns[vol_states == 1]
        fig.add_trace(go.Histogram(x=returns_low, marker_color='#2196F3',
                                   opacity=0.7, name='Low Vol', nbinsx=50), row=3, col=1)
        fig.add_trace(go.Histogram(x=returns_high, marker_color='#FF5722',
                                   opacity=0.7, name='High Vol', nbinsx=50), row=3, col=1)

        # 6. ACF of squared returns
        acf_sq = calculate_acf(returns ** 2, nlags=30)
        fig.add_trace(go.Bar(y=acf_sq, marker_color='purple', name='ACF(r²)'),
                      row=3, col=2)
        ci = 1.96 / np.sqrt(n)
        fig.add_hline(y=ci, line_dash="dash", line_color="gray", row=3, col=2)
        fig.add_hline(y=-ci, line_dash="dash", line_color="gray", row=3, col=2)

        fig.update_layout(height=1000, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Mean Return", f"{np.mean(returns):.4f}")
            st.metric("Std Dev", f"{np.std(returns):.4f}")

        with col2:
            st.metric("Skewness", f"{stats.skew(returns):.3f}")
            st.metric("Kurtosis", f"{stats.kurtosis(returns):.3f}")

        with col3:
            st.metric("Mean σ (Low)", f"{np.mean(volatility[vol_states == 0]):.4f}")
            st.metric("Mean σ (High)", f"{np.mean(volatility[vol_states == 1]):.4f}")

        with col4:
            st.metric("% Low Vol", f"{100 * np.mean(vol_states == 0):.1f}%")
            st.metric("% High Vol", f"{100 * np.mean(vol_states == 1):.1f}%")

    elif app_type == "Interest Rate Modeling":
        st.markdown("""
        ### Interest Rate Modeling with STAR

        Interest rates may show smooth adjustment toward a target, with adjustment speed 
        depending on distance from target.
        """)

        st.latex(r"""
        \Delta r_t = (\alpha_1 + \alpha_2 G(r_{t-1}; \gamma, c))(r^* - r_{t-1}) + \varepsilon_t
        """)

        st.latex(r"""
        G(r; \gamma, c) = \frac{1}{1 + \exp(-\gamma(r - c))}
        """)

        # Parameters
        col1, col2 = st.columns(2)

        with col1:
            r_star = st.slider("Target rate (r*)", 1.0, 5.0, 3.0, 0.25, key='r_star')
            alpha1 = st.slider("α₁ (slow adjustment)", 0.01, 0.3, 0.1, 0.01, key='alpha1_ir')
            alpha2 = st.slider("α₂ (fast adjustment)", 0.1, 0.8, 0.4, 0.05, key='alpha2_ir')

        with col2:
            gamma_ir = st.slider("γ (transition speed)", 0.5, 10.0, 3.0, 0.5, key='gamma_ir')
            c_ir = st.slider("c (threshold)", 0.0, 5.0, 3.0, 0.25, key='c_ir')
            sigma_ir = st.slider("σ (shock std)", 0.05, 0.5, 0.1, 0.05, key='sigma_ir')

        # Generate interest rate series
        np.random.seed(42)
        n = 600
        interest_rate = np.zeros(n)
        G_ir = np.zeros(n)
        adjustment_speed_ir = np.zeros(n)
        epsilon = np.random.normal(0, sigma_ir, n)

        interest_rate[0] = r_star + np.random.normal(0, 1)

        for t in range(1, n):
            # Transition function
            G = 1 / (1 + np.exp(-gamma_ir * (interest_rate[t - 1] - c_ir)))
            G_ir[t] = G

            # Adjustment speed
            alpha_t = alpha1 + alpha2 * G
            adjustment_speed_ir[t] = alpha_t

            # Interest rate change
            interest_rate[t] = interest_rate[t - 1] + alpha_t * (r_star - interest_rate[t - 1]) + epsilon[t]

        # Visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Interest Rate Path", "Adjustment Speed",
                "Transition Function G(r)", "Phase Diagram"
            ),
            vertical_spacing=0.15
        )

        # 1. Interest rate
        fig.add_trace(go.Scatter(y=interest_rate, mode='lines',
                                 line=dict(color='#1976D2', width=2),
                                 name='Rate'), row=1, col=1)
        fig.add_hline(y=r_star, line_dash="dash", line_color="red",
                      annotation_text="Target", row=1, col=1)

        # 2. Adjustment speed
        fig.add_trace(go.Scatter(y=adjustment_speed_ir, mode='lines',
                                 line=dict(color='#FF9800', width=2),
                                 name='α(t)'), row=1, col=2)

        # 3. Transition function
        r_range = np.linspace(0, 6, 200)
        G_range = 1 / (1 + np.exp(-gamma_ir * (r_range - c_ir)))
        fig.add_trace(go.Scatter(x=r_range, y=G_range, mode='lines',
                                 line=dict(color='#4CAF50', width=3),
                                 name='G(r)'), row=2, col=1)
        fig.add_vline(x=c_ir, line_dash="dash", line_color="black", row=2, col=1)

        # 4. Phase diagram
        fig.add_trace(go.Scatter(x=interest_rate[:-1], y=interest_rate[1:],
                                 mode='markers',
                                 marker=dict(color=G_ir[1:], colorscale='Viridis',
                                             size=4, showscale=True,
                                             colorbar=dict(title="G(r)")),
                                 name='Phase'), row=2, col=2)
        # 45-degree line
        fig.add_trace(go.Scatter(x=[0, 6], y=[0, 6], mode='lines',
                                 line=dict(dash='dash', color='gray'),
                                 name='45°'), row=2, col=2)

        fig.update_layout(height=700, showlegend=False)
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_xaxes(title_text="r", row=2, col=1)
        fig.update_xaxes(title_text="r(t)", row=2, col=2)
        fig.update_yaxes(title_text="Rate (%)", row=1, col=1)
        fig.update_yaxes(title_text="α", row=1, col=2)
        fig.update_yaxes(title_text="G(r)", row=2, col=1)
        fig.update_yaxes(title_text="r(t+1)", row=2, col=2)

        st.plotly_chart(fig, use_container_width=True)

        st.info(f"""
        **Interpretation:**
        - When far from target (high r): faster adjustment (α ≈ {alpha1 + alpha2:.3f})
        - When near target (low r): slower adjustment (α ≈ {alpha1:.3f})
        - Half-life at high adjustment: {np.log(0.5) / np.log(1 - (alpha1 + alpha2)):.1f} periods
        - Half-life at low adjustment: {np.log(0.5) / np.log(1 - alpha1):.1f} periods
        """)

    else:  # Monte Carlo Comparison
        st.markdown("""
        ### Monte Carlo Comparison of Models

        Compare forecasting performance of different models through simulation.
        """)

        # Simulation settings
        n_simulations = st.slider("Number of simulations", 100, 1000, 500, 100)
        forecast_horizon = st.slider("Forecast horizon", 1, 20, 10, 1)

        if st.button("Run Monte Carlo Simulation"):
            with st.spinner("Running simulations..."):
                np.random.seed(42)

                # True DGP: TAR model
                true_threshold = 0
                true_phi_low = 0.7
                true_phi_high = -0.4

                rmse_linear = []
                rmse_tar = []
                rmse_ms = []

                progress_bar = st.progress(0)

                for sim in range(n_simulations):
                    # Generate data
                    n_train = 300
                    n_total = n_train + forecast_horizon

                    y_true = np.zeros(n_total)
                    epsilon = np.random.normal(0, 1, n_total)

                    y_true[0] = epsilon[0]
                    for t in range(1, n_total):
                        if y_true[t - 1] <= true_threshold:
                            y_true[t] = true_phi_low * y_true[t - 1] + epsilon[t]
                        else:
                            y_true[t] = true_phi_high * y_true[t - 1] + epsilon[t]

                    # Training data
                    y_train = y_true[:n_train]
                    y_test = y_true[n_train:]

                    # Model 1: Linear AR(1)
                    phi_linear = np.cov(y_train[1:], y_train[:-1])[0, 1] / np.var(y_train[:-1])
                    forecast_linear = np.zeros(forecast_horizon)
                    forecast_linear[0] = phi_linear * y_train[-1]
                    for h in range(1, forecast_horizon):
                        forecast_linear[h] = phi_linear * forecast_linear[h - 1]

                    # Model 2: TAR (assume known threshold)
                    mask_low = y_train[:-1] <= true_threshold
                    mask_high = y_train[:-1] > true_threshold

                    if np.sum(mask_low) > 0:
                        phi_tar_low = np.mean(y_train[1:][mask_low] / y_train[:-1][mask_low])
                    else:
                        phi_tar_low = true_phi_low

                    if np.sum(mask_high) > 0:
                        phi_tar_high = np.mean(y_train[1:][mask_high] / y_train[:-1][mask_high])
                    else:
                        phi_tar_high = true_phi_high

                    forecast_tar = np.zeros(forecast_horizon)
                    last_val = y_train[-1]
                    for h in range(forecast_horizon):
                        if last_val <= true_threshold:
                            forecast_tar[h] = phi_tar_low * last_val
                        else:
                            forecast_tar[h] = phi_tar_high * last_val
                        last_val = forecast_tar[h]

                    # Model 3: Simple MS (2-state, equal probability)
                    forecast_ms = np.zeros(forecast_horizon)
                    forecast_ms[0] = 0.5 * (phi_tar_low + phi_tar_high) * y_train[-1]
                    for h in range(1, forecast_horizon):
                        forecast_ms[h] = 0.5 * (phi_tar_low + phi_tar_high) * forecast_ms[h - 1]

                    # RMSE
                    rmse_linear.append(np.sqrt(np.mean((y_test - forecast_linear) ** 2)))
                    rmse_tar.append(np.sqrt(np.mean((y_test - forecast_tar) ** 2)))
                    rmse_ms.append(np.sqrt(np.mean((y_test - forecast_ms) ** 2)))

                    progress_bar.progress((sim + 1) / n_simulations)

                # Results
                st.success("Simulation complete!")

                # Visualization
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("RMSE Distribution", "Mean RMSE by Model")
                )

                # Box plots
                fig.add_trace(go.Box(y=rmse_linear, name='Linear AR', marker_color='#2196F3'),
                              row=1, col=1)
                fig.add_trace(go.Box(y=rmse_tar, name='TAR', marker_color='#4CAF50'),
                              row=1, col=1)
                fig.add_trace(go.Box(y=rmse_ms, name='MS', marker_color='#FF9800'),
                              row=1, col=1)

                # Bar chart of means
                mean_rmse = [np.mean(rmse_linear), np.mean(rmse_tar), np.mean(rmse_ms)]
                fig.add_trace(go.Bar(x=['Linear', 'TAR', 'MS'], y=mean_rmse,
                                     marker_color=['#2196F3', '#4CAF50', '#FF9800']),
                              row=1, col=2)

                fig.update_layout(height=400, showlegend=False)
                fig.update_yaxes(title_text="RMSE", row=1, col=1)
                fig.update_yaxes(title_text="Mean RMSE", row=1, col=2)

                st.plotly_chart(fig, use_container_width=True)

                # Statistics table
                results_df = pd.DataFrame({
                    'Model': ['Linear AR', 'TAR', 'Markov Switching'],
                    'Mean RMSE': [np.mean(rmse_linear), np.mean(rmse_tar), np.mean(rmse_ms)],
                    'Std RMSE': [np.std(rmse_linear), np.std(rmse_tar), np.std(rmse_ms)],
                    'Min RMSE': [np.min(rmse_linear), np.min(rmse_tar), np.min(rmse_ms)],
                    'Max RMSE': [np.max(rmse_linear), np.max(rmse_tar), np.max(rmse_ms)]
                })

                st.dataframe(results_df.style.highlight_min(subset=['Mean RMSE'], color='lightgreen'),
                             use_container_width=True)

                # Relative performance
                st.markdown("### Relative Performance")

                col1, col2 = st.columns(2)

                with col1:
                    improvement_tar = 100 * (1 - np.mean(rmse_tar) / np.mean(rmse_linear))
                    st.metric("TAR vs Linear", f"{improvement_tar:.1f}% better" if improvement_tar > 0
                    else f"{-improvement_tar:.1f}% worse")

                with col2:
                    improvement_ms = 100 * (1 - np.mean(rmse_ms) / np.mean(rmse_linear))
                    st.metric("MS vs Linear", f"{improvement_ms:.1f}% better" if improvement_ms > 0
                    else f"{-improvement_ms:.1f}% worse")

# Section 9: Conclusion & References
else:  # Conclusion & References
    st.header("9. Conclusion & References")

    st.markdown("""
    ### Key Takeaways

    Throughout this lecture, we have explored the rich world of nonlinear time series models 
    and their applications in economics and finance.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **What We Learned:**

        1. **Nonlinearity is pervasive** in economic data
           - Asymmetric business cycles
           - Regime-dependent dynamics
           - Threshold effects

        2. **Testing is crucial**
           - BDS test for general nonlinearity
           - Specific tests for ARCH, TAR, etc.
           - Visual diagnostics complement formal tests

        3. **Model variety**
           - TAR for threshold effects
           - MS for unobserved regimes
           - STAR for smooth transitions
        """)

    with col2:
        st.markdown("""
        **Practical Considerations:**

        1. **Model selection**
           - Theory should guide specification
           - Data characteristics matter
           - Information criteria help

        2. **Estimation challenges**
           - Computational intensity
           - Parameter identification
           - Initial values sensitivity

        3. **Forecasting**
           - Regime uncertainty
           - Simulation-based methods
           - Density forecasts valuable
        """)

    st.markdown("---")

    st.markdown("""
    ### Advanced Topics for Further Study

    Building on this foundation, you are now prepared to explore:
    """)

    advanced_topics = {
        "Multivariate Extensions": [
            "Vector TAR (VTAR) models",
            "Multivariate Markov Switching",
            "Regime-dependent VAR",
            "Threshold cointegration"
        ],
        "Testing & Specification": [
            "Linearity tests with nuisance parameters",
            "Threshold estimation and inference",
            "Model selection in MS models",
            "Bootstrap methods"
        ],
        "Estimation Methods": [
            "Maximum likelihood for MS models",
            "Bayesian estimation",
            "Sequential Monte Carlo",
            "EM algorithm applications"
        ],
        "Applications": [
            "Monetary policy analysis",
            "Asset pricing with regimes",
            "Exchange rate intervention",
            "Credit risk modeling"
        ]
    }

    for topic, subtopics in advanced_topics.items():
        with st.expander(f"📚 {topic}"):
            for subtopic in subtopics:
                st.markdown(f"- {subtopic}")

    st.markdown("---")

    st.markdown("""
    ### Essential References

    #### Foundational Papers
    """)

    references = pd.DataFrame({
        'Authors': [
            'Hamilton (1989)',
            'Tong & Lim (1980)',
            'Teräsvirta (1994)',
            'Hansen (1996)',
            'Koop & Potter (1999)',
            'Brock et al. (1996)'
        ],
        'Title': [
            'A New Approach to the Economic Analysis of Nonstationary Time Series',
            'Threshold Autoregression, Limit Cycles and Cyclical Data',
            'Specification, Estimation, and Evaluation of Smooth Transition Autoregressive Models',
            'Inference When a Nuisance Parameter is Not Identified Under the Null',
            'Dynamic Asymmetries in U.S. Unemployment',
            'A Test for Independence Based on the Correlation Dimension'
        ],
        'Journal': [
            'Econometrica',
            'Journal of the Royal Statistical Society',
            'Journal of the American Statistical Association',
            'Econometrica',
            'Journal of Econometrics',
            'Econometric Reviews'
        ]
    })

    st.dataframe(references, use_container_width=True, hide_index=True)

    st.markdown("""
    #### Textbooks

    - **Tong, H.** (1990). *Non-linear Time Series: A Dynamical System Approach*. Oxford University Press.
    - **Franses, P.H. & van Dijk, D.** (2000). *Non-linear Time Series Models in Empirical Finance*. Cambridge University Press.
    - **Teräsvirta, T., Tjøstheim, D. & Granger, C.W.J.** (2010). *Modelling Nonlinear Economic Time Series*. Oxford University Press.
    - **Hamilton, J.D.** (1994). *Time Series Analysis*. Princeton University Press.
    - **Tsay, R.S.** (2010). *Analysis of Financial Time Series*, 3rd Edition. Wiley.

    #### Recent Developments

    - Machine learning approaches to regime identification
    - Time-varying transition probabilities
    - High-dimensional nonlinear models
    - Real-time regime monitoring
    - Nonlinear forecasting combinations
    """)

    st.markdown("---")

    st.markdown("""
    ### Software and Implementation

    **R Packages:**
    - `tsDyn` - Nonlinear time series models
    - `MSwM` - Markov switching models
    - `vars` - Vector autoregression (with structural breaks)
    - `forecast` - Forecasting framework

    **Python Libraries:**
    - `statsmodels` - Markov switching models
    - `arch` - ARCH/GARCH models
    - `PyFlux` - Bayesian time series

    **MATLAB:**
    - Econometrics Toolbox
    - MFE Toolbox (Kevin Sheppard)
    """)

    st.markdown("---")

    # Final message
    st.success("""
    ### 🎓 Congratulations!

    You have completed this comprehensive introduction to nonlinearity in economic time series.
    You now have the foundation to:

    ✅ Identify nonlinear patterns in economic data  
    ✅ Apply appropriate statistical tests  
    ✅ Specify and estimate nonlinear models  
    ✅ Generate forecasts from nonlinear specifications  
    ✅ Interpret regime-dependent dynamics  

    **Next Steps:** Apply these techniques to real economic and financial data, 
    and explore the advanced topics that interest you most!
    """)

    st.markdown("---")

    st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: #e3f2fd; border-radius: 10px;'>
    <h3>📧 Contact Information</h3>
    <p><b>Dr. Merwan Roudane</b></p>
    <p>For questions, comments, or further discussion on nonlinear time series analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Download option for notes
    st.markdown("---")
    st.markdown("### 📥 Download Lecture Notes")

    lecture_summary = f"""
    NONLINEARITY IN ECONOMIC TIME SERIES
    Lecture Notes by Dr. Merwan Roudane

    1. INTRODUCTION
    - Sources of nonlinearity in economics
    - Linear vs nonlinear models
    - Importance of proper specification

    2. TESTING FOR NONLINEARITY
    - BDS test for general nonlinearity
    - Tsay test for threshold effects
    - McLeod-Li test for ARCH effects
    - Ramsey RESET test

    3. THRESHOLD MODELS (TAR)
    - Self-exciting TAR (SETAR)
    - Threshold estimation
    - Regime-dependent dynamics

    4. MARKOV SWITCHING MODELS
    - Hamilton filter and smoother
    - Transition probabilities
    - Business cycle applications

    5. SMOOTH TRANSITION MODELS
    - LSTAR models
    - ESTAR models
    - Transition function specification

    6. APPLICATIONS
    - Business cycles
    - Exchange rates
    - Interest rates
    - Volatility modeling

    KEY REFERENCES:
    - Hamilton (1989) - Markov Switching
    - Tong (1990) - Nonlinear Time Series
    - Teräsvirta et al. (2010) - Comprehensive treatment

    Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
    """

    st.download_button(
        label="Download Summary (TXT)",
        data=lecture_summary,
        file_name="nonlinear_timeseries_notes.txt",
        mime="text/plain"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
<p><b>Nonlinearity in Economic Time Series</b></p>
<p>An Interactive Lecture by Dr. Merwan Roudane</p>
<p style='font-size: 0.9em;'>© 2025 | Built with Streamlit & Plotly</p>
</div>
""", unsafe_allow_html=True)