import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="IrisAI | K-Means Analytics Dashboard",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS (DARK GLASSMORPHISM WITH OPTIMIZED HIGH CONTRAST)
# =========================================================
st.markdown("""
<style>
    /* Main Background & Base Styling */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background:
            linear-gradient(rgba(5, 10, 24, 0.55), rgba(5, 10, 24, 0.65)),
            url("https://images.unsplash.com/photo-1563241527-3004b7be0ffd?q=80&w=1920&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        color: #ffffff;
    }

    /* Strict High-Contrast Color Palette for Ultra-Legibility */
    .stApp p, .stApp label, .stApp span, .stApp div,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #ffffff !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.9);
    }

    /* Subtext & Captions using High-Contrast Light Cyan */
    .stApp .stCaption, .stApp small {
        color: #a5f3fc !important;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.9);
    }

    /* Native Widget Label Contrast Fixes (Sliders, Radio, Selects) */
    .stApp label p {
        color: #38bdf8 !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }

    /* Sidebar Contrast with Darkened Overlay & Crisp Pure White/Cyan Text */
    section[data-testid="stSidebar"] {
        background:
            linear-gradient(rgba(3, 7, 18, 0.85), rgba(3, 7, 18, 0.92)),
            url("https://images.unsplash.com/photo-1563241527-3004b7be0ffd?q=80&w=1920&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        border-right: 1px solid rgba(0, 242, 254, 0.3);
    }

    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] li {
        color: #f1f5f9 !important;
    }

    /* Hero Banner with Strong Backdrop Glass Tinting */
    .hero {
        padding: 2.2rem 2.5rem;
        border-radius: 24px;
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(0, 242, 254, 0.5);
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.7);
        margin-bottom: 2rem;
    }

    .hero h1 {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #00f2fe 0%, #38bdf8 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .hero p {
        font-size: 1.1rem;
        color: #e2e8f0 !important;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }

    /* Translucent Glass Cards with Dynamic Contrast Borders */
    .glass-card {
        background: rgba(8, 15, 30, 0.78);
        backdrop-filter: blur(14px);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.25);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
        margin-bottom: 1.2rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .glass-card:hover {
        border: 1px solid rgba(0, 242, 254, 0.6);
        box-shadow: 0 12px 35px rgba(0, 242, 254, 0.3);
    }

    /* Metric Container */
    .metric-container {
        background: rgba(8, 15, 30, 0.82);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 1.2rem 1rem;
        text-align: center;
        border: 1px solid rgba(0, 242, 254, 0.3);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
    }

    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f2fe, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-lbl {
        color: #7dd3fc !important;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.3rem;
        font-weight: 800;
    }

    /* Result Box with High Contrast Highlighting */
    .result-badge {
        padding: 2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.90) 0%, rgba(30, 41, 59, 0.85) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(0, 242, 254, 0.6);
        text-align: center;
        box-shadow: 0 0 30px rgba(0, 242, 254, 0.3);
    }

    .result-title {
        font-size: 0.95rem;
        color: #cbd5e1 !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 700;
    }

    .result-cluster {
        font-size: 3.2rem;
        font-weight: 900;
        color: #00f2fe !important;
        margin: 0.4rem 0;
        text-shadow: 0 0 25px rgba(0, 242, 254, 0.6);
    }

    .result-species {
        font-size: 1.3rem;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 0.5rem;
    }

    .result-desc {
        color: #e2e8f0 !important;
        font-size: 0.92rem;
    }

    /* High-Contrast Interactive Buttons */
    .stButton > button {
        border-radius: 12px;
        font-weight: 800;
        background: linear-gradient(135deg, #00f2fe 0%, #38bdf8 100%);
        color: #020617 !important;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        box-shadow: 0 4px 20px rgba(0, 242, 254, 0.5);
        transition: all 0.25s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 242, 254, 0.8);
        color: #000000 !important;
    }

    /* Streamlit Navigation Tabs Contrast Override */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 16px;
        background-color: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(8px);
        color: #cbd5e1 !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .stTabs [aria-selected="true"] {
        background-color: #00f2fe !important;
        color: #020617 !important;
        font-weight: bold;
    }

    /* Table Contrast Fixes */
    .stDataFrame {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border-radius: 12px;
    }

    /* Hide Default Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA & MODEL LOADING (WITH AUTOMATIC FALLBACK TRAINER)
# =========================================================
FEATURE_COLS = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

@st.cache_data
def load_dataset():
    if os.path.exists("Iris.csv"):
        df = pd.read_csv("Iris.csv")
    else:
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=FEATURE_COLS)
        df['Species'] = [iris.target_names[i] for i in iris.target]
    return df

@st.cache_resource
def get_model_and_scaler():
    df = load_dataset()
    X = df[FEATURE_COLS]
    y = df['Species'] if 'Species' in df.columns else None

    # Load from pkl if available
    if os.path.exists("iris_kmeans_model.pkl"):
        try:
            with open("iris_kmeans_model.pkl", "rb") as file:
                saved = pickle.load(file)
            model = saved.get("model")
            scaler = saved.get("scaler", None)
            
            # Create scaler if absent in saved pickle
            if scaler is None:
                scaler = StandardScaler()
                scaler.fit(X)
            return model, scaler, X, y
        except Exception:
            pass

    # Fallback: Train clean Scaled K-Means model on Iris dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    model.fit(X_scaled)
    return model, scaler, X, y

df_raw = load_dataset()
model, scaler, X_raw, y_raw = get_model_and_scaler()
n_clusters = getattr(model, 'n_clusters', 3)

# Build Cluster -> Species Mapping via Majority Vote if species labels exist
cluster_species_map = {}
X_scaled = scaler.transform(X_raw)
cluster_labels = model.predict(X_scaled)

if y_raw is not None:
    df_mapped = pd.DataFrame({'Cluster': cluster_labels, 'Species': y_raw})
    for c in range(n_clusters):
        sub = df_mapped[df_mapped['Cluster'] == c]
        if len(sub) > 0:
            majority_species = sub['Species'].mode()[0]
            cluster_species_map[c] = majority_species.replace("Iris-", "")
        else:
            cluster_species_map[c] = f"Cluster {c}"

# Initialize session state variables for input values
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
    st.session_state.sepal_length = 5.8
    st.session_state.sepal_width = 3.0
    st.session_state.petal_length = 3.8
    st.session_state.petal_width = 1.2

# =========================================================
# SIDEBAR NAVIGATION & DETAILS
# =========================================================
with st.sidebar:
    st.markdown("<h2 style='color:#00f2fe !important; text-shadow: 0 0 10px rgba(0,242,254,0.5);'>🌸 IrisAI</h2>", unsafe_allow_html=True)
    st.caption("Unsupervised K-Means Dashboard")
    st.divider()

    st.markdown("### 📌 Overview")
    st.write(
        "Explore **K-Means Clustering** applied to the famous Iris dataset. "
        "Test input flower dimensions to predict cluster identity and inspect 2D PCA cluster spaces."
    )

    st.markdown("### 🤖 Model Specs")
    st.info(
        f"**Algorithm:** K-Means\n\n"
        f"**Optimal K:** {n_clusters}\n\n"
        f"**Preprocessing:** StandardScaler\n\n"
        f"**Dataset Size:** {len(df_raw)} samples"
    )

    st.markdown("### 📊 Features")
    st.markdown("""
    - **Sepal Length** (4.3 - 7.9 cm)
    - **Sepal Width** (2.0 - 4.4 cm)
    - **Petal Length** (1.0 - 6.9 cm)
    - **Petal Width** (0.1 - 2.5 cm)
    """)

    st.divider()
    st.caption("Built with Python • Scikit-Learn • Plotly • Streamlit")

# =========================================================
# HERO BANNER
# =========================================================
st.markdown("""
<div class="hero">
    <h1>🌸 Iris Flower K-Means Clustering</h1>
    <p>Machine Learning Dashboard with interactive cluster space projections and real-time prediction.</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# METRIC CARDS
# =========================================================
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="metric-container">
        <div class="metric-val">K-Means</div>
        <div class="metric-lbl">Algorithm</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-val">{n_clusters}</div>
        <div class="metric-lbl">Clusters (K)</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-container">
        <div class="metric-val">4</div>
        <div class="metric-lbl">Features</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-val">{len(df_raw)}</div>
        <div class="metric-lbl">Total Samples</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# =========================================================
# MAIN CONTENT AREA (TABS)
# =========================================================
tab1, tab2, tab3 = st.tabs(["🧪 Predict Cluster", "📈 PCA Cluster Space", "📊 Cluster Profiling & Evaluation"])

# ---------------------------------------------------------
# TAB 1: PREDICTION & INPUT
# ---------------------------------------------------------
with tab1:
    left_col, right_col = st.columns([1.1, 0.9], gap="large")

    with left_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("⚙️ Enter Flower Measurements")
        st.write("Adjust the slider values below to specify the flower's physical dimensions.")

        c1, c2 = st.columns(2)
        with c1:
            sepal_length = st.slider("🌿 Sepal Length (cm)", 4.0, 8.0, st.session_state.sepal_length, 0.1)
            petal_length = st.slider("🌺 Petal Length (cm)", 1.0, 7.0, st.session_state.petal_length, 0.1)

        with c2:
            sepal_width = st.slider("🌿 Sepal Width (cm)", 2.0, 4.5, st.session_state.sepal_width, 0.1)
            petal_width = st.slider("🌺 Petal Width (cm)", 0.1, 2.5, st.session_state.petal_width, 0.1)

        st.write("")
        if st.button("🔍 Analyze & Classify", use_container_width=True):
            st.session_state.sepal_length = sepal_length
            st.session_state.sepal_width = sepal_width
            st.session_state.petal_length = petal_length
            st.session_state.petal_width = petal_width
            st.session_state.analyzed = True

        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🎯 Clustering Output")

        # Active evaluation variables
        cur_sl = st.session_state.sepal_length
        cur_sw = st.session_state.sepal_width
        cur_pl = st.session_state.petal_length
        cur_pw = st.session_state.petal_width

        input_raw = np.array([[cur_sl, cur_sw, cur_pl, cur_pw]])
        input_scaled = scaler.transform(input_raw)
        pred_cluster = int(model.predict(input_scaled)[0])

        mapped_species = cluster_species_map.get(pred_cluster, "Unknown Species")

        st.markdown(f"""
        <div class="result-badge">
            <div class="result-title">Predicted Assignment</div>
            <div class="result-cluster">Cluster {pred_cluster}</div>
            <div class="result-species">Dominant Species: <i>{mapped_species}</i></div>
            <div class="result-desc">
                Calculated in 4D feature space via Euclidean distance to cluster centroid.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        # Input Summary Table
        summary_df = pd.DataFrame({
            "Feature": ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
            "User Value": [f"{cur_sl} cm", f"{cur_sw} cm", f"{cur_pl} cm", f"{cur_pw} cm"]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 2: PCA CLUSTER SPACE VISUALIZATION
# ---------------------------------------------------------
with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🌌 2D PCA Projection of K-Means Clusters")
    st.write(
        "Below is a 2-component Principal Component Analysis (PCA) plot projecting the 4-dimensional "
        "Iris measurements into 2D space. The user's input sample is marked dynamically."
    )

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    input_pca = pca.transform(input_scaled)

    pca_df = pd.DataFrame(X_pca, columns=["PCA Component 1", "PCA Component 2"])
    pca_df = pd.concat([pca_df, X_raw.reset_index(drop=True)], axis=1)
    pca_df['Cluster'] = [f"Cluster {c} ({cluster_species_map.get(c, '')})" for c in cluster_labels]
    
    if y_raw is not None:
        pca_df['Species'] = y_raw.reset_index(drop=True)

    fig_pca = px.scatter(
        pca_df,
        x="PCA Component 1",
        y="PCA Component 2",
        color="Cluster",
        symbol="Species" if y_raw is not None else None,
        hover_data=FEATURE_COLS if y_raw is None else ["Species"] + FEATURE_COLS,
        color_discrete_sequence=["#00f2fe", "#f472b6", "#38bdf8"],
        template="plotly_dark",
        opacity=0.90
    )

    fig_pca.add_trace(go.Scatter(
        x=[input_pca[0, 0]],
        y=[input_pca[0, 1]],
        mode="markers",
        marker=dict(size=18, color="#ff007f", symbol="star", line=dict(width=2, color="#ffffff")),
        name="📍 Your Input Flower"
    ))

    fig_pca.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_pca, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 3: CLUSTER PROFILING & EVALUATION
# ---------------------------------------------------------
with tab3:
    c_left, c_right = st.columns(2, gap="large")

    with c_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📊 Average Feature Values per Cluster")

        df_profile = X_raw.copy()
        df_profile['Cluster'] = [f"Cluster {c}" for c in cluster_labels]
        profile_means = df_profile.groupby('Cluster').mean().reset_index()

        fig_bar = px.bar(
            profile_means.melt(id_vars="Cluster", var_name="Feature", value_name="Mean (cm)"),
            x="Feature",
            y="Mean (cm)",
            color="Cluster",
            barmode="group",
            color_discrete_sequence=["#00f2fe", "#f472b6", "#38bdf8"],
            template="plotly_dark"
        )
        fig_bar.update_layout(
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📉 Elbow & Silhouette Evaluation")

        K_range = range(2, 7)
        inertias = []
        silhouettes = []

        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(X_scaled, km.labels_))

        eval_df = pd.DataFrame({
            "K": list(K_range),
            "Inertia": inertias,
            "Silhouette Score": silhouettes
        })

        fig_eval = px.line(
            eval_df,
            x="K",
            y="Silhouette Score",
            markers=True,
            title="Silhouette Score vs. K (Optimal K = 3)",
            template="plotly_dark"
        )
        fig_eval.update_traces(line_color="#00f2fe", marker_size=10)
        fig_eval.update_layout(
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_eval, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div style="text-align: center; color: #e2e8f0; font-size: 0.85rem; padding: 2rem 0 1rem 0; text-shadow: 0 1px 3px rgba(0,0,0,0.9);">
    🌸 <b>IrisAI Analytics</b> &nbsp;|&nbsp; Developed with Python, Scikit-Learn, Plotly & Streamlit
</div>
""", unsafe_allow_html=True)