import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# ----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="US Urban Connectivity Dashboard",
    page_icon="🗺️",
    layout="wide",
)

st.title("US Urban Connectivity")
st.caption(
    "Dashboard-only analytical product: mobility, parks/public space, amenities, and equity access across major US cities."
)

DATA_PATH = "data/urban_connectivity.csv"


# -----------------------------
# Utilities
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Standardize a few common quirks
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Clean column names lightly (keep original readable, but remove double spaces)
    df.columns = [c.strip().replace("  ", " ") for c in df.columns]

    # Ensure City/State exist
    for col in ["City", "State"]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in dataset.")

    # Create a display label
    df["City_Label"] = df["City"].astype(str) + ", " + df["State"].astype(str)

    # Convert object columns that are numeric-ish into numeric
    for c in df.columns:
        if df[c].dtype == "object":
            # Try numeric conversion; keep as object if it fails widely
            converted = pd.to_numeric(df[c], errors="coerce")
            # If enough values convert, adopt numeric
            if converted.notna().mean() > 0.6:
                df[c] = converted

    return df


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = np.nanmean(s)
    sd = np.nanstd(s)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=series.index)
    return (s - mu) / sd


def winsorize(s: pd.Series, lower=0.02, upper=0.98) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    lo = x.quantile(lower)
    hi = x.quantile(upper)
    return x.clip(lo, hi)


def percent_fmt(x):
    if pd.isna(x):
        return "—"
    return f"{x*100:.1f}%"


def safe_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().sum() > 0


# -----------------------------
# Load
# -----------------------------
try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(
        f"Could not load dataset at `{DATA_PATH}`.\n\n"
        f"Fix: make sure the file exists in your repo at `data/urban_connectivity.csv`.\n\n"
        f"Error: {e}"
    )
    st.stop()


# -----------------------------
# Feature engineering (senior dashboard layer)
# -----------------------------
# Mobility pillars (Walkscore)
mob_cols = [c for c in ["Walk Score", "Transit Score", "Bike Score"] if safe_col(df, c)]

# Parks / access pillars (TPL)
park_cols_candidates = [
    "Walkable_Park_Access_all_residents",
    "Parkland_Stats_by_City_parks_as__city_area",
    "Parkland_Stats_by_City_total_acres",
]
park_cols = [c for c in park_cols_candidates if safe_col(df, c)]

# Amenities: use counts if present (normalize per 100k residents)
amenity_count_cols = [c for c in df.columns if c.startswith("Number_of_")]
trail_cols = [c for c in df.columns if c.startswith("Trail_Miles_")]
amenity_cols = [c for c in amenity_count_cols + trail_cols if safe_col(df, c)]

# Equity / disparity: park access by race + income distribution columns
equity_cols_candidates = [
    "Walkable_Park_Access_white",
    "Walkable_Park_Access_all_people_of_color",
    "Walkable_Park_Access_black",
    "Walkable_Park_Access_hispanic_latinx",
    "Walkable_Park_Access_asian",
    "Distribution_of_Park_Space_low_income",
    "Distribution_of_Park_Space_high_income",
    "Distribution_of_Park_Space_neighborhoods_of_color",
    "Distribution_of_Park_Space_white",
]
equity_cols = [c for c in equity_cols_candidates if safe_col(df, c)]

# Population column for per-capita normalization
pop_col = None
for candidate in ["Population_2022_Census", "Population_2021"]:
    if safe_col(df, candidate):
        pop_col = candidate
        break

# Build per-100k amenity rates if we can
df_feat = df.copy()
if pop_col and len(amenity_cols) > 0:
    pop = pd.to_numeric(df_feat[pop_col], errors="coerce")
    pop = pop.replace(0, np.nan)
    for c in amenity_cols:
        df_feat[c + "_per_100k"] = (pd.to_numeric(df_feat[c], errors="coerce") / pop) * 100000.0

    amenity_rate_cols = [c + "_per_100k" for c in amenity_cols if safe_col(df_feat, c + "_per_100k")]
else:
    amenity_rate_cols = []

# Create composite indices
# Mobility Index: z-scored components (winsorized)
if len(mob_cols) > 0:
    mob_z = np.vstack([zscore(winsorize(df_feat[c])) for c in mob_cols]).T
    df_feat["Mobility_Index"] = np.nanmean(mob_z, axis=1)
else:
    df_feat["Mobility_Index"] = np.nan

# Parks Index
if len(park_cols) > 0:
    park_z = np.vstack([zscore(winsorize(df_feat[c])) for c in park_cols]).T
    df_feat["Parks_Index"] = np.nanmean(park_z, axis=1)
else:
    df_feat["Parks_Index"] = np.nan

# Amenities Index (per 100k if possible, else raw)
if len(amenity_rate_cols) > 0:
    amen_z = np.vstack([zscore(winsorize(df_feat[c])) for c in amenity_rate_cols]).T
    df_feat["Amenities_Index"] = np.nanmean(amen_z, axis=1)
elif len(amenity_cols) > 0:
    amen_z = np.vstack([zscore(winsorize(df_feat[c])) for c in amenity_cols]).T
    df_feat["Amenities_Index"] = np.nanmean(amen_z, axis=1)
else:
    df_feat["Amenities_Index"] = np.nan

# Equity gap metrics (lower gap is better)
if safe_col(df_feat, "Walkable_Park_Access_white") and safe_col(df_feat, "Walkable_Park_Access_all_people_of_color"):
    df_feat["Park_Access_Gap_White_vs_POC"] = (
        pd.to_numeric(df_feat["Walkable_Park_Access_white"], errors="coerce")
        - pd.to_numeric(df_feat["Walkable_Park_Access_all_people_of_color"], errors="coerce")
    )
else:
    df_feat["Park_Access_Gap_White_vs_POC"] = np.nan

if safe_col(df_feat, "Distribution_of_Park_Space_high_income") and safe_col(df_feat, "Distribution_of_Park_Space_low_income"):
    df_feat["Park_Space_Gap_High_vs_Low_Income"] = (
        pd.to_numeric(df_feat["Distribution_of_Park_Space_high_income"], errors="coerce")
        - pd.to_numeric(df_feat["Distribution_of_Park_Space_low_income"], errors="coerce")
    )
else:
    df_feat["Park_Space_Gap_High_vs_Low_Income"] = np.nan

# Equity Index (we want smaller gaps => invert z-scores)
gap_cols = ["Park_Access_Gap_White_vs_POC", "Park_Space_Gap_High_vs_Low_Income"]
gap_cols = [c for c in gap_cols if safe_col(df_feat, c)]

if len(gap_cols) > 0:
    gap_z = np.vstack([zscore(winsorize(df_feat[c])) for c in gap_cols]).T
    # invert so higher = better equity (smaller gaps)
    df_feat["Equity_Index"] = -np.nanmean(gap_z, axis=1)
else:
    df_feat["Equity_Index"] = np.nan

# Percentile ranks (easier to interpret than z)
for idx_col in ["Mobility_Index", "Parks_Index", "Amenities_Index", "Equity_Index"]:
    if safe_col(df_feat, idx_col):
        df_feat[idx_col + "_pct"] = df_feat[idx_col].rank(pct=True)
    else:
        df_feat[idx_col + "_pct"] = np.nan


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Controls")

    city_options = df_feat["City_Label"].dropna().unique().tolist()
    city_options = sorted(city_options)
    focus_city = st.selectbox("Focus city", city_options, index=0)

    st.subheader("Connectivity Score Weights")
    w_mob = st.slider("Mobility (Walk/Transit/Bike)", 0.0, 1.0, 0.35, 0.05)
    w_parks = st.slider("Parks / Access", 0.0, 1.0, 0.30, 0.05)
    w_amen = st.slider("Amenities (per-capita where available)", 0.0, 1.0, 0.20, 0.05)
    w_eq = st.slider("Equity (smaller gaps → higher score)", 0.0, 1.0, 0.15, 0.05)

    w_sum = w_mob + w_parks + w_amen + w_eq
    if w_sum == 0:
        w_sum = 1.0

    normalize_weights = st.checkbox("Normalize weights to 1.0", value=True)
    if normalize_weights:
        w_mob, w_parks, w_amen, w_eq = [w / w_sum for w in [w_mob, w_parks, w_amen, w_eq]]

    st.subheader("Compare")
    compare_n = st.slider("Number of comparator cities", 3, 15, 7, 1)
    comparator_mode = st.radio(
        "Comparator method",
        ["Nearest in overall score", "Top performers overall"],
        index=0,
    )

    st.subheader("Segmentation")
    k = st.slider("City clusters (KMeans)", 3, 8, 5, 1)

    st.divider()
    st.caption("Dataset: Walk Score + Trust for Public Land (TPL), 100+ US cities.")


# -----------------------------
# Build overall score
# -----------------------------
score_components = {
    "Mobility_Index": w_mob,
    "Parks_Index": w_parks,
    "Amenities_Index": w_amen,
    "Equity_Index": w_eq,
}

# Weighted score using z indices (already centered)
score = np.zeros(len(df_feat))
weight_total = 0.0

for col, w in score_components.items():
    if safe_col(df_feat, col) and w > 0:
        score = score + df_feat[col].fillna(df_feat[col].median()) * w
        weight_total += w

if weight_total == 0:
    df_feat["Connectivity_Score"] = np.nan
else:
    df_feat["Connectivity_Score"] = score / weight_total

df_feat["Connectivity_Score_pct"] = df_feat["Connectivity_Score"].rank(pct=True)
df_feat["Connectivity_Rank"] = df_feat["Connectivity_Score"].rank(ascending=False, method="min").astype(int)


# -----------------------------
# Focus selection
# -----------------------------
focus_row = df_feat.loc[df_feat["City_Label"] == focus_city].iloc[0]
focus_score = focus_row["Connectivity_Score"]
focus_rank = int(focus_row["Connectivity_Rank"])
focus_pct = focus_row["Connectivity_Score_pct"]

# Comparator selection
df_sorted = df_feat.sort_values("Connectivity_Score", ascending=False).reset_index(drop=True)

if comparator_mode == "Top performers overall":
    comp_df = df_sorted.head(compare_n)
else:
    # nearest by score
    df_tmp = df_feat.copy()
    df_tmp["dist"] = (df_tmp["Connectivity_Score"] - focus_score).abs()
    comp_df = df_tmp.sort_values("dist").head(compare_n)

# Ensure focus city included
if focus_city not in comp_df["City_Label"].values:
    comp_df = pd.concat([comp_df, df_feat[df_feat["City_Label"] == focus_city]], ignore_index=True)

comp_df = comp_df.drop_duplicates(subset=["City_Label"]).copy()


# -----------------------------
# Layout
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Connectivity Rank", f"#{focus_rank}")
c2.metric("Percentile", f"{focus_pct*100:.0f}th")
c3.metric("Connectivity Score (z)", f"{focus_score:.2f}" if pd.notna(focus_score) else "—")
if pop_col:
    c4.metric("Population", f"{int(focus_row[pop_col]):,}" if pd.notna(focus_row[pop_col]) else "—")
else:
    c4.metric("Population", "—")


tab1, tab2, tab3, tab4 = st.tabs(["Overview", "City Deep Dive", "Equity Lens", "Segmentation"])


# -----------------------------
# TAB 1: Overview
# -----------------------------
with tab1:
    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Overall ranking (interactive)")
        rank_view = df_feat[[
            "City_Label",
            "Connectivity_Score",
            "Connectivity_Rank",
            "Mobility_Index",
            "Parks_Index",
            "Amenities_Index",
            "Equity_Index",
        ]].sort_values("Connectivity_Rank")

        fig_rank = px.scatter(
            rank_view,
            x="Connectivity_Rank",
            y="Connectivity_Score",
            hover_name="City_Label",
            hover_data={
                "Connectivity_Rank": True,
                "Connectivity_Score": ":.2f",
                "Mobility_Index": ":.2f",
                "Parks_Index": ":.2f",
                "Amenities_Index": ":.2f",
                "Equity_Index": ":.2f",
            },
        )
        fig_rank.update_layout(height=420, xaxis_title="Rank (1 = best)", yaxis_title="Connectivity Score (z)")
        fig_rank.add_hline(y=focus_score, line_dash="dot")
        st.plotly_chart(fig_rank, use_container_width=True)

    with right:
        st.subheader("Score composition (for focus city)")
        comp = []
        for col, w in score_components.items():
            val = float(focus_row[col]) if pd.notna(focus_row[col]) else np.nan
            comp.append({"Component": col.replace("_", " "), "Weight": w, "Value (z)": val, "Contribution": w * val})

        comp_df2 = pd.DataFrame(comp)
        comp_df2["Contribution"] = comp_df2["Contribution"].fillna(0)

        fig_bar = px.bar(
            comp_df2,
            x="Contribution",
            y="Component",
            orientation="h",
            hover_data={"Weight": ":.2f", "Value (z)": ":.2f", "Contribution": ":.2f"},
        )
        fig_bar.update_layout(height=420, xaxis_title="Weighted contribution", yaxis_title="")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Compare cities (radar)")
    radar_cols = ["Mobility_Index", "Parks_Index", "Amenities_Index", "Equity_Index"]
    radar_cols = [c for c in radar_cols if safe_col(comp_df, c)]

    if len(radar_cols) >= 2:
        # Normalize to 0-100 for radar readability
        radar_norm = comp_df[["City_Label"] + radar_cols].copy()
        for c in radar_cols:
            x = radar_norm[c]
            radar_norm[c] = (x - x.min()) / (x.max() - x.min() + 1e-9) * 100

        fig_radar = go.Figure()
        for _, r in radar_norm.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[r[c] for c in radar_cols],
                theta=[c.replace("_", " ") for c in radar_cols],
                fill="toself" if r["City_Label"] == focus_city else None,
                name=r["City_Label"],
                opacity=0.9 if r["City_Label"] == focus_city else 0.45,
            ))

        fig_radar.update_layout(
            height=520,
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("Not enough index columns available for radar comparison.")


# -----------------------------
# TAB 2: City Deep Dive
# -----------------------------
with tab2:
    st.subheader(f"Deep dive: {focus_city}")

    a, b = st.columns([1.1, 1])

    with a:
        st.markdown("**Key indicator distribution**")
        # Choose a few high-signal metrics if present
        candidate_metrics = [
            "Walk Score", "Transit Score", "Bike Score",
            "Walkable_Park_Access_all_residents",
            "Parkland_Stats_by_City_parks_as__city_area",
            "City_Population_Stats_density__people_acre_",
        ]
        available_metrics = [m for m in candidate_metrics if safe_col(df_feat, m)]

        metric = st.selectbox("Metric", available_metrics, index=0)
        fig_hist = px.histogram(df_feat, x=metric, nbins=25, marginal="box")
        # Add focus line
        focus_val = focus_row[metric]
        if pd.notna(focus_val):
            fig_hist.add_vline(x=float(focus_val), line_dash="dot")
        fig_hist.update_layout(height=420)
        st.plotly_chart(fig_hist, use_container_width=True)

    with b:
        st.markdown("**Where this city sits vs the field**")

        # Percentile gauges for the indices
        gauges = [
            ("Mobility", focus_row.get("Mobility_Index_pct", np.nan)),
            ("Parks", focus_row.get("Parks_Index_pct", np.nan)),
            ("Amenities", focus_row.get("Amenities_Index_pct", np.nan)),
            ("Equity", focus_row.get("Equity_Index_pct", np.nan)),
        ]

        fig_g = go.Figure()
        for i, (name, pct) in enumerate(gauges, start=1):
            if pd.isna(pct):
                pct = 0
            fig_g.add_trace(go.Indicator(
                mode="gauge+number",
                value=float(pct * 100),
                title={"text": name},
                gauge={"axis": {"range": [0, 100]}},
                domain={"row": (i-1)//2, "column": (i-1) % 2},
            ))

        fig_g.update_layout(
            grid={"rows": 2, "columns": 2, "pattern": "independent"},
            height=420,
        )
        st.plotly_chart(fig_g, use_container_width=True)

    st.subheader("Top drivers and weak spots (simple narrative)")
    # Identify strongest/weakest components for the focus city
    comp_vals = {k: focus_row.get(k, np.nan) for k in ["Mobility_Index", "Parks_Index", "Amenities_Index", "Equity_Index"]}
    comp_vals_clean = {k: v for k, v in comp_vals.items() if pd.notna(v)}

    if comp_vals_clean:
        best = max(comp_vals_clean, key=comp_vals_clean.get)
        worst = min(comp_vals_clean, key=comp_vals_clean.get)

        st.write(
            f"- **Relative strength:** {best.replace('_', ' ')} (higher than most cities under current weighting)\n"
            f"- **Relative gap:** {worst.replace('_', ' ')} (largest opportunity area under current weighting)\n"
        )
    else:
        st.write("Index components are not available for narrative drivers.")


# -----------------------------
# TAB 3: Equity Lens
# -----------------------------
with tab3:
    st.subheader("Equity access and gaps")

    colL, colR = st.columns([1.05, 1])

    with colL:
        st.markdown("**Park access by demographic group (focus city)**")

        access_cols = [c for c in df_feat.columns if c.startswith("Walkable_Park_Access_")]
        access_cols = [c for c in access_cols if safe_col(df_feat, c)]

        # keep a curated order if present
        preferred = [
            "Walkable_Park_Access_all_residents",
            "Walkable_Park_Access_white",
            "Walkable_Park_Access_all_people_of_color",
            "Walkable_Park_Access_black",
            "Walkable_Park_Access_hispanic_latinx",
            "Walkable_Park_Access_asian",
        ]
        access_cols_sorted = [c for c in preferred if c in access_cols]
        # add any remaining
        access_cols_sorted += [c for c in access_cols if c not in access_cols_sorted]

        focus_access = pd.DataFrame({
            "Group": [c.replace("Walkable_Park_Access_", "").replace("_", " ").title() for c in access_cols_sorted],
            "Access": [focus_row.get(c, np.nan) for c in access_cols_sorted],
        }).dropna()

        if not focus_access.empty:
            fig_access = px.bar(
                focus_access,
                x="Access",
                y="Group",
                orientation="h",
                text=focus_access["Access"].apply(percent_fmt),
            )
            fig_access.update_layout(height=520, xaxis_title="Share of residents within walkable park access", yaxis_title="")
            st.plotly_chart(fig_access, use_container_width=True)
        else:
            st.info("No walkable park access fields available for this city.")

    with colR:
        st.markdown("**Gap diagnostics across all cities**")

        gap_pick = st.selectbox(
            "Gap metric",
            [c for c in ["Park_Access_Gap_White_vs_POC", "Park_Space_Gap_High_vs_Low_Income"] if safe_col(df_feat, c)],
        )

        if gap_pick:
            temp = df_feat[["City_Label", gap_pick, "Connectivity_Score"]].copy().dropna()
            # color by connectivity for context
            fig_gap = px.scatter(
                temp,
                x="Connectivity_Score",
                y=gap_pick,
                hover_name="City_Label",
                trendline="ols",
            )
            # Focus marker
            fv = focus_row.get(gap_pick, np.nan)
            if pd.notna(fv) and pd.notna(focus_score):
                fig_gap.add_trace(go.Scatter(
                    x=[focus_score],
                    y=[fv],
                    mode="markers+text",
                    text=[focus_city],
                    textposition="top center",
                    name="Focus city",
                ))
            fig_gap.update_layout(height=520, xaxis_title="Connectivity Score (z)", yaxis_title=gap_pick.replace("_", " "))
            st.plotly_chart(fig_gap, use_container_width=True)
        else:
            st.info("No gap fields available in this dataset for equity diagnostics.")

    st.subheader("Distribution lens: park space share")
    dist_cols = [c for c in df_feat.columns if c.startswith("Distribution_of_Park_Space_")]
    dist_cols = [c for c in dist_cols if safe_col(df_feat, c)]

    if dist_cols:
        # Let user choose two distributions to compare
        d1 = st.selectbox("Distribution metric A", dist_cols, index=0)
        d2 = st.selectbox("Distribution metric B", dist_cols, index=min(1, len(dist_cols)-1))

        temp = df_feat[["City_Label", d1, d2]].dropna()
        fig_dist = px.scatter(
            temp,
            x=d1,
            y=d2,
            hover_name="City_Label",
        )
        # Focus marker
        if pd.notna(focus_row.get(d1, np.nan)) and pd.notna(focus_row.get(d2, np.nan)):
            fig_dist.add_trace(go.Scatter(
                x=[focus_row[d1]],
                y=[focus_row[d2]],
                mode="markers+text",
                text=[focus_city],
                textposition="top center",
                name="Focus city",
            ))
        fig_dist.update_layout(height=520, xaxis_title=d1.replace("_", " "), yaxis_title=d2.replace("_", " "))
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("No Distribution_of_Park_Space_* fields found.")


# -----------------------------
# TAB 4: Segmentation (PCA + KMeans)
# -----------------------------
with tab4:
    st.subheader("City segmentation")

    # Choose features for clustering: indices + a few raw signals if present
    cluster_features = []
    for c in ["Mobility_Index", "Parks_Index", "Amenities_Index", "Equity_Index"]:
        if safe_col(df_feat, c):
            cluster_features.append(c)

    # Add a couple of raw measures if present to give segmentation texture
    for c in [
        "City_Population_Stats_density__people_acre_",
        "Walk Score",
        "Walkable_Park_Access_all_residents",
        "Parkland_Stats_by_City_parks_as__city_area",
    ]:
        if safe_col(df_feat, c) and c not in cluster_features:
            cluster_features.append(c)

    if len(cluster_features) < 3:
        st.info("Not enough usable numeric features to run segmentation.")
    else:
        X = df_feat[cluster_features].copy()
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median(numeric_only=True))

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        pca = PCA(n_components=2, random_state=42)
        pcs = pca.fit_transform(Xs)

        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(Xs)

        seg = df_feat[["City_Label", "Connectivity_Score"]].copy()
        seg["Cluster"] = labels.astype(str)
        seg["PC1"] = pcs[:, 0]
        seg["PC2"] = pcs[:, 1]

        expl = pca.explained_variance_ratio_
        st.caption(f"PCA variance explained: PC1={expl[0]*100:.1f}%, PC2={expl[1]*100:.1f}%")

        fig_seg = px.scatter(
            seg,
            x="PC1",
            y="PC2",
            color="Cluster",
            hover_name="City_Label",
            hover_data={"Connectivity_Score": ":.2f"},
        )

        # Focus marker
        focus_seg = seg.loc[seg["City_Label"] == focus_city]
        if not focus_seg.empty:
            fig_seg.add_trace(go.Scatter(
                x=focus_seg["PC1"],
                y=focus_seg["PC2"],
                mode="markers+text",
                text=[focus_city],
                textposition="top center",
                name="Focus city",
            ))

        fig_seg.update_layout(height=560, xaxis_title="PC1", yaxis_title="PC2")
        st.plotly_chart(fig_seg, use_container_width=True)

        st.subheader("Cluster profiles (what makes each segment different)")
        profile = df_feat.copy()
        profile["Cluster"] = labels.astype(int)

        cols_for_profile = ["Connectivity_Score"] + cluster_features
        cluster_summary = profile.groupby("Cluster")[cols_for_profile].mean(numeric_only=True).reset_index()

        # Show as heatmap (normalized) for interpretability
        heat = cluster_summary.set_index("Cluster").copy()
        heat_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)

        fig_heat = px.imshow(
            heat_norm.T,
            aspect="auto",
            labels=dict(x="Cluster", y="Metric", color="Normalized mean"),
        )
        fig_heat.update_layout(height=520)
        st.plotly_chart(fig_heat, use_container_width=True)

        st.caption("Interpretation tip: segments group cities with similar mobility/parks/amenities/equity patterns, not just overall score.")


# -----------------------------
# Download section (professional touch)
# -----------------------------
with st.expander("Download (clean export)"):
    export_cols = [
        "City_Label", "Connectivity_Rank", "Connectivity_Score", "Connectivity_Score_pct",
        "Mobility_Index", "Parks_Index", "Amenities_Index", "Equity_Index",
        "Park_Access_Gap_White_vs_POC", "Park_Space_Gap_High_vs_Low_Income"
    ]
    export_cols = [c for c in export_cols if c in df_feat.columns]

    out = df_feat[export_cols].sort_values("Connectivity_Rank")
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download city scores (CSV)",
        data=csv,
        file_name="urban_connectivity_scores.csv",
        mime="text/csv",
    )

