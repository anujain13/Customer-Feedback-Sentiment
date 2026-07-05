"""
Women's Clothing E-Commerce Reviews Dashboard
A modern, interactive Streamlit dashboard for the Women's Clothing E-Commerce
Reviews dataset.

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Women's Clothing Reviews Dashboard",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# MODERN STYLING (dark theme, gradient KPI cards, clean fonts)
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0f1117 0%, #161a23 100%);
    }
    h1, h2, h3, h4, h5, p, span, label, div { color: #eaeaf0; }

    /* Title banner */
    .dashboard-title {
        background: linear-gradient(90deg, #f53e9a 0%, #b03ef5 50%, #6a3ef5 100%);
        padding: 22px 30px;
        border-radius: 16px;
        margin-bottom: 22px;
        box-shadow: 0 4px 20px rgba(245, 62, 154, 0.25);
    }
    .dashboard-title h1 {
        color: white;
        font-size: 30px;
        font-weight: 800;
        letter-spacing: 0.5px;
        margin: 0;
    }
    .dashboard-title p {
        color: rgba(255,255,255,0.85);
        margin: 4px 0 0 0;
        font-size: 14px;
    }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(145deg, #1d2130 0%, #262b3d 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 18px 20px;
        text-align: left;
        box-shadow: 0 4px 14px rgba(0,0,0,0.35);
        height: 100%;
    }
    .kpi-label {
        font-size: 13px;
        color: #9aa0b4;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 2px;
    }
    .kpi-delta-pos { color: #3ddc97; font-size: 13px; font-weight: 600; }
    .kpi-delta-neg { color: #ff5c7a; font-size: 13px; font-weight: 600; }

    /* Section card wrapper for charts */
    .chart-card {
        background: #1a1e29;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 14px 16px 4px 16px;
        margin-bottom: 18px;
    }
    .chart-title {
        font-size: 15px;
        font-weight: 700;
        color: #eaeaf0;
        margin-bottom: 4px;
    }

    section[data-testid="stSidebar"] {
        background: #11141d;
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    /* ---- Multiselect: input box ---- */
    div[data-baseweb="select"] > div {
        background-color: #1d2130;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.10);
    }

    /* Cap the height of the selected-tags box so it doesn't grow huge,
       and let it scroll internally instead of pushing the dropdown panel down */
    div[data-baseweb="select"] > div:first-child {
        max-height: 120px;
        overflow-y: auto;
    }

    /* Selected option "pills" */
    span[data-baseweb="tag"] {
        background-color: #f53e9a !important;
        border-radius: 6px !important;
        margin: 2px !important;
    }

    /* ---- Dropdown menu (the floating list with options / "No results") ---- */
    div[data-baseweb="popover"] ul,
    div[data-baseweb="menu"] {
        background-color: #1d2130 !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="popover"] ul li,
    div[data-baseweb="menu"] li {
        background-color: #1d2130 !important;
        color: #eaeaf0 !important;
    }
    div[data-baseweb="popover"] ul li:hover,
    div[data-baseweb="menu"] li:hover {
        background-color: #2a2f42 !important;
    }
    /* "No results" / empty-state text inside the dropdown */
    div[data-baseweb="popover"] div,
    div[data-baseweb="popover"] span {
        color: #eaeaf0;
    }

    /* Sidebar multiselect labels need breathing room above each widget */
    section[data-testid="stSidebar"] label {
        font-weight: 600;
        margin-bottom: 4px;
    }

    /* Keep dropdown panel from getting clipped by sidebar overflow */
    section[data-testid="stSidebar"] .block-container,
    section[data-testid="stSidebar"] > div:first-child {
        overflow: visible;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_TEMPLATE = "plotly_dark"
ACCENT_SEQ = ["#f53e9a", "#b03ef5", "#6a3ef5", "#39c0ff", "#3ddc97", "#ffb84d"]


# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------
@st.cache_data
def load_data(path: str = "Womens_Clothing_E-Commerce_Reviews.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop the stray index column that ships with this dataset
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Clean up missing categorical values
    for col in ["Division Name", "Department Name", "Class Name"]:
        df[col] = df[col].fillna("Unknown")

    df["Title"] = df["Title"].fillna("")
    df["Review Text"] = df["Review Text"].fillna("")

    df["Recommended"] = df["Recommended IND"].map({1: "Recommended", 0: "Not Recommended"})
    df["Review Length"] = df["Review Text"].str.len()
    df["Word Count"] = df["Review Text"].str.split().str.len().fillna(0)

    bins = [0, 25, 35, 45, 55, 65, 120]
    labels = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
    df["Age Group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)

    return df


df_raw = load_data()

# ----------------------------------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------------------------------
st.sidebar.markdown("## 🔎 Filters")
st.sidebar.caption("Leave a filter empty to include all values. Charts and KPIs update live.")

min_age, max_age = int(df_raw["Age"].min()), int(df_raw["Age"].max())
age_range = st.sidebar.slider(
    "Customer Age",
    min_value=min_age,
    max_value=max_age,
    value=(min_age, max_age),
)

all_divisions = sorted(df_raw["Division Name"].unique())
all_departments = sorted(df_raw["Department Name"].unique())
all_ratings = sorted(df_raw["Rating"].unique())
all_recommended = sorted(df_raw["Recommended"].unique())

divisions = st.sidebar.multiselect(
    "Division",
    options=all_divisions,
    default=[],
    placeholder="All divisions",
)

departments = st.sidebar.multiselect(
    "Department",
    options=all_departments,
    default=[],
    placeholder="All departments",
)

class_pool = df_raw[df_raw["Department Name"].isin(departments)] if departments else df_raw
class_options = sorted(class_pool["Class Name"].unique())
classes = st.sidebar.multiselect(
    "Class",
    options=class_options,
    default=[],
    placeholder="All classes",
)

ratings = st.sidebar.multiselect(
    "Rating",
    options=all_ratings,
    default=[],
    placeholder="All ratings",
)

recommended = st.sidebar.multiselect(
    "Recommended",
    options=all_recommended,
    default=[],
    placeholder="All",
)

st.sidebar.markdown("---")
reset = st.sidebar.button("🔄 Reset all filters", use_container_width=True)
if reset:
    st.rerun()

# ----------------------------------------------------------------------------
# APPLY FILTERS
# (An empty selection means "no filter" — i.e. include every value)
# ----------------------------------------------------------------------------
mask = (
    (df_raw["Age"] >= age_range[0])
    & (df_raw["Age"] <= age_range[1])
    & (df_raw["Division Name"].isin(divisions) if divisions else True)
    & (df_raw["Department Name"].isin(departments) if departments else True)
    & (df_raw["Class Name"].isin(classes) if classes else True)
    & (df_raw["Rating"].isin(ratings) if ratings else True)
    & (df_raw["Recommended"].isin(recommended) if recommended else True)
)

df = df_raw[mask].copy()

# ----------------------------------------------------------------------------
# TITLE
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="dashboard-title">
        <h1>👗 WOMEN'S CLOTHING E-COMMERCE REVIEWS DASHBOARD</h1>
        <p>Interactive overview of customer ratings, sentiment &amp; recommendations across divisions, departments &amp; classes</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if df.empty:
    st.warning("No data matches the selected filters. Try widening your filter selection.")
    st.stop()

# ----------------------------------------------------------------------------
# KPI CALCULATIONS
# ----------------------------------------------------------------------------
total_reviews = len(df)
avg_rating = df["Rating"].mean()
recommend_rate = df["Recommended IND"].mean() * 100
avg_age = df["Age"].mean()
avg_positive_feedback = df["Positive Feedback Count"].mean()


def fmt_num(x: float, decimals: int = 1) -> str:
    return f"{x:,.{decimals}f}"


kpi_cols = st.columns(5)
kpi_data = [
    ("📝 Total Reviews", f"{total_reviews:,}"),
    ("⭐ Avg Rating", f"{avg_rating:.2f} / 5"),
    ("👍 Recommend Rate", f"{recommend_rate:.1f}%"),
    ("🎂 Avg Customer Age", f"{avg_age:.0f} yrs"),
    ("💬 Avg Helpful Votes", f"{avg_positive_feedback:.1f}"),
]
for col, (label, value) in zip(kpi_cols, kpi_data):
    col.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# ROW 1 — Rating Distribution & Recommend Split (full width)
# ----------------------------------------------------------------------------
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown('<div class="chart-title">⭐ Rating Distribution &amp; Recommendation Split</div>', unsafe_allow_html=True)

r1, r2 = st.columns([2, 1])

with r1:
    rating_counts = df["Rating"].value_counts().sort_index().reset_index()
    rating_counts.columns = ["Rating", "Count"]
    fig_rating = px.bar(
        rating_counts, x="Rating", y="Count",
        color="Rating", color_continuous_scale=["#6a3ef5", "#b03ef5", "#f53e9a"],
        text="Count",
    )
    fig_rating.update_traces(textposition="outside")
    fig_rating.update_layout(
        template=PLOTLY_TEMPLATE, height=320, showlegend=False, coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Rating (stars)", showgrid=False, dtick=1),
        yaxis=dict(title="Number of Reviews", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
    )
    st.plotly_chart(fig_rating, use_container_width=True)

with r2:
    rec_counts = df["Recommended"].value_counts().reset_index()
    rec_counts.columns = ["Recommended", "Count"]
    fig_rec = px.pie(
        rec_counts, names="Recommended", values="Count", hole=0.55,
        color="Recommended",
        color_discrete_map={"Recommended": "#3ddc97", "Not Recommended": "#ff5c7a"},
    )
    fig_rec.update_traces(textinfo="percent+label", textfont_size=12)
    fig_rec.update_layout(
        template=PLOTLY_TEMPLATE, height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    st.plotly_chart(fig_rec, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# ROW 2 — Department-wise & Division-wise
# ----------------------------------------------------------------------------
c1, c2 = st.columns(2)

with c1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">🗂️ Avg Rating by Department</div>', unsafe_allow_html=True)
    dept_rating = df.groupby("Department Name")["Rating"].mean().reset_index().sort_values("Rating", ascending=True)
    fig_dept = px.bar(
        dept_rating, x="Rating", y="Department Name", orientation="h",
        color="Department Name", color_discrete_sequence=ACCENT_SEQ, text="Rating",
    )
    fig_dept.update_traces(texttemplate="%{x:.2f}", textposition="outside")
    fig_dept.update_layout(
        template=PLOTLY_TEMPLATE, height=300, showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, title=None, range=[0, 5.5]), yaxis=dict(title=None),
    )
    st.plotly_chart(fig_dept, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">🌎 Review Volume by Division</div>', unsafe_allow_html=True)
    div_counts = df["Division Name"].value_counts().reset_index()
    div_counts.columns = ["Division Name", "Count"]
    fig_div = px.pie(
        div_counts, names="Division Name", values="Count", hole=0.55,
        color_discrete_sequence=ACCENT_SEQ,
    )
    fig_div.update_traces(textinfo="percent+label", textfont_size=12)
    fig_div.update_layout(
        template=PLOTLY_TEMPLATE, height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    st.plotly_chart(fig_div, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# ROW 3 — Top Classes & Sub-Category (Department > Class) Breakdown
# ----------------------------------------------------------------------------
c3, c4 = st.columns(2)

with c3:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">🏆 Top 10 Classes by Review Count</div>', unsafe_allow_html=True)
    top_classes = df["Class Name"].value_counts().reset_index().head(10)
    top_classes.columns = ["Class Name", "Count"]
    fig_top = px.bar(
        top_classes.sort_values("Count"), x="Count", y="Class Name", orientation="h",
        color="Count", color_continuous_scale=["#3a2a6e", "#b03ef5", "#f53e9a"],
    )
    fig_top.update_layout(
        template=PLOTLY_TEMPLATE, height=380, showlegend=False, coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Reviews", showgrid=False), yaxis=dict(title=None),
    )
    st.plotly_chart(fig_top, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c4:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">📊 Review Volume by Department &amp; Class</div>', unsafe_allow_html=True)
    dept_class = (
        df.groupby(["Department Name", "Class Name"]).size()
        .reset_index(name="Count").sort_values("Count", ascending=False)
    )
    fig_sub = px.treemap(
        dept_class, path=["Department Name", "Class Name"], values="Count",
        color="Count", color_continuous_scale=["#1d2130", "#6a3ef5", "#f53e9a"],
    )
    fig_sub.update_layout(
        template=PLOTLY_TEMPLATE, height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_sub, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# ROW 4 — Age Group performance & Review Length vs Rating
# ----------------------------------------------------------------------------
c5, c6 = st.columns(2)

with c5:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">👥 Avg Rating &amp; Recommend Rate by Age Group</div>', unsafe_allow_html=True)
    age_perf = (
        df.groupby("Age Group", observed=True)
        .agg(Avg_Rating=("Rating", "mean"), Recommend_Rate=("Recommended IND", "mean"))
        .reset_index()
    )
    age_perf["Recommend_Rate"] = age_perf["Recommend_Rate"] * 100
    fig_age = go.Figure()
    fig_age.add_trace(go.Bar(
        name="Avg Rating", x=age_perf["Age Group"], y=age_perf["Avg_Rating"],
        marker_color="#f53e9a", yaxis="y1",
    ))
    fig_age.add_trace(go.Scatter(
        name="Recommend Rate (%)", x=age_perf["Age Group"], y=age_perf["Recommend_Rate"],
        mode="lines+markers", line=dict(color="#3ddc97", width=3), yaxis="y2",
    ))
    fig_age.update_layout(
        template=PLOTLY_TEMPLATE, height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(showgrid=False, title=None),
        yaxis=dict(title="Avg Rating", range=[0, 5.5], showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        yaxis2=dict(title="Recommend %", overlaying="y", side="right", range=[0, 105], showgrid=False),
    )
    st.plotly_chart(fig_age, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c6:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">💬 Review Length by Rating</div>', unsafe_allow_html=True)
    fig_len = px.box(
        df, x="Rating", y="Word Count", color="Rating",
        color_discrete_sequence=ACCENT_SEQ,
    )
    fig_len.update_layout(
        template=PLOTLY_TEMPLATE, height=320, showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Rating (stars)", showgrid=False, dtick=1),
        yaxis=dict(title="Word Count", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
    )
    st.plotly_chart(fig_len, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# ROW 5 — Raw data explorer
# ----------------------------------------------------------------------------
with st.expander("🔍 View filtered raw data"):
    st.dataframe(
        df[
            ["Clothing ID", "Age", "Title", "Review Text", "Rating", "Recommended",
             "Positive Feedback Count", "Division Name", "Department Name", "Class Name"]
        ].sort_values("Rating", ascending=False),
        use_container_width=True,
        height=320,
    )
    st.download_button(
        "⬇️ Download filtered data as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_reviews_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown(
    "<p style='text-align:center; color:#5a5f72; font-size:12px; margin-top:10px;'>"
    "Women's Clothing E-Commerce Reviews Dashboard · Built with Streamlit &amp; Plotly</p>",
    unsafe_allow_html=True,
)
