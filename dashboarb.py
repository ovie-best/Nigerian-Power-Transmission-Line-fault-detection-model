import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ─── Page Config ───
st.set_page_config(
    page_title="Nigerian Transmission Line Fault Dashboard",
    page_icon="⚡",
    layout="wide"
)

# ─── Title ───
st.title("⚡ Nigerian Transmission Line Fault Prediction Dashboard")
st.markdown("Visualizing predicted transmission line faults — **April–June 2025**")

# ─── Load Data ───
@st.cache_data
def load_data():
    df = pd.read_csv("nigerian_test_data_with_predictions.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df

df = load_data()

# ─── Label map (consistent with A.1/A.2) ───
label_map = {
    0: "Class 0 – Normal A",
    1: "Class 1 – Normal B",
    2: "Class 2 – LG Fault",
    3: "Class 3 – LL Fault",
    4: "Class 4 – LLG Fault",
    5: "Class 5 – Healthy"
}
df["Fault_Label"] = df["Predicted_Fault_Type"].map(label_map)

# ─── Sidebar Filters ───
st.sidebar.header("Filter Options")

start_date = st.sidebar.date_input("Start Date", df["Timestamp"].min().date())
end_date   = st.sidebar.date_input("End Date",   df["Timestamp"].max().date())

all_fault_labels = sorted(df["Fault_Label"].unique())
selected_labels  = st.sidebar.multiselect(
    "Filter by Fault Type",
    options=all_fault_labels,
    default=all_fault_labels
)

# ─── Filter ───
mask = (
    (df["Timestamp"].dt.date >= start_date) &
    (df["Timestamp"].dt.date <= end_date) &
    (df["Fault_Label"].isin(selected_labels))
)
filtered_df = df.loc[mask].copy()

st.markdown(f"Showing **{len(filtered_df):,}** records from **{start_date}** to **{end_date}**")

# ─── KPI Cards ───
st.subheader("Summary Metrics")
col1, col2, col3, col4 = st.columns(4)

total       = len(filtered_df)
fault_mask  = filtered_df["Predicted_Fault_Type"].isin([2, 3, 4])
fault_count = fault_mask.sum()
normal_count = (~fault_mask).sum()
fault_rate  = 100 * fault_count / total if total > 0 else 0

col1.metric("Total Records",    f"{total:,}")
col2.metric("Fault Events",     f"{fault_count:,}")
col3.metric("Normal/Healthy",   f"{normal_count:,}")
col4.metric("Fault Rate",       f"{fault_rate:.1f}%")

st.divider()

# ─── Row 1: Fault trend + Distribution ───
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Fault Frequency Over Time")
    fault_trend = (
        filtered_df
        .groupby(filtered_df["Timestamp"].dt.date)["Predicted_Fault_Type"]
        .value_counts()
        .unstack(fill_value=0)
    )
    fault_trend.columns = [label_map.get(c, c) for c in fault_trend.columns]
    st.line_chart(fault_trend)

with col_right:
    st.subheader("Fault Type Distribution")
    fault_counts = filtered_df["Fault_Label"].value_counts().sort_index()

    fig1, ax1 = plt.subplots(figsize=(5, 4))
    colors = ['#2c5f8a' if 'Fault' in l else '#7fb3d3' for l in fault_counts.index]
    bars = ax1.barh(fault_counts.index, fault_counts.values, color=colors, edgecolor='none')
    for bar, val in zip(bars, fault_counts.values):
        ax1.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                 str(val), va='center', fontsize=9)
    ax1.set_xlabel("Count")
    ax1.set_facecolor('#f8f8f8')
    ax1.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    ax1.tick_params(left=False)
    fig1.patch.set_facecolor('#f8f8f8')
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

st.divider()

# ─── Row 2: Electrical signals ───
st.subheader("Electrical Signal Overview (Filtered Period)")

features = ['Ia', 'Ib', 'Ic', 'Va']
available = [f for f in features if f in filtered_df.columns]

if available:
    fig2, axes = plt.subplots(1, len(available), figsize=(14, 3))
    if len(available) == 1:
        axes = [axes]
    for ax, feat in zip(axes, available):
        ax.plot(filtered_df["Timestamp"].values[:500],
                filtered_df[feat].values[:500],
                linewidth=0.6, color='royalblue')
        ax.set_title(feat, fontsize=11)
        ax.set_facecolor('#f8f8f8')
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.spines[['top', 'right']].set_visible(False)
    fig2.suptitle("Electrical Signals (first 500 samples of filtered range)",
                  fontsize=11, y=1.02)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()
else:
    st.info("Electrical feature columns (Ia, Ib, Ic, Va) not found in dataset.")

st.divider()

# ─── Row 3: Fault breakdown table ───
st.subheader("Predicted Fault Type Breakdown")
breakdown = (
    filtered_df["Fault_Label"]
    .value_counts()
    .rename_axis("Fault Type")
    .reset_index(name="Count")
)
breakdown["Percentage"] = (100 * breakdown["Count"] / total).round(2).astype(str) + "%"
st.dataframe(breakdown, use_container_width=True)

st.divider()

# ─── Row 4: Raw data + download ───
st.subheader("Raw Predictions")
st.dataframe(filtered_df[["Timestamp"] + available + ["Predicted_Fault_Type", "Fault_Label"]].head(100),
             use_container_width=True)

st.subheader("Download Filtered Data")
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇ Download Filtered Data as CSV",
    data=csv,
    file_name="filtered_fault_predictions.csv",
    mime="text/csv"
)