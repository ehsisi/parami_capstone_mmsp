import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.markdown("## 📊 Sales Forecast App")

    with st.expander("👤 Student Info"):
        st.write("**Name:** Eh Si")
        st.write("**Course:** Capstone MMSP")
        st.write("**Project:** Smartstock: a data-drive sales forecasting web-application for grocery retailers")

    st.info("Select a product family and forecast horizon")

# -----------------------------
# LOAD FILES
# -----------------------------

from joblib import load

@st.cache_resource
def load_pipeline():
    return load("sales_pipeline.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("family_df.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

pipeline = load_pipeline()
family_df = load_data()



# -----------------------------
# TITLE
# -----------------------------
st.markdown("""
# 📊 Sales Forecasting Dashboard
Analyze patterns & predict future sales
""")

# -----------------------------
# 📊 EDA SECTION
# -----------------------------

st.header("📊 Exploratory Data Analysis")

eda_df = family_df.copy()
eda_df['family_name'] = eda_df['family']

# -------------------------
# FILTER
# -------------------------
family_labels = sorted(eda_df['family_name'].unique())

selected_family_eda = st.selectbox(
    "Select Product Family for Analysis",
    ["All"] + family_labels
)

if selected_family_eda != "All":
    eda_df = eda_df[eda_df['family_name'] == selected_family_eda]

# -------------------------
# 1. Perishable vs Non-Perishable
# -------------------------
st.subheader("📦 Perishable vs Non-Perishable")

group1 = eda_df.groupby(['family_name', 'perishable'])['unit_sales'].mean().reset_index()

fig1 = px.bar(
    group1,
    x='family_name',
    y='unit_sales',
    color='perishable',
    barmode='group',
    title="Average Sales by Perishable Type"
)

st.plotly_chart(fig1, use_container_width=True)

# -------------------------
# 2. Holiday vs Workday
# -------------------------
st.subheader("🎉 Holiday vs Workday")

group2 = eda_df.groupby(['family_name', 'holiday_type'])['unit_sales'].mean().reset_index()

fig2 = px.bar(
    group2,
    x='family_name',
    y='unit_sales',
    color='holiday_type',
    barmode='group',
    title="Average Sales by Day Type"
)

st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# 3. Promotion vs No Promotion
# -------------------------
st.subheader("🏷️ Promotion vs No Promotion")

group3 = eda_df.groupby(['family_name', 'onpromotion'])['unit_sales'].mean().reset_index()

fig3 = px.bar(
    group3,
    x='family_name',
    y='unit_sales',
    color='onpromotion',
    barmode='group',
    title="Average Sales by Promotion"
)

st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# MODEL INFO
# -----------------------------
st.header("📈 Model Info")
st.success("Model: XGBoost + Pipeline (Auto Encoding)")

# -----------------------------
# FORECAST FUNCTION
# -----------------------------
def forecast_family(pipeline, df, days):
    df = df.copy()
    forecasts = []

    for f in df['family'].unique():
        group = df[df['family'] == f].sort_values('date').copy()

        for _ in range(days):
            last_row = group.iloc[-1:].copy()
            next_date = last_row['date'].values[0] + np.timedelta64(1, 'D')

            new_row = last_row.copy()
            new_row['date'] = next_date

            dt = pd.to_datetime(next_date)
            new_row['day'] = dt.day
            new_row['month'] = dt.month
            new_row['dayofweek'] = dt.dayofweek
            new_row['is_weekend'] = int(dt.dayofweek in [5, 6])

            # Lag features
            new_row['lag_7'] = group['unit_sales'].iloc[-7]
            new_row['lag_14'] = group['unit_sales'].iloc[-14]
            new_row['rolling_7'] = group['unit_sales'].iloc[-7:].mean()

            X_new = new_row.drop(columns=['unit_sales', 'date'])

            # 🔥 PIPELINE handles encoding automatically
            new_row['unit_sales'] = pipeline.predict(X_new)[0]

            group = pd.concat([group, new_row], ignore_index=True)
            forecasts.append(new_row)

    return pd.concat(forecasts)

# -----------------------------
# 🔮 FORECAST SECTION
# -----------------------------
st.header("🔮 Forecasting")

family_labels = sorted(family_df['family'].unique())

selected_family = st.selectbox("Select Product Family", family_labels)
days = st.selectbox("Forecast Horizon (days)", [7, 14, 30])

if st.button("Generate Forecast"):

    forecast = forecast_family(pipeline, family_df, days)

    subset = forecast[forecast['family'] == selected_family]

    st.subheader(f"Forecast for {selected_family}")

    # 📈 Line Chart
    fig = px.line(
        subset,
        x="date",
        y="unit_sales",
        title="Future Sales Prediction"
    )

    st.plotly_chart(fig, use_container_width=True)

    # 📋 Table
    st.dataframe(subset[['date', 'unit_sales']])

    # 📥 Download
    csv = subset.to_csv(index=False).encode('utf-8')
    st.download_button("Download Forecast CSV", csv, "forecast.csv", "text/csv")