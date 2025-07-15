import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from PIL import Image
import base64

# === PAGE SETUP ===
st.set_page_config(page_title="Dymra Market Research", layout="wide", page_icon="📈")

# === LOGO + TITLE ===
def show_logo_and_title():
    with open("dymra_logo.jpg", "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    st.markdown(f"""
        <div style='background-color: #0e1117; padding: 25px 15px; display: flex; align-items: center;'>
            <img src='data:image/jpg;base64,{encoded}' style='height: 80px; margin-right: 25px;'/>
            <div>
                <h1 style='color: white; margin: 0;'>HS Code Classifier</h1>
                <p style='color: #cfcfcf; margin-top: 5px;'>Enter product description</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

show_logo_and_title()

# === MAIN TITLE ===
st.title("📈 AI-Powered Market Research – Dymra Tech")
st.caption("Analyze trade leads, uncover trends, and find market expansion opportunities.")

# === FILE UPLOAD ===
uploaded_file = st.file_uploader("📤 Upload your trade leads CSV", type="csv")

# === LOAD DATA ===
df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        st.stop()
else:
    try:
        df = pd.read_csv("extended_trade_leads.csv")
        st.info("Using demo data from extended_trade_leads.csv")
    except FileNotFoundError:
        st.error("No uploaded file and 'extended_trade_leads.csv' not found. Please upload a CSV to proceed.")
        st.stop()

# === CLEAN & FILTER ===
if df is not None:
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    st.sidebar.header("🔍 Filter Leads")
    regions = ["All"] + sorted(df["region"].dropna().unique())
    products = ["All"] + sorted(df["product"].dropna().unique())

    selected_region = st.sidebar.selectbox("🌍 Region", regions)
    selected_product = st.sidebar.selectbox("📦 Product", products)

    filtered_df = df.copy()
    if selected_region != "All":
        filtered_df = filtered_df[filtered_df["region"] == selected_region]
    if selected_product != "All":
        filtered_df = filtered_df[filtered_df["product"] == selected_product]

    # === DISPLAY TABLE ===
    st.subheader("📋 Filtered Trade Leads")
    st.dataframe(filtered_df, use_container_width=True)

    # === DOWNLOAD CSV ===
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Filtered Leads", csv, "filtered_leads.csv", "text/csv")

    # === TRADE VALUE CHART ===
    if "annual_trade_value_(usd)" in filtered_df.columns:
        st.subheader("💰 Trade Value by Region")
        value_chart = filtered_df.groupby("region")["annual_trade_value_(usd)"].sum().reset_index()
        bar_fig = px.bar(value_chart, x="region", y="annual_trade_value_(usd)", color="region",
                         title="Total Annual Trade Value by Region")
        st.plotly_chart(bar_fig, use_container_width=True)

    # === LEAD DISTRIBUTION ===
    st.subheader("📊 Lead Distribution by Region")
    region_counts = filtered_df["region"].value_counts().reset_index()
    region_counts.columns = ["region", "count"]
    pie_fig = px.pie(region_counts, names="region", values="count", title="Leads by Region")
    st.plotly_chart(pie_fig, use_container_width=True)

    # === TOP PRODUCTS ===
    st.subheader("📦 Top Products by Lead Count")
    product_stats = filtered_df["product"].value_counts().head(10).reset_index()
    product_stats.columns = ["Product", "Leads"]
    prod_fig = px.bar(product_stats, x="Product", y="Leads", color="Product", title="Top 10 Products")
    st.plotly_chart(prod_fig, use_container_width=True)

    # === DEMAND FORECASTING ===
    st.subheader("⏰ Product Demand Forecast (Experimental)")
    forecast_product = st.selectbox("Select Product for Forecasting", df["product"].dropna().unique())
    df_forecast = df[df["product"] == forecast_product]

    if not df_forecast.empty and "year" in df_forecast.columns:
        yearly = df_forecast.groupby("year")["annual_trade_value_(usd)"].sum().reset_index()
        yearly.columns = ["ds", "y"]
        try:
            model = Prophet()
            model.fit(yearly)
            future = model.make_future_dataframe(periods=3, freq='Y')
            forecast = model.predict(future)
            fig_forecast = px.line(forecast, x='ds', y='yhat', title=f"Forecasted Demand for {forecast_product}")
            st.plotly_chart(fig_forecast, use_container_width=True)
        except Exception as e:
            st.warning("Forecasting failed. Check if there’s enough data.")
            st.text(str(e))
    else:
        st.info("No yearly trade data available for this product.")
















