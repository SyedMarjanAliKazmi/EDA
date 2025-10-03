import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import calendar
import streamlit as st

# ===============================
# Streamlit App Setup
# ===============================
st.set_page_config(page_title="Ecommerce EDA Dashboard", layout="wide")
st.title("Ecommerce Data Analysis Dashboard")

# ===============================
# File Uploader
# ===============================
uploaded_file = st.file_uploader("Upload your Ecommerce CSV file", type=["csv"])

if uploaded_file is not None:
    # Read dataset
    df = pd.read_csv(uploaded_file)

    # ===============================
    # Data Understanding
    # ===============================
    st.subheader("Data Preview")
    st.write(df.head())
    st.write("Shape of dataset:", df.shape)

    st.subheader("Missing & Duplicate Values")
    st.write("Missing Values:", df.isnull().sum())
    st.write("Duplicate Records:", df.duplicated().sum())

    # ===============================
    # Data Cleaning
    # ===============================
    if "price" in df.columns:
        df["price"] = df["price"].astype(float)

    if "quantity" in df.columns and "discount" in df.columns:
        df["sales"] = (df["price"] * df["quantity"] * (1 - df["discount"])).astype(float)

    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        df["month"] = df["order_date"].dt.month
        df["month_name"] = df["month"].apply(lambda x: calendar.month_abbr[x] if pd.notnull(x) else x)
        df["day_name"] = df["order_date"].dt.day_name()

    # ===============================
    # Visualizations
    # ===============================
    st.subheader("Visualizations")

    # 1. Orders by Category
    if "category" in df.columns:
        st.write("Orders by Category")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df, x="category", palette="Set2", ax=ax)
        ax.set_title("Number of Orders by Category")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # 2. Sales Share by Region (Donut chart)
    if "region" in df.columns and "sales" in df.columns:
        region_sales = df.groupby("region")["sales"].sum().reset_index()
        fig = px.pie(region_sales, names="region", values="sales",
                     title="Sales Contribution by Region",
                     hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig)

    # 3. Quantity Sold by Category (Boxplot)
    if "category" in df.columns and "quantity" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x="category", y="quantity", palette="coolwarm", ax=ax)
        ax.set_title("Distribution of Quantity Sold by Category")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # 4. Sales Distribution by Category
    if "category" in df.columns and "sales" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=df, x="sales", hue="category", kde=True, element="step", ax=ax)
        ax.set_title("Sales Distribution by Category")
        st.pyplot(fig)

    # 5. Category vs Payment Method (Heatmap)
    if "category" in df.columns and "payment_method" in df.columns:
        heatmap_grouped = pd.crosstab(df["category"], df["payment_method"])
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(heatmap_grouped, cmap="YlGnBu", annot=True, fmt="d", ax=ax)
        ax.set_title("Category vs Payment Method")
        st.pyplot(fig)

    # 6. Sales by Weekday
    if "day_name" in df.columns and "sales" in df.columns:
        weekday_sales = df.groupby("day_name")["sales"].sum().reset_index()
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=weekday_sales, x="day_name", y="sales", order=order, palette="mako", ax=ax)
        ax.set_title("Total Sales by Weekday")
        st.pyplot(fig)

    # 7. Sales by Category Across Regions
    if "category" in df.columns and "region" in df.columns and "sales" in df.columns:
        cat_region_sales = df.groupby(["category", "region"])["sales"].sum().reset_index()
        fig = px.bar(cat_region_sales, x="category", y="sales", color="region", barmode="stack",
                     title="Sales by Category Across Regions")
        st.plotly_chart(fig)

    # 8. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

    # 9. Scatter Plot
    if "price" in df.columns and "sales" in df.columns:
        fig = px.scatter(df, x="price", y="sales", color="category" if "category" in df.columns else None,
                         size="quantity" if "quantity" in df.columns else None,
                         hover_data=["region", "payment_method"] if "region" in df.columns and "payment_method" in df.columns else None,
                         title="Price vs Sales by Category")
        st.plotly_chart(fig)

else:
    st.warning("Please upload a CSV file to begin analysis.")
