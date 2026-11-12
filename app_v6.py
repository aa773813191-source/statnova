
# ============================================================
# STATNOVA v7
# Intelligent Research Analytics
# Developed by Amatalelah
# ============================================================

import io
import secrets
from datetime import datetime
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import requests

from scipy import stats
from sklearn.linear_model import LinearRegression

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="STATNOVA | Research Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------- DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background-color: #f5f8fc;
}
.main-title {
    color: #102c50;
    font-size: 42px;
    font-weight: 800;
}
.subtitle {
    color: #3986bd;
    font-size: 17px;
}
div[data-testid="stMetric"] {
    background: white;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #dce7f2;
}
section[data-testid="stSidebar"] {
    background-color: #102c50;
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------- SESSION STATE ----------------

if "data" not in st.session_state:
    st.session_state.data = None

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = []

if "orcid_id" not in st.session_state:
    st.session_state.orcid_id = None

if "orcid_name" not in st.session_state:
    st.session_state.orcid_name = None

if "orcid_state" not in st.session_state:
    st.session_state.orcid_state = None


# ---------------- ORCID CONFIGURATION ----------------

def get_orcid_config():
    try:
        return st.secrets["orcid"]
    except Exception:
        return None


def get_orcid_base(config):
    environment = config.get("environment", "production")

    if environment == "sandbox":
        return "https://sandbox.orcid.org"

    return "https://orcid.org"


def process_orcid_callback():

    config = get_orcid_config()

    if not config:
        return

    params = st.query_params

    if params.get("error"):
        st.error("ORCID authorization was not completed.")
        st.session_state.orcid_state = None
        st.query_params.clear()
        return

    code = params.get("code")

    if not code:
        return

    returned_state = params.get("state", "")
    expected_state = st.session_state.get("orcid_state")

    if not expected_state or not secrets.compare_digest(
        str(returned_state), str(expected_state)
    ):
        st.error("ORCID security verification failed. Please reconnect.")
        st.session_state.orcid_state = None
        st.query_params.clear()
        return

    try:
        base_url = get_orcid_base(config)

        response = requests.post(
            f"{base_url}/oauth/token",
            data={
                "client_id": config["client_id"],
                "client_secret": config["client_secret"],
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": config["redirect_uri"]
            },
            headers={"Accept": "application/json"},
            timeout=20
        )

        response.raise_for_status()
        token_data = response.json()

        orcid_id = token_data.get("orcid")

        if not orcid_id:
            raise ValueError("ORCID iD was not returned.")

        st.session_state.orcid_id = orcid_id
        st.session_state.orcid_name = token_data.get("name", "")
        st.session_state.orcid_state = None

        # The access token is deliberately not stored.
        st.query_params.clear()

        st.success("ORCID connected successfully.")

    except Exception:
        st.error(
            "ORCID connection failed. Check the credentials "
            "and redirect URI in Streamlit Secrets."
        )
        st.session_state.orcid_state = None
        st.query_params.clear()


process_orcid_callback()


# ---------------- DATA FUNCTIONS ----------------

def load_file(uploaded_file):

    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin1")

    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file format.")


def numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def categorical_columns(df):
    return df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()


def clean_numeric(series):
    return pd.to_numeric(series, errors="coerce").dropna()


def descriptive_statistics(df):

    nums = df[numeric_columns(df)]

    if nums.empty:
        return pd.DataFrame()

    result = nums.describe().T

    result["median"] = nums.median()
    result["variance"] = nums.var()
    result["skewness"] = nums.skew()
    result["kurtosis"] = nums.kurtosis()

    return result.round(4)


def normality_test(series):

    x = clean_numeric(series)

    if len(x) < 3:
        return None

    if len(x) > 5000:
        x = x.sample(5000, random_state=42)

    return stats.shapiro(x)


def add_result(title, statistic, p_value, interpretation):

    result = {
        "Analysis": title,
        "Statistic": float(statistic),
        "P-value": float(p_value),
        "Interpretation": interpretation
    }

    st.session_state.analysis_results.append(result)

    return result


def interpret_p(p):
    if p < 0.05:
        return "Statistically significant at alpha = 0.05."
    return "Not statistically significant at alpha = 0.05."


# ---------------- EXPORT FUNCTIONS ----------------

def create_excel(df, description, results):

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:

        df.to_excel(writer, sheet_name="Data", index=False)

        if not description.empty:
            description.to_excel(
                writer, sheet_name="Descriptive Statistics"
            )

        if results:
            pd.DataFrame(results).to_excel(
                writer, sheet_name="Analysis Results", index=False
            )

    output.seek(0)
    return output


def create_pdf(description, results):

    output = io.BytesIO()

    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=35,
        leftMargin=35,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(
        "STATNOVA - Research Analytics Report",
        styles["Title"]
    ))

    story.append(Spacer(1, 12))

    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        styles["Normal"]
    ))

    story.append(Spacer(1, 18))

    story.append(Paragraph(
        "Descriptive Statistics",
        styles["Heading2"]
    ))

    if not description.empty:

        table_data = [
            ["Variable"] + list(description.columns)
        ]

        for index, row in description.iterrows():
            table_data.append(
                [str(index)] + [str(v) for v in row.tolist()]
            )

        table = Table(table_data, repeatRows=1)

        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#102c50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
        ]))

        story.append(table)

    story.append(Spacer(1, 20))

    story.append(Paragraph(
        "Statistical Analysis Results",
        styles["Heading2"]
    ))

    for result in results:

        paragraph = (
            f"<b>{result['Analysis']}</b><br/>"
            f"Statistic: {result['Statistic']:.4f}<br/>"
            f"P-value: {result['P-value']:.6f}<br/>"
            f"{result['Interpretation']}"
        )

        story.append(Paragraph(paragraph, styles["Normal"]))
        story.append(Spacer(1, 10))

    document.build(story)

    output.seek(0)
    return output


# ---------------- SIDEBAR ----------------

st.sidebar.markdown("# 📊 STATNOVA")
st.sidebar.caption("Intelligent Research Analytics")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Upload & Explore",
        "Data Cleaning",
        "Descriptive Statistics",
        "Statistical Tests",
        "Correlation",
        "Regression",
        "Visualization",
        "Reports & Export",
        "Researcher Profile",
        "About"
    ]
)


# ---------------- HOME ----------------

if page == "Home":

    st.markdown(
        '<div class="main-title">STATNOVA</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Intelligent Research Analytics'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("### Welcome to STATNOVA")

    st.write(
        "A research-oriented platform for data management, "
        "statistical analysis, visualization, and scientific reporting."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Analysis Modules", "10")
    col2.metric("Supported Formats", "CSV / Excel")
    col3.metric("Research Workflow", "Integrated")

    st.info(
        "Upload your dataset to begin exploring and analyzing "
        "your research data."
    )

    st.write("### Main capabilities")

    st.markdown("""
    - Data upload and exploration
    - Data cleaning and preparation
    - Descriptive statistics
    - Parametric and non-parametric tests
    - Correlation and regression
    - Scientific visualization
    - Excel and PDF reports
    - ORCID researcher identity connection
    """)


# ---------------- UPLOAD & EXPLORE ----------------

elif page == "Upload & Explore":

    st.title("Upload & Explore Data")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file:

        try:
            df = load_file(uploaded_file)
            st.session_state.data = df
            st.session_state.analysis_results = []

            st.success("Dataset uploaded successfully.")

        except Exception as error:
            st.error(f"Could not read the file: {error}")

    df = st.session_state.data

    if df is not None:

        st.subheader("Dataset Preview")
        st.dataframe(df.head(100), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Rows", len(df))
        c2.metric("Columns", len(df.columns))
        c3.metric("Numeric Variables", len(numeric_columns(df)))
        c4.metric("Missing Values", int(df.isna().sum().sum()))

        st.subheader("Column Information")

        info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isna().sum().values,
            "Unique Values": df.nunique().values
        })

        st.dataframe(info, use_container_width=True)


# ---------------- DATA CLEANING ----------------

elif page == "Data Cleaning":

    st.title("Data Cleaning")

    df = st.session_state.data

    if df is None:
        st.warning("Upload a dataset first.")

    else:

        st.write("### Missing values")
        st.dataframe(df.isna().sum().to_frame("Missing Values"))

        missing_method = st.selectbox(
            "Choose missing-value treatment",
            [
                "Keep missing values",
                "Remove rows with missing values",
                "Fill numeric missing values with mean",
                "Fill numeric missing values with median"
            ]
        )

        remove_duplicates = st.checkbox(
            "Remove duplicate rows",
            value=False
        )

        if st.button("Apply Cleaning"):

            cleaned = df.copy()

            if missing_method == "Remove rows with missing values":
                cleaned = cleaned.dropna()

            elif missing_method == "Fill numeric missing values with mean":
                for col in numeric_columns(cleaned):
                    cleaned[col] = cleaned[col].fillna(
                        cleaned[col].mean()
                    )

            elif missing_method == "Fill numeric missing values with median":
                for col in numeric_columns(cleaned):
                    cleaned[col] = cleaned[col].fillna(
                        cleaned[col].median()
                    )

            if remove_duplicates:
                cleaned = cleaned.drop_duplicates()

            st.session_state.data = cleaned

            st.success("Cleaning completed.")
            st.dataframe(cleaned.head(100), use_container_width=True)


# ---------------- DESCRIPTIVE STATISTICS ----------------

elif page == "Descriptive Statistics":

    st.title("Descriptive Statistics")

    df = st.session_state.data

    if df is None:
        st.warning("Upload a dataset first.")

    else:

        description = descriptive_statistics(df)

        if description.empty:
            st.warning("No numeric variables were found.")
        else:
            st.dataframe(description, use_container_width=True)

            st.download_button(
                "Download Descriptive Statistics CSV",
                description.to_csv().encode("utf-8"),
                "descriptive_statistics.csv",
                "text/csv"
            )


# ---------------- STATISTICAL TESTS ----------------

elif page == "Statistical Tests":

    st.title("Statistical Tests")

    df = st.session_state.data

    if df is None:
        st.warning("Upload a dataset first.")

    else:

        nums = numeric_columns(df)

        if not nums:
            st.warning("No numeric variables were found.")

        else:

            test = st.selectbox(
                "Select statistical test",
                [
                    "Independent Samples T-test",
                    "Paired Samples T-test",
                    "One Sample T-test",
                    "Mann-Whitney U",
                    "Wilcoxon Signed-Rank",
                    "One-way ANOVA",
                    "Kruskal-Wallis"
                ]
            )

            if test in [
                "One-way ANOVA",
                "Kruskal-Wallis"
            ]:

                group_col = st.selectbox(
                    "Grouping variable",
                    categorical_columns(df)
                ) if categorical_columns(df) else None

                value_col = st.selectbox(
                    "Numeric variable",
                    nums
                )

                if group_col and st.button("Run Test"):

                    groups = [
                        clean_numeric(group[value_col]).values
                        for _, group in df.groupby(group_col)
                    ]

                    groups = [g for g in groups if len(g) > 0]

                    if len(groups) < 2:
                        st.error("At least two non-empty groups are required.")

                    elif test == "One-way ANOVA":

                        result = stats.f_oneway(*groups)

                        add_result(
                            test,
                            result.statistic,
                            result.pvalue,
                            interpret_p(result.pvalue)
                        )

                        st.write(result)

                    else:

                        result = stats.kruskal(*groups)

                        add_result(
                            test,
                            result.statistic,
                            result.pvalue,
                            interpret_p(result.pvalue)
                        )

                        st.write(result)

            else:

                x_col = st.selectbox(
                    "First variable",
                    nums,
                    key="test_x"
                )

                if test in [
                    "Independent Samples T-test",
                    "Mann-Whitney U"
                ]:

                    group_col = st.selectbox(
                        "Grouping variable",
                        categorical_columns(df),
                        key="test_group"
                    ) if categorical_columns(df) else None

                    if group_col and st.button("Run Test"):

                        categories = df[group_col].dropna().unique()

                        if len(categories) != 2:
                            st.error(
                                "This test requires exactly two groups."
                            )

                        else:

                            a = clean_numeric(
                                df.loc[df[group_col] == categories[0], x_col]
                            )

                            b = clean_numeric(
                                df.loc[df[group_col] == categories[1], x_col]
                            )

                            if len(a) < 2 or len(b) < 2:
                                st.error(
                                    "Each group needs at least two observations."
                                )

                            else:

                                if test == "Independent Samples T-test":
                                    result = stats.ttest_ind(
                                        a, b, equal_var=False
                                    )
                                else:
                                    result = stats.mannwhitneyu(
                                        a, b, alternative="two-sided"
                                    )

                                add_result(
                                    test,
                                    result.statistic,
                                    result.pvalue,
                                    interpret_p(result.pvalue)
                                )

                                st.write(result)

                                st.caption(
                                    f"Groups compared: {categories[0]} and "
                                    f"{categories[1]}"
                                )

                elif test in [
                    "Paired Samples T-test",
                    "Wilcoxon Signed-Rank"
                ]:

                    y_col = st.selectbox(
                        "Second / paired variable",
                        [c for c in nums if c != x_col]
                    )

                    if st.button("Run Test"):

                        paired = df[[x_col, y_col]].dropna()

                        if len(paired) < 2:
                            st.error("At least two complete pairs are required.")

                        else:

                            a = paired[x_col]
                            b = paired[y_col]

                            if test == "Paired Samples T-test":
                                result = stats.ttest_rel(a, b)
                            else:
                                result = stats.wilcoxon(a, b)

                            add_result(
                                test,
                                result.statistic,
                                result.pvalue,
                                interpret_p(result.pvalue)
                            )

                            st.write(result)

 
