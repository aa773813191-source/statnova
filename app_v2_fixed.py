import io
import math
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import (
    ttest_ind, ttest_rel, ttest_1samp, f_oneway, mannwhitneyu,
    wilcoxon, kruskal, chi2_contingency, pearsonr, spearmanr, kendalltau
)
from sklearn.linear_model import LinearRegression

# STATNOVA — development build
st.set_page_config(
    page_title="STATNOVA | Intelligent Research Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
:root { color-scheme: light; }
.stApp { background: linear-gradient(135deg,#f7fbff 0%,#f7fff9 55%,#fff8fc 100%); }
.block-container { padding-top: 1.5rem; max-width: 1250px; }
.hero {
  padding: 1.6rem; border-radius: 22px;
  background: linear-gradient(110deg,#e7f4ff,#eaf9f0,#fff0f7);
  border: 1px solid #dcecf2; margin-bottom: 1rem;
}
.hero h1 { color:#244e68; margin:0; font-size:2.35rem; }
.hero p { color:#496779; margin:.35rem 0 0 0; }
h2,h3 { color:#315d70; }
div[data-testid="stMetric"] {
  background:#ffffffcc; border:1px solid #e1edf0;
  padding:12px; border-radius:14px;
}
.small-note { color:#58717d; font-size:.9rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>STATNOVA</h1>
  <p><b>Intelligent Research Analytics</b> · A practical workspace for research data</p>
  <p>Developed by <b>Amatalelah Abdulazez Mashhor Al-Ahdel</b> · M.Sc. Physics</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("📊 STATNOVA")
    page = st.radio(
        "Workspace",
        ["Home", "Upload & Explore", "Data Quality", "Descriptive Statistics",
         "Statistical Tests", "Relationships & Regression", "Visualizations",
         "Reports & Export", "About"]
    )
    st.divider()
    st.caption("Development build · Verify results before research or clinical use.")

def load_data(uploaded):
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded)
        except UnicodeDecodeError:
            uploaded.seek(0)
            return pd.read_csv(uploaded, encoding="latin-1")
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded)
    raise ValueError("Supported formats are CSV and Excel (.xlsx, .xls).")

def numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()

def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan

def p_text(p):
    if p is None or not np.isfinite(p):
        return "Not available"
    return f"{p:.5g}"

def result_card(title, statistic, pvalue, note=""):
    st.subheader(title)
    c1, c2 = st.columns(2)
    c1.metric("Test statistic", f"{statistic:.5g}" if np.isfinite(statistic) else "N/A")
    c2.metric("p-value", p_text(pvalue))
    if np.isfinite(pvalue):
        st.info("At α = 0.05, the result is statistically significant." if pvalue < .05
                else "At α = 0.05, the result is not statistically significant.")
    if note:
        st.caption(note)

if "data" not in st.session_state:
    st.session_state.data = None
if "source_name" not in st.session_state:
    st.session_state.source_name = ""

if page == "Home":
    st.title("Welcome to STATNOVA")
    st.write(
        "A research-focused platform for importing datasets, exploring data quality, "
        "summarizing variables, running selected statistical tests, visualizing results, "
        "and exporting working reports."
    )
    a, b, c = st.columns(3)
    a.metric("Analysis areas", "7")
    b.metric("Input formats", "CSV / Excel")
    c.metric("Interface", "English")
    st.subheader("Getting started")
    st.markdown("""
    1. Open **Upload & Explore** and import a CSV or Excel file.
    2. Review **Data Quality** before analysis.
    3. Explore **Descriptive Statistics** and select an appropriate test.
    4. Create figures and export a report.
    """)
    st.warning("STATNOVA is an early development build. It does not replace expert statistical review.")

elif page == "Upload & Explore":
    st.title("Upload & Explore")
    uploaded = st.file_uploader("Upload a CSV or Excel dataset", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        try:
            df = load_data(uploaded)
            st.session_state.data = df
            st.session_state.source_name = uploaded.name
            st.success(f"Loaded: {uploaded.name}")
        except Exception as exc:
            st.error(f"Could not read this file: {exc}")
    df = st.session_state.data
    if df is not None:
        st.subheader("Dataset preview")
        st.dataframe(df.head(100), use_container_width=True)
        x, y, z = st.columns(3)
        x.metric("Rows", f"{df.shape[0]:,}")
        y.metric("Columns", f"{df.shape[1]:,}")
        z.metric("Numeric columns", len(numeric_columns(df)))
        st.caption(f"Current dataset: {st.session_state.source_name}")
    else:
        st.info("Choose a CSV or Excel file to begin.")

elif page == "Data Quality":
    st.title("Data Quality Review")
    df = st.session_state.data
    if df is None:
        st.info("Upload a dataset first.")
    else:
        missing = df.isna().sum()
        dup = int(df.duplicated().sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Duplicate rows", f"{dup:,}")
        c3.metric("Missing cells", f"{int(missing.sum()):,}")
        quality = pd.DataFrame({
            "Column": df.columns.astype(str),
            "Data type": df.dtypes.astype(str).values,
            "Missing count": missing.values,
            "Missing %": (missing.values / max(len(df), 1) * 100).round(2)
        })
        st.dataframe(quality, use_container_width=True)
        nums = numeric_columns(df)
        if nums:
            st.subheader("Potential outliers (IQR screening)")
            col = st.selectbox("Numeric variable", nums)
            series = df[col].dropna()
            if len(series) >= 4:
                q1, q3 = series.quantile(.25), series.quantile(.75)
                iqr = q3 - q1
                low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                flagged = series[(series < low) | (series > high)]
                st.write(f"Lower screening bound: **{low:.5g}** · Upper screening bound: **{high:.5g}**")
                st.metric("Flagged observations", len(flagged))
                if len(flagged):
                    st.dataframe(flagged.rename("Flagged value").to_frame(), use_container_width=True)
            else:
                st.caption("At least four non-missing numeric observations are needed for this screening.")
        st.warning("Missing values, duplicates, and IQR flags require context. Do not delete observations automatically.")

elif page == "Descriptive Statistics":
    st.title("Descriptive Statistics")
    df = st.session_state.data
    if df is None:
        st.info("Upload a dataset first.")
    else:
        nums = numeric_columns(df)
        if not nums:
            st.warning("No numeric columns were detected.")
        else:
            selected = st.multiselect("Choose numeric variables", nums, default=nums[:min(5, len(nums))])
            if selected:
                summary = df[selected].describe().T
                summary["median"] = df[selected].median()
                summary["variance"] = df[selected].var()
                summary["skewness"] = df[selected].skew()
                summary["kurtosis"] = df[selected].kurtosis()
                st.dataframe(summary.round(5), use_container_width=True)
                st.download_button(
                    "Download descriptive statistics (CSV)",
                    summary.to_csv().encode("utf-8-sig"),
                    "statnova_descriptive_statistics.csv",
                    "text/csv"
                )

elif page == "Statistical Tests":
    st.title("Statistical Tests")
    df = st.session_state.data
    if df is None:
        st.info("Upload a dataset first.")
    else:
        nums = numeric_columns(df)
        if not nums:
            st.warning("No numeric columns were detected.")
        else:
            test = st.selectbox("Select a test", [
                "Independent samples t-test (Welch)",
                "Paired samples t-test",
                "One-sample t-test",
                "One-way ANOVA",
                "Mann–Whitney U",
                "Wilcoxon signed-rank",
                "Kruskal–Wallis",
                "Chi-square test of independence"
            ])
            alpha = st.number_input("Significance level (α)", min_value=.001, max_value=.20,
                                    value=.05, step=.005, format="%.3f")
            if test == "Chi-square test of independence":
                cats = df.select_dtypes(exclude=np.number).columns.tolist()
                if len(cats) < 2:
                    st.warning("This test requires two categorical columns.")
                else:
                    c1, c2 = st.columns(2)
                    v1 = c1.selectbox("First categorical variable", cats, key="chi1")
                    v2 = c2.selectbox("Second categorical variable", [x for x in cats if x != v1], key="chi2")
                    if st.button("Run chi-square test"):
                        table = pd.crosstab(df[v1], df[v2])
                        chi, p, dof, expected = chi2_contingency(table)
                        st.dataframe(table, use_container_width=True)
                        result_card("Chi-square test", chi, p, f"Degrees of freedom: {dof}. Check expected cell counts.")
                        st.caption(f"Smallest expected frequency: {expected.min():.3g}")
            else:
                col1, col2 = st.columns(2)
                xcol = col1.selectbox("Numeric variable", nums, key="test_x")
                groups = df.select_dtypes(exclude=np.number).columns.tolist()
                group_col = None
                if test not in ["One-sample t-test"]:
                    group_col = col2.selectbox("Grouping variable", groups, key="test_group") if groups else None
                if test == "One-sample t-test":
                    mu = st.number_input("Reference mean", value=0.0)
                elif test == "Paired samples t-test" or test == "Wilcoxon signed-rank":
                    second = st.selectbox("Second paired numeric variable", [x for x in nums if x != xcol])
                elif test in ["Independent samples t-test (Welch)", "Mann–Whitney U"]:
                    if group_col:
                        levels = df[group_col].dropna().unique().tolist()
                        if len(levels) >= 2:
                            g1, g2 = st.columns(2)
                            level1 = g1.selectbox("Group 1", levels, key="g1")
                            level2 = g2.selectbox("Group 2", [v for v in levels if v != level1], key="g2")
                        else:
                            levels = []
                    else:
                        levels = []
                elif test in ["One-way ANOVA", "Kruskal–Wallis"] and group_col:
                    st.caption("The selected grouping column should contain at least two independent groups.")
                if st.button("Run selected test"):
                    try:
                        if test == "One-sample t-test":
                            x = df[xcol].dropna()
                            stat, p = ttest_1samp(x, mu)
                        elif test in ["Paired samples t-test", "Wilcoxon signed-rank"]:
                            pair = df[[xcol, second]].dropna()
                            if test == "Paired samples t-test":
                                stat, p = ttest_rel(pair[xcol], pair[second])
                            else:
                                stat, p = wilcoxon(pair[xcol], pair[second])
                        elif test in ["Independent samples t-test (Welch)", "Mann–Whitney U"]:
                            if len(levels) < 2:
                                raise ValueError("Choose a grouping column with at least two groups.")
                            a = df.loc[df[group_col] == level1, xcol].dropna()
                            b = df.loc[df[group_col] == level2, xcol].dropna()
                            if test == "Independent samples t-test (Welch)":
                                stat, p = ttest_ind(a, b, equal_var=False)
                            else:
                                stat, p = mannwhitneyu(a, b, alternative="two-sided")
                        else:
                            if not group_col:
                                raise ValueError("A categorical grouping column is required.")
                            arrays = [g[xcol].dropna().values for _, g in df.groupby(group_col)]
                            arrays = [a for a in arrays if len(a)]
                            if len(arrays) < 2:
                                raise ValueError("At least two non-empty groups are required.")
                            if test == "One-way ANOVA":
                                stat, p = f_oneway(*arrays)
                            else:
                                stat, p = kruskal(*arrays)
                        result_card(test, stat, p, f"Decision threshold α = {alpha:.3f}. This is not a substitute for checking assumptions.")
                    except Exception as exc:
                        st.error(f"Test could not be completed: {exc}")

elif page == "Relationships & Regression":
    st.title("Relationships & Regression")
    df = st.session_state.data
    if df is None:
        st.info("Upload a dataset first.")
    else:
        nums = numeric_columns(df)
        if len(nums) < 2:
            st.warning("At least two numeric columns are required.")
        else:
            method = st.selectbox("Correlation method", ["Pearson", "Spearman", "Kendall"])
            xcol, ycol = st.columns(2)
            x = xcol.selectbox("X variable", nums, key="corrx")
            y = ycol.selectbox("Y variable", [v for v in nums if v != x], key="corry")
            pair = df[[x, y]].dropna()
            if st.button("Calculate correlation"):
                if len(pair) < 3:
                    st.error("At least three paired observations are required.")
                else:
                    funcs = {"Pearson": pearsonr, "Spearman": spearmanr, "Kendall": kendalltau}
                    r, p = funcs[method](pair[x], pair[y])
                    result_card(f"{method} correlation", r, p, f"Complete paired observations: {len(pair)}.")
            st.divider()
            st.subheader("Simple linear regression")
            if st.button("Fit linear regression"):
                pair = df[[x, y]].dropna()
                if len(pair) < 3:
                    st.error("At least three paired observations are required.")
                else:
                    model = LinearRegression().fit(pair[[x]], pair[y])
                    pred = model.predict(pair[[x]])
                    r2 = model.score(pair[[x]], pair[y])
                    st.metric("R-squared", f"{r2:.5f}")
                    st.write(f"Intercept: **{model.intercept_:.5g}**")
                    st.write(f"Slope for {x}: **{model.coef_[0]:.5g}**")
                    st.caption("This is a simple descriptive model. Review assumptions, influential points, and study design before interpretation.")

elif page == "Visualizations":
    st.title("Visualizations")
    df = st.session_state.data
    if df is None:
        st.info("Upload a dataset first.")
    else:
        nums = numeric_columns(df)
        if not nums:
            st.warning("No numeric variables available.")
        else:
            kind = st.selectbox("Chart type", ["Histogram", "Boxplot", "Scatter plot", "Correlation heatmap", "Line plot"])
            if kind in ["Histogram", "Boxplot", "Line plot"]:
                col = st.selectbox("Variable", nums)
            elif kind == "Scatter plot":
                c1, c2 = st.columns(2)
                x = c1.selectbox("X axis", nums)
                y = c2.selectbox("Y axis", [v for v in nums if v != x])
            if st.button("Create chart"):
                fig, ax = plt.subplots(figsize=(8, 4.5))
                if kind == "Histogram":
                    ax.hist(df[col].dropna(), bins="auto", edgecolor="black")
                    ax.set_xlabel(col); ax.set_ylabel("Frequency")
                elif kind == "Boxplot":
                    ax.boxplot(df[col].dropna(), vert=True)
                    ax.set_ylabel(col)
                elif kind == "Scatter plot":
                    pair = df[[x, y]].dropna()
                    ax.scatter(pair[x], pair[y], alpha=.7)
                    ax.set_xlabel(x); ax.set_ylabel(y)
                elif kind == "Correlation heatmap":
                    corr = df[nums].corr(numeric_only=True)
                    im = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto")
                    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
                    ax.set_yticks(range(len(corr.index))); ax.set_yticklabels(corr.index)
                    fig.colorbar(im, ax=ax, label="Correlation")
                else:
                    ax.plot(df[col].dropna().values)
                    ax.set_xlabel("Observation order"); ax.set_ylabel(col)
                ax.set_title(kind)
                fig.tight_layout()
                st.pyplot(fig)
                buffer = io.BytesIO()
                fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
                st.download_button("Download figure (PNG)", buffer.getvalue(), "statnova_figure.png", "image/png")
                plt.close(fig)

elif page == "Reports & Export":
    st.title("Reports & Export")
    df = st.session_state.data
    if df is None:
        st.info("Upload a dataset first.")
    else:
        st.write(f"Dataset: **{st.session_state.source_name}**")
        st.subheader("Export cleaned working copy")
        st.caption("This export preserves the uploaded values; it does not automatically remove missing values or outliers.")
        st.download_button("Download dataset as CSV", df.to_csv(index=False).encode("utf-8-sig"),
                           "statnova_dataset.csv", "text/csv")
        excel = io.BytesIO()
        with pd.ExcelWriter(excel, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Data", index=False)
            df.describe(include="all").transpose().to_excel(writer, sheet_name="Summary")
        st.download_button("Download Excel workbook", excel.getvalue(),
                           "statnova_report.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        report = [
            "STATNOVA — Dataset Summary",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Source: {st.session_state.source_name}",
            f"Rows: {df.shape[0]}",
            f"Columns: {df.shape[1]}",
            f"Duplicate rows: {int(df.duplicated().sum())}",
            f"Missing cells: {int(df.isna().sum().sum())}",
            "",
            "Column types:",
            df.dtypes.astype(str).to_string(),
            "",
            "Descriptive statistics:",
            df.describe(include="all").transpose().to_string()
        ]
        st.download_button("Download text summary", "\n".join(report).encode("utf-8"),
                           "statnova_summary.txt", "text/plain")

elif page == "About":
    st.title("About STATNOVA")
    st.write("STATNOVA is being developed as a general research data analysis workspace.")
    st.markdown("""
    **Developer:** Amatalelah Abdulazez Mashhor Al-Ahdel  
    **Qualification:** Master of Science in Physics  
    **Research interests:** Medical Physics · Nanotechnology · Statistical Analysis · Scientific Computing

    **Current development modules**
    - Dataset upload and exploration
    - Data quality screening
    - Descriptive statistics
    - Selected parametric and non-parametric tests
    - Correlation and simple linear regression
    - Basic scientific visualizations
    - CSV and Excel exports
    """)
    st.warning("""
    This is a development version, not a clinically validated or production-certified system.
    Statistical outputs depend on data quality, study design, assumptions, and correct test selection.
    User accounts, payment subscriptions, multilingual interface, advanced effect sizes,
    confidence intervals, post-hoc procedures, and a formal privacy workflow are not included yet.
    """)

st.divider()
st.caption("© STATNOVA · Intelligent Research Analytics · Development build")
