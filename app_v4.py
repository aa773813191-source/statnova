# STATNOVA v3 — Intelligent Research Analytics
# Development build: validate statistical choices and outputs before research use.

import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import (
    ttest_ind, ttest_rel, ttest_1samp, f_oneway, mannwhitneyu,
    wilcoxon, kruskal, chi2_contingency, pearsonr, spearmanr, kendalltau
)
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="STATNOVA | Intelligent Research Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Visual identity ----------
st.markdown("""
<style>
:root { color-scheme: light; }
.stApp {
    background: linear-gradient(135deg, #f4f8fc 0%, #f3faf7 55%, #f8f7fc 100%);
}
.block-container { padding-top: 1.35rem; max-width: 1320px; }
.hero {
    padding: 1.7rem 1.9rem; border-radius: 18px;
    background: linear-gradient(110deg, #102f4b, #126b70);
    border: 1px solid #d4e3e8; margin-bottom: 1.2rem;
}
.hero h1 { color: #ffffff; margin: 0; font-size: 2.45rem; letter-spacing: .04em; }
.hero p { color: #e5f1f4; margin: .35rem 0 0 0; }
h1, h2, h3 { color: #173f59; }
div[data-testid="stMetric"] {
    background: #ffffff; border: 1px solid #dce7ed;
    padding: 13px; border-radius: 12px;
}
.stButton > button, .stDownloadButton > button {
    border-radius: 9px; font-weight: 600;
}
.small-note { color: #526b78; font-size: .9rem; }
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
    st.caption("Development build · Verify methods and results before research or clinical use.")

# ---------- Helpers ----------
def load_data(uploaded):
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        try:
            uploaded.seek(0)
            return pd.read_csv(uploaded)
        except UnicodeDecodeError:
            uploaded.seek(0)
            return pd.read_csv(uploaded, encoding="latin-1")
    if name.endswith((".xlsx", ".xls")):
        uploaded.seek(0)
        return pd.read_excel(uploaded)
    raise ValueError("Supported formats are CSV and Excel (.xlsx, .xls).")

def numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()

def categorical_columns(df):
    return df.select_dtypes(exclude=np.number).columns.tolist()

def p_text(p):
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "N/A"
    return f"{p:.5g}" if np.isfinite(p) else "N/A"

def result_card(title, statistic, pvalue, alpha=0.05, note=""):
    st.subheader(title)
    c1, c2, c3 = st.columns(3)
    c1.metric("Test statistic", f"{statistic:.5g}" if np.isfinite(statistic) else "N/A")
    c2.metric("p-value", p_text(pvalue))
    c3.metric("Significance level (α)", f"{alpha:.3f}")
    if np.isfinite(pvalue):
        if pvalue < alpha:
            st.success(f"Statistically significant at α = {alpha:.3f}.")
        else:
            st.info(f"Not statistically significant at α = {alpha:.3f}.")
    if note:
        st.caption(note)

def clean_numeric(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

def safe_filename(name):
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))
    return cleaned[:80] or "statnova_export"

if "data" not in st.session_state:
    st.session_state.data = None
if "source_name" not in st.session_state:
    st.session_state.source_name = ""

# ---------- Home ----------
if page == "Home":
    st.title("Welcome to STATNOVA")
    st.write(
        "A research-focused workspace for importing datasets, reviewing data quality, "
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
    3. Explore **Descriptive Statistics** and select a test that matches your study design.
    4. Create figures and export a report.
    """)
    st.warning("STATNOVA is a development build. It does not replace expert statistical review.")

# ---------- Upload ----------
elif page == "Upload & Explore":
    st.title("Upload & Explore")
    uploaded = st.file_uploader("Upload a CSV or Excel dataset", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        try:
            df_new = load_data(uploaded)
            if df_new.empty:
                st.error("The uploaded file contains no data rows.")
            else:
                st.session_state.data = df_new
                st.session_state.source_name = uploaded.name
                st.success(f"Loaded successfully: {uploaded.name}")
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
        if st.button("Clear current dataset"):
            st.session_state.data = None
            st.session_state.source_name = ""
            st.rerun()
    else:
        st.info("Choose a CSV or Excel file to begin.")

# ---------- Data quality ----------
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
            col = st.selectbox("Numeric variable", nums, key="quality_numeric")
            series = clean_numeric(df[col])
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
                st.caption("At least four valid numeric observations are needed for this screening.")
        st.warning("Missing values, duplicates, and IQR flags require context. STATNOVA does not delete observations automatically.")

# ---------- Descriptive statistics ----------
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
            selected = st.multiselect(
                "Choose numeric variables", nums,
                default=nums[:min(5, len(nums))]
            )
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

# ---------- Statistical tests ----------
elif page == "Statistical Tests":
    st.title("Statistical Tests")
    df = st.session_state.data
    if df is None:
        st.info("Upload a dataset first.")
    else:
        nums = numeric_columns(df)
        cats = categorical_columns(df)
        if not nums:
            st.warning("No numeric columns were detected.")
        else:
            available_tests = [
                "Independent samples t-test (Welch)",
                "Paired samples t-test",
                "One-sample t-test",
                "One-way ANOVA",
                "Mann–Whitney U",
                "Wilcoxon signed-rank",
                "Kruskal–Wallis",
                "Chi-square test of independence"
            ]

            st.subheader("Smart Statistical Test Advisor")
            st.caption(
                "أجب عن أسئلة تصميم الدراسة للحصول على اقتراح مبدئي. "
                "الاقتراح لا يستبدل حكم الباحث أو مراجعة افتراضات الاختبار."
            )
            design = st.selectbox(
                "Study design",
                [
                    "Independent groups — two groups",
                    "Independent groups — three or more groups",
                    "Paired measurements — same participants",
                    "One sample compared with a reference value",
                    "Association between two categorical variables",
                    "Association between two numeric variables"
                ],
                key="advisor_design"
            )

            recommendation = None
            reason = ""
            caution = ""

            if design == "Independent groups — two groups":
                recommendation = "Independent samples t-test (Welch)"
                reason = (
                    "يقارن متوسط متغير رقمي بين مجموعتين مستقلتين. "
                    "اختبار Welch لا يشترط تساوي التباينات، لذلك هو خيار مبدئي مناسب."
                )
                caution = (
                    "تأكد من استقلال المشاهدات وأن المتغير الرقمي مناسب للمقارنة. "
                    "إذا كانت البيانات رتبية أو شديدة الانحراف مع عينات صغيرة، ناقش Mann–Whitney U."
                )
            elif design == "Independent groups — three or more groups":
                recommendation = "One-way ANOVA"
                reason = "يقارن متوسط متغير رقمي بين ثلاث مجموعات مستقلة أو أكثر."
                caution = (
                    "تحقق من استقلال المشاهدات، وشكل البواقي، وتجانس التباين. "
                    "إذا لم تلائم الافتراضات، فقد يكون Kruskal–Wallis مناسبًا؛ "
                    "وعند عدم تجانس التباين يلزم اختبار Welch ANOVA غير المتاح حاليًا."
                )
            elif design == "Paired measurements — same participants":
                recommendation = "Paired samples t-test"
                reason = "يقارن قياسين مرتبطين للمشاركين أنفسهم أو أزواج متطابقة."
                caution = (
                    "افحص توزيع الفروق بين القياسين، لا توزيع كل عمود منفردًا. "
                    "قد يكون Wilcoxon signed-rank بديلًا عند عدم ملاءمة افتراضات اختبار t."
                )
            elif design == "One sample compared with a reference value":
                recommendation = "One-sample t-test"
                reason = "يقارن متوسط متغير رقمي بقيمة مرجعية محددة."
                caution = "حدد القيمة المرجعية علميًا، وافحص ملاءمة توزيع المتغير وحجم العينة."
            elif design == "Association between two categorical variables":
                recommendation = "Chi-square test of independence"
                reason = "يفحص وجود ارتباط بين متغيرين تصنيفيين باستخدام جدول تقاطعي."
                caution = (
                    "راجع التكرارات المتوقعة في خلايا الجدول. إذا كانت صغيرة، "
                    "قد تحتاج إلى Fisher exact أو أسلوب آخر؛ Fisher غير متاح في هذه النسخة."
                )
            elif design == "Association between two numeric variables":
                reason = (
                    "اختر Pearson للعلاقة الخطية بين متغيرين رقميين مع مراعاة القيم الشاذة، "
                    "أو Spearman/Kendall للعلاقة الرتبية أو الرتيبة."
                )
                recommendation = "Pearson correlation"
                caution = "هذه الصفحة تعرض معاملات الارتباط في قسم Relationships & Regression."

            if recommendation == "Pearson correlation":
                st.info("**الاقتراح:** Pearson correlation — انتقل إلى Relationships & Regression لاستخدامه.")
                st.caption(reason)
                st.warning(caution)
            elif recommendation:
                st.success(f"**الاختبار المقترح:** {recommendation}")
                st.write(reason)
                st.warning(caution)
            else:
                st.info("لا يمكن تقديم اقتراح قبل تحديد تصميم الدراسة.")

            st.divider()
            st.subheader("Manual Test Selection")
            st.caption("يمكنك تجاهل الاقتراح واختيار أي اختبار متاح بنفسك.")
            default_index = available_tests.index(recommendation) if recommendation in available_tests else 0
            test = st.selectbox(
                "Select a test",
                available_tests,
                index=default_index,
                key="manual_test_selection"
            )
            alpha = st.number_input(
                "Significance level (α)", min_value=0.001, max_value=0.20,
                value=0.05, step=0.005, format="%.3f"
            )

            if test == "Chi-square test of independence":
                if len(cats) < 2:
                    st.warning("This test requires two categorical columns.")
                else:
                    c1, c2 = st.columns(2)
                    v1 = c1.selectbox("First categorical variable", cats, key="chi_v1")
                    v2_options = [v for v in cats if v != v1]
                    v2 = c2.selectbox("Second categorical variable", v2_options, key="chi_v2")
                    if st.button("Run chi-square test"):
                        try:
                            table = pd.crosstab(df[v1], df[v2])
                            if table.shape[0] < 2 or table.shape[1] < 2:
                                raise ValueError("Both variables must contain at least two observed categories.")
                            chi, p, dof, expected = chi2_contingency(table)
                            st.dataframe(table, use_container_width=True)
                            result_card(
                                "Chi-square test of independence", chi, p, alpha,
                                f"Degrees of freedom: {dof}. Smallest expected frequency: {expected.min():.3g}. "
                                "Review expected cell counts before interpretation."
                            )
                        except Exception as exc:
                            st.error(f"Test could not be completed: {exc}")
            else:
                c1, c2 = st.columns(2)
                xcol = c1.selectbox("Numeric variable", nums, key="test_numeric")
                group_col = None
                if test not in ["One-sample t-test", "Paired samples t-test", "Wilcoxon signed-rank"]:
                    if not cats:
                        st.warning("This test requires a categorical grouping variable.")
                    else:
                        group_col = c2.selectbox("Grouping variable", cats, key="test_grouping")

                second = None
                if test in ["Paired samples t-test", "Wilcoxon signed-rank"]:
                    second_options = [v for v in nums if v != xcol]
                    if second_options:
                        second = st.selectbox("Second paired numeric variable", second_options)
                    else:
                        st.warning("A paired test requires at least two numeric variables.")

                mu = None
                if test == "One-sample t-test":
                    mu = st.number_input("Reference mean", value=0.0)

                level1 = level2 = None
                levels = []
                if test in ["Independent samples t-test (Welch)", "Mann–Whitney U"] and group_col:
                    levels = df[group_col].dropna().unique().tolist()
                    if len(levels) >= 2:
                        g1, g2 = st.columns(2)
                        level1 = g1.selectbox("Group 1", levels, key="ind_group_1")
                        level2_options = [v for v in levels if v != level1]
                        level2 = g2.selectbox("Group 2", level2_options, key="ind_group_2")
                    else:
                        st.warning("The grouping variable must contain at least two observed groups.")

                if test in ["One-way ANOVA", "Kruskal–Wallis"] and group_col:
                    st.caption("The selected grouping variable should represent independent groups.")

                if st.button("Run selected test"):
                    try:
                        if test == "One-sample t-test":
                            x = clean_numeric(df[xcol])
                            if len(x) < 2:
                                raise ValueError("At least two valid observations are required.")
                            stat, p = ttest_1samp(x, mu)
                            note = f"Valid observations: {len(x)}; reference mean: {mu:g}."

                        elif test in ["Paired samples t-test", "Wilcoxon signed-rank"]:
                            if second is None:
                                raise ValueError("Select a second paired numeric variable.")
                            pair = df[[xcol, second]].apply(pd.to_numeric, errors="coerce").replace(
                                [np.inf, -np.inf], np.nan
                            ).dropna()
                            if len(pair) < 2:
                                raise ValueError("At least two complete pairs are required.")
                            if test == "Paired samples t-test":
                                stat, p = ttest_rel(pair[xcol], pair[second])
                            else:
                                differences = pair[xcol] - pair[second]
                                if np.allclose(differences, 0):
                                    raise ValueError("All paired differences are zero; the Wilcoxon test is not informative.")
                                stat, p = wilcoxon(pair[xcol], pair[second])
                            note = f"Complete pairs: {len(pair)}; paired variables: {xcol} and {second}."

                        elif test in ["Independent samples t-test (Welch)", "Mann–Whitney U"]:
                            if group_col is None or level1 is None or level2 is None:
                                raise ValueError("Select a grouping variable with at least two groups.")
                            a = clean_numeric(df.loc[df[group_col] == level1, xcol])
                            b = clean_numeric(df.loc[df[group_col] == level2, xcol])
                            if len(a) < 2 or len(b) < 2:
                                raise ValueError("Each selected group needs at least two valid observations.")
                            if test == "Independent samples t-test (Welch)":
                                stat, p = ttest_ind(a, b, equal_var=False)
                            else:
                                stat, p = mannwhitneyu(a, b, alternative="two-sided")
                            note = f"{level1}: n = {len(a)}; {level2}: n = {len(b)}."

                        else:
                            if group_col is None:
                                raise ValueError("A categorical grouping variable is required.")
                            grouped = df[[group_col, xcol]].copy()
                            grouped[xcol] = pd.to_numeric(grouped[xcol], errors="coerce")
                            grouped[xcol] = grouped[xcol].replace([np.inf, -np.inf], np.nan)
                            arrays = [
                                g[xcol].dropna().to_numpy()
                                for _, g in grouped.dropna(subset=[group_col]).groupby(group_col)
                            ]
                            arrays = [arr for arr in arrays if len(arr) > 0]
                            if len(arrays) < 2:
                                raise ValueError("At least two non-empty groups are required.")
                            if test == "One-way ANOVA":
                                if any(len(arr) < 2 for arr in arrays):
                                    raise ValueError("Each ANOVA group needs at least two valid observations.")
                                stat, p = f_oneway(*arrays)
                            else:
                                stat, p = kruskal(*arrays)
                            note = f"Number of groups: {len(arrays)}; group sample sizes: {', '.join(map(str, map(len, arrays)))}."

                        result_card(test, stat, p, alpha, note)
                        st.caption("A p-value does not measure effect size or clinical importance. Check assumptions, study design, and missing-data handling.")
                    except Exception as exc:
                        st.error(f"Test could not be completed: {exc}")

# ---------- Relationships ----------
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
            c1, c2 = st.columns(2)
            x = c1.selectbox("X variable", nums, key="corr_x")
            y = c2.selectbox("Y variable", [v for v in nums if v != x], key="corr_y")
            pair = df[[x, y]].apply(pd.to_numeric, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).dropna()

            if st.button("Calculate correlation"):
                try:
                    if len(pair) < 3:
                        raise ValueError("At least three complete paired observations are required.")
                    if pair[x].nunique() < 2 or pair[y].nunique() < 2:
                        raise ValueError("Both variables must have variation; a constant variable cannot be correlated.")
                    funcs = {"Pearson": pearsonr, "Spearman": spearmanr, "Kendall": kendalltau}
                    r, p = funcs[method](pair[x], pair[y])
                    result_card(f"{method} correlation", r, p, 0.05, f"Complete paired observations: {len(pair)}.")
                    st.caption("Correlation describes association, not causation.")
                except Exception as exc:
                    st.error(f"Correlation could not be calculated: {exc}")

            st.divider()
            st.subheader("Simple linear regression")
            st.caption(f"Predictor (X): {x} · Outcome (Y): {y}")
            if st.button("Fit linear regression"):
                try:
                    if len(pair) < 3:
                        raise ValueError("At least three complete paired observations are required.")
                    if pair[x].nunique() < 2:
                        raise ValueError("The predictor must contain at least two distinct values.")
                    model = LinearRegression().fit(pair[[x]], pair[y])
                    pred = model.predict(pair[[x]])
                    r2 = model.score(pair[[x]], pair[y])
                    c1, c2, c3 = st.columns(3)
                    c1.metric("R-squared", f"{r2:.5f}")
                    c2.metric("Intercept", f"{model.intercept_:.5g}")
                    c3.metric("Slope", f"{model.coef_[0]:.5g}")
                    st.caption(f"Complete observations: {len(pair)}. This is a simple descriptive model; assess assumptions and influential observations.")
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.scatter(pair[x], pair[y], alpha=0.7, label="Observed")
                    order = np.argsort(pair[x].to_numpy())
                    ax.plot(pair[x].to_numpy()[order], pred[order], label="Fitted line")
                    ax.set_xlabel(x)
                    ax.set_ylabel(y)
                    ax.legend()
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as exc:
                    st.error(f"Regression could not be fitted: {exc}")

# ---------- Visualizations ----------
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
            col = x = y = None
            if kind in ["Histogram", "Boxplot", "Line plot"]:
                col = st.selectbox("Variable", nums)
            elif kind == "Scatter plot":
                c1, c2 = st.columns(2)
                x = c1.selectbox("X axis", nums)
                y = c2.selectbox("Y axis", [v for v in nums if v != x])
            if st.button("Create chart"):
                try:
                    fig, ax = plt.subplots(figsize=(8, 4.7))
                    if kind == "Histogram":
                        ax.hist(clean_numeric(df[col]), bins="auto", edgecolor="black")
                        ax.set_xlabel(col)
                        ax.set_ylabel("Frequency")
                    elif kind == "Boxplot":
                        ax.boxplot(clean_numeric(df[col]), vert=True)
                        ax.set_ylabel(col)
                    elif kind == "Scatter plot":
                        pair = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
                        ax.scatter(pair[x], pair[y], alpha=.7)
                        ax.set_xlabel(x)
                        ax.set_ylabel(y)
                    elif kind == "Correlation heatmap":
                        corr = df[nums].corr(numeric_only=True)
                        im = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto")
                        ax.set_xticks(range(len(corr.columns)))
                        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
                        ax.set_yticks(range(len(corr.index)))
                        ax.set_yticklabels(corr.index)
                        fig.colorbar(im, ax=ax, label="Correlation")
                    else:
                        ax.plot(clean_numeric(df[col]).to_numpy())
                        ax.set_xlabel("Observation order")
                        ax.set_ylabel(col)
                    ax.set_title(kind)
                    fig.tight_layout()
                    st.pyplot(fig)
                    buffer = io.BytesIO()
                    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
                    st.download_button(
                        "Download figure (PNG)", buffer.getvalue(),
                        "statnova_figure.png", "image/png"
                    )
                    plt.close(fig)
                except Exception as exc:
                    st.error(f"Chart could not be created: {exc}")

# ---------- Reports ----------
elif page == "Reports & Export":
    st.title("Reports & Export")
    df = st.session_state.data
    if df is None:
        st.info("Upload a dataset first.")
    else:
        st.write(f"Dataset: **{st.session_state.source_name}**")
        st.subheader("Export dataset and summary")
        st.caption("Exports preserve the uploaded values; missing values and outliers are not removed automatically.")
        st.download_button(
            "Download dataset as CSV",
            df.to_csv(index=False).encode("utf-8-sig"),
            "statnova_dataset.csv", "text/csv"
        )
        excel = io.BytesIO()
        with pd.ExcelWriter(excel, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Data", index=False)
            df.describe(include="all").transpose().to_excel(writer, sheet_name="Summary")
        st.download_button(
            "Download Excel workbook", excel.getvalue(),
            "statnova_report.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
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
        st.download_button(
            "Download text summary", "\n".join(report).encode("utf-8"),
            "statnova_summary.txt", "text/plain"
        )

# ---------- About ----------
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
