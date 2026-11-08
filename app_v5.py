# STATNOVA v3 — Intelligent Research Analytics
# Development build: validate statistical choices and outputs before research use.

import io
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import (
    ttest_ind, ttest_rel, ttest_1samp, f_oneway, mannwhitneyu,
    wilcoxon, kruskal, chi2_contingency, pearsonr, spearmanr, kendalltau, studentized_range
)
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.linear_model import LinearRegression
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

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
  background-color: #f4f8fb;
  background-image: linear-gradient(115deg, rgba(247,250,252,.94), rgba(241,249,247,.91)), url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNjAwIiBoZWlnaHQ9IjEwMDAiIHZpZXdCb3g9IjAgMCAxNjAwIDEwMDAiPjxkZWZzPjxsaW5lYXJHcmFkaWVudCBpZD0iZyIgeDE9IjAiIHkxPSIwIiB4Mj0iMSIgeTI9IjEiPjxzdG9wIHN0b3AtY29sb3I9IiMwZDU5NjUiLz48c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiM2Y2E5YTAiLz48L2xpbmVhckdyYWRpZW50PjwvZGVmcz48ZyBmaWxsPSJub25lIiBzdHJva2U9InVybCgjZykiIHN0cm9rZS13aWR0aD0iMi4yIiBvcGFjaXR5PSIuNzIiPjxwYXRoIGQ9Ik0wIDc2MCBDMTgwIDY1MCAyNTAgODUwIDQzMCA3MTAgUzcwMCA2MjAgODYwIDcyMCAxMTIwIDg1MCAxMzAwIDY4MCAxNDgwIDYyMCAxNjAwIDcwMCIvPjxwYXRoIGQ9Ik0wIDgyMCBDMTgwIDcxMCAyNTAgOTEwIDQzMCA3NzAgUzcwMCA2ODAgODYwIDc4MCAxMTIwIDkxMCAxMzAwIDc0MCAxNDgwIDY4MCAxNjAwIDc2MCIvPjxwYXRoIGQ9Ik0xMDMwIDAgTDEyMDAgMTUwIDEzNzAgNDAgMTUzMCAyMTAgMTYwMCAxNTAiLz48cGF0aCBkPSJNMCAxMzAgTDE1MCAyNTAgMzAwIDE3MCA0NDAgMzEwIDU3MCAyMjAiLz48L2c+PGcgZmlsbD0iIzJhOGM5MSIgb3BhY2l0eT0iLjQ4Ij48Y2lyY2xlIGN4PSIxNTAiIGN5PSIyNTAiIHI9IjgiLz48Y2lyY2xlIGN4PSIzMDAiIGN5PSIxNzAiIHI9IjgiLz48Y2lyY2xlIGN4PSI0NDAiIGN5PSIzMTAiIHI9IjgiLz48Y2lyY2xlIGN4PSIxMjAwIiBjeT0iMTUwIiByPSI4Ii8+PGNpcmNsZSBjeD0iMTM3MCIgY3k9IjQwIiByPSI4Ii8+PGNpcmNsZSBjeD0iMTUzMCIgY3k9IjIxMCIgcj0iOCIvPjwvZz48ZyBmaWxsPSJub25lIiBzdHJva2U9IiM3MGFhYTMiIHN0cm9rZS13aWR0aD0iMS41IiBvcGFjaXR5PSIuNDUiPjxjaXJjbGUgY3g9IjEzNTAiIGN5PSI3MzAiIHI9Ijg1Ii8+PGNpcmNsZSBjeD0iMTM1MCIgY3k9IjczMCIgcj0iMTI1Ii8+PGNpcmNsZSBjeD0iMjMwIiBjeT0iNzkwIiByPSI3NSIvPjxjaXJjbGUgY3g9IjIzMCIgY3k9Ijc5MCIgcj0iMTE1Ii8+PC9nPjwvc3ZnPg==");
  background-size: cover, cover; background-position: center, center; background-attachment: fixed;
}
.block-container { padding-top: 1.05rem; padding-bottom: 2.5rem; max-width: 1360px; }
.hero { padding: 1.35rem 1.7rem; border-radius: 20px; background: linear-gradient(112deg,#102f4b 0%,#126b70 72%,#23827d 100%); border: 1px solid rgba(255,255,255,.28); margin-bottom: 1.15rem; box-shadow: 0 12px 30px rgba(16,47,75,.13); }
.hero h1 { color:#fff; margin:0; font-size:2.15rem; letter-spacing:.045em; }
.hero p { color:#e5f1f4; margin:.32rem 0 0 0; }
h1,h2,h3 { color:#173f59; }
h1 { font-size:1.85rem !important; }
h2 { font-size:1.42rem !important; }
div[data-testid="stMetric"] { background:rgba(255,255,255,.91); border:1px solid #dce7ed; padding:13px 15px; border-radius:14px; box-shadow:0 4px 14px rgba(22,62,82,.045); }
div[data-testid="stExpander"], div[data-testid="stDataFrame"], div[data-testid="stFileUploader"] { background:rgba(255,255,255,.82); border-radius:13px; }
.stButton>button,.stDownloadButton>button { border-radius:10px; font-weight:650; border:1px solid #c9dce2; transition:all .18s ease; }
.stButton>button:hover,.stDownloadButton>button:hover { border-color:#167b7b; color:#125f68; box-shadow:0 4px 12px rgba(18,107,112,.12); }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#edf5f6 0%,#f7fafb 100%); border-right:1px solid #dce8eb; }
section[data-testid="stSidebar"] .block-container { padding-top:1rem; }
section[data-testid="stSidebar"] div[role="radiogroup"] label { padding:.32rem .45rem; border-radius:8px; }
.small-note { color:#526b78; font-size:.9rem; }
@media (max-width: 700px) { .block-container { padding-left:1rem; padding-right:1rem; } .hero { padding:1.1rem 1.15rem; } .hero h1 { font-size:1.8rem; } h1 { font-size:1.55rem !important; } }
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
    st.markdown("<div style='font-size:1.2rem;font-weight:800;letter-spacing:.04em;color:#16465b'>📊 STATNOVA</div><div style='font-size:.76rem;color:#66818b;margin:.1rem 0 .8rem'>Research Analytics Workspace</div>", unsafe_allow_html=True)
    page = st.radio(
        "Workspace",
        ["Home", "User Guide", "Upload & Explore", "Data Quality", "Descriptive Statistics",
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

def games_howell(groups, alpha=0.05):
    """Games-Howell pairwise comparisons for independent groups with unequal variances."""
    if not 0 < float(alpha) < 1:
        raise ValueError("alpha must be between 0 and 1.")
    groups = [(name, np.asarray(values, dtype=float)) for name, values in groups]
    groups = [(name, values[np.isfinite(values)]) for name, values in groups]
    groups = [(name, values) for name, values in groups if len(values) > 0]
    if len(groups) < 3:
        raise ValueError("Games–Howell requires at least three non-empty independent groups.")
    if any(len(values) < 2 for _, values in groups):
        raise ValueError("Each group needs at least two valid observations for Games–Howell.")
    rows = []
    k = len(groups)
    for (name1, x), (name2, y) in combinations(groups, 2):
        n1, n2 = len(x), len(y)
        if n1 < 2 or n2 < 2:
            continue
        v1, v2 = np.var(x, ddof=1) / n1, np.var(y, ddof=1) / n2
        se = np.sqrt(v1 + v2)
        diff = float(np.mean(x) - np.mean(y))
        if not np.isfinite(se) or se == 0:
            p_adj, low, high, df = (1.0, diff, diff, np.inf) if diff == 0 else (0.0, diff, diff, np.inf)
        else:
            df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
            q_stat = np.sqrt(2) * abs(diff) / se
            p_adj = float(studentized_range.sf(q_stat, k, df))
            q_crit = float(studentized_range.ppf(1 - alpha, k, df))
            margin = q_crit * np.sqrt((v1 + v2) / 2)
            low, high = diff - margin, diff + margin
        rows.append({
            "Group 1": str(name1), "Group 2": str(name2),
            "Mean difference": diff, "Adjusted p-value": p_adj,
            f"{100*(1-alpha):.1f}% CI lower": low,
            f"{100*(1-alpha):.1f}% CI upper": high,
            "Welch df": df
        })
    return pd.DataFrame(rows)



def build_pdf_report(df, source_name):
    """Create a concise, reproducible dataset overview PDF."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=42, bottomMargin=42)
    styles = getSampleStyleSheet()
    story = [Paragraph("STATNOVA — Research Data Summary", styles["Title"]), Spacer(1, 10)]
    story.append(Paragraph(f"Source file: {str(source_name)}", styles["Normal"]))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 12))
    metrics = [["Rows", "Columns", "Duplicate rows", "Missing cells"],
               [str(df.shape[0]), str(df.shape[1]), str(int(df.duplicated().sum())), str(int(df.isna().sum().sum()))]]
    tbl = Table(metrics, hAlign="LEFT")
    tbl.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e7f0f3")),
                             ("GRID", (0,0), (-1,-1), .5, colors.HexColor("#b8cbd2")),
                             ("PADDING", (0,0), (-1,-1), 7)]))
    story.extend([tbl, Spacer(1, 14), Paragraph("Column information", styles["Heading2"])])
    rows = [["Variable", "Data type", "Missing", "Missing %"]]
    for col in df.columns:
        miss = int(df[col].isna().sum())
        rows.append([str(col)[:45], str(df[col].dtype), str(miss), f"{miss / max(len(df), 1) * 100:.1f}%"])
    info = Table(rows, repeatRows=1, colWidths=[210, 100, 65, 75])
    info.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e7f0f3")),
                              ("GRID", (0,0), (-1,-1), .35, colors.grey),
                              ("FONTSIZE", (0,0), (-1,-1), 8), ("PADDING", (0,0), (-1,-1), 5)]))
    story.extend([info, Spacer(1, 14), Paragraph("Descriptive statistics", styles["Heading2"])])
    numeric = df.select_dtypes(include=np.number)
    if len(numeric.columns):
        desc = numeric.describe().T.reset_index().rename(columns={"index": "Variable"})
        data = [[str(c) for c in ["Variable", "count", "mean", "std", "min", "50%", "max"]]]
        for _, row in desc.iterrows():
            data.append([str(row.get("Variable", ""))[:32]] + [f"{row.get(c, np.nan):.4g}" if pd.notna(row.get(c, np.nan)) else "N/A" for c in ["count", "mean", "std", "min", "50%", "max"]])
        summary = Table(data, repeatRows=1, colWidths=[125, 45, 55, 55, 55, 55, 55])
        summary.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e7f0f3")),
                                     ("GRID", (0,0), (-1,-1), .35, colors.grey),
                                     ("FONTSIZE", (0,0), (-1,-1), 7), ("PADDING", (0,0), (-1,-1), 4)]))
        story.append(summary)
    story.extend([Spacer(1, 14), Paragraph("Interpretation note", styles["Heading2"]),
                  Paragraph("This report summarizes the uploaded dataset only. It does not certify data quality, select a study design, or replace statistical review. Missing values and potential outliers have not been removed automatically.", styles["BodyText"])])
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def safe_filename(name):
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))
    return cleaned[:80] or "statnova_export"

if "data" not in st.session_state:
    st.session_state.data = None
if "source_name" not in st.session_state:
    st.session_state.source_name = ""

# ---------- Home ----------
if page == "Home":
    st.title("Welcome")
    st.caption("Your research data workspace · مساحة عملك لتحليل البيانات البحثية")
    st.write(
        "A research-focused workspace for importing datasets, reviewing data quality, "
        "summarizing variables, running selected statistical tests, visualizing results, "
        "and exporting working reports."
    )
    a, b, c = st.columns(3)
    a.metric("Analysis areas", "8")
    b.metric("Input formats", "CSV / Excel")
    c.metric("Interface", "English")
    st.subheader("Getting started")
    st.markdown("""
    1. Open **Upload & Explore** and import a CSV or Excel file.
    2. Review **Data Quality** before analysis.
    3. Explore **Descriptive Statistics** and select a test that matches your study design.
    4. Create figures and export a report.
    """)
    st.info("New here? Open **User Guide | دليل الاستخدام** from the sidebar for a guided workflow.")
    st.warning("Research-use note: verify assumptions, study design, and statistical interpretations before publication or clinical use.")

# ---------- User guide ----------
elif page == "User Guide":
    st.title("STATNOVA User Guide | دليل الاستخدام")
    st.markdown("""
    ### 1. Upload your dataset | رفع البيانات
    Open **Upload & Explore** and upload a CSV or Excel file. Confirm that each column has a clear variable name and numeric values are stored as numbers.

    افتحي صفحة **Upload & Explore** لرفع ملف CSV أو Excel، وتأكدي من وضوح أسماء المتغيرات وأن القيم الرقمية محفوظة كأرقام.

    ### 2. Review data quality | فحص جودة البيانات
    Check missing values, duplicate rows, data types, and potential IQR outliers. These flags are prompts for review, not automatic deletion instructions.

    راجعي القيم المفقودة والتكرارات وأنواع البيانات والقيم المتطرفة المحتملة. التنبيهات للمراجعة فقط ولا تعني حذف البيانات تلقائياً.

    ### 3. Describe variables | الإحصاءات الوصفية
    Review count, mean, standard deviation, quartiles, median, variance, skewness, and kurtosis for numeric variables.

    افحصي العدد والمتوسط والانحراف المعياري والوسيط والربيعات والتباين والالتواء والتفرطح للمتغيرات الرقمية.

    ### 4. Select a statistical test | اختيار الاختبار
    Use the Smart Statistical Test Advisor as an initial guide, then confirm the study design. Independent groups, paired measurements, and repeated observations are not interchangeable. Manual test selection remains available.

    استخدمي مستشار الاختبار الذكي كاقتراح أولي، ثم تحققي من تصميم الدراسة. المجموعات المستقلة والقياسات المزدوجة والقياسات المتكررة ليست متطابقة. ويمكنك اختيار الاختبار يدوياً.

    ### 5. Interpret results | تفسير النتائج
    Read the test statistic and p-value together with sample sizes, assumptions, effect size, confidence intervals, and the scientific context. Statistical significance does not by itself establish practical or clinical importance.

    فسّري قيمة الاختبار وp-value مع أحجام العينات والافتراضات وحجم الأثر وفترات الثقة والسياق العلمي. الدلالة الإحصائية وحدها لا تثبت الأهمية العملية أو السريرية.

    ### 6. Export | تصدير النتائج
    Open **Reports & Export** to download the dataset, an Excel workbook, a text summary, or a PDF overview. Protect confidential or identifiable research data before uploading.

    من صفحة **Reports & Export** يمكنك تنزيل البيانات أو ملف Excel أو ملخص نصي أو تقرير PDF. لا ترفعي بيانات شخصية أو سرية دون حماية مناسبة.

    **Important:** STATNOVA is a development tool. Verify analyses independently before using results in a thesis, publication, or clinical decision.
    """)

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
                "Welch ANOVA",
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
                    "إذا لم تلائم الافتراضات، فقد يكون Kruskal–Wallis مناسبًا للبيانات الرتبية أو الملائمة له. "
                    "وعند عدم تجانس التباين، اختر Welch ANOVA من قائمة الاختبارات."
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

                if test in ["One-way ANOVA", "Welch ANOVA", "Kruskal–Wallis"] and group_col:
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
                            named_arrays = [
                                (label, g[xcol].dropna().to_numpy(dtype=float))
                                for label, g in grouped.dropna(subset=[group_col]).groupby(group_col, sort=False)
                            ]
                            named_arrays = [(label, arr) for label, arr in named_arrays if len(arr) > 0]
                            arrays = [arr for _, arr in named_arrays]
                            if len(arrays) < 2:
                                raise ValueError("At least two non-empty groups are required.")
                            if test in ["One-way ANOVA", "Welch ANOVA"]:
                                if any(len(arr) < 2 for arr in arrays):
                                    raise ValueError("Each ANOVA group needs at least two valid observations.")
                                if test == "One-way ANOVA":
                                    stat, p = f_oneway(*arrays)
                                else:
                                    welch_result = anova_oneway(arrays, use_var="unequal", welch_correction=True)
                                    stat, p = float(welch_result.statistic), float(welch_result.pvalue)
                            else:
                                stat, p = kruskal(*arrays)
                            note = f"Number of groups: {len(arrays)}; group sample sizes: {', '.join(map(str, map(len, arrays)))}."

                        result_card(test, stat, p, alpha, note)
                        if test == "One-way ANOVA" and len(arrays) >= 3:
                            st.subheader("Post-hoc pairwise comparisons — Tukey HSD")
                            st.caption("Tukey HSD controls the family-wise error rate after classical one-way ANOVA.")
                            pooled_values = np.concatenate(arrays)
                            pooled_labels = np.concatenate([[str(label)] * len(arr) for label, arr in named_arrays])
                            tukey = pairwise_tukeyhsd(pooled_values, pooled_labels, alpha=alpha)
                            tukey_table = pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0])
                            st.dataframe(tukey_table, use_container_width=True)
                        elif test == "Welch ANOVA" and len(arrays) >= 3:
                            st.subheader("Post-hoc pairwise comparisons — Games–Howell")
                            st.caption("Games–Howell is designed for pairwise comparisons when group variances and sample sizes may differ.")
                            gh_table = games_howell(named_arrays, alpha=alpha)
                            st.dataframe(gh_table.round(6), use_container_width=True)
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
        try:
            pdf_bytes = build_pdf_report(df, st.session_state.source_name)
            st.download_button(
                "Download PDF report", pdf_bytes,
                "statnova_research_summary.pdf", "application/pdf"
            )
        except Exception as exc:
            st.error(f"PDF report could not be generated: {exc}")

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
    User accounts, payment subscriptions, a complete multilingual interface, and a formal
    privacy workflow are not included yet. Effect sizes and confidence intervals are still
    being expanded and validated across the available statistical procedures.
    """)

st.divider()
st.caption("© STATNOVA · Intelligent Research Analytics · Developed by Amatalelah")
