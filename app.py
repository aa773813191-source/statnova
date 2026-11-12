
import streamlit as st
import pandas as pd
import numpy as np
import io

# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="STATNOVA",
    page_icon="📊",
    layout="wide"
)

ORCID_ID = "0009-0006-5666-3091"
ORCID_URL = f"https://orcid.org/{ORCID_ID}"

# ==============================
# DESIGN AND STYLE
# ==============================

st.markdown("""
<style>

.stApp {
    background-color: #F4F7F6;
}

section[data-testid="stSidebar"] {
    background-color: #102B35;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

h1, h2, h3 {
    color: #123D3A;
}

.stat-card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border-left: 5px solid #159B78;
    box-shadow: 0 3px 12px #00000010;
}

.orcid-link {
    background: white;
    padding: 15px;
    border-radius: 12px;
    border-left: 5px solid #A6CE39;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# MAIN HEADER
# ==============================

st.markdown("""
<div style="
text-align:center;
padding:20px;
background:linear-gradient(120deg,#102B35,#126B61);
border-radius:18px;
color:white;
">

<h1 style="color:white;font-size:42px;">
STATNOVA
</h1>

<h3 style="color:#D9C58B;">
Intelligent Research Analytics
</h3>

<p>
Advanced Statistical Analysis & Scientific Research
</p>

<p>
<b>Amatalelah Abdulazez Mashhor Al-Ahdel</b><br>
Master of Physics (M.Sc.)
</p>

</div>
""", unsafe_allow_html=True)

st.write("")

# ==============================
# SIDEBAR MENU
# ==============================

st.sidebar.title("STATNOVA")

page = st.sidebar.radio(
    "Main Menu",
    [
        "Home",
        "Data Analysis",
        "Data Quality",
        "Statistical Reports",
        "About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption("Research Analytics Platform")

# ORCID LINK IN SIDEBAR

st.sidebar.markdown("---")

st.sidebar.markdown(
    f"""
    ### 🔬 Researcher Profile

    [View My ORCID Profile]({ORCID_URL})

    ORCID: {ORCID_ID}
    """
)

# ==============================
# HOME PAGE
# ==============================

if page == "Home":

    st.title("Welcome to STATNOVA")

    st.write(
        "Your intelligent workspace for statistical analysis "
        "and scientific research."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Analysis Modules", "5")
    c2.metric("Supported Formats", "CSV / Excel")
    c3.metric("Platform Status", "Ready")

    st.subheader("Our Research Tools")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 📊 Statistical Analysis")

        st.write(
            "Descriptive statistics and hypothesis testing."
        )

        st.markdown("### 🔍 Data Quality")

        st.write(
            "Missing values, duplicates and unusual observations."
        )

    with col2:

        st.markdown("### 📈 Data Visualization")

        st.write(
            "Scientific charts and graphical summaries."
        )

        st.markdown("### 📄 Scientific Reports")

        st.write(
            "Organized results and downloadable reports."
        )

    st.markdown("---")

    st.subheader("Researcher")

    st.write("Developed by Amatalelah Abdulazez Mashhor Al-Ahdel")

    st.markdown(
        f"[🔬 View Researcher ORCID Profile]({ORCID_URL})"
    )

# ==============================
# DATA ANALYSIS / QUALITY / REPORTS
# ==============================

elif page in [
    "Data Analysis",
    "Data Quality",
    "Statistical Reports"
]:

    st.title(page)

    uploaded_file = st.file_uploader(
        "Upload your research dataset",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        try:

            if uploaded_file.name.endswith(".csv"):

                df = pd.read_csv(uploaded_file)

            else:

                df = pd.read_excel(uploaded_file)

            st.success("Dataset uploaded successfully!")

            st.subheader("Data Preview")

            st.dataframe(
                df.head(10),
                use_container_width=True
            )

            numeric = df.select_dtypes(
                include=np.number
            )

            # ==============================
            # DATA QUALITY
            # ==============================

            if page == "Data Quality":

                st.subheader("Data Quality Assessment")

                c1, c2, c3 = st.columns(3)

                c1.metric("Rows", len(df))

                c2.metric("Columns", len(df.columns))

                c3.metric(
                    "Duplicates",
                    int(df.duplicated().sum())
                )

                quality = pd.DataFrame({

                    "Missing Values": df.isnull().sum(),

                    "Missing (%)": (
                        df.isnull().mean() * 100
                    ).round(2),

                    "Data Type": df.dtypes.astype(str)

                })

                st.dataframe(
                    quality,
                    use_container_width=True
                )

                st.info(
                    "Quality checks identify potential problems. "
                    "They do not prove whether data are authentic."
                )

            # ==============================
            # STATISTICAL ANALYSIS
            # ==============================

            elif page == "Data Analysis":

                st.subheader("Descriptive Statistics")

                if not numeric.empty:

                    st.dataframe(
                        numeric.describe().T,
                        use_container_width=True
                    )

                else:

                    st.warning(
                        "No numeric columns found."
                    )

                st.subheader("Choose a variable")

                if not numeric.empty:

                    variable = st.selectbox(
                        "Numeric variable",
                        numeric.columns
                    )

                    st.write(
                        "Mean:",
                        numeric[variable].mean()
                    )

                    st.write(
                        "Standard Deviation:",
                        numeric[variable].std()
                    )

                    st.write(
                        "Median:",
                        numeric[variable].median()
                    )

                    st.write(
                        "Minimum:",
                        numeric[variable].min()
                    )

                    st.write(
                        "Maximum:",
                        numeric[variable].max()
                    )

                    st.subheader("Data Visualization")

                    st.bar_chart(
                        numeric[variable]
                    )

            # ==============================
            # STATISTICAL REPORTS
            # ==============================

            elif page == "Statistical Reports":

                st.subheader("Download Your Dataset")

                csv = df.to_csv(
                    index=False
                ).encode("utf-8-sig")

                st.download_button(
                    "Download CSV",
                    csv,
                    "statnova_data.csv",
                    "text/csv"
                )

                output = io.BytesIO()

                with pd.ExcelWriter(
                    output,
                    engine="openpyxl"
                ) as writer:

                    df.to_excel(
                        writer,
                        index=False,
                        sheet_name="Data"
                    )

                st.download_button(
                    "Download Excel",
                    output.getvalue(),
                    "statnova_data.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        except Exception as e:

            st.error(
                f"Error reading dataset: {e}"
            )

# ==============================
# ABOUT PAGE
# ==============================

elif page == "About":

    st.title("About STATNOVA")

    st.write("""
    STATNOVA is an independent research analytics platform
    designed to support students and researchers.

    Our vision is to provide reliable statistical analysis,
    scientific interpretation, data quality assessment,
    and professional research reporting.
    """)

    st.markdown("---")

    st.subheader("Developed by")

    st.write(
        "Amatalelah Abdulazez Mashhor Al-Ahdel"
    )

    st.write("Master of Physics (M.Sc.)")

    st.markdown("---")

    # ORCID RESEARCHER PROFILE

    st.subheader("🔬 Researcher Profile")

    st.write(
        "Connect with the developer's verified academic researcher profile."
    )

    st.markdown(
        f"""
        <div class="orcid-link">

        <h3>ORCID Researcher ID</h3>

        <p>{ORCID_ID}</p>

        <a href="{ORCID_URL}" target="_blank">
        View My Verified ORCID Profile
        </a>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.caption(
        "STATNOVA | Intelligent Research Analytics"
    )

    st.caption(
        "Turning Data into Research Insights."
    )

</>
