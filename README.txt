STATNOVA - app_v6 update

Upload these three files together to the root of the existing GitHub repository:
1) app_v6.py
2) requirements.txt
3) NotoNaskhArabic-Regular.ttf

Replace the old files with these versions. Keep the Streamlit Main file setting as app_v6.py.
Do not upload this ZIP file to the repository; extract it first.

This update includes:
- p-value display that avoids presenting tiny values as zero
- automatic paired-analysis safeguards for recognized participant IDs
- repeated participant IDs block the automatic paired analysis rather than guessing matches
- paired analysis excludes missing participant IDs when an ID column is detected
- improved report display for missing table cells and p-value formatting
- warnings/context for very large standardized effect sizes
- no automatic deletion, imputation, or alteration of uploaded observations

Validation performed before packaging: Python syntax compilation and ZIP integrity check.
