STATNOVA PDF EXPORT FIX

This ZIP contains the corrected app_v10_fixed.py only.

Fix applied: changed the Smart Research Analyzer PDF export font filename from
NotoSansArabic-Regular.ttf to NotoNaskhArabic-Regular.ttf, matching the Arabic
font file used in the STATNOVA repository.

Upload/replace app_v10_fixed.py at the repository root. Keep the existing
NotoNaskhArabic-Regular.ttf font file and requirements.txt at the repository root.
Do not rename the Streamlit main file or replace it with app.py.

The Python file passed a syntax compilation check. The PDF export itself must
be tested after deployment in Streamlit.
