STATNOVA — Quality Upgrade

Included files:
- app.py: updated Streamlit application
- requirements.txt: application dependencies (includes pyreadstat for SPSS .sav export and xlrd for .xls input)
- NotoNaskhArabic-Regular.ttf: Arabic font required by the bilingual PDF report

Deployment:
1. Keep a backup of the current GitHub repository files.
2. Replace the current main Python file with this app.py (or rename app.py to the filename configured in Streamlit Cloud).
3. Replace requirements.txt with the included requirements.txt.
4. Upload NotoNaskhArabic-Regular.ttf beside app.py in the repository root.
5. Commit changes and wait for Streamlit Cloud to rebuild.
6. Open the existing STATNOVA URL and rerun the same test dataset.

Changes in this build:
- Very small p-values display as <0.0001 instead of a misleading 0.0000/0.
- Added name-informed checks for common range/unit concerns (percentages, sleep hours, ages, satisfaction scales, and 0–100 score-like variables).
- Added identifier integrity checks for missing and repeated IDs.
- Exports now include potential range/unit flags in the text and Excel research reports.
- No observations are automatically deleted, imputed, or modified.

Important:
Range checks are review prompts, not proof of invalid data. Confirm each variable's actual scale and protocol. This remains a research-support tool; verify statistical methods and study design before publication.
