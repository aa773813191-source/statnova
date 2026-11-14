STATNOVA V11.6 — Bilingual Statistical Interpretation Upgrade

STATNOVA V11.5 — Smart + Standard Individual and Complete Reports

محتويات الحزمة:
- app_v11_final.py: ملف التطبيق الرئيسي، استبدليه في GitHub مكان الملف الحالي.
- requirements.txt: متطلبات التطبيق.
- NotoNaskhArabic-Regular.ttf: خط التقارير العربية.

التحديثات:
1) في Standard Analysis: تنزيل Word وPDF أسفل كل اختبار منفذ، مع تقرير شامل يجمع الاختبارات المسجلة في الجلسة.
2) في Smart Analysis: تنزيل Word وPDF أسفل نتيجة كل اختبار.
3) سجل اختبارات Smart Analysis يحتفظ بالاختبارات المختلفة أثناء التنقل بين إعدادات التحليل وإعادة تشغيل واجهة Streamlit في الجلسة نفسها، ويُدرجها في التقارير الشاملة.
4) التنزيلات الشاملة في Smart Analysis تشمل Excel وCSV وWord وPDF، وملف SPSS SAV عند توفر pyreadstat.
5) لا يتم إنشاء نتائج إحصائية غير منفذة أو استنتاج قيم غير محسوبة.

طريقة التحديث:
- افتحي مستودع GitHub الحالي.
- ارفعي app_v11_final.py وrequirements.txt والخط المرفق إذا لم يكن موجودًا، مع الاستبدال.
- اتركي Streamlit Cloud يعيد تشغيل التطبيق.
- لا تنشئي تطبيقًا جديدًا ولا تغيّري رابط التطبيق.

ملاحظة:
- SAV هو ملف بيانات للتحليل في SPSS، وليس تقرير مخرجات SPSS كاملًا.
- سجل الاختبارات محفوظ في جلسة التطبيق الحالية؛ إعادة تحميل الصفحة/الجلسة أو فتح جلسة جديدة قد تبدأ سجلًا جديدًا.


V11.6 changes:
- Method-specific coefficient labels for Pearson (r), Spearman (rho), and Kendall (tau).
- Automatic Arabic and English interpretation with direction/strength where applicable, p-value decision, and sample size when available.
- Academic wording in Arabic and English.
- Interpretation included in individual Word/PDF exports for Standard and Smart analyses.
- No uncomputed confidence intervals or effect sizes are invented.
