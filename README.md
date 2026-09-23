# ⚙️ Uskuna Nosozligi va Sababini Bashorat Qilish Tizimi

XGBoost va early stopping texnikasi asosida qurilgan ikki bosqichli ML tizimi: 
avval uskunaning buzilish ehtimolini, so'ng (agar xavf yuqori bo'lsa) buzilishning 
ehtimoliy sababini bashorat qiladi.

## 📊 Dataset

**AI4I 2020 Predictive Maintenance Dataset**

- 10,000 ta sanoat uskunasi o'lchovi (harorat, aylanish tezligi, moment, asbob eskirishi)
- Binary target: normal ishlash (0) vs buzilish (1)
- Nomutanosib dataset: ~3.4% buzilish holatlari

## 🧠 Model 1 — Buzilish ehtimoli (Binary Classification)

- **Algoritm:** XGBoost Classifier
- **Early stopping:** validatsiya to'plamidagi `logloss` metrikasi bo'yicha
- **Imbalance bilan ishlash:** `scale_pos_weight` parametri orqali (~27.6)
- **Natijalar:** F1-score ≈ 0.71, ROC-AUC ≈ 0.97

## 🧠 Model 2 — Buzilish sababi (Multi-class Classification)

- Faqat buzilgan holatlar (306 ta, bitta aniq sababli) ustida o'qitilgan
- 4 ta sabab turi: TWF (asbob eskirishi), HDF (issiqlik chiqmasligi), 
  PWF (quvvat muammosi), OSF (haddan tashqari zo'riqish)
- **Natijalar:** Accuracy ≈ 0.98, F1 (macro) ≈ 0.98

## ⚠️ Cheklovlar (halol e'tirof)

- Sabab-model uchun test to'plami kichik (46 namuna) — kattaroq ma'lumot bilan 
  qo'shimcha tekshirish tavsiya etiladi
- Model tasodifiy buzilishlarni (RNF) bashorat qila olmaydi — bu turdagi 
  buzilish ta'rifi bo'yicha hech qanday sensor ko'rsatkichiga bog'liq emas
- Model yakuniy qaror qabul qiluvchi emas — u xavf signali beruvchi 
  yordamchi vosita, yakuniy qarorni mutaxassis qabul qilishi kerak

## 🖥️ Ilova (Streamlit)

Foydalanuvchi 6 ta sensor ko'rsatkichini kiritadi → model buzilish ehtimolini 
chiqaradi → agar xavf yuqori bo'lsa, ikkinchi model ehtimoliy sababni ko'rsatadi.

## 🚀 Ishga tushirish

\`\`\`bash
git clone https://github.com/SIZNING_USERNAME/failure-cause-predictor.git
cd failure-cause-predictor
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## 🛠️ Texnologiyalar

Python · XGBoost · Streamlit · Pandas · Scikit-learn
