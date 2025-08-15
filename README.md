# 🩺 Medical Appointment No-Show Intelligence

An end-to-end analytics + machine-learning project on 110K+ real medical appointments from Brazil's public health system, delivered as an interactive **Streamlit** web app: exploratory dashboard, a trained no-show prediction model with a full evaluation report, and a live "what-if" risk calculator.

**🔗 Live app:** _add your Streamlit Community Cloud link here after deployment_
**📊 Dataset:** [Medical Appointment No Shows — Kaggle](https://www.kaggle.com/datasets/joniarroba/noshowappointments)

---

## Why this project

~20% of scheduled medical appointments in the dataset are no-shows — a costly problem for any healthcare provider. This project goes beyond a static notebook: it turns the analysis into a **usable, deployed tool** that answers two questions a clinic operations team would actually ask:

1. **What patterns drive no-shows?** (Exploratory Data Analysis)
2. **Which upcoming appointments are at risk, right now?** (Predictive model + interactive risk calculator)

## ✨ Features

- **Interactive filtering** — date range, gender, age, neighbourhood, SMS reminder status, scholarship status, all live-updating every chart.
- **5-tab dashboard**
  - 📊 **Overview** — KPIs, monthly trend, show/no-show split, auto-generated insight callouts.
  - 👥 **Demographics** — age distribution, no-show rate by age group, gender, top neighbourhoods.
  - 🗓️ **Scheduling Patterns** — waiting-time effect, day-of-week heatmap, SMS reminder effect, booking-hour effect.
  - 💊 **Health & Behaviour** — chronic conditions & socioeconomic status vs. no-show rate, patient history effect, downloadable filtered dataset.
  - 🤖 **Prediction Model** — ROC & Precision-Recall curves, adjustable-threshold confusion matrix, permutation feature importance, and a live form to score a hypothetical appointment.
- **Machine learning, done properly:**
  - Three candidate models (Logistic Regression, Random Forest, Hist Gradient Boosting) trained and compared — not just one model assumed to be best.
  - **Chronological train/test split** (trained on earlier appointments, tested on later ones) instead of a random split, to avoid the optimistic bias random splitting causes with repeat patients.
  - **Leakage-safe feature engineering**: each patient's "prior appointments" and "prior no-show rate" are computed using only appointments strictly *before* the current one.
  - Class-imbalance handling (`class_weight="balanced"`) since only ~20% of appointments are no-shows.
  - Evaluated with ROC-AUC and PR-AUC (more informative than accuracy under class imbalance), plus a confusion matrix with an adjustable decision threshold.

## 📈 Model results

| Model | ROC-AUC | PR-AUC | Train time |
|---|---|---|---|
| Logistic Regression | 0.673 | 0.300 | 0.1s |
| Random Forest | 0.735 | 0.351 | 14.0s |
| **Hist Gradient Boosting (selected)** | **0.737** | **0.360** | 1.3s |

Evaluated on a held-out chronological test set of 26,450 appointments (all appointments scheduled on/after 2016-06-01), trained on the 84,076 before that date. Full metrics (confusion matrix, per-class precision/recall) are generated in `models/metrics.json` and viewable live in the app's **Prediction Model** tab.

The strongest single predictor is **waiting time** between scheduling and the appointment date — the longer a patient waits, the more likely they are to miss it — followed by **age** and a patient's own **past no-show history**.

## 🧱 Tech stack

`Python` · `Streamlit` · `pandas` · `scikit-learn` · `Plotly` · `joblib`

## 📁 Project structure

```
medical-appointment-dashboard/
├── app.py                      # Streamlit application (UI + charts + model tab)
├── src/
│   ├── features.py             # Shared data cleaning & feature engineering
│   └── train_model.py          # Offline training script (candidate models, evaluation, artifacts)
├── data/
│   └── appointments.csv        # Raw dataset (Kaggle)
├── models/
│   ├── model.joblib             # Trained champion model (loaded by the app)
│   ├── metrics.json             # Full evaluation report
│   └── feature_importance.csv   # Permutation feature importance
├── .streamlit/
│   └── config.toml              # App theme
├── requirements.txt
└── README.md
```

## 🚀 Run it locally

```bash
git clone https://github.com/<your-username>/medical-appointment-dashboard.git
cd medical-appointment-dashboard
pip install -r requirements.txt

# optional: regenerate the model artifact from scratch
python -m src.train_model

streamlit run app.py
```

The app opens at `http://localhost:8501`.

## ☁️ Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (public repo).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, click **New app**.
3. Select this repo, branch `main`, main file path `app.py`, then **Deploy**.
4. Once live, copy the app URL into the "Live app" link at the top of this README.

## 🔮 Possible extensions

- Hyperparameter tuning with `Optuna` / `GridSearchCV`
- SHAP values for per-prediction explainability
- A/B-testable "intervention" simulator (e.g. impact of an extra SMS reminder)
- Multi-clinic support with per-clinic filtering, if extended to a real dataset with that field

## 📄 Data source & license

Dataset originally published on Kaggle: *Medical Appointment No Shows* (100k+ appointments, Brazil, 2016), collected by JoniHoppen / Aquarela Advanced Analytics. Used here for educational/portfolio purposes.

## 👤 Author

Built as part of a data analytics & machine learning portfolio.
Feel free to connect, fork, or open an issue with suggestions.
