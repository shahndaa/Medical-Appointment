# Medical Appointment No-Show Intelligence

An end-to-end analytics and machine-learning project on 110,000+ real medical appointments from Brazil's public health system, delivered as an interactive Streamlit web application. The project combines an exploratory dashboard, a trained no-show prediction model with a full evaluation report, and a live risk calculator.

**Live app:** [medical-appointment.streamlit.app](https://medical-appointment.streamlit.app/)
**Dataset:** [Medical Appointment No Shows, Kaggle](https://www.kaggle.com/datasets/joniarroba/noshowappointments)

---

## Why this project

Roughly 20% of scheduled medical appointments in the dataset are no-shows, a costly problem for any healthcare provider. This project goes beyond a static notebook: it turns the analysis into a usable, deployed tool that answers two questions a clinic operations team would actually ask:

1. What patterns drive no-shows? (exploratory data analysis)
2. Which upcoming appointments are at risk, right now? (predictive model and interactive risk calculator)

## Features

- **Interactive filtering** across date range, gender, age, neighbourhood, SMS reminder status, and scholarship status, with every chart updating live.
- **Five-tab dashboard:**
  - **Overview** — KPIs, monthly trend, show/no-show split, auto-generated insight callouts.
  - **Demographics** — age distribution, no-show rate by age group, gender, top neighbourhoods.
  - **Scheduling Patterns** — waiting-time effect, day-of-week heatmap, SMS reminder effect, booking-hour effect.
  - **Health & Behaviour** — chronic conditions and socioeconomic status versus no-show rate, patient history effect, downloadable filtered dataset.
  - **Prediction Model** — ROC and precision-recall curves, adjustable-threshold confusion matrix, permutation feature importance, and a live form to score a hypothetical appointment.
- **Machine learning, done properly:**
  - Three candidate models (logistic regression, random forest, histogram gradient boosting) trained and compared, rather than assuming a single model is best.
  - Chronological train/test split (trained on earlier appointments, tested on later ones) instead of a random split, avoiding the optimistic bias a random split causes with repeat patients.
  - Leakage-safe feature engineering: each patient's prior appointment count and prior no-show rate are computed using only appointments strictly before the current one.
  - Class-imbalance handling (`class_weight="balanced"`), since only about 20% of appointments are no-shows.
  - Evaluation with ROC-AUC and PR-AUC, which are more informative than accuracy under class imbalance, plus a confusion matrix with an adjustable decision threshold.
  - The app trains the selected model live on startup (cached for the session, roughly one to two seconds to fit) instead of loading a pre-pickled artifact. This avoids a class of deployment failure where a pickle file built with one Python or scikit-learn version fails to unpickle on a host running a different version, which is a real issue on Streamlit Community Cloud. `src/train_model.py` is kept as a standalone script for offline experimentation and reproducibility, but the deployed app does not depend on its output.

## Model results

| Model | ROC-AUC | PR-AUC | Train time |
|---|---|---|---|
| Logistic Regression | 0.673 | 0.300 | 0.1s |
| Random Forest | 0.735 | 0.351 | 14.0s |
| Hist Gradient Boosting (selected) | 0.737 | 0.360 | 1.3s |

Evaluated on a held-out chronological test set of 26,450 appointments (all appointments scheduled on or after 2016-06-01), trained on the 84,076 appointments before that date. Full metrics, including the confusion matrix and per-class precision and recall, are generated in `models/metrics.json` and viewable live in the app's Prediction Model tab.

The strongest single predictor is waiting time between scheduling and the appointment date: the longer a patient waits, the more likely they are to miss it, followed by age and a patient's own past no-show history.

## Tech stack

Python, Streamlit, pandas, scikit-learn, Plotly, joblib

## Project structure

```
medical-appointment-dashboard/
├── app.py                      # Streamlit application (UI, charts, model tab)
├── src/
│   ├── features.py             # Shared data cleaning and feature engineering
│   └── train_model.py          # Offline training script (candidate models, evaluation, artifacts)
├── data/
│   └── appointments.csv        # Raw dataset (Kaggle)
├── models/
│   ├── metrics.json            # Evaluation report from a reference offline run
│   └── feature_importance.csv  # Permutation feature importance from that run
│       (model.joblib is generated locally by train_model.py but git-ignored;
│        the deployed app trains its own copy live, see app.py)
├── .streamlit/
│   └── config.toml             # App theme
├── requirements.txt
└── README.md
```

## Run it locally

```bash
git clone https://github.com/<your-username>/medical-appointment-dashboard.git
cd medical-appointment-dashboard
pip install -r requirements.txt

# optional: regenerate the model artifact from scratch
python -m src.train_model

streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (public repo).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and click New app.
3. Select this repo, branch `main`, main file path `app.py`, then Deploy.
4. Once live, copy the app URL into the Live app link at the top of this README.

## Possible extensions

- Hyperparameter tuning with Optuna or GridSearchCV
- SHAP values for per-prediction explainability
- An A/B-testable intervention simulator (for example, the impact of an extra SMS reminder)
- Multi-clinic support with per-clinic filtering, if extended to a dataset with that field

## Data source and license

Dataset originally published on Kaggle as Medical Appointment No Shows (100,000+ appointments, Brazil, 2016), collected by JoniHoppen and Aquarela Advanced Analytics. Used here for educational and portfolio purposes.

## Author

Built as part of a data analytics and machine learning portfolio. Feel free to connect, fork, or open an issue with suggestions.
