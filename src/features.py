"""
Data loading and feature engineering for the Medical Appointment No-Show project.

This module is imported by both the Streamlit app (app.py) and the offline
training script (src/train_model.py) so that the exact same preprocessing
logic is used everywhere -> no train/serve skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RAW_COLUMNS_RENAME = {
    "PatientId": "patient_id",
    "AppointmentID": "appointment_id",
    "Gender": "gender",
    "ScheduledDay": "scheduled_day",
    "AppointmentDay": "appointment_day",
    "Age": "age",
    "Neighbourhood": "neighbourhood",
    "Scholarship": "scholarship",
    "Hipertension": "hypertension",
    "Diabetes": "diabetes",
    "Alcoholism": "alcoholism",
    "Handcap": "handicap",
    "SMS_received": "sms_received",
    "No-show": "no_show",
}

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

MODEL_FEATURES = [
    "age",
    "gender_code",
    "scholarship",
    "hypertension",
    "diabetes",
    "alcoholism",
    "handicap_flag",
    "sms_received",
    "waiting_days",
    "appointment_dow",
    "appointment_month_num",
    "scheduled_hour",
    "neighbourhood_freq",
    "prior_appointments",
    "prior_no_show_rate",
]


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw Kaggle CSV file."""
    df = pd.read_csv(path)
    df = df.rename(columns=RAW_COLUMNS_RENAME)
    return df


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw data and build all engineered features.

    Returns a new dataframe - the input is not mutated.
    """
    df = df.copy()

    # ---- Basic type fixes -------------------------------------------------
    df["scheduled_day"] = pd.to_datetime(df["scheduled_day"])
    df["appointment_day"] = pd.to_datetime(df["appointment_day"])

    # ---- Data quality fixes ------------------------------------------------
    # A handful of rows have age == -1, which is not physically possible.
    df = df[df["age"] >= 0].copy()
    # Cap unrealistic ages (data entry errors, e.g. age 115 appears once)
    df["age"] = df["age"].clip(upper=100)

    # ---- Target -------------------------------------------------------------
    df["no_show_flag"] = (df["no_show"] == "Yes").astype(int)

    # ---- Date-derived features ----------------------------------------------
    # Normalise to date-only before subtracting (the raw timestamps in
    # appointment_day are always midnight, but scheduled_day has a real time)
    sched_date = df["scheduled_day"].dt.normalize()
    appt_date = df["appointment_day"].dt.normalize()
    df["waiting_days"] = (appt_date - sched_date).dt.days.clip(lower=0)

    df["appointment_dow"] = df["appointment_day"].dt.dayofweek  # 0=Mon
    df["appointment_dow_name"] = pd.Categorical(
        df["appointment_day"].dt.day_name(), categories=DAY_ORDER, ordered=True
    )
    df["appointment_month"] = df["appointment_day"].dt.tz_localize(None).dt.to_period("M").astype(str)
    df["appointment_month_num"] = df["appointment_day"].dt.month
    df["scheduled_hour"] = df["scheduled_day"].dt.hour

    # ---- Categorical encodings -----------------------------------------------
    df["gender_code"] = (df["gender"] == "M").astype(int)
    df["handicap_flag"] = (df["handicap"] > 0).astype(int)

    df["age_group"] = pd.cut(
        df["age"],
        bins=[-1, 12, 18, 30, 45, 60, 100],
        labels=["0-12 (Child)", "13-18 (Teen)", "19-30", "31-45", "46-60", "60+"],
    )

    waiting_bins = [-1, 0, 3, 7, 14, 30, 10_000]
    waiting_labels = ["Same day", "1-3 days", "4-7 days", "8-14 days", "15-30 days", "30+ days"]
    df["waiting_bucket"] = pd.cut(df["waiting_days"], bins=waiting_bins, labels=waiting_labels)

    # Frequency encoding for neighbourhood (81 unique values -> too many for
    # one-hot, and it avoids leaking the target the way mean-encoding would)
    freq_map = df["neighbourhood"].value_counts(normalize=True)
    df["neighbourhood_freq"] = df["neighbourhood"].map(freq_map)

    # ---- Patient history features (computed causally, no leakage) -----------
    # For every patient, look only at THEIR earlier appointments (sorted by
    # scheduled date) to build "how many times have they shown up / no-showed
    # before this appointment". shift(1) guarantees the current row's own
    # outcome is never used to build its own feature.
    df = df.sort_values(["patient_id", "scheduled_day"]).reset_index(drop=True)
    grp = df.groupby("patient_id")["no_show_flag"]
    df["prior_appointments"] = grp.cumcount()
    df["prior_no_shows"] = grp.transform(lambda s: s.shift(fill_value=0).cumsum())
    df["prior_no_show_rate"] = np.where(
        df["prior_appointments"] > 0, df["prior_no_shows"] / df["prior_appointments"], 0.0
    )

    df = df.sort_values("appointment_day").reset_index(drop=True)
    return df


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Split the data chronologically by appointment_day.

    A random split would let information from a patient's future visit leak
    into the training set for an earlier visit's neighbours in time. Splitting
    by date mimics how the model would actually be used in production:
    trained on the past, evaluated on the future.
    """
    df_sorted = df.sort_values("appointment_day")
    cutoff_idx = int(len(df_sorted) * (1 - test_frac))
    cutoff_date = df_sorted.iloc[cutoff_idx]["appointment_day"]
    train_df = df_sorted[df_sorted["appointment_day"] < cutoff_date]
    test_df = df_sorted[df_sorted["appointment_day"] >= cutoff_date]
    return train_df, test_df, cutoff_date
