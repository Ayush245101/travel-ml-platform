import logging
from typing import List
import pandas as pd
import numpy as np

DATE_COL = "date"

def setup_logging(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        fmt = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger

def read_csv_with_date(path: str, date_cols: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",", engine="python")
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
    return df

def ensure_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def add_date_parts(df: pd.DataFrame, date_col: str = DATE_COL) -> pd.DataFrame:
    if date_col in df.columns:
        dt = df[date_col].dt
        df = df.assign(
            year=dt.year,
            month=dt.month,
            day=dt.day,
            dayofweek=dt.dayofweek,
            is_weekend=dt.dayofweek.isin([5, 6]).astype(int),
        )
    return df

def safe_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan