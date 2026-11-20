"""Reading of the raw data."""
import os
import pandas as pd

from src.config import RAW_FILE, RAW_URL


def load_raw():
    """Read the raw file."""
    if os.path.exists(RAW_FILE):
        return pd.read_csv(RAW_FILE, encoding="latin-1", low_memory=False)
    return pd.read_csv(RAW_URL, encoding="latin-1", low_memory=False)
