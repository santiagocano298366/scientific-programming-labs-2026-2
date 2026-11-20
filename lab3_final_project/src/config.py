"""Paths and constants of the project."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_FILE = os.path.join(ROOT, "data", "raw", "WRC_dataset_surya_et_al_2021_final.csv")
RAW_URL = ("https://raw.githubusercontent.com/santiagocano298366/"
           "scientific-programming-labs-2026-2/main/data_raw/WRC_dataset_surya_et_al_2021_final.csv")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
FIGURES_DIR = os.path.join(ROOT, "results", "figures")
TABLES_DIR = os.path.join(ROOT, "results", "tables")

GROUPS = ["coarse", "medium", "fine"]
COLORS = {"coarse": "orange", "medium": "green", "fine": "blue"}

TEXTURE_MAP = {
    "sand": "coarse", "loamy sand": "coarse", "sandy loam": "coarse",
    "sandy clay loam": "medium", "loam": "medium", "silt loam": "medium",
    "silt": "medium", "silty clay loam": "medium", "clay loam": "medium",
    "clay": "fine", "sandy clay": "fine", "silty clay": "fine",
}
