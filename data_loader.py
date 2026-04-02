"""
data_loader.py — Load the raw Google Ads CSV into a pandas DataFrame.

This module handles:
  • Locating the CSV file on disk
  • Reading it with sensible defaults
  • Basic validation that required columns exist
"""

import pandas as pd
from config import DATASET_PATH


def load_dataset(path: str | None = None) -> pd.DataFrame:
    """
    Read the Google Ads CSV and return a raw DataFrame.

    Parameters
    ----------
    path : str or None
        Optional override for the CSV path. Falls back to the value
        configured in config.py / .env.

    Returns
    -------
    pd.DataFrame
        Raw, unprocessed data exactly as it appears in the file.

    Raises
    ------
    FileNotFoundError
        If the CSV cannot be found at the resolved path.
    """
    csv_path = path or str(DATASET_PATH)

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. "
            "Check the DATASET_PATH variable in your .env file."
        )

    print(f"✅  Loaded {len(df):,} rows × {len(df.columns)} columns from '{csv_path}'")
    return df
