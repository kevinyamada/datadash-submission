"""Load the competition data.

By default the files are downloaded once from Hugging Face and kept in a cache on
your computer, so later runs are instant. If you already have the files in a folder,
point DATA_DIR at that folder instead and nothing is downloaded:

    macOS / Linux:       DATA_DIR=/path/to/folder python run.py validate
    Windows PowerShell:  $env:DATA_DIR = "C:/path/to/folder"; python run.py validate
"""
import os
from pathlib import Path
import pandas as pd
HF_DATASET = 'Baliyang/SSU_data_dash'
FILE_NAMES = {'listings': 'listings', 'off_market': 'test_off_market', 'list_aware': 'test_list_aware'}
DATE_COLUMNS = ['date_listed', 'date_sold', 'external_avm_estimate_date']
AFTER_SALE_COLUMNS = ['sold_price', 'date_sold', 'days_on_market', 'external_avm_estimate', 'external_avm_estimate_date']
DOWNLOAD_HELP = '\nCould not download {file} from Hugging Face.\n\nCheck your internet connection and try again. If it still fails, download the files by hand from\nhttps://huggingface.co/datasets/{repo} and set DATA_DIR to their folder (see data.py).\n\nThe error from Hugging Face was: {error}\n'

def find_file(name):
    """Return the path of one data file, downloading it the first time."""
    stem = FILE_NAMES[name]
    folder = os.environ.get('DATA_DIR')
    if not folder and (Path(__file__).parent / 'data_local').is_dir():
        folder = str(Path(__file__).parent / 'data_local')
    if folder:
        for ending in ('.parquet', '.csv'):
            path = Path(folder) / (stem + ending)
            if path.exists():
                return path
        raise FileNotFoundError(f'DATA_DIR is {folder}, but {stem}.parquet or {stem}.csv is not in it.')
    from huggingface_hub import hf_hub_download
    try:
        return Path(hf_hub_download(HF_DATASET, stem + '.parquet', repo_type='dataset', revision='96a258439e2a85f650699be8339e52d8ac5614b7'))
    except Exception as error:
        message = DOWNLOAD_HELP.format(file=stem + '.parquet', repo=HF_DATASET, error=f'{type(error).__name__}: {error}')
        raise RuntimeError(message) from None

def load(name):
    """Load "listings", "off_market" or "list_aware" as a pandas DataFrame."""
    path = find_file(name)
    if path.suffix == '.parquet':
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    for column in DATE_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column])
    return df

def hide_answers(homes, track):
    """Keep only the columns a test file of this track has.

    Validation uses this so that a model is never shown the answer (sold_price) or
    anything else that is only known after the sale. Off-market homes also lose
    their list price.
    """
    hidden = AFTER_SALE_COLUMNS + (['list_price'] if track == 'off_market' else [])
    return homes.drop(columns=[c for c in hidden if c in homes.columns])
