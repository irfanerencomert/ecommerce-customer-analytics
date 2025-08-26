from pathlib import Path
import os

# Temel yollar
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_PATH = DATA_DIR / 'raw' / 'online_retail_II.csv'
PROCESSED_DATA_PATH = DATA_DIR / 'processed' / 'cleaned_online_retail.csv'
RFM_DATA_PATH = DATA_DIR / 'processed' / 'rfm_data.csv'
SEGMENTS_PATH = DATA_DIR / 'processed' / 'customer_segments.csv'

# Model yolları
MODELS_DIR = BASE_DIR / 'models'
KMEANS_MODEL_PATH = MODELS_DIR / 'kmeans_model.pkl'
SALES_MODEL_PATH = MODELS_DIR / 'sales_prediction_model.pkl'

# Klasörleri oluştur
for path in [DATA_DIR / 'raw', DATA_DIR / 'processed', MODELS_DIR]:
    os.makedirs(path, exist_ok=True)