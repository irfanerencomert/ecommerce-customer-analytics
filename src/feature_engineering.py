import pandas as pd
import numpy as np
from config import PROCESSED_DATA_PATH, RFM_DATA_PATH
from tqdm import tqdm

def calculate_rfm():
    """RFM hesaplamalarını yapar"""
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=['InvoiceDate'])
        
        # RFM Analizi
        snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
        
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
            'Invoice': 'nunique',
            'TotalPrice': 'sum'
        }).round(2)
        
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        
        # RFM Skorlama
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        
        # Kaydet
        rfm.to_csv(RFM_DATA_PATH)
        print(f"✅ RFM verisi kaydedildi: {RFM_DATA_PATH}")
        return rfm
        
    except Exception as e:
        print(f"❌ RFM hesaplama hatası: {str(e)}")
        return None