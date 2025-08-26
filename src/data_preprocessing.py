import pandas as pd
import numpy as np
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from tqdm import tqdm
import os

def load_and_clean_data():
    """Veriyi yükleyip temizler"""
    # Bellek optimizasyonu
    dtypes = {
        'Invoice': 'category',
        'StockCode': 'category',
        'Description': 'category',
        'Quantity': 'int32',
        'InvoiceDate': 'str',
        'Price': 'float32',
        'Customer ID': 'float32',
        'Country': 'category'
    }
    
    try:
        # Büyük veri için chunk'larla okuma
        chunks = []
        for chunk in tqdm(pd.read_csv(RAW_DATA_PATH, encoding='ISO-8859-1', 
                                     dtype=dtypes, chunksize=100000)):
            chunks.append(chunk)
        df = pd.concat(chunks)
        
        # Temizleme işlemleri
        df = df.rename(columns={'Customer ID': 'CustomerID'})
        df = df.dropna(subset=['CustomerID'])
        df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['TotalPrice'] = df['Quantity'] * df['Price']
        
        # Kaydetme
        df.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"✅ Temizlenmiş veri kaydedildi: {PROCESSED_DATA_PATH}")
        return df
    
    except FileNotFoundError:
        print(f"❌ Hata: Veri dosyası bulunamadı: {RAW_DATA_PATH}")
        return None