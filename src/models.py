import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from config import RFM_DATA_PATH, SEGMENTS_PATH, KMEANS_MODEL_PATH, SALES_MODEL_PATH, PROCESSED_DATA_PATH
import matplotlib.pyplot as plt
from datetime import datetime

def segment_customers():
    """Müşteri segmentasyonu yapar"""
    try:
        rfm = pd.read_csv(RFM_DATA_PATH, index_col='CustomerID')
        
        # Ölçeklendirme
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
        
        # K-Means modeli
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
        
        # Segment isimlendirme
        segment_map = {
            0: 'At Risk',
            1: 'Champions',
            2: 'Potential Loyalists',
            3: 'New Customers'
        }
        rfm['Segment'] = rfm['Cluster'].map(segment_map)
        
        # Kaydet
        rfm.to_csv(SEGMENTS_PATH)
        joblib.dump(kmeans, KMEANS_MODEL_PATH)
        print(f"✅ Segmentasyon tamamlandı ve kaydedildi")
        return rfm
        
    except Exception as e:
        print(f"❌ Segmentasyon hatası: {str(e)}")
        return None

def train_sales_model():
    """Satış tahmin modelini eğitir"""
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=['InvoiceDate'])
        
        # Özellik mühendisliği
        df['Year'] = df['InvoiceDate'].dt.year
        df['Month'] = df['InvoiceDate'].dt.month
        df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
        df['Hour'] = df['InvoiceDate'].dt.hour

        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        # Model için veri hazırlama
        X = df[['Quantity', 'Price', 'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 'Month_sin', 'Month_cos']]
        y = df['TotalPrice']
        
        # Eğitim-test ayırma
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Model eğitimi
        model = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=8,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Değerlendirme
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"✅ Model eğitildi - MAE: {mae:.2f}, R²: {r2:.2f}")
        
        # Kaydet (versiyonlu)
        version = datetime.now().strftime("%Y%m%d_%H%M")
        model_path = SALES_MODEL_PATH.parent / f'sales_model_v{version}.pkl'
        joblib.dump(model, model_path)
        print(f"💾 Model kaydedildi: {model_path}")
        
        return model
        
    except Exception as e:
        print(f"❌ Model eğitim hatası: {str(e)}")
        return None