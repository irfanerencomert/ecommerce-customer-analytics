import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
from pathlib import Path
from config import SEGMENTS_PATH, MODELS_DIR
from datetime import datetime

# Sayfa ayarları
st.set_page_config(
    page_title="E-Ticaret Analitik Paneli",
    layout="wide",
    page_icon="📊"
)

# Başlık
st.title("📊 E-Ticaret Müşteri Analitiği ve Satış Tahmini")


# Yardımcı fonksiyonlar
@st.cache_data
def load_segment_data():
    try:
        return pd.read_csv(SEGMENTS_PATH, index_col='CustomerID')
    except FileNotFoundError:
        st.error("⚠️ Segment verisi bulunamadı. Lütfen önce segmentasyon modelini çalıştırın.")
        return None


@st.cache_resource
def load_latest_sales_model():
    try:
        # En yeni model dosyasını bul
        model_files = list(MODELS_DIR.glob("sales_model_*.pkl"))
        if not model_files:
            raise FileNotFoundError

        latest_model = max(model_files, key=os.path.getctime)
        return joblib.load(latest_model), latest_model.stem
    except FileNotFoundError:
        st.error("⚠️ Tahmin modeli bulunamadı. Lütfen önce model eğitimini yapın.")
        return None, None


# Verileri yükle
rfm = load_segment_data()
model, model_name = load_latest_sales_model()

# Yan menü
st.sidebar.title("Navigasyon")
page = st.sidebar.radio("Sayfalar", ["Genel Bakış", "Müşteri Segmentasyonu", "Satış Tahmini"])

# Genel Bakış Sayfası
if page == "Genel Bakış":
    st.header("Proje Genel Bakış")

    if rfm is not None:
        # Metrikler
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Toplam Müşteri", f"{rfm.shape[0]:,}")
        col2.metric("Toplam Ciro", f"€{rfm['Monetary'].sum() / 1e6:.2f}M")
        col3.metric("Ort. Sipariş Değeri", f"€{rfm['Monetary'].sum() / rfm['Frequency'].sum():.2f}")
        col4.metric("Tekrar Oranı", f"%{(rfm[rfm['Frequency'] > 1].shape[0] / rfm.shape[0] * 100):.1f}")

    st.divider()

    # Proje Bilgileri
    st.subheader("Proje Detayları")
    st.markdown("""
    **Proje Amacı:**  
    Bu E-Ticaret Analitik Paneli, müşteri davranışlarını anlamak ve satışları optimize etmek amacıyla geliştirilmiştir. Müşteri segmentasyonu ile hedef kitle analizleri yapılırken, gerçek zamanlı satış tahminleri ile iş kararları desteklenmektedir.

    **Kullanılan Teknolojiler:**  
    Python ve Streamlit kullanılarak hızlı ve interaktif bir web uygulaması geliştirilmiştir. Veri işleme için Pandas, görselleştirme için Plotly, makine öğrenmesi modelleri için scikit-learn ve joblib tercih edilmiştir.

    **Analiz & Modelleme Süreci:**  
    Müşteri segmentasyonu için Recency, Frequency, Monetary (RFM) analizi uygulanmıştır. Satış tahmini ise Random Forest algoritmasıyla gerçekleştirilmiş ve model performansı düzenli olarak güncellenmektedir.

    **Uygulama Özellikleri:**  
    Kullanıcılar, genel satış ve müşteri özetlerini görebilir, segmentasyon sonuçlarını detaylı inceleyebilir ve farklı parametrelerle satış tahminleri yapabilirler.
    """)

# Müşteri Segmentasyonu Sayfası
elif page == "Müşteri Segmentasyonu":
    st.header("Müşteri Segmentasyon Analizi")

    if rfm is not None:
        # Segment özeti
        st.subheader("Segment İstatistikleri")
        segment_summary = rfm.groupby('Segment').agg({
            'Recency': ['mean', 'median'],
            'Frequency': ['mean', 'median'],
            'Monetary': ['mean', 'median', 'sum']
        }).round(2)
        segment_summary.columns = [f"{col[0]}_{col[1]}" for col in segment_summary.columns]
        st.dataframe(segment_summary.style.background_gradient(cmap='Blues'))

        # 3D Görselleştirme
        st.subheader("RFM Segmentasyonu")
        with st.expander("3D Segmentasyon Haritası"):
            fig = px.scatter_3d(
                rfm.reset_index(),
                x='Recency',
                y='Frequency',
                z='Monetary',
                color='Segment',
                hover_name='CustomerID',
                height=700
            )
            st.plotly_chart(fig, use_container_width=True)

        # Müşteri detayları
        st.subheader("Segment Detayları")
        selected_segment = st.selectbox("Segment Seçin", rfm['Segment'].unique())
        segment_data = rfm[rfm['Segment'] == selected_segment]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Müşteri Sayısı", segment_data.shape[0])
            st.dataframe(segment_data[['Recency', 'Frequency', 'Monetary']].describe())

        with col2:
            top_customers = segment_data['Monetary'].sort_values(ascending=False).head(10)
            others_sum = segment_data['Monetary'].sum() - top_customers.sum()

            # Yeni bir Series oluştur, 'Diğer' adlı dilimi ekle
            pie_data = top_customers.append(pd.Series({'Diğer': others_sum}))

            fig = px.pie(
                names=pie_data.index,
                values=pie_data.values,
                title=f'{selected_segment} Segmenti - Gelir Dağılımı'
            )
            st.plotly_chart(fig, use_container_width=True)

# Satış Tahmini Sayfası
elif page == "Satış Tahmini":
    st.header("Gerçek Zamanlı Satış Tahmini")

    if model and model_name:
        st.success(f"Model Yüklendi: {model_name}")

        # Giriş formu
        with st.form("prediction_form"):
            st.subheader("Tahmin Parametreleri")

            col1, col2 = st.columns(2)
            with col1:
                quantity = st.slider("Ürün Miktarı", 1, 100, 10)
                unit_price = st.number_input("Birim Fiyat (€)", min_value=0.01, value=25.0, step=0.5)

            with col2:
                month = st.select_slider("Ay", options=list(range(1, 13)), value=6)
                day_of_week = st.selectbox(
                    "Haftanın Günü",
                    options=list(range(7)),
                    format_func=lambda x: ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"][x]
                )
                hour = st.select_slider("Saat", options=list(range(24)), value=14)

            submit = st.form_submit_button("Tahmin Et")

            if submit:
                # input_data'yu dataframe olarak oluştur
                input_df = pd.DataFrame({
                    'Quantity': [quantity],
                    'Price': [unit_price],
                    'Month': [month],
                    'DayOfWeek': [day_of_week],
                    'Hour': [hour]
                })

                # Özellik mühendisliği: trigonometrik dönüşümler
                input_df['Hour_sin'] = np.sin(2 * np.pi * input_df['Hour'] / 24)
                input_df['Hour_cos'] = np.cos(2 * np.pi * input_df['Hour'] / 24)
                input_df['DayOfWeek_sin'] = np.sin(2 * np.pi * input_df['DayOfWeek'] / 7)
                input_df['DayOfWeek_cos'] = np.cos(2 * np.pi * input_df['DayOfWeek'] / 7)
                input_df['Month_sin'] = np.sin(2 * np.pi * input_df['Month'] / 12)
                input_df['Month_cos'] = np.cos(2 * np.pi * input_df['Month'] / 12)

                # Modelin beklediği özellikler
                features = ['Quantity', 'Price', 'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 'Month_sin',
                            'Month_cos']

                # Tahmin yap
                prediction = model.predict(input_df[features])[0]

                st.success(f"**Tahmini Satış Tutarı:** €{prediction:.2f}")
                st.metric("Toplam Tahmin", f"€{prediction:.2f}")

                # Detaylı açıklama
                with st.expander("Tahmin Detayları"):
                    st.write(f"- **Miktar:** {quantity} adet")
                    st.write(f"- **Birim Fiyat:** €{unit_price:.2f}")
                    st.write(
                        f"- **Dönem:** {month}. ay, {['Pzt', 'Sal', 'Çar', 'Per', 'Cum', 'Cmt', 'Paz'][day_of_week]}, {hour}:00"
                    )
                    st.write(f"- **Model:** {model_name}")

    # Model yönetimi
    st.divider()
    st.subheader("Model Yönetimi")

    if st.button("Modeli Yeniden Eğit"):
        with st.spinner("Model eğitiliyor..."):
            from src.models import train_sales_model

            new_model = train_sales_model()
            if new_model:
                st.experimental_rerun()

# Alt bilgi
st.divider()
st.caption("© 2025 E-Ticaret Analitik Paneli | İrfan Eren CÖMERT")