import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Store Customer Clustering App", layout="centered")

st.title("Store Customer Clustering App")
st.write(
    "Bu uygulama, mağaza müşterilerinin yaş, toplam harcama ve satın alma miktarı metriklerine göre K-Means modeli ile hangi müşteri grubunda yer aldığını belirler."
)


@st.cache_resource
def load_model():
    return joblib.load("real_data_clustering_model.pkl")


model = load_model()

st.subheader("Musteri Metriklerini Giriniz:")

age = st.number_input("Musteri Yasi (Age)", min_value=15.0, max_value=100.0, value=30.0)
total_purchase = st.number_input("Toplam Satin Alma Tutari (Total Purchase Amount)", min_value=0.0, max_value=50000.0, value=500.0)
quantity = st.number_input("Satin Alma Miktari (Quantity)", min_value=1.0, max_value=500.0, value=5.0)

if st.button("Musteri Segmentini Bul", type="primary"):
    try:
        input_data = pd.DataFrame({
            "age": [age],
            "Total_Purchase_Amount": [total_purchase],
            "quantity": [quantity]
        })
        
        cluster = model.predict(input_data)
        st.success(f"Analiz Sonucu: Bu müşteri **Cluster {cluster[0]}** grubuna aittir.")
    except Exception as e:
        st.error(f"Tahmin sirasinda bir hata olustu: {e}")