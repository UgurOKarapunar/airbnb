import streamlit as st
import pandas as pd
import joblib




# Modeli ve ön işleme nesnelerini yükleme
classifier = joblib.load('decision_tree_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Streamlit uygulaması
st.title('Airbnb Price Prediction')

# Room Type seçeneklerini tanımlama ve metin tabanlı seçici
room_types = ['Entire home/apt', 'Private room', 'Shared room']  # Metin bazlı seçenekler
selected_room_type = st.selectbox('Room Type', room_types)

# Room Type değerini sayısal formata dönüştürme
room_type_encoded = label_encoder.transform([selected_room_type])[0]

# Kullanıcıdan diğer girişleri al
latitude = st.number_input('Latitude', min_value=-90.0, max_value=90.0, value=41.144)
longitude = st.number_input('Longitude', min_value=-180.0, max_value=180.0, value=29.091)
minimum_nights = st.number_input('Minimum Nights', min_value=1, value=3)

# Giriş verilerini hazırlama
new_data = pd.DataFrame([[room_type_encoded, latitude, longitude, minimum_nights]],
                        columns=["room_type", "latitude", "longitude", "minimum_nights"])

# Tahmin yapma
predicted_price = classifier.predict(new_data)
st.write(f"Predicted Price: {predicted_price[0]}")

