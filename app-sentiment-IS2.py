import streamlit as st
import joblib
import nltk
from googletrans import Translator
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# ✅ โหลดโมเดล NLTK ที่จำเป็นก่อนใช้งาน
nltk.download('punkt')
nltk.download('stopwords')

# ✅ โหลดโมเดล Machine Learning
model = joblib.load("sentiment_model_IS2.pkl")

st.title("Samsung Galaxy S25 Sentiment Analysis")

# Input text
user_input = st.text_area("Enter a review:")

if st.button("Analyze Sentiment"):
    if user_input:
        # แปลภาษา
        translator = Translator()
        try:
            translated_text = translator.translate(user_input, src='auto', dest='en').text
        except:
            translated_text = user_input  # ถ้าแปลไม่ได้ ให้ใช้ข้อความเดิม
        
        stop_words = set(stopwords.words('english'))

        @st.cache_data  # ใช้ caching เพื่อดาวน์โหลด punkt แค่ครั้งเดียว
        def download_punkt():
            try:
                nltk.download('punkt')
                return True  # สำเร็จ
            except LookupError:
                return False  # ไม่สำเร็จ
        
        def preprocess_text(text):  # บรรทัดที่ 39
        #"""ฟังก์ชันสำหรับทำ Text Preprocessing"""
        if not download_punkt():
            st.error("Error: punkt resource not found. Please check your internet connection and try again. If the problem persists, consider using a pre-built Docker image with NLTK resources.")
        return ""

            tokens = word_tokenize(text.lower())  # เยื้อง 4 ช่องว่าง
            tokens = [word for word in tokens if word.isalnum()]  # เยื้อง 4 ช่องว่าง
            tokens = [word for word in tokens if word not in stop_words]  # เยื้อง 4 ช่องว่าง
            return " ".join(tokens)  # เยื้อง 4 ช่องว่าง

        # 🔹 ทำ Preprocessing ข้อความ
        processed_text = preprocess_text(translated_text)

        # 🔹 ทำนายผลด้วยโมเดล Naive Bayes
        prediction = model.predict([processed_text])[0]

        sentiment_mapping = {1: "Positive 😊", 0: "Neutral 😐", -1: "Negative 😞"}
        st.write("Sentiment:", sentiment_mapping[prediction])

    else:
        st.write("Please enter some text.")
