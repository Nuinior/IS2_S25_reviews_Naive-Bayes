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

        def preprocess_text(text):
            """ฟังก์ชันสำหรับทำ Text Preprocessing"""
            nltk.download('punkt')  # ✅ โหลด punkt ภายในฟังก์ชัน เพื่อป้องกันปัญหา
            tokens = word_tokenize(text.lower())  # Tokenization
            tokens = [word for word in tokens if word.isalnum()]  # ลบเครื่องหมาย
            tokens = [word for word in tokens if word not in stop_words]  # ลบ Stopwords
            return " ".join(tokens)

        # 🔹 ทำ Preprocessing ข้อความ
        processed_text = preprocess_text(translated_text)

        # 🔹 ทำนายผลด้วยโมเดล Naive Bayes
        prediction = model.predict([processed_text])[0]

        sentiment_mapping = {1: "Positive 😊", 0: "Neutral 😐", -1: "Negative 😞"}
        st.write("Sentiment:", sentiment_mapping[prediction])

    else:
        st.write("Please enter some text.")
