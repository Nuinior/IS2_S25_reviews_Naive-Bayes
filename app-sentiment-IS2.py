import streamlit as st
import joblib

# Load model
model = joblib.load("sentiment_model_IS2.pkl")

st.title("Samsung Galaxy S25 Sentiment Analysis")

# Input text
user_input = st.text_area("Enter a review:")

if st.button("Analyze Sentiment"):
    if user_input:
        # Process the input text
        from googletrans import Translator
        translator = Translator()

        def translate_text(text):
            try:
                return translator.translate(text, src='auto', dest='en').text
            except:
                return text  # Return original text if translation fails

        translated_text = translate_text(user_input)

        import nltk
        try:
             nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        
        # ดาวน์โหลด punkt และ stopwords ก่อนใช้งาน
        nltk.download('punkt')
        nltk.download('stopwords')
        
        stop_words = set(stopwords.words('english'))

        def preprocess_text(text):
        #"""ฟังก์ชันสำหรับทำ Text Preprocessing"""
            tokens = word_tokenize(text.lower())  # Tokenization
            tokens = [word for word in tokens if word.isalnum()]  # ลบเครื่องหมาย
            tokens = [word for word in tokens if word not in stop_words]  # ลบ Stopwords
            return " ".join(tokens)

        processed_text = preprocess_text(translated_text)
        prediction = model.predict([processed_text])[0]

        sentiment_mapping = {1: "Positive 😊", 0: "Neutral 😐", -1: "Negative 😞"}
        st.write("Sentiment:", sentiment_mapping[prediction])
    else:
        st.write("Please enter some text.")
