import streamlit as st
import joblib
import nltk
from googletrans import Translator
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# ✅ Load necessary NLTK models once
nltk.download('punkt')
nltk.download('stopwords')

# ✅ Load ML model
model = joblib.load("sentiment_model_IS2.pkl")

st.title("Samsung Galaxy S25 Sentiment Analysis")

# Input text
user_input = st.text_area("Enter a review:")

if st.button("Analyze Sentiment"):
    if user_input:
        # ✅ Translate text
        translator = Translator()
        try:
            translated_text = translator.translate(user_input, src='auto', dest='en').text
        except:
            translated_text = user_input  # Fallback if translation fails
        
        stop_words = set(stopwords.words('english'))

        @st.cache_data  # Caches downloaded resources
        def download_punkt():
            """Downloads 'punkt' tokenizer if not available"""
            nltk.download('punkt')
            return True
        
        # ✅ Download punkt tokenizer
        download_punkt()

        def preprocess_text(text):
            """Text Preprocessing"""
            tokens = word_tokenize(text.lower())  # Tokenization
            tokens = [word for word in tokens if word.isalnum()]  # Remove punctuation
            tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
            return " ".join(tokens)

        # ✅ Process the text
        processed_text = preprocess_text(translated_text)

        # ✅ Predict sentiment
        prediction = model.predict([processed_text])[0]

        sentiment_mapping = {1: "Positive 😊", 0: "Neutral 😐", -1: "Negative 😞"}
        st.write("Sentiment:", sentiment_mapping[prediction])

    else:
        st.write("Please enter some text.")
