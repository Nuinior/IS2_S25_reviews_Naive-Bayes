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
        nltk.download('punkt')  # ⬅️ โหลด punkt tokenization model   
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        import string
        

        nltk.download('punkt')
        nltk.download('stopwords')
        
        stop_words = set(stopwords.words('english'))

        def preprocess_text(text):
            tokens = word_tokenize(text.lower())
            tokens = [word for word in tokens if word.isalnum()]
            tokens = [word for word in tokens if word not in stop_words]
            return " ".join(tokens)

        processed_text = preprocess_text(translated_text)
        prediction = model.predict([processed_text])[0]

        sentiment_mapping = {1: "Positive 😊", 0: "Neutral 😐", -1: "Negative 😞"}
        st.write("Sentiment:", sentiment_mapping[prediction])
    else:
        st.write("Please enter some text.")
