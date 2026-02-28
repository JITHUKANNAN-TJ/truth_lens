import streamlit as st
import joblib
import os
import math

# Set page config
st.set_page_config(page_title="TruthLens - Fake News Detector", page_icon="🔍", layout="centered")

# Custom CSS for UI mimicking the image
# Updated CSS to match TruthLens UI exactly
from flask import Flask, render_template, request, jsonify
import joblib
import math

app = Flask(__name__)

# Load the AI models
model = joblib.load('fake_news_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

@app.route('/')
def home():
    # This looks into the /templates folder for index.html
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # AI Logic
    text_vectorized = vectorizer.transform([text])
    prediction = int(model.predict(text_vectorized)[0]) # 0 for Real, 1 for Fake
    
    # Calculate confidence score
    decision = model.decision_function(text_vectorized)[0]
    confidence = 1 / (1 + math.exp(-abs(decision))) 
    
    return jsonify({
        'prediction': 'REAL' if prediction == 0 else 'FAKE',
        'confidence': round(confidence * 100, 2)
    })

if __name__ == '__main__':
    # use_reloader=False prevents the "signal" error seen in your screenshot
    app.run(debug=True, use_reloader=False)
# Load the model and vectorizer at startup
@st.cache_resource
def load_models():
    model_path = 'fake_news_model.joblib'
    vectorizer_path = 'tfidf_vectorizer.joblib'
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        clf = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return clf, vectorizer
    else:
        return None, None

clf, vectorizer = load_models()

if not clf or not vectorizer:
    st.error("Error: Model (`fake_news_model.joblib`) or vectorizer (`tfidf_vectorizer.joblib`) not found. Please train the model first.")
else:
    # Use HTML wrapper to mimic the dark container background
    st.markdown('<div class="form-box">', unsafe_allow_html=True)
    
    text_input = st.text_area("", height=200, placeholder="Paste article text here to verify its authenticity...", label_visibility="collapsed")
    
    submit_button = st.button("Analyze Content")
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    if submit_button:
        if not text_input.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing content authenticity..."):
                try:
                    # Vectorize the input text
                    text_vectorized = vectorizer.transform([text_input])
                    prediction = clf.predict(text_vectorized)[0]
                    
                    # Calculate a pseudo-confidence score
                    dist = abs(clf.decision_function(text_vectorized)[0])
                    confidence = 1 / (1 + math.exp(-dist))
                    confidence_percent = round(confidence * 100, 2)
                    
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 0:
                            st.markdown('<div class="result-real">✓ REAL CONTENT</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="result-fake">✗ FAKE CONTENT</div>', unsafe_allow_html=True)
                            
                    with col2:
                        st.metric(label="Confidence Score", value=f"{confidence_percent}%")
                        st.progress(confidence)
                        
                except Exception as e:
                    st.error(f"An error occurred during prediction: {str(e)}")

# Footer
st.markdown('<div class="footer-text">Powered by LinearSVC & TF-IDF • IBM Project 2026</div>', unsafe_allow_html=True)


