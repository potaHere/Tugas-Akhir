import streamlit as st
from backend import load_model_and_vectorizer, predict_sentiment

def app():
    # Dropdown untuk memilih model
    model_choice = st.selectbox('Pilih Model', ['SVM', 'Naive Bayes', 'KNN'])

    # Input teks dari user
    user_input = st.text_area('Masukkan teks untuk analisis sentimen')

    # Tombol untuk melakukan prediksi
    if st.button('Prediksi Sentimen'):
        # if model_choice == 'SVM':
        #     model_path = 'models/svm_model.pkl'
        #     vectorizer_path = 'models/datasets-tfidf.pkl'
        # elif model_choice == 'Naive Bayes':
        #     model_path = 'models/nb_model.pkl'
        #     vectorizer_path = 'models/datasets-tfidf.pkl'
        # elif model_choice == 'KNN':
        #     model_path = 'models/knn_model.pkl'
        #     vectorizer_path = 'models/datasets-tfidf.pkl'
            
        # Load model dan vectorizer
        # model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)
        
        # Prediksi sentimen
        # prediction = predict_sentiment(model, vectorizer, user_input)
        st.write(f'#### Prediksi Sentimen:')

    
if __name__ == '__main__':
    app()