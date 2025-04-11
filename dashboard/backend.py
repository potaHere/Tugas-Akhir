import pandas as pd
from wordcloud import WordCloud, get_single_color_func
import joblib

def get_label_counts(sentiment_data: pd.DataFrame) -> pd.DataFrame:
    """
    Menghasilkan DataFrame dengan jumlah masing-masing label.
    """
    label_counts = sentiment_data['label'].value_counts().reset_index()
    label_counts.columns = ['label', 'count']
    return label_counts

def get_sentiment_distribution(y_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Menghasilkan DataFrame dengan jumlah masing-masing label dari hasil prediksi.
    """
    sentiment_counts = y_pred['predicted_label'].value_counts().reset_index()
    sentiment_counts.columns = ['label', 'count']
    return sentiment_counts


def get_yearly_sentiment(sentiment_data: pd.DataFrame) -> pd.DataFrame:
    """
    Menghasilkan DataFrame dengan jumlah label sentimen per tahun.
    """
    # Pastikan kolom 'created_at' bertipe datetime
    # Tidak perlu parameter utc karena kita menggunakan tanggal tanpa timezone
    sentiment_data['created_at'] = pd.to_datetime(sentiment_data['created_at'])
    
    # Ekstraksi tahun dari kolom 'created_at'
    sentiment_data['year'] = sentiment_data['created_at'].dt.year
    
    # Group by tahun dan label, lalu hitung jumlahnya
    yearly_sentiment = sentiment_data.groupby(['year', 'label']).size().reset_index(name='count')
    # Mengatur urutan kategori label
    yearly_sentiment['label'] = pd.Categorical(yearly_sentiment['label'], categories=['Negatif', 'Positif', 'Netral'], ordered=True)
    # Urutkan DataFrame berdasarkan tahun dan label
    yearly_sentiment = yearly_sentiment.sort_values(by=['year', 'label'])

    return yearly_sentiment

def get_keyword_sentiment_distribution(sentiment_data: pd.DataFrame) -> pd.DataFrame:
    """
    Menghasilkan DataFrame dengan distribusi sentimen per keyword.
    """
    keyword_sentiment_counts = sentiment_data.groupby(['keyword', 'label']).size().reset_index(name='count')
    
    # Mengatur urutan kategori label
    keyword_sentiment_counts['label'] = pd.Categorical(keyword_sentiment_counts['label'], categories=['Negatif', 'Positif', 'Netral'], ordered=True)
    
    return keyword_sentiment_counts

def get_pivot_sentiment(sentiment_data: pd.DataFrame) -> pd.DataFrame:
    """
    Menghasilkan DataFrame pivot_sentiment dengan jumlah label positif, netral, dan negatif per tahun.
    """
    # Pastikan kolom 'created_at' bertipe datetime
    # Tidak perlu parameter utc karena kita menggunakan tanggal tanpa timezone
    sentiment_data['created_at'] = pd.to_datetime(sentiment_data['created_at'])

    # Ekstraksi tahun dari kolom 'created_at'
    sentiment_data['Year'] = sentiment_data['created_at'].dt.year

    # Group by Year dan Label, lalu hitung jumlahnya
    yearly_sentiment = sentiment_data.groupby(['Year', 'label']).size().reset_index(name='count')

    # Pivot agar setiap label menjadi kolom tersendiri
    pivot_sentiment = yearly_sentiment.pivot(index='Year', columns='label', values='count').fillna(0)

    # Reset index agar kolom 'Year' tersedia sebagai kolom biasa
    pivot_sentiment = pivot_sentiment.reset_index()

    return pivot_sentiment

def extract_avg_metrics(report: str) -> dict:
    """
    Mengekstrak nilai rata-rata dari Accuracy, Precision, Recall, dan F1-score dari kolom Classification Report.
    """
    report_dict = eval(report)
    return {
        'Accuracy': report_dict['accuracy'],
        'Precision': report_dict['macro avg']['precision'],
        'Recall': report_dict['macro avg']['recall'],
        'F1-score': report_dict['macro avg']['f1-score']
    }

def get_avg_metrics(performance_data: pd.DataFrame) -> pd.DataFrame:
    """
    Menghasilkan DataFrame dengan nilai rata-rata dari Accuracy, Precision, Recall, dan F1-score.
    """
    performance_data['Classification Report'] = performance_data['Classification Report'].apply(extract_avg_metrics)
    avg_metrics_df = performance_data['Classification Report'].apply(pd.Series)
    avg_metrics_df['model'] = performance_data['model']
    return avg_metrics_df

def generate_wordclouds(wordcloud_data: pd.DataFrame, label_colors: dict) -> dict:
    """
    Menghasilkan wordcloud untuk setiap label dalam wordcloud_data.
    """
    wordclouds = {}
    labels = wordcloud_data['label'].unique()
    for label in labels:
        words = wordcloud_data[wordcloud_data['label'] == label].set_index('word')['count'].to_dict()
        wordcloud = WordCloud(width=1000, height=500, background_color='white', color_func=get_single_color_func(label_colors[label])).generate_from_frequencies(words)
        wordclouds[label] = wordcloud
    return wordclouds

# ======================================
# Memuat Model dan Prediksi Sentimen
# ======================================

# vectorizer = joblib.load('models/datasets-tfidf.pkl')

# def load_model_and_vectorizer(model_path, vectorizer_path):
#     """
#     Memuat model dari file pickle.
#     """
#     try:
#         model = joblib.load(model_path)
#         text_vectorizer = joblib.load(vectorizer_path)
#         return model, text_vectorizer
#     except Exception as e:
#         print(f"Error loading model or vectorizer: {e}")
#         return None, None

# def predict_sentiment(model, text_vectorizer, text):
#     """
#     Melakukan prediksi sentimen terhadap teks yang diberikan menggunakan model yang dipilih.
#     """
#     try:
#         text_vectorized = text_vectorizer.transform([text])
#         prediction = model.predict(text_vectorized)
#         return prediction[0]
#     except Exception as e:
#         print(f"Error predicting sentiment: {e}")
#         return None