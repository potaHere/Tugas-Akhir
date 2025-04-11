import streamlit as st
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.graph_objects as go
import os
from backend import get_pivot_sentiment, get_label_counts, get_keyword_sentiment_distribution, get_avg_metrics, generate_wordclouds
from datetime import timedelta

def inject_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def app():
    inject_css()

    # Load semua data dengan format tanggal yang benar
    sentiment_data = pd.read_csv(
        'datasets/datasets-keyword-label.csv',
        parse_dates=['created_at'],
        date_format="%d %b %Y"  # Format yang sesuai dengan "30 Dec 2024"
    )
    
    # Tidak perlu normalisasi timezone karena sudah menggunakan format tanpa jam
    # sentiment_data['created_at'] = pd.to_datetime(sentiment_data['created_at']).dt.tz_localize(None)
    
    # Tambahkan filter di sidebar
    with st.sidebar:
        st.markdown("## 🔍 Filter Dashboard")
        
        # Mendapatkan tanggal minimum dan maksimum dari dataset
        min_date = sentiment_data['created_at'].min().date()
        max_date = sentiment_data['created_at'].max().date()
        
        # Menampilkan opsi filter waktu
        st.markdown("### ⏱️ Filter Waktu")
        
        # Filter waktu menggunakan slider tanggal
        date_range = st.slider(
            "Pilih Rentang Waktu:",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="DD-MMM-YYYY"
        )
        
        # Convert date range to datetime for filtering
        start_datetime = pd.Timestamp(date_range[0])
        end_datetime = pd.Timestamp(date_range[1]) + timedelta(days=1) - timedelta(seconds=1)  # end of day
        
        # Menampilkan periode waktu yang dipilih
        st.info(f"Menampilkan data dari: {start_datetime.strftime('%d %B %Y')} hingga {end_datetime.strftime('%d %B %Y')}")
        
        # Filter keyword
        st.markdown("### 🏷️ Filter Keyword")
        keywords = sorted(sentiment_data['keyword'].unique())
        selected_keywords = st.multiselect(
            "Pilih Keyword",
            options=keywords,
            default=keywords
        )
        
        # # Filter sentimen
        # st.markdown("### 😊 Filter Sentimen")
        # sentiments = ['Positif', 'Negatif', 'Netral']
        # selected_sentiments = st.multiselect(
        #     "Pilih Sentimen",
        #     options=sentiments,
        #     default=sentiments
        # )

    # Terapkan filter
    filtered_data = sentiment_data[
        (sentiment_data['created_at'] >= start_datetime) & 
        (sentiment_data['created_at'] <= end_datetime) &
        (sentiment_data['keyword'].isin(selected_keywords)) #&
        # (sentiment_data['label'].isin(selected_sentiments))
    ]
    
    # # Tampilkan jumlah data yang ditampilkan
    # st.write(f"Menampilkan {len(filtered_data)} dari {len(sentiment_data)} data")
    
    # Data wordcloud - tetap menggunakan semua data
    wordcloud_data = pd.read_csv('datasets/word_count_labeled.csv')
    
    # Performance data - tetap menggunakan semua data
    performance_data = pd.read_csv('datasets/evaluation_results_combine.csv')
    
    # Row 1: Pie Chart 
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Sentiment Distribution')
        label_counts = get_label_counts(filtered_data)
        pie_fig = px.pie(label_counts, names='label', values='count')
        pie_fig.update_traces(textinfo='percent+label')
        pie_fig.update_layout(showlegend=False)
        st.plotly_chart(pie_fig, use_container_width=True)
        
    # Distribusi Sentimen bedasarkan Model
    with col2:        
        st.subheader('Keyword Sentiment Distribution')
        keyword_sentiment_counts = get_keyword_sentiment_distribution(filtered_data)
        
        # Mengatur urutan kategori label di frontend
        keyword_sentiment_counts['label'] = pd.Categorical(keyword_sentiment_counts['label'], 
                                                            categories=['Negatif', 'Positif', 'Netral'], 
                                                            ordered=True)
        
        # Membuat bar chart dengan urutan label yang diatur
        bar_fig = px.bar(keyword_sentiment_counts, x='keyword', y='count', color='label', barmode='group',
                        category_orders={'label': ['Negatif', 'Positif', 'Netral']})
        st.plotly_chart(bar_fig, use_container_width=True)

    # Row 2: Wordclouds (tidak difilter)
    label_colors = {
        'positif': 'green',
        'negatif': 'red',
        'netral': 'gray'
    }
    
    wordclouds = generate_wordclouds(wordcloud_data, label_colors)

    cols = st.columns(len(wordclouds))
    for col, (label, wordcloud) in zip(cols, wordclouds.items()):
        with col:
            st.subheader(f'Word Cloud {label}')
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
            
    # cols = st.columns(len(wordclouds))
    # for col, (label, wordcloud) in zip(cols, wordclouds.items()):
    #     with col:
    #         st.subheader(f'Word Cloud {label}')
    #         fig = px.imshow(wordcloud, binary_string=True)
    #         fig.update_layout(
    #             coloraxis_showscale=False,
    #             xaxis=dict(visible=False),
    #             yaxis=dict(visible=False),
    #             margin=dict(l=0, r=0, t=0, b=0)
    #             )
    #         st.plotly_chart(fig, use_container_height=False)
            
            st.markdown('<div class="column-costum"></div>', unsafe_allow_html=True)

    # Row 3: Line Chart & Grouped Bar Chart
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader('Sentiment Trends Over Time')
        pivot_sentiment = get_pivot_sentiment(filtered_data)
        line_fig = px.line(pivot_sentiment, x='Year', y=['Negatif', 'Positif', 'Netral'], markers=True)
        st.plotly_chart(line_fig, use_container_width=True)
    
    with col6:
        st.subheader('Model Performance')
        avg_metrics_df = get_avg_metrics(performance_data)
        bar_group_fig = px.bar(avg_metrics_df.melt(id_vars='model', value_vars=['Accuracy', 'Precision', 'Recall', 'F1-score']),
                            x='variable', y='value', color='model', barmode='group')
        st.plotly_chart(bar_group_fig, use_container_width=True)

    # Data table
    st.subheader('Data Tables')
    columns_to_display = ['created_at', 'cleanning_text', 'keyword', 'label']
    display_df = filtered_data[columns_to_display].copy()
    
    # Urutkan data berdasarkan tanggal (created_at) dari yang terbaru
    display_df = display_df.sort_values(by='created_at', ascending=False)
    
    # Format kolom created_at menjadi format DD-BULAN-YYYY (contoh: 30-December-2024)
    display_df['created_at'] = display_df['created_at'].dt.strftime('%d-%B-%Y')
    
    display_df = display_df.rename(columns={
    'created_at': 'Tanggal',
    'cleanning_text': 'Tweet',
    'keyword': 'Kata Kunci',
    'label': 'Sentimen'
    })

    st.dataframe(display_df, hide_index=True, use_container_width=True)

if __name__ == "__main__":
    app()