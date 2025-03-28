import streamlit as st
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.graph_objects as go
from backend import get_pivot_sentiment, get_label_counts, get_keyword_sentiment_distribution, get_avg_metrics, generate_wordclouds

def inject_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def app():
    inject_css()

    # Placeholder Data (Will be replaced with actual backend data)
    sentiment_data = pd.read_csv(
        'datasets/datasets-keyword-label.csv',
        parse_dates=['created_at'],
        infer_datetime_format=True
    )
    
    # data wordcloud
    wordcloud_data = pd.read_csv('datasets/word_count_labeled.csv')
    
    # performance data
    performance_data = pd.read_csv('datasets/evaluation_results_combine.csv')
    
    # Row 1: Pie Chart 
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Sentiment Distribution')
        label_counts = get_label_counts(sentiment_data)  # Panggil fungsi backend
        pie_fig = px.pie(label_counts, names='label', values='count')
        pie_fig.update_traces(textinfo='percent+label')
        pie_fig.update_layout(showlegend=False)
        st.plotly_chart(pie_fig, use_container_width=True)
        
    # Distribusi Sentimen bedasarkan Model
    with col2:        
        st.subheader('Keyword Sentiment Distribution')
        keyword_sentiment_counts = get_keyword_sentiment_distribution(sentiment_data)  # Panggil fungsi backend
        
        # Mengatur urutan kategori label di frontend
        keyword_sentiment_counts['label'] = pd.Categorical(keyword_sentiment_counts['label'], categories=['Negatif', 'Positif', 'Netral'], ordered=True)
        
        # Membuat bar chart dengan urutan label yang diatur
        bar_fig = px.bar(keyword_sentiment_counts, x='keyword', y='count', color='label', barmode='group',
                        category_orders={'label': ['Negatif', 'Positif', 'Netral']})
        st.plotly_chart(bar_fig, use_container_width=True)

    # Row 2: Wordclouds
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

    # Row 3: Line Chart & Grouped Bar Chart
    col5, col6 = st.columns(2)
    
    with col5:
        st.markdown('<div class="column-costum"></div>', unsafe_allow_html=True)
        st.subheader('Sentiment Trends Over Time')
        pivot_sentiment = get_pivot_sentiment(sentiment_data)
        line_fig = px.line(pivot_sentiment, x='Year', y=['Negatif', 'Positif', 'Netral'], markers=True)
        st.plotly_chart(line_fig, use_container_width=True)
    
    with col6:
        st.markdown('<div class="column-costum"></div>', unsafe_allow_html=True)
        st.subheader('Model Performance')
        avg_metrics_df = get_avg_metrics(performance_data)  # Panggil fungsi backend
        bar_group_fig = px.bar(avg_metrics_df.melt(id_vars='model', value_vars=['Accuracy', 'Precision', 'Recall', 'F1-score']),
                            x='variable', y='value', color='model', barmode='group')
        st.plotly_chart(bar_group_fig, use_container_width=True)

    # with col8:
    st.subheader('Data Tables')
    columns_to_display = ['created_at', 'full_text', 'keyword', 'label']
    st.dataframe(sentiment_data[columns_to_display], use_container_width=True)

if __name__ == "__main__":
    app()