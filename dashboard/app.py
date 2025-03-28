import streamlit as st
from multiapp import MultiApp
from apps import frontend,test 

# Set page configuration
st.set_page_config(page_title='Sentiment Analysis Dashboard', layout='wide')

app = MultiApp()

st.markdown("""
# Sentiment Analysis Dashboard
Dashboard ini berisi analisis sentimen komentar netizen di Twitter terhadap gaji dan kesehatan mental generasi Z.
""")

app.add_app("Home", frontend.app)
app.add_app("Test", test.app)
app.run()
