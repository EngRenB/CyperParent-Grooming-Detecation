import streamlit as st
import pandas as pd
import re
import string
from PIL import Image
import nltk
from nltk.corpus import stopwords
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from pathlib import Path

# Download stopwords
nltk.download('stopwords')

# Initialize the stemmer
stemmer = nltk.SnowballStemmer("english")

#image path
logo = Path("2.FYPapp") / "logo.png"

# Import logo
logo = Image.open(logo)

# Page title and logo
st.set_page_config(page_title='Grooming Detection', page_icon=logo)

# Columns to organize the page
col1, col2 = st.columns([2, 2])

# Header
st.title("**Grooming Detection**")
st.markdown("#### is an ML model to detect grooming in textual and image files")
st.write("If you’re worried about the content of your child's chats please upload the file and **we will take care of it**")

# Receive a file as an input
file = st.file_uploader("Upload a file", type=["csv"])

# Function to clean text
def clean(text):
    if isinstance(text, str):
        stop_words = stopwords.words('english')
        more_stopwords = ['u', 'im', 'c', 'apos']
        stop_words = stop_words + more_stopwords
        text = text.lower()  # Convert text to lowercase
        text = re.sub(r'\[.*?\]', '', text)  # Remove text within square brackets
        text = re.sub(r'http\S+', '', text)  # Remove URLs starting with http
        text = re.sub(r'www\.\S+', '', text)  # Remove URLs starting with www
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\b\w*\d\w*\b', '', text)  # Remove words containing numbers
        text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stopwords
        return text.strip()
    return ""  # Return an empty string for non-string data

if file is not None:
    if file.name.endswith('.csv'):
        # Read CSV file
        df = pd.read_csv(file)
        # Assume the text data is in the 'text' column and labels are in 'Grooming Detection'
        df['cleaned_text'] = df['Text'].apply(clean)
    elif file.name.endswith('.txt'):
        # Read text file
        text = file.read().decode("utf-8")
        cleaned_text = clean(text)
        df = pd.DataFrame([cleaned_text], columns=['cleaned_text'])
        df['Grooming Detection'] = [0]  # Dummy label for txt file

    # Log intermediate cleaned text for debugging
    st.write(" **Preprocessed File:**")
    st.write(df['cleaned_text'])

    # Filter out empty cleaned_text
    df = df[df['cleaned_text'].str.strip().astype(bool)]

    if not df.empty:
        # Vectorization
        vectorizer = TfidfVectorizer()
        file_vectorized = vectorizer.fit_transform(df['cleaned_text'])

        # Extract labels for training
        labels = df['Label']

        # Train the Gaussian Naive Bayes classifier
        clf = GaussianNB()
        clf.fit(file_vectorized.toarray(), labels)

        # Classify the text
        pred = clf.predict(file_vectorized.toarray())

        # Get the probability of the prediction
        grooming_detection = clf.predict_proba(file_vectorized.toarray())[:, 1]

        # Sentiment analysis using VADER
        vader = SentimentIntensityAnalyzer()
        sentiment_scores = df['cleaned_text'].apply(vader.polarity_scores)

        # Aggregate sentiment scores
        avg_sentiment = pd.DataFrame(sentiment_scores.tolist()).mean()
        avg_grooming_detection = grooming_detection.mean()

        # Display overall insights
        st.write("#### **Overall Insights**")
        st.write("##### **Sentiments**")
        st.write(f"Positive: {avg_sentiment['pos']:.2%}")
        st.write(f"Negative: {avg_sentiment['neg']:.2%}")
        st.write(f"Neutral: {avg_sentiment['neu']:.2%}")
        st.write(f"##### **Grooming Detection likelihood {avg_grooming_detection:.2%}**")

        # Visualization
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Tone distribution
        tone_labels = ['Positive', 'Negative', 'Neutral']
        tone_values = [avg_sentiment['pos'], avg_sentiment['neg'], avg_sentiment['neu']]
        ax[0].pie(tone_values, labels=tone_labels, autopct='%1.1f%%', colors=['#ccd5ae','#ffe5ec','#e8e8e4'])#pos = green, neg = pink, neut = grey/blue
        ax[0].set_title('Sentiments Distribution')

        # Grooming Detection likelihood
        ax[1].bar(['Grooming Detection'], [avg_grooming_detection], color='#b392ac')
        ax[1].set_ylim(0, 1)
        ax[1].set_ylabel('Likelihood')
        ax[1].set_title('Grooming Detection Percentage')

        st.pyplot(fig)
    else:
        st.write("The cleaned text data is empty after preprocessing. Please check the input file.")

# Hide made by Streamlit and main menu
hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

#python3 -m streamlit run /Users/Ren/fypvenv/2.FYPapp/01_🏠_Homepage.py
