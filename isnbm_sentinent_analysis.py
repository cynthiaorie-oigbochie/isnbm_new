import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from google_play_scraper import reviews, Sort
import re
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- Setup ---
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

ISNBM = joblib.load("svm.pkl")
pca_model = joblib.load("pca_model_3.pkl")
# Emoji sentiment dictionary (expand as needed)
emoji_dict = {
    #positive Emojis
'😀': "positive",# Grinning Face – Happiness, friendliness
'😃': "positive",# Big Eyes Grin – Excitement, joy
'😄': "positive",# Smiling Eyes Grin – Cheerful, friendly
'😊': "positive",# Smiling Face – Warmth, positivity
'🥳': "positive",# Party Face – Celebration, fun
'😎': "positive",# Sunglasses Face – Confidence, coolness
'✨': "positive",# Sparkles – Magic, excitement
'😍': "positive",# Heart Eyes – Adoration, admiration
'😘': "positive",# Kiss Face – Affection, gratitude
'🥰': "positive",# Smiling Hearts – Feeling loved
'😂': "positive",# Tears of Joy – Extreme laughter
'🤣': "positive",# Rolling Laugh – Hysterical laughter
'👏': "positive",# Clapping Hands – Applause, approval
'🔥': "positive",# Fire – Excitement, excellence
'👍': "positive",#
    # Negative MEmogies
'😢': "negative",# Crying Face – Sadness, disappointment
'😭': "negative",# Loudly Crying – Intense grief or despair
'😔': "negative",# Pensive Face – Reflection, regret
'😞': "negative",# Disappointed Face – Sadness, letdown
'😡': "negative",# Angry Face – Anger, frustration
'😤': "negative",# Steam Nose – Irritation, determination
'🙄': "negative",# Rolling Eyes – Sarcasm, boredom
'😩': "negative",# Weary Face – Exhaustion, frustration
'😱': "negative",# Screaming Face – Shock, fear
'😓': "negative",# Sweat Face – Stress, worry
'😖': "negative"# Confounded Face – Confusion, frustration
    
}


# --- Preprocessing ---

# 2. Emoji mapping
def map_emojis(text):
    for emoji, sentiment in emoji_dict.items():
        text = text.replace(emoji, f" {sentiment} ")
    return text


def clean_tweet(tweet):
    tweet = str(tweet).lower()
    tweet = re.sub(r'\b\d+(\.\d+)?\b', '', tweet)
    return tweet.strip()

def remove_noise(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    return text

def remove_punct_num(text):
    return re.sub(r'[^a-zA-Z\s]', '', text)

def tokenize(text):
    return text.split()

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

def lemmatize(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

def preprocess_text(text):
    text = clean_tweet(text)
    text = map_emojis(text)
    text = remove_noise(text)
    text = remove_punct_num(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)

def get_mean_pooling_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state
    mask_expanded = inputs["attention_mask"].unsqueeze(-1).expand(embeddings.size()).float()
    sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
    return mean_embeddings.numpy().flatten()

def fetch_and_process_reviews(app_id, num_reviews):
    scraped_reviews, _ = reviews(app_id, lang="en", country="us", sort=Sort.NEWEST, count=num_reviews)
    df = pd.DataFrame(scraped_reviews).dropna(subset=['content', 'score'])
    if df.empty:
        return df
    df = df[df['score'] != 3].copy()
    df['score_new'] = df['score'].replace({1: 0, 2: 0, 4: 1, 5: 1})
    df['cleaned_content'] = df['content'].apply(preprocess_text)
    X_embed_raw = df["cleaned_content"].apply(get_mean_pooling_embedding)
    X_embed = np.vstack(X_embed_raw.values)
    X_reduced = pca_model.transform(X_embed)
    df["Predicted Sentiment"] = ISNBM.predict(X_reduced)
    return df

# --- Streamlit UI ---
st.title("Google App Review Sentiment Predictor with Real-Time Scraping by Orie Cynthia(Mrs)")

app_id = st.text_input("Enter Google Play App ID:", value="com.google.android.apps.maps")
num_reviews = st.slider("Number of recent reviews to fetch", min_value=10, max_value=200, value=50, step=10)

if st.button("Fetch and Analyze Reviews"):
    df = fetch_and_process_reviews(app_id, num_reviews)
    if df.empty:
        st.warning("No reviews found.")
    else:
        st.subheader("Sample Reviews")
        st.write(df[["content", "score", "Predicted Sentiment"]].head(20))
        st.text("For Score     : 1 Star → Very dissatisfied, 2 Stars → Poor experience, 3 Stars → Average, 4 Stars → Good, 5 Stars → Excellent")
        st.text("For Predicted: 0→ Negative (dissatisfied/poor experoience), 1 → Positive (Good/Excellent)")

        # Sentiment Distribution
        st.subheader("Sentiment Distribution")
        fig, ax = plt.subplots()
        df["Predicted Sentiment"].value_counts().reindex([0, 1], fill_value=0).plot(
            kind="bar", ax=ax, color=["red", "green"]
        )
        ax.set_xticklabels(["Negative (0)", "Positive (1)"], rotation=0)
        ax.set_title("Predicted Sentiment Distribution")
        st.pyplot(fig)

        # Word Cloud
        st.subheader("Word Cloud")
        combined_text = " ".join(df["cleaned_content"].astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(combined_text)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wordcloud, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(df["score"], df["Predicted Sentiment"], labels=[0, 1])
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Negative", "Positive"],
                    yticklabels=["Negative", "Positive"], ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

        # Download CSV
        st.download_button(
            label="Download Predictions as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="predicted_reviews.csv",
            mime="text/csv"
        )
