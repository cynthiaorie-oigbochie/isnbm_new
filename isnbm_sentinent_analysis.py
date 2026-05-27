import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from google_play_scraper import reviews, Sort
from sklearn.svm import SVC
import re
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import emoji
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from stopwordsiso import stopwords as iso_stopwords
from langdetect import detect
from nltk.tokenize import word_tokenize 
from googletrans import Translator

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

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

# Download NLTK resources (run once)
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
translator = Translator()
# 1. Lowercasing
def to_lower(text):
    return text.lower()
def clean_tweet(tweet): 
    tweet = str(tweet) # ensure it's a string
    # Convert to lowercase 
    tweet = tweet.lower() 
    # Remove integers and floats (e.g., 123, 45.67) 
    tweet = re.sub(r'\b\d+(\.\d+)?\b', '', tweet) 
    return tweet.strip()


def remove_numbers(tweet):
    # Remove all digits
    return re.sub(r'\d+', '', tweet)


# 2. Emoji mapping
def map_emojis(text):
    for emoji, sentiment in emoji_dict.items():
        text = text.replace(emoji, f" {sentiment} ")
    return text

# 3. Remove URLs, mentions, hashtags
def remove_noise(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # URLs
    text = re.sub(r'@\w+', '', text)                     # mentions
    text = re.sub(r'#\w+', '', text)                     # hashtags
    return text

# 4. Remove punctuation & numbers
def remove_punct_num(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # keep only letters
    return text

# 5. Tokenization
def tokenize(text):
    return text.split()

# 6. Stopword removal
def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

# 7. Lemmatization
def lemmatize(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]
# 8. Remove non-english words
def filter_non_english(tokens): 
    # Keep only tokens detected as English 
    english_tokens = [] 
    for word in tokens: 
        try: 
            if detect(word) == "en": 
                english_tokens.append(word) 
        except: pass # skip words that can't be detected 
    return english_tokens

# 9 Translate non english word to english words
def translate_non_english(tokens): 
    translated_tokens = [] 
    for word in tokens: 
        try: 
            if detect(word) != "en": 
                translated = translator.translate(word, src=detect(word), dest="en").text 
                translated_tokens.append(translated) 
            else: 
                translated_tokens.append(word) 
        except: 
            translated_tokens.append(word) # fallback if detection fails 
        return translated_tokens    

# 8. Rejoin tokens
def join_tokens(tokens):
    return " ".join(tokens)

# Full preprocessing pipeline
def preprocess_text(text):
    text = clean_tweet(text)
    #text = to_lower(text)
    
    text = map_emojis(text)
    text = remove_noise(text)
    text = remove_punct_num(text)
    tokens = tokenize(text)
   # tokens = translate_non_english(tokens)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    #tokens = filter_non_english(tokens)
    return join_tokens(tokens)

def list_to_str(lst):
    result_str = ''
    for word in lst:
        result_str += word + ' '  # Append each word with a space
    return result_str.rstrip()  # Remove trailing space before returning

# Mean pooling function
def get_mean_pooling_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    # outputs.last_hidden_state shape: (batch_size, seq_len, hidden_dim)
    embeddings = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]

    # Expand mask to match embeddings shape
    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()

    # Apply mask and compute mean
    sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask

    return mean_embeddings.numpy().flatten()




# Load your models
model2 = joblib.load("svm.pkl")
#embedder = joblib.load("embeddings.pkl")
pca = joblib.load("pca_model_3.pkl")

st.title("Google App Review Sentiment Predictor with Real-Time Scraping by Orie Cynthia(Mrs)")

# --- Scrape Reviews ---
st.header("Scrape Google Play Reviews")
app_id = st.text_input("Enter Google Play App ID (e.g., com.facebook.katana):",value="com.google.android.apps.maps")

num_reviews = st.slider("Number of reviews to fetch", min_value=10, max_value=1000, value=50, step=10)

if st.button("Fetch Reviews"):
    if app_id.strip():
        # Scrape reviews
        scraped_reviews, _ = reviews(
            app_id,
            lang="en",     # language
            country="us",  # country
            sort=Sort.NEWEST,
            count=num_reviews
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(scraped_reviews)
        
       
        # Drop column with label '3'
       # df = df[df['score'] != 3]

        
        st.subheader("Sample Reviews")
        st.write(df[["content", "at", "score"]].head(20))
        st.text("1 Star → Very dissatisfied, 2 Stars → Poor experience, 3 Stars → Average.")
        st.text("4 Stars → Good, 5 Stars → Excellent")
        # Discretization

       # df["clean_text"] = df["content"].apply(preprocess_text)
        df= df.dropna()

        df['score'] = df['score'].replace({1: 0, 2: 0, 4: 1, 5: 1})
        #df['score'] = df['score'].replace({1: 1, 2: 1, 4: 0, 5: 0})
           


        
        # Embedding + PCA + prediction
        #X_embed = embedder.transform(df["content"].astype(str))
        X_embed_raw= df["content"].apply(get_mean_pooling_embedding)
        # Stack into matrix
        X_embed = np.vstack(X_embed_raw.values)
        X_reduced = pca.transform(X_embed)
         # Apply to dataframe
       

    
        
        df["Predicted Sentiment"] = model2.predict(X_reduced)#model2.predict_proba(X_reduced)[:,1]
        
        st.subheader("Predictions")
        st.write(df[["content", "Predicted Sentiment"]].head(20))
        st.caption("This is a small caption for extra context.")
        
        # Visualization
        st.subheader("Sentiment Distribution")
        sentiment_counts = df["Predicted Sentiment"].value_counts()
        
        fig, ax = plt.subplots()
        sentiment_counts.plot(kind="bar", ax=ax, color=["green", "red", "blue"])
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Predicted Sentiments")
        st.pyplot(fig)
        
        # Word Cloud
        st.subheader("Word Cloud of Reviews")
        combined_text = " ".join(df["content"].astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(combined_text)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wordcloud, interpolation="bilinear")
        ax_wc.axis("off")
        ax_wc.set_title("Most Frequent Words in Reviews", fontsize=16, fontweight="bold")
        st.pyplot(fig_wc)
        
        # Download predictions
        st.download_button(
            label="Download Predictions as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="scraped_predicted_reviews.csv",
            mime="text/csv"
        )
    else:
        st.warning("Please enter a valid app ID.")
