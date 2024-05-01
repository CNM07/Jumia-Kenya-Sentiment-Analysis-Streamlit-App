# importing libraries
import re
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from textblob import TextBlob
import pickle

def preprocess_data(reviews):
    all_reviews = []
    lines = reviews.values.tolist()
    for document in lines:
        document = document.lower()
        document = re.sub(r"[,:;!@<>!~%$#*&//\\]", " ", document)
        tokens = word_tokenize(document)
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        word = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(w) for w in word if not w in stop_words]
        words = " ".join(words)
        all_reviews.append(words)
    return all_reviews

def get_sentiment_polarity(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        sentiment = 'Positive'
    elif polarity < 0:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    return polarity, sentiment



def main():
    # Load data
    jumia_reviews = pd.read_csv("customer_reviews.csv")

    # Data Preprocessing
    jumia_reviews['Processed Text'] = preprocess_data(jumia_reviews['Text'])

    # Sentiment Analysis/Polarity by Text Blob
    jumia_reviews['Score TextBlob'] = jumia_reviews['Processed Text'].apply(get_sentiment_polarity)
    jumia_reviews['Sentiment TextBlob'] = ['Positive' if score[0] > 0 else 'Negative' if score[0] < 0 else 'Neutral' for score in jumia_reviews['Score TextBlob']]

    # Feature Engineering
    label_encoder = LabelEncoder()
    jumia_reviews['Sentiment_TextBlob_Encoded'] = label_encoder.fit_transform(jumia_reviews['Sentiment TextBlob'])
    X = jumia_reviews['Processed Text']
    y = jumia_reviews['Sentiment_TextBlob_Encoded']
    vectorizer = TfidfVectorizer()
    X_vect = vectorizer.fit_transform(X)

    def sentiment_modeling(X, y):
        np.random.seed(42)
        random_forest = RandomForestClassifier()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        random_forest.fit(X_train, y_train)
        random_forest_pred = random_forest.predict(X_test)
        accuracy = accuracy_score(y_test, random_forest_pred)
        return random_forest, accuracy

    # Save TF-IDF vectorizer
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # Sentiment Modeling
    random_forest_model, accuracy = sentiment_modeling(X_vect, y), accuracy

    # Save models and processed data
    jumia_reviews.to_csv("processed_reviews.csv", index=False)
    with open("random_forest_model.pkl", "wb") as f:
        pickle.dump(random_forest_model, f)

if __name__ == "__main__":
    main()


data = pd.read_csv("processed_reviews.csv")
print(data.head())
