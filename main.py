import re
import pandas as pd
import streamlit as st
from google_play_scraper import Sort, reviews
from time import sleep
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator
import matplotlib.pyplot as plt
import seaborn as sns

def scrape_reviews_batched(app_id, lang='id', country='id', sort=Sort.NEWEST, filter_score_with=""):
    all_reviews_content = []

    for _ in range(5):  # Adjust as needed for more batches
        result, continuation_token = reviews(app_id, lang=lang, country=country, sort=sort, count=200, filter_score_with=filter_score_with)
        all_reviews_content.extend(review['content'] for review in result)
        if not continuation_token:
            break  # No more pages to fetch, exit loop
        sleep(1)  # Delay for 1 second between batches

    return all_reviews_content

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def filter_reviews_by_keywords(reviews, keywords):
    filtered_reviews = []
    for review in reviews:
        for keyword in keywords:
            if re.search(r'\b{}\b'.format(re.escape(keyword)), review):
                filtered_reviews.append(review)
                break
    return filtered_reviews

def sentiment_to_likert(sentiment_score, scale=5):
    if scale == 5:
        if sentiment_score >= 0.6:
            return 5  # Sangat Puas Sekali
        elif sentiment_score >= 0.2:
            return 4  # Sangat Puas
        elif sentiment_score >= -0.2:
            return 3  # Cukup Puas
        elif sentiment_score >= -0.6:
            return 2  # Tidak Puas
        else:
            return 1  # Sangat Tidak Puas

def likert_label(score):
    labels = {
        1: "Sangat Tidak Puas",
        2: "Tidak Puas",
        3: "Cukup Puas",
        4: "Sangat Puas",
        5: "Sangat Puas Sekali"
    }
    return labels.get(score, "Unknown")

def translate_reviews(reviews, target_lang='en'):
    translator = Translator()
    translated_reviews = [translator.translate(review, dest=target_lang).text for review in reviews]
    return translated_reviews

def main():
    st.title("Filter Ulasan Aplikasi dengan Kata Kunci")

    app_id = st.text_input("Masukkan ID Aplikasi Google Play Store:")

    if app_id:
        reviews_content = scrape_reviews_batched(app_id)
        normalized_reviews_content = [normalize_text(review) for review in reviews_content]

        keywords = ['penggunaannya', 'penggunaan', 'memudahkan', 'pengguna', 'informasi', 'sistem informasi', 'dapat memudahkan pengguna', 'sistem']
        filter_reviews = st.radio("Filter Ulasan", ("y", "n")).lower() == 'y'

        if filter_reviews:
            reviews_with_keywords = filter_reviews_by_keywords(normalized_reviews_content, keywords)

            # Translate reviews
            translated_reviews = translate_reviews(reviews_with_keywords)

            analyzer = SentimentIntensityAnalyzer()
            sentiments = [analyzer.polarity_scores(review)['compound'] for review in translated_reviews]

            likert_scale = [sentiment_to_likert(sentiment, scale=5) for sentiment in sentiments]

            df_reviews_with_keywords = pd.DataFrame({
                "Review Number": range(1, len(reviews_with_keywords) + 1),
                "Review": reviews_with_keywords,
                "Translated Review": translated_reviews,
                "Sentiment Score": sentiments,
                "Likert Scale": likert_scale,
                "Sentiment Label": [likert_label(score) for score in likert_scale]
            })

            # Calculate average sentiment score
            avg_sentiment_score = df_reviews_with_keywords["Sentiment Score"].mean()

            st.markdown("## Reviews containing keywords:")
            st.dataframe(df_reviews_with_keywords)

            # Display the average sentiment score
            st.markdown(f"## Skor Sentimen Rata-rata: {avg_sentiment_score:.2f}")

            # Calculate the counts for each sentiment label
            sentiment_counts = df_reviews_with_keywords["Sentiment Label"].value_counts().to_dict()

            # Display the descriptive results
            st.markdown("## Deskripsi Sentimen dari Ulasan Terfilter:")
            for label, count in sentiment_counts.items():
                st.markdown(f"- **{label}:** {count} ulasan")

            # Plot sentiment analysis
            plt.figure(figsize=(10, 6))
            sns.barplot(x=list(sentiment_counts.keys()), y=list(sentiment_counts.values()))
            plt.title('Sentiment Analysis of Filtered Reviews')
            plt.xlabel('Sentiment Label')
            plt.ylabel('Number of Reviews')
            st.pyplot(plt)

if __name__ == "__main__":
    main()
