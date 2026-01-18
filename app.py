import os
from flask import Flask, render_template, request, send_file, redirect, url_for
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from collections import Counter
import re
import json

# NLTK downloads
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

app = Flask(__name__)   # <-- FIXED (templates folder auto-loaded)

analyzer = SentimentIntensityAnalyzer()
STOPWORDS = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens

def analyze_feedback_list(feedback_list):
    results = []
    all_tokens = []

    for fb in feedback_list:
        fb_text = str(fb)
        scores = analyzer.polarity_scores(fb_text)
        compound = scores['compound']

        if compound >= 0.05:
            sentiment = 'Positive'
        elif compound <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        results.append({'feedback': fb_text, 'sentiment': sentiment, 'score': compound})
        all_tokens.extend(preprocess_text(fb_text))

    df_result = pd.DataFrame(results)

    sentiment_counts = df_result['sentiment'].value_counts().to_dict()
    total = len(df_result)

    positive_pct = round((sentiment_counts.get('Positive', 0) / total) * 100, 1) if total else 0
    neutral_pct = round((sentiment_counts.get('Neutral', 0) / total) * 100, 1) if total else 0
    negative_pct = round((sentiment_counts.get('Negative', 0) / total) * 100, 1) if total else 0
    avg_score = round(df_result['score'].mean(), 3) if total else 0

    top_n = 12
    word_counts = Counter(all_tokens)
    top_words = word_counts.most_common(top_n)
    words, word_freqs = zip(*top_words) if top_words else ([], [])

    return {
        'df_result': df_result,
        'sentiment_counts': sentiment_counts,
        'labels': list(sentiment_counts.keys()),
        'values': list(sentiment_counts.values()),
        'total': total,
        'positive_pct': positive_pct,
        'neutral_pct': neutral_pct,
        'negative_pct': negative_pct,
        'avg_score': avg_score,
        'top_words': list(words),
        'top_word_freqs': list(word_freqs)
    }

@app.route('/')
def index():
    return render_template("index.html")   # <-- FIXED

@app.route('/analyze', methods=['POST'])
def analyze():
    feedback_list = []

    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return f"Error reading CSV: {e}"

        if 'feedback' not in df.columns:
            return "CSV must contain a 'feedback' column"

        feedback_list = df['feedback'].dropna().astype(str).tolist()

    else:
        text = request.form.get('feedback', '').strip()
        if not text:
            return redirect(url_for('index'))
        feedback_list = [text]

    analysis = analyze_feedback_list(feedback_list)
    df_result = analysis['df_result']

    tables = df_result[['feedback', 'sentiment', 'score']].values.tolist()

    return render_template(
        'result.html',        # <-- FIXED (NO ABSOLUTE PATH)
        tables=tables,
        sentiment_counts=analysis['sentiment_counts'],
        labels=json.dumps(analysis['labels']),
        values=json.dumps(analysis['values']),
        total=analysis['total'],
        positive_pct=analysis['positive_pct'],
        neutral_pct=analysis['neutral_pct'],
        negative_pct=analysis['negative_pct'],
        avg_score=analysis['avg_score'],
        top_words=json.dumps(analysis['top_words']),
        top_word_freqs=json.dumps(analysis['top_word_freqs'])
    )

if __name__ == '__main__':
    app.run(debug=True)
