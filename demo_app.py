import os
import re
import json
from collections import Counter

import pandas as pd
import nltk
from flask import Flask, render_template_string, request, redirect, url_for
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

# --- NLTK INITIALIZATION ---
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

app = Flask(__name__)

# --- HTML TEMPLATES AS STRINGS ---

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Customer Feedback Analyzer</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background: #f8f9fa; }
        .hero { margin-top: 40px; margin-bottom: 40px; }
        .card-quick { border-radius: 12px; box-shadow: 0 6px 20px rgba(0,0,0,0.05); }
    </style>
</head>
<body class="p-4">
    <div class="container">
        <div class="text-center hero">
            <h1 class="mb-2">🧠 Customer Feedback Analyzer</h1>
            <p class="text-muted">Paste feedback or upload a CSV file with a <b>'feedback'</b> column.</p>
        </div>
        <div class="row justify-content-center">
            <div class="col-md-10">
                <div class="card card-quick p-4">
                    <form action="/analyze" method="POST" enctype="multipart/form-data">
                        <div class="mb-3">
                            <label for="feedback" class="form-label">Type feedback manually (single entry):</label>
                            <textarea name="feedback" class="form-control" rows="4" placeholder="Type one piece of feedback here..."></textarea>
                        </div>
                        <div class="mb-3">
                            <p class="mb-1 text-muted">OR upload a CSV file with a <b>'feedback'</b> column:</p>
                            <input type="file" name="file" class="form-control">
                        </div>
                        <div class="d-flex gap-2">
                            <button type="submit" class="btn btn-primary">Analyze</button>
                            <a href="#" class="btn btn-outline-secondary" onclick="location.reload();return false;">Reset</a>
                        </div>
                    </form>
                </div>
            </div>
        </div>
        <footer class="text-center mt-4 text-muted small">Built with VADER + Flask</footer>
    </div>
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Feedback Results</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    <style>
        body { background: #ffffff; }
        .card-quick { border-radius: 12px; box-shadow: 0 6px 20px rgba(0,0,0,0.04); }
        .stat { font-size: 1.25rem; font-weight: 600; }
    </style>
</head>
<body class="p-4">
    <div class="container">
        <div class="d-flex justify-content-between align-items-center mb-3">
            <h2>Sentiment Analysis Results</h2>
            <a href="/" class="btn btn-outline-secondary">← Analyze More</a>
        </div>

        <div class="row g-3 mb-4">
            <div class="col-md-3"><div class="card p-3 card-quick text-center"><div class="text-muted small">Total feedback</div><div class="stat">{{ total }}</div></div></div>
            <div class="col-md-3"><div class="card p-3 card-quick text-center"><div class="text-muted small">Positive</div><div class="stat">{{ positive_pct }}%</div></div></div>
            <div class="col-md-3"><div class="card p-3 card-quick text-center"><div class="text-muted small">Neutral</div><div class="stat">{{ neutral_pct }}%</div></div></div>
            <div class="col-md-3"><div class="card p-3 card-quick text-center"><div class="text-muted small">Negative</div><div class="stat">{{ negative_pct }}%</div></div></div>
        </div>

        <div class="row mb-4">
            <div class="col-md-6 mb-3"><div class="card p-3 card-quick"><h6>Sentiment distribution</h6><canvas id="sentimentPie" height="220"></canvas></div></div>
            <div class="col-md-6 mb-3"><div class="card p-3 card-quick"><h6>Sentiment counts</h6><canvas id="sentimentBar" height="220"></canvas></div></div>
        </div>

        <div class="row mb-4">
            <div class="col-md-12"><div class="card p-3 card-quick"><h6>Top words (most frequent)</h6><canvas id="wordBar" height="120"></canvas></div></div>
        </div>

        <div class="card p-3 card-quick">
            <h5 class="mb-3">Detailed Results</h5>
            <table id="resultsTable" class="display" style="width:100%">
                <thead><tr><th>Feedback</th><th>Sentiment</th><th>Score</th></tr></thead>
                <tbody>
                    {% for fb, sentiment, score in tables %}
                    <tr><td>{{ fb }}</td><td>{{ sentiment }}</td><td>{{ "%.3f"|format(score) }}</td></tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        <footer class="text-center mt-4 text-muted small">Average score: {{ avg_score }}</footer>
    </div>

    <script>
        const labels = {{ labels|safe }};
        const values = {{ values|safe }};
        const topWords = {{ top_words|safe }};
        const topWordFreqs = {{ top_word_freqs|safe }};

        new Chart(document.getElementById('sentimentPie'), {
            type: 'pie',
            data: { labels: labels, datasets: [{ data: values, backgroundColor: ['#36A2EB', '#FFCE56', '#FF6384'] }] }
        });

        new Chart(document.getElementById('sentimentBar'), {
            type: 'bar',
            data: { labels: labels, datasets: [{ label: 'Count', data: values, backgroundColor: '#36A2EB' }] },
            options: { scales: { y: { beginAtZero: true } } }
        });

        new Chart(document.getElementById('wordBar'), {
            type: 'bar',
            data: { labels: topWords, datasets: [{ label: 'Frequency', data: topWordFreqs, backgroundColor: '#4BC0C0' }] },
            options: { indexAxis: 'y', scales: { x: { beginAtZero: true } } }
        });

        $(document).ready(function() {
            $('#resultsTable').DataTable({ pageLength: 10, order: [[2, 'desc']] });
        });
    </script>
</body>
</html>
"""

# --- LOGIC & HELPERS ---

analyzer = SentimentIntensityAnalyzer()
STOPWORDS = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = nltk.word_tokenize(text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

def analyze_feedback_list(feedback_list):
    results = []
    all_tokens = []
    for fb in feedback_list:
        fb_text = str(fb)
        scores = analyzer.polarity_scores(fb_text)
        compound = scores['compound']
        sentiment = 'Positive' if compound >= 0.05 else 'Negative' if compound <= -0.05 else 'Neutral'
        results.append({'feedback': fb_text, 'sentiment': sentiment, 'score': compound})
        all_tokens.extend(preprocess_text(fb_text))

    df_result = pd.DataFrame(results)
    counts = df_result['sentiment'].value_counts().to_dict()
    total = len(df_result)
    
    top_words_data = Counter(all_tokens).most_common(12)
    words, freqs = zip(*top_words_data) if top_words_data else ([], [])

    return {
        'df_result': df_result,
        'labels': list(counts.keys()),
        'values': list(counts.values()),
        'total': total,
        'positive_pct': round((counts.get('Positive', 0) / total) * 100, 1) if total else 0,
        'neutral_pct': round((counts.get('Neutral', 0) / total) * 100, 1) if total else 0,
        'negative_pct': round((counts.get('Negative', 0) / total) * 100, 1) if total else 0,
        'avg_score': round(df_result['score'].mean(), 3) if total else 0,
        'top_words': list(words),
        'top_word_freqs': list(freqs)
    }

# --- ROUTES ---

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/analyze', methods=['POST'])
def analyze():
    feedback_list = []
    if 'file' in request.files and request.files['file'].filename != '':
        try:
            df = pd.read_csv(request.files['file'])
            if 'feedback' not in df.columns: return "CSV must contain a 'feedback' column"
            feedback_list = df['feedback'].dropna().astype(str).tolist()
        except Exception as e: return f"Error: {e}"
    else:
        text = request.form.get('feedback', '').strip()
        if not text: return redirect(url_for('index'))
        feedback_list = [text]

    analysis = analyze_feedback_list(feedback_list)
    return render_template_string(
        RESULT_HTML,
        tables=analysis['df_result'][['feedback', 'sentiment', 'score']].values.tolist(),
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