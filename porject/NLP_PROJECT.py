import os
import json
import re
import datetime
from collections import Counter
import nltk
import numpy as np
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.express as px
import networkx as nx
from youtube_comment_downloader import YoutubeCommentDownloader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download("stopwords")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_comments(video_id: str, max_comments: int = 1000):
    """
    Step 1: Download up to max_comments from YouTube video and save to raw_comments.json.
    P.S: I do NOT pass sort_by_time here, to avoid the TypeError.
    """
    downloader = YoutubeCommentDownloader()
    comments = []
    
    for c in downloader.get_comments(video_id):
        comments.append({
            "author": c["author"],
            "text": c["text"],
            "time": c["time"],  
        })
        if len(comments) >= max_comments:
            break
    with open(f"{DATA_DIR}/raw_comments.json", "w", encoding="utf-8") as f:
        json.dump(comments, f, ensure_ascii=False, indent=2)
    print(f"[1/5] Fetched {len(comments)} comments.")

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower().strip()
    stops = set(nltk.corpus.stopwords.words("english"))
    return " ".join(tok for tok in text.split() if tok not in stops)


def preprocess():
    """Step 2: Read raw_comments.json → add clean_text → write clean_comments.json."""
    with open(f"{DATA_DIR}/raw_comments.json", encoding="utf-8") as f:
        data = json.load(f)
    for c in data:
        c["clean_text"] = clean_text(c["text"])
    with open(f"{DATA_DIR}/clean_comments.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("[2/5] Preprocessing done.")


def analyze_sentiment():
    """Step 3: Read clean_comments.json → VADER analysis → write sentiment_comments.json."""
    analyzer = SentimentIntensityAnalyzer()
    with open(f"{DATA_DIR}/clean_comments.json", encoding="utf-8") as f:
        data = json.load(f)
    for c in data:
        c["sentiment"] = analyzer.polarity_scores(c["clean_text"])
    with open(f"{DATA_DIR}/sentiment_comments.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("[3/5] Sentiment analysis done.")

def load_data():
    with open(f"{DATA_DIR}/sentiment_comments.json", encoding="utf-8") as f:
        return json.load(f)

def sentiment_distribution(data):
    dist = Counter()
    for c in data:
        comp = c["sentiment"]["compound"]
        if comp >= 0.05:      dist["positive"] += 1
        elif comp <= -0.05:   dist["negative"] += 1
        else:                 dist["neutral"]  += 1
    return dist

def plot_sentiment(dist):
    labels, counts = zip(*dist.items())
    fig = go.Figure(
        data=[go.Bar(x=labels, y=counts, marker_color=["#2ca02c" if l=="positive" else ("#d62728" if l=="negative" else "#7f7f7f") for l in labels])]
    )
    fig.update_layout(title="Sentiment Distribution", yaxis_title="Count")
    out = f"{DATA_DIR}/sentiment_dist.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"[4a/5] Saved interactive sentiment distribution → {out}")

def length_distribution(data):
    lengths = [len(c["clean_text"].split()) for c in data]
    # Use Plotly's create_distplot for a smooth density curve (no histogram)
    fig = ff.create_distplot([lengths], ["Comment length"], show_hist=False, show_rug=False)
    fig.update_layout(title="Comment Length Density", xaxis_title="Number of Tokens")
    out = f"{DATA_DIR}/length_density.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"[4b/5] Saved comment length density → {out}")


def time_series(data):
    """
    Step 4: Try to create a time series plot of comment counts per day.
    """
    valid_dates = []
    for c in data:
        try:
            # Convert milliseconds to seconds, then to datetime.date
            ts = int(c["time"]) / 1000
            date = datetime.datetime.fromtimestamp(ts).date()
            valid_dates.append(date)
        except (KeyError, TypeError, ValueError):
            continue  

    if not valid_dates:
        print("⚠️ No valid timestamps found. Skipping time_series plot.")
        return

    # Count comments per date
    date_counts = Counter(valid_dates)
    sorted_dates = sorted(date_counts.items())

    x, y = zip(*sorted_dates)

    # Use Plotly for interactive time series
    fig = go.Figure(data=go.Scatter(x=list(x), y=list(y), mode='lines+markers'))
    fig.update_layout(title='Comments Over Time', xaxis_title='Date', yaxis_title='Number of Comments')
    out = f"{DATA_DIR}/time_series.html"
    fig.write_html(out, include_plotlyjs='cdn')
    print(f"[4c/5] Saved interactive time series → {out}")

def plot_top_keywords(data, top_n=10):
    """
    Step 4d/5: Plot top N keywords from clean_text.
    """
    # For compatibility keep a function that produces a top-keywords/emoji visualization.
    # Here we will try to plot top emojis. If none found, fall back to top keywords bar chart.
    def extract_emojis(text):
        # broad unicode ranges for most emoji characters
        emoji_pattern = re.compile("[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF\U0001F1E0-\U0001F1FF]")
        return emoji_pattern.findall(text)

    emojis = []
    tokens = []
    for c in data:
        emojis.extend(extract_emojis(c.get("text", "")))
        tokens.extend(c["clean_text"].split())

    if emojis:
        freq = Counter(emojis).most_common(top_n)
        items, counts = zip(*freq)
        fig = go.Figure(data=[go.Bar(x=list(items), y=list(counts), marker_color=px.colors.qualitative.Dark24)])
        fig.update_layout(title=f"Top {top_n} Emojis in Comments", xaxis_title="Emoji", yaxis_title="Frequency")
        out = f"{DATA_DIR}/top_emojis.html"
        fig.write_html(out, include_plotlyjs="cdn")
        print(f"[4d/5] Saved top emojis → {out}")
    else:
        freq = Counter(tokens).most_common(top_n)
        if not freq:
            print("[4d/5] No keywords found to plot.")
            return
        words, counts = zip(*freq)
        fig = go.Figure(data=[go.Bar(x=list(words), y=list(counts))])
        fig.update_layout(title=f"Top {top_n} Keywords in Comments", xaxis_title="Keyword", yaxis_title="Frequency")
        out = f"{DATA_DIR}/top_keywords.html"
        fig.write_html(out, include_plotlyjs="cdn")
        print(f"[4d/5] Saved top keywords → {out}")

def plot_keyword_network(data, top_n=50, window_size=2, min_edge_weight=2):
    # Build token co-occurrence counts within a sliding window per comment
    all_tokens = []
    for c in data:
        toks = c["clean_text"].split()
        all_tokens.extend(toks)

    # focus on top tokens to reduce graph size
    top_tokens = set([t for t, _ in Counter(all_tokens).most_common(top_n)])

    pair_counts = Counter()
    for c in data:
        toks = [t for t in c["clean_text"].split() if t in top_tokens]
        for i, tok in enumerate(toks):
            for j in range(i+1, min(i+1+window_size, len(toks))):
                pair = tuple(sorted((tok, toks[j])))
                pair_counts[pair] += 1

    G = nx.Graph()
    for (a, b), w in pair_counts.items():
        if w >= min_edge_weight:
            G.add_edge(a, b, weight=w)

    if G.number_of_nodes() == 0:
        print("[4e/5] Keyword network: not enough co-occurrence data to build network.")
        return

    pos = nx.spring_layout(G, seed=42)

    edge_x = []
    edge_y = []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x = []
    node_y = []
    node_text = []
    node_size = []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_text.append(n)
        node_size.append(5 + 2 * G.degree(n))

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text,
                            textposition='top center', hoverinfo='text', marker=dict(size=node_size, color='#1f77b4'))

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title='Keyword Co-occurrence Network', showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    out = f"{DATA_DIR}/keyword_network.html"
    fig.write_html(out, include_plotlyjs='cdn')
    print(f"[4e/5] Saved keyword network → {out}")

def top_keywords(data, top_n=10):
    tokens = []
    for c in data:
        tokens.extend(c["clean_text"].split())
    return Counter(tokens).most_common(top_n)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="NLP project plotting")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip fetching and analysis; use existing data files")
    parser.add_argument("--video-id", default="fK85SQzm0Z0", help="YouTube video id to fetch comments for")
    parser.add_argument("--max-comments", type=int, default=1000, help="Max comments to fetch")
    args = parser.parse_args()

    video_id = args.video_id
    if not args.skip_fetch:
        fetch_comments(video_id, max_comments=args.max_comments)
        preprocess()
        analyze_sentiment()

    data = load_data()
    dist = sentiment_distribution(data)
    print(f"[5/5] Sentiment counts = {dict(dist)}")

    plot_sentiment(dist)
    length_distribution(data)
    time_series(data)
    plot_top_keywords(data)
    plot_keyword_network(data)

    top10 = top_keywords(data)
    print("Top 10 keywords:", top10)


if __name__ == "__main__":
    main()


