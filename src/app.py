import plotly.express as px
import json
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fetch_tweet_text import fetch_tweet_text
import matplotlib as mpl
import os
from screenshot_tweet import download_tweet_image


base_dir = os.path.abspath(os.path.join(os.getcwd()))
screenshots_dir = os.path.join(base_dir, "screenshots")
results_dir = os.path.join(base_dir, "results")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

MODELS = {
    "BERTweet" : os.path.join(base_dir, "models", "bertweet-MiDe22"),
    "TwHIN-BERT" : os.path.join(base_dir, "models", "twhin-bert-MiDe22"),
    "ELECTRA" : os.path.join(base_dir, "models", "electra-MiDe22"),
    "RoBERTa-irony" : os.path.join(base_dir, "models", "roberta-irony-MiDe22"),
    "XLNet" : os.path.join(base_dir, "models", "xlnet-MiDe22"),
}
NUM_LABELS = 3
LABELS_TXT = ["True", "False", "Unverified"]

@st.cache_resource(show_spinner=False)
def load_metrics():
    with open("results/model_metrics.json") as f:
        return json.load(f)

metrics = load_metrics()

@st.cache_resource(show_spinner=False)
def load_clf(model_id):
    cfg = AutoConfig.from_pretrained(model_id)
    cfg.id2label = {0: "True", 1: "False", 2: "Unverified"}
    cfg.label2id = {v: k for k, v in cfg.id2label.items()}

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        model_id, ignore_mismatched_sizes=True, config=cfg
    )
    return pipeline("text-classification", model=mdl, tokenizer=tok, return_all_scores=True)

def prob_to_color(scores):
    p_true = scores[0]["score"]
    p_false = scores[1]["score"]
    p_unverified = scores[2]["score"]
    red = 255 * p_false + 255 * p_unverified / 2
    green = 255 * p_true + 255 * p_unverified / 2

    return mpl.colors.to_hex((red / 255, green / 255, 0))

def clean_text(text):
    text = text.lower()

    text = re.sub(r'url', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)

    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)

st.set_page_config(page_title="Twitter Misinformation Demo", layout="wide")
st.title("Twitter Misinformation Detection")

with st.sidebar:
    model_name = st.selectbox("Select model", list(MODELS.keys()))
    st.write("Current model:", model_name)

    model_metrics = metrics.get(model_name, {})

    if not model_metrics:
        st.info("No metric file for this model.")
    else:
        accs = [d["accuracy"] for d in model_metrics.values()]
        f1s = [d["macro_f1"] for d in model_metrics.values()]
        vls = [d["val_loss"] for d in model_metrics.values()]

        st.markdown("### Metrics")
        st.metric("Accuracy", f"{sum(accs) / len(accs):.3f}")
        st.metric("Macro-F1", f"{sum(f1s) / len(f1s):.3f}")
        st.metric("Val loss", f"{sum(vls) / len(vls):.3f}")

        bar_df = pd.DataFrame({
            "Dataset": list(model_metrics.keys()),
            "Accuracy": [d["accuracy"] for d in model_metrics.values()],
            "Macro-F1": [d["macro_f1"] for d in model_metrics.values()],
        })
        st.markdown("### Per-dataset accuracy")
        fig = px.bar(bar_df, x="Dataset", y="Accuracy", text="Accuracy",
                     range_y=[0.7, 1.1], height=360, width=300,  color="Accuracy",
                     color_continuous_scale="Blues")
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(showlegend=False, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

model = MODELS[model_name]
clf = load_clf(model)

col1, col2 = st.columns([2, 3])

with col1:
    tweet_url = st.text_area("Paste tweet url here", height=150)
    tweet_url = tweet_url.strip()
    img_path = None
    if tweet_url:
        img_path = os.path.join(screenshots_dir, f"{tweet_url.split('/')[-1]}.png")

    if img_path and not os.path.exists(img_path):
        download_tweet_image("photo", tweet_url, tweet_url.split('/')[-1], img_path)

    if st.button("Classify", type="primary") and tweet_url:
        tweet_txt = fetch_tweet_text(tweet_url)
        cleaned_text = clean_text(tweet_txt)
        preds   = clf(cleaned_text)[0]

        label_i = max(range(len(preds)), key=lambda i: preds[i]["score"])
        label   = LABELS_TXT[label_i] if label_i < len(LABELS_TXT) else str(label_i)
        conf    = preds[label_i]["score"]

        badge_col = prob_to_color(preds)

        st.markdown(
            f"<div style='"
            f"background:{badge_col};color:white;padding:0.4em 0.8em;"
            f"border-radius:6px;display:inline-block;font-weight:600'>"
            f"{label} – {conf:.1%}"
            f"</div>",
            unsafe_allow_html=True
        )

        st.subheader("Class probabilities")
        prob_df = pd.DataFrame({
            "label": LABELS_TXT[:NUM_LABELS],
            "prob":  [round(p["score"]*100, 1) for p in preds]
        })
        st.bar_chart(prob_df.set_index("label"))

with col2:
    st.subheader("Tweet screenshot")
    if tweet_url:
        st.image(img_path, caption="Captured tweet", use_container_width=True)
    else:
        st.info("Paste a tweet URL in the left panel to see its screenshot.")