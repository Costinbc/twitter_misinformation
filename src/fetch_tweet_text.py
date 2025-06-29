import sys
import os
import subprocess
import pandas as pd

base_dir = os.path.abspath(os.path.join(os.getcwd()))
tweets_dir = os.path.join(base_dir, "_")

def download_tweet(url: str):
    try:
        result = subprocess.run(
            ['gallery-dl', '--sleep-request', '4-6', url],
        )
        if result:
            print(f"Tweet downloaded with: {result}")
    except subprocess.CalledProcessError as e:
        print(f"Error fetching tweet: {e}")

def fetch_tweet_text(tweet_id: str) -> str:
    tweet_url = tweet_id

    if not tweet_url.startswith("https://"):
        tweet_url = f"https://twitter.com/i/web/status/{tweet_id}"
    else:
        tweet_id = int(tweet_url.split('/')[-1])

    download_tweet(tweet_url)

    try:
        df =  pd.read_json(os.path.join(tweets_dir, "tweets.jsonl"),lines=True)
    except ValueError as e:
        print(f"Error reading JSON data: {e}")
        return ""
    if df.empty:
        print("DataFrame empty.")
        return ""

    content = df[df['tweet_id'] == tweet_id]['content']

    if content.empty:
        print(f"No tweet found with ID: {tweet_id}")
        return ""

    content = content.iloc[0]

    return content

if __name__ == '__main__':

    if sys.argv[1:]:
        tweet_id = sys.argv[1]
        text = fetch_tweet_text(tweet_id)
        if text:
            print(f"Tweet text: {text}")
        else:
            print("No tweet text found.")
    else:
        print("Usage fetch_tweet_text.py <tweet_id>")
        sys.exit(1)