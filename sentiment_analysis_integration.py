import pandas as pd
from textblob import TextBlob
import tweepy

class SentimentAnalysisIntegration:
    def __init__(self, twitter_api_keys):
        """
        Initializes the sentiment analysis module with Twitter API keys.

        :param twitter_api_keys: Dictionary containing Twitter API keys and tokens.
        """
        self.auth = tweepy.OAuthHandler(twitter_api_keys['api_key'], twitter_api_keys['api_secret_key'])
        self.auth.set_access_token(twitter_api_keys['access_token'], twitter_api_keys['access_token_secret'])
        self.api = tweepy.API(self.auth)

    def fetch_tweets(self, query, count=100):
        """
        Fetches recent tweets based on a query.

        :param query: Search term for fetching tweets.
        :param count: Number of tweets to fetch.
        :return: List of tweet texts.
        """
        tweets = self.api.search(q=query, count=count, lang='en', tweet_mode='extended')
        return [tweet.full_text for tweet in tweets]

    def analyze_sentiment(self, texts):
        """
        Analyzes the sentiment of a list of texts.

        :param texts: List of strings to analyze.
        :return: DataFrame with texts and their corresponding sentiment scores.
        """
        sentiments = []
        for text in texts:
            blob = TextBlob(text)
            sentiments.append(blob.sentiment.polarity)
        return pd.DataFrame({'text': texts, 'sentiment': sentiments})

    def get_market_sentiment(self, query, count=100):
        """
        Fetches tweets and analyzes their sentiment to gauge market sentiment.

        :param query: Search term for fetching tweets.
        :param count: Number of tweets to fetch.
        :return: Average sentiment score.
        """
        tweets = self.fetch_tweets(query, count)
        sentiment_df = self.analyze_sentiment(tweets)
        return sentiment_df['sentiment'].mean()
