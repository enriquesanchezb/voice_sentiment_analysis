""" 
Get sentiment analysis results for the given text
"""

from transformers import pipeline

sentiment_pipeline = pipeline(model="SamLowe/roberta-base-go_emotions")


def analyze_sentiment(text: str) -> dict:
    """Returns the sentiment analysis results for the given text"""
    try:
        results = sentiment_pipeline(text)
        sentiment_results = {result["label"]: result["score"] for result in results}
        return sentiment_results
    except Exception as e:
        print(f"Error in analyze_sentiment: {e}")
        return {}
