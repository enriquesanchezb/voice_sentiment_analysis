import pytest
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from src.sentiment import analyze_sentiment

SENTIMENTS = [
    "disappointment",
    "sadness",
    "annoyance",
    "neutral",
    "disapproval",
    "realization",
    "nervousness",
    "approval",
    "joy",
    "anger",
    "embarrassment",
    "caring",
    "remorse",
    "disgust",
    "grief",
    "confusion",
    "relief",
    "desire",
    "admiration",
    "optimism",
    "fear",
    "love",
    "excitement",
    "curiosity",
    "amusement",
    "surprise",
    "gratitude",
    "pride",
]


@pytest.mark.parametrize("sentiment", SENTIMENTS)
def test_analyze_sentiment(sentiment):
    # Positive sentiment
    prompt = PromptTemplate(
        template="Generate a basic sentence with this sentiment: {sentiment}.",
        input_variables=["sentiment"],
    )
    model = Ollama(model="mistral")
    output_parser = StrOutputParser()

    chain = prompt | model | output_parser

    text = chain.invoke({"sentiment": sentiment})
    result = analyze_sentiment(text)
    assert all(key in result.keys() for key in [sentiment])
