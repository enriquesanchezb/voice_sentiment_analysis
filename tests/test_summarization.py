import os
import tempfile

import pytest
from langchain_community.llms import Ollama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from src.generate_conversation import generate_conversation
from src.summarization import summarize
from src.summarization import topics_for_text

# SENTIMENTS = ["neutral", "sadness", "joy"]
# TOPICS = ["ai", "soccer", "art"]
SENTIMENTS = ["joy"]
TOPICS = ["ai"]


@pytest.mark.parametrize("sentiment", SENTIMENTS)
@pytest.mark.parametrize("topic", TOPICS)
def test_topics_conversation(sentiment, topic):
    # Call the function
    conversation = generate_conversation(topic, sentiment, llm="llama2")

    # Save the conversation to a temporary text file
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as file:
        file.write(conversation)
        temp_filepath = file.name

    new_topics = topics_for_text(temp_filepath)
    os.remove(temp_filepath)
    assert any(topic in word for word in new_topics)


@pytest.mark.parametrize("sentiment", SENTIMENTS)
@pytest.mark.parametrize("topic", TOPICS)
def test_summary_conversation(sentiment, topic):
    # Call the function
    conversation = generate_conversation(topic, sentiment, llm="llama2")

    # Save the conversation to a temporary text file
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as file:
        file.write(conversation)
        temp_filepath = file.name

    summary = summarize(temp_filepath)
    os.remove(temp_filepath)

    assert summary != ""

    model = Ollama(model="llama2")
    prompt = PromptTemplate(
        template="""You will read a summary of a conversation with a sentiment and a topic. Your task is to analyze the conversation and the summary and returns a json object where the key is summary, topic, sentiment and the value is True if the sentiment and the topic are correct and False otherwise. The conversation is the following: {conversation} The summary is the following: {summary}, the topic is {topic} and the sentiment is the following: {sentiment}
        JSON:""",
        input_variables=["conversation", "summary", "topic", "sentiment"],
    )
    output_parser = JsonOutputParser()
    chain = prompt | model | output_parser
    result = chain.invoke(
        {
            "conversation": conversation,
            "topic": topic,
            "summary": summary,
            "sentiment": sentiment,
        }
    )
    assert result["summary"], "The summary is not correct"
    assert result["topic"], "The topic is not correct"
    assert result["sentiment"], "The sentiment is not correct"
