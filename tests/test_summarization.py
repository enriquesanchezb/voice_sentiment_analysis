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
    conversation = generate_conversation(topic, sentiment, llm="mistral")

    # Save the conversation to a temporary text file
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as file:
        file.write(conversation)
        temp_filepath = file.name

    summary = summarize(temp_filepath)
    os.remove(temp_filepath)

    assert summary != ""

    model = Ollama(model="llama2")
    output_parser = JsonOutputParser()
    prompt = PromptTemplate(
        template="""Given the following conversation: "{conversation}" and its summary: "{summary}", where the topic is stated as "{topic}" and the sentiment as "{sentiment}", evaluate whether the summary, topic, and sentiment are accurate in relation to the conversation. {format_instructions} The JSON object would have the keys 'summary', 'topic', 'sentiment', assigning True if they are correct and False otherwise to all of them. Dont return other keys or values or any other information.""",
        input_variables=["conversation", "summary", "topic", "sentiment"],
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        },
    )
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
