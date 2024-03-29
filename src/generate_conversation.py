from langchain.prompts import FewShotPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_experimental.tabular_synthetic_data.base import SyntheticDataGenerator
from langchain_experimental.tabular_synthetic_data.prompts import (
    SYNTHETIC_FEW_SHOT_PREFIX,
)
from langchain_experimental.tabular_synthetic_data.prompts import (
    SYNTHETIC_FEW_SHOT_SUFFIX,
)


examples = [
    {
        "example": """"Hello, how are you?", "Great! you?", "Good, good thanks", "Great! See you later!", "See you!"]"""
    }
]


def generate_conversation(
    topic: str, sentiment: str = "positive", llm: str = "mistral"
) -> str:
    """Generate a conversation about a topic with a given sentiment.

    Args:
        topic (str): The topic of the conversation.
        sentiment (str, optional): The sentiment of the conversation. Defaults to "positive".

    Returns:
        str: A conversation between two people about the topic.
    """
    template = FewShotPromptTemplate(
        prefix=SYNTHETIC_FEW_SHOT_PREFIX,
        examples=examples,
        suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
        input_variables=["topic", "extra"],
        example_prompt=PromptTemplate(
            input_variables=["example"], template="{example}"
        ),
    )

    model = Ollama(model=llm)

    generator = SyntheticDataGenerator(template=template, llm=model)
    results = generator.generate(
        subject=f"Create only one phone conversation between two people about {topic}. The sentiment for the conversation has to be {sentiment}. only a few sentences are needed for the conversation, every sentence should be separated by a '\n' character.",
        runs=1,
        extra="A:<list of sentences>",
    )
    return results[0]
