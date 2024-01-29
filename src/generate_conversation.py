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
        "example": """"Hola, como estas?", "Bien, y tu?, "Muy bien, gracias, y tu?", "Yo tambien muy bien, gracias", "Que bueno, nos vemos luego!", "Adios!"]"""
    }
]


def generate_response(topic: str):
    template = FewShotPromptTemplate(
        prefix=SYNTHETIC_FEW_SHOT_PREFIX,
        examples=examples,
        suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
        input_variables=["topic", "extra"],
        example_prompt=PromptTemplate(
            input_variables=["example"], template="{example}"
        ),
    )

    model = Ollama(model="mistral")

    generator = SyntheticDataGenerator(template=template, llm=model)
    results = generator.generate(
        subject=f"Create only one phone conversation between two people about {topic}. only a few sentences are needed for the conversation, every sentence should be separated by a '\n' character.",
        runs=1,
        extra="A:<list of sentences>",
    )
    return results[0]
