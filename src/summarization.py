from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import Ollama


def topics_for_text(file_conv: str) -> str:
    """Get the topics for a text.

    Args:
        file_conv (str): The file with the conversation.

    Returns:
        str: The topics for the text.
    """
    prompt_template = """The next text is a conversation between two people about a topic. Your task is to summarize the conversation in only a list of words that describe the conversation. The list of words should be separated by a comma. The conversation is the following:
    "{text}"
    TOPICS:"""

    prompt = PromptTemplate.from_template(prompt_template)
    llm = Ollama(model="mistral")

    loader = TextLoader(file_conv)
    docs = loader.load()
    # Define StuffDocumentsChain
    chain = load_summarize_chain(
        llm, chain_type="stuff", prompt=prompt, input_key="text"
    )
    result = chain({"text": docs}, return_only_outputs=True)
    return [element.strip().lower() for element in result["output_text"].split(", ")]
