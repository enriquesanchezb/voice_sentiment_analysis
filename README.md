# AI project for testing

This repository contains examples and guidelines on how to approach testing for a simple AI application designed to recognize emotions in phone conversations.

The primary objective is to explore various testing methodologies for different models and steps involved in the AI development process, and to establish a comprehensive testing pipeline.

## How to Run the App
Ensure you have Python 3.10 installed on your system. This project utilizes [Poetry](https://python-poetry.org/) as its package manager. If you haven't installed Poetry yet, please follow the instructions [here](https://python-poetry.org/docs/#installation).

First, install [Ollama](https://ollama.ai/) and the required models. By default, this project uses the [Mistral](https://ollama.ai/library/mistral) and [Llama2](https://ollama.ai/library/lama2) models.

### Dependencies
To install the necessary dependencies, run the following commands:

```bash
ollama pull mistral
ollama pull llama2
poetry install
```
### Running the Project
To start the application, execute:

```bash
poetry run python app.py
```

## Testing
Testing is a crucial and exciting part of AI development. This project's codebase, including the app, summarization, and sentiment analysis components, utilizes models directly without fine-tuning. This approach is intentional to facilitate the creation of a diverse set of test cases for each use case.

To execute all test cases, simply run:

```bash
poetry run pytest tests
```

### Testing Approach
The testing strategy is designed to accommodate the inherently slow nature of model-based tests. It involves:

- Generating prompts that define the necessary data for each test. The generate_conversation helper method is available to create synthetic text conversations between two individuals, based on specified topics and emotions.
- Applying the synthetic data within the function or pipeline under test.
- Utilizing an alternative model to evaluate the output, ensuring the functionality works as intended.

It's important to note that, due to the reliance on multiple models for validation, there is a possibility of encountering false negatives or positives in the results.
