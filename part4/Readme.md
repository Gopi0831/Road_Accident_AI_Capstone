# Part 4 - Road Safety AI Assistant

## Overview

This part of the project extends the road accident project into a conversational road safety assistant.

The assistant accepts road safety questions, retrieves relevant information from a local knowledge file and sends the selected context to an LLM API.

The system also includes simple production guardrails for input handling, conversation size and API failures.

## RAG-Style Retrieval

The project uses a lightweight retrieval approach.

Road safety information is stored in `road_safety_knowledge.json`. Each knowledge entry contains a topic, keywords and road safety information.

When the user enters a question, the program checks the question against the keywords in the knowledge base.

A maximum of three matching knowledge entries are selected and added to the LLM prompt.

This is a simple keyword-based RAG-style approach. It does not use embeddings or a vector database.

## Conversation History

The assistant stores user and assistant messages in a conversation list.

This allows previous messages to be included in later API requests and supports a basic multi-turn conversation.

## Context Size Guard

A character-based context guard is used to control the size of the conversation history.

The program counts the number of characters in stored messages. If the configured limit is exceeded, the oldest messages are removed until the history is within the limit.

This reduces the chance of sending an unnecessarily large conversation to the API.

## Production Guardrails

The assistant includes the following safeguards:

- Empty input validation
- Maximum question length of 500 characters
- API request timeout of 30 seconds
- Up to three API request attempts
- Short delay before retrying a failed request
- Conversation history size control
- Graceful message when the API cannot return a response
- API key loaded from an environment variable

## API Key Security

The Groq API key is stored locally in a `.env` file.

The Python code reads the following environment variable:

```text
GROQ_API_KEY
```

The `.env` file is excluded from Git using `.gitignore`.

The API key is not hardcoded in the Python source file.

## Files

`road_safety_assistant.py` contains the assistant logic, retrieval process, conversation handling and API request code.

`road_safety_knowledge.json` contains the local road safety knowledge used during retrieval.

## How to Run

Install the required libraries:

```bash
pip install requests python-dotenv
```

Create a `.env` file in the main project folder and add:

```text
GROQ_API_KEY=your_api_key_here
```

Run the assistant from the main project folder:

```bash
python part4/road_safety_assistant.py
```

Enter a road safety question when the program displays the input prompt.

Type `exit` to stop the assistant.

## Example Questions

The assistant can be tested with questions such as:

- Why are wet roads dangerous?
- What should a driver do in this situation?
- Why is speeding risky?
- How does darkness affect road safety?
- Why are junctions dangerous?

## Design Decisions

A JSON file was used as a small local knowledge base because it is simple to inspect and maintain.

Keyword matching was selected as the retrieval method to keep the retrieval process clear and lightweight.

The assistant sends only matching road safety information instead of the complete knowledge base.

Conversation history is retained for multi-turn interaction, while a character-based guard prevents the stored context from growing without control.

## Libraries Used

- Requests
- Python-dotenv
- JSON
- OS
- Time