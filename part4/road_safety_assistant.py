import os
import json
import time
import requests
from dotenv import load_dotenv

# Load the API key
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("GROQ_API_KEY was not found")
    raise SystemExit

url = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Load the local road safety knowledge
with open("part4/road_safety_knowledge.json", "r", encoding="utf-8") as file:
    knowledge = json.load(file)


def find_relevant_knowledge(question):
    """Find road safety information related to the question."""

    question = question.lower()
    matches = []

    for item in knowledge:
        for keyword in item["keywords"]:
            if keyword in question:
                matches.append(item["information"])
                break

    if matches:
        return matches[:3]

    return [
        "No closely matching local road safety information was found."
    ]


def trim_history(messages, max_characters=4000):
    """Remove old messages when conversation history becomes too large."""

    total_characters = sum(
        len(message["content"]) for message in messages
    )

    while total_characters > max_characters and len(messages) > 2:
        messages.pop(0)

        total_characters = sum(
            len(message["content"]) for message in messages
        )

    return messages


def ask_llm(messages):
    """Send the conversation to the LLM API."""

    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "temperature": 0.2
    }

    for attempt in range(3):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()

                return result["choices"][0]["message"]["content"]

            print(
                "API request failed with status:",
                response.status_code
            )

        except requests.exceptions.RequestException as error:
            print("Request error:", error)

        if attempt < 2:
            print("Retrying...")
            time.sleep(2)

    return "The assistant could not get a response from the API."


def main():
    """Run the road safety assistant."""

    print("===== ROAD SAFETY AI ASSISTANT =====")
    print("Ask a road safety question.")
    print("Type 'exit' to stop the program.")

    conversation = []

    while True:
        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            print("Assistant: Goodbye")
            break

        if not question:
            print("Assistant: Please enter a question.")
            continue

        if len(question) > 500:
            print("Assistant: Please keep the question below 500 characters.")
            continue

        relevant_information = find_relevant_knowledge(question)

        context = "\n".join(relevant_information)

        user_message = f"""
Road safety context:
{context}

User question:
{question}

Answer the question using the road safety context when it is relevant.
If the local context does not contain enough information, clearly say that
the answer is based on general road safety knowledge.
Keep the answer clear and practical.
"""

        conversation.append({
            "role": "user",
            "content": user_message
        })

        conversation = trim_history(conversation)

        answer = ask_llm(conversation)

        print("\nAssistant:", answer)

        conversation.append({
            "role": "assistant",
            "content": answer
        })


if __name__ == "__main__":
    main()