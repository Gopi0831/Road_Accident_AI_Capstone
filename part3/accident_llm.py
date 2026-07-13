import os
import json
import requests
from dotenv import load_dotenv

# Load the API key from the .env file
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("GROQ_API_KEY was not found in the .env file")
    raise SystemExit

# Groq chat completion API
url = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Sample road accident details
accident_data = {
    "severity": "Serious",
    "number_of_vehicles": 2,
    "number_of_casualties": 2,
    "speed_limit": 60,
    "weather_condition": "Raining",
    "light_condition": "Darkness",
    "road_surface": "Wet"
}

prompt = f"""
Study the road accident details below.

Accident details:
{json.dumps(accident_data, indent=2)}

Return a JSON object with exactly these fields:
risk_level
summary
main_factors
recommendation

risk_level must be Low, Medium, or High.
main_factors must be a list.
Do not add markdown or extra text.
"""

data = {
    "model": "llama-3.3-70b-versatile",
    "messages": [
        {
            "role": "system",
            "content": "You are a road safety analysis assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    "response_format": {
        "type": "json_object"
    },
    "temperature": 0.2
}

try:
    response = requests.post(
        url,
        headers=headers,
        json=data,
        timeout=30
    )

    print("HTTP Status Code:", response.status_code)

    if response.status_code == 200:
        api_response = response.json()

        content = api_response["choices"][0]["message"]["content"]
        report = json.loads(content)

        required_fields = [
            "risk_level",
            "summary",
            "main_factors",
            "recommendation"
        ]

        missing_fields = []

        for field in required_fields:
            if field not in report:
                missing_fields.append(field)

        if missing_fields:
            print("Missing fields:", missing_fields)
        else:
            print("\n===== ROAD SAFETY LLM REPORT =====")
            print("Risk Level:", report["risk_level"])
            print("Summary:", report["summary"])
            print("Main Factors:", report["main_factors"])
            print("Recommendation:", report["recommendation"])

    else:
        print("API request failed")
        print(response.text)

except requests.exceptions.RequestException as error:
    print("Request error:", error)

except json.JSONDecodeError:
    print("The model response was not valid JSON")