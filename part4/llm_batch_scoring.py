import os
import re
import json

import pandas as pd
import requests

from dotenv import load_dotenv
from jsonschema import validate, ValidationError


# --------------------------------------------------
# Load API settings
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("LLM_API_KEY")

if not api_key:
    raise ValueError(
        "LLM_API_KEY was not found in the .env file."
    )


url = "https://openrouter.ai/api/v1/chat/completions"

model_name = "openai/gpt-4o-mini"


# --------------------------------------------------
# PII guardrail
# --------------------------------------------------

def has_pii(text):
    email_pattern = (
        r"[a-zA-Z0-9_.+-]+@"
        r"[a-zA-Z0-9-]+\."
        r"[a-zA-Z0-9-.]+"
    )

    phone_pattern = (
        r"\b\d{10}\b|"
        r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"
    )

    return bool(
        re.search(email_pattern, text)
        or re.search(phone_pattern, text)
    )


# --------------------------------------------------
# Reusable LLM function
# --------------------------------------------------

def call_llm(
    system_prompt,
    user_prompt,
    temperature=0.0,
    max_tokens=512,
):

    if has_pii(user_prompt):
        print("Input blocked: PII detected.")
        return None

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
    )

    if response.status_code != 200:
        print(
            "HTTP Status Code:",
            response.status_code,
        )

        print(response.text)

        return None

    response_data = response.json()

    return response_data[
        "choices"
    ][0]["message"]["content"]


# --------------------------------------------------
# Simple API test
# --------------------------------------------------

print("===== SIMPLE LLM TEST =====")

test_system_prompt = (
    "Follow the user instruction exactly."
)

test_response = call_llm(
    test_system_prompt,
    "Reply with only the word: hello",
    temperature=0,
)

print("Test response:", test_response)


# --------------------------------------------------
# Track B scoring schema
# --------------------------------------------------

assessment_schema = {
    "type": "object",
    "properties": {
        "risk_tier": {
            "type": "string",
            "enum": [
                "low",
                "medium",
                "high",
            ],
        },
        "flag_for_review": {
            "type": "boolean",
        },
        "primary_signal": {
            "type": "string",
        },
        "confidence": {
            "type": "string",
            "enum": [
                "low",
                "medium",
                "high",
            ],
        },
        "recommended_action": {
            "type": "string",
        },
    },
    "required": [
        "risk_tier",
        "flag_for_review",
        "primary_signal",
        "confidence",
        "recommended_action",
    ],
    "additionalProperties": False,
}


fallback_assessment = {
    "risk_tier": None,
    "flag_for_review": None,
    "primary_signal": None,
    "confidence": None,
    "recommended_action": None,
}


# --------------------------------------------------
# System prompt
# Exactly one worked input-output example
# --------------------------------------------------

system_prompt = """
You are a road safety record scoring assistant.

Output only one valid JSON object.
Do not use Markdown or code fences.

Score each collision record using this rubric:

HIGH RISK:
- serious or fatal collision severity, or
- multiple casualties combined with hazardous road,
  weather, or light conditions.

MEDIUM RISK:
- moderate warning signals such as multiple vehicles,
  higher speed limits, poor light, bad weather, or
  adverse road surface conditions.

LOW RISK:
- few warning signals and ordinary driving conditions.

Set flag_for_review to true for high-risk records.
It may also be true for medium-risk records when
multiple warning signals are present.

Confidence must be low, medium, or high.

Required JSON fields:
risk_tier
flag_for_review
primary_signal
confidence
recommended_action

Worked example:

Input:
{"collision_severity": 1, "number_of_vehicles": 3,
"number_of_casualties": 4, "speed_limit": 60,
"light_conditions": 4, "weather_conditions": 2,
"road_surface_conditions": 2}

Output:
{"risk_tier":"high","flag_for_review":true,
"primary_signal":"Severe collision with multiple casualties",
"confidence":"high",
"recommended_action":"Prioritize the record for detailed safety review"}
"""


# --------------------------------------------------
# Validation function
# --------------------------------------------------

def parse_and_validate(raw_response):

    if raw_response is None:
        return fallback_assessment.copy(), "fail - no response"

    cleaned_response = raw_response.strip()

    try:
        parsed_response = json.loads(
            cleaned_response
        )

    except json.JSONDecodeError as error:
        print(
            "JSON decode error:",
            error,
        )

        return (
            fallback_assessment.copy(),
            f"fail - JSON decode error: {error}",
        )

    try:
        validate(
            instance=parsed_response,
            schema=assessment_schema,
        )

    except ValidationError as error:
        print(
            "Schema validation error:",
            error.message,
        )

        return (
            fallback_assessment.copy(),
            f"fail - validation error: {error.message}",
        )

    return parsed_response, "pass"


# --------------------------------------------------
# Load three dataset records
# --------------------------------------------------

df = pd.read_csv(
    "part1/cleaned_data.csv"
)


record_columns = [
    "collision_severity",
    "number_of_vehicles",
    "number_of_casualties",
    "day_of_week",
    "road_type",
    "speed_limit",
    "junction_detail",
    "light_conditions",
    "weather_conditions",
    "road_surface_conditions",
    "urban_or_rural_area",
]


records = (
    df[record_columns]
    .iloc[[0, 1, 2]]
    .to_dict(orient="records")
)


# --------------------------------------------------
# Main three-record scoring pipeline
# --------------------------------------------------

print("\n===== TRACK B - BATCH SCORING =====")


batch_results = []


for record_number, record in enumerate(
    records,
    start=1,
):

    user_prompt = (
        "Assess this road collision record and "
        "return only the required JSON object.\n\n"
        f"Record:\n{json.dumps(record)}"
    )

    print(
        f"\n----- RECORD {record_number} -----"
    )

    print("Input record:")
    print(json.dumps(record, indent=2))

    raw_response = call_llm(
        system_prompt,
        user_prompt,
        temperature=0,
    )

    print("\nRaw LLM response:")
    print(raw_response)

    assessment, validation_status = (
        parse_and_validate(raw_response)
    )

    print("\nValidated assessment:")
    print(
        json.dumps(
            assessment,
            indent=2,
        )
    )

    print(
        "\nValidation outcome:",
        validation_status,
    )

    batch_results.append({
        "Input": record_number,
        "LLM Output": json.dumps(assessment),
        "Valid JSON": validation_status,
        "Pass/Block": "pass",
    })


# --------------------------------------------------
# Temperature A/B comparison
# --------------------------------------------------

print("\n===== TEMPERATURE A/B COMPARISON =====")


temperature_results = []


for record_number, record in enumerate(
    records,
    start=1,
):

    user_prompt = (
        "Assess this road collision record and "
        "return only the required JSON object.\n\n"
        f"Record:\n{json.dumps(record)}"
    )

    output_temp_zero = call_llm(
        system_prompt,
        user_prompt,
        temperature=0,
    )

    output_temp_seven = call_llm(
        system_prompt,
        user_prompt,
        temperature=0.7,
    )

    print(
        f"\nRecord {record_number}"
    )

    print("\nOutput at temperature 0:")
    print(output_temp_zero)

    print("\nOutput at temperature 0.7:")
    print(output_temp_seven)

    temperature_results.append({
        "Input": record_number,
        "Output at temp=0": output_temp_zero,
        "Output at temp=0.7": output_temp_seven,
    })


# --------------------------------------------------
# PII guardrail tests
# --------------------------------------------------

print("\n===== PII GUARDRAIL TEST =====")


pii_input = (
    "Assess this record for user test@example.com"
)

print("\nEmail-containing input:")

blocked_response = call_llm(
    system_prompt,
    pii_input,
    temperature=0,
)

print("Guardrail result:", blocked_response)


clean_input = (
    "Assess a clean road safety record with no "
    "personal information. Return only the word safe."
)

print("\nClean input:")

clean_response = call_llm(
    "Follow the user instruction exactly.",
    clean_input,
    temperature=0,
)

print("Clean response:", clean_response)


# --------------------------------------------------
# Three-row demonstration table
# --------------------------------------------------

print("\n===== THREE-ROW DEMONSTRATION TABLE =====")


demonstration_table = pd.DataFrame(
    batch_results
)


print(
    demonstration_table.to_string(
        index=False
    )
)


print("\nPart 4 completed successfully.")