# Part 4 - Road Safety AI Assistant

## Overview

This part of the project extends the road accident project into a conversational road safety assistant.

The assistant accepts road safety questions, retrieves relevant information from a local knowledge file and sends the selected context to an LLM API.

The system also includes simple production guardrails for input handling, conversation size and API failures.

## RAG-Style Retrieval
# Part 4 - LLM-Powered Tabular Record Batch Scoring

## Chosen Track

**Track B - Tabular Record Batch Scoring**

This part adds an LLM-powered road safety assessment feature to the cleaned collision dataset.

Three records from the cleaned dataset are converted into JSON objects and scored individually using a public LLM API. The LLM returns a structured road safety assessment for every record.

The returned JSON is parsed and validated against a predefined JSON schema before it is accepted.

## Feature Overview

The pipeline performs the following steps:

1. Loads three road collision records from `part1/cleaned_data.csv`.
2. Formats each record as a JSON object.
3. Checks the user prompt for personally identifiable information.
4. Sends one API request per record to the LLM.
5. Requests a structured JSON road safety assessment.
6. Strips whitespace from the raw LLM response.
7. Parses the response using `json.loads()`.
8. Validates the parsed object using `jsonschema.validate()`.
9. Applies a fallback assessment when JSON parsing or schema validation fails.
10. Compares LLM responses at temperature 0 and temperature 0.7.

## How to Run

Run the script from the main project folder:

```bash
python part4/llm_batch_scoring.py
```

The API key must be stored in a local `.env` file.

Required environment variable:

```text
LLM_API_KEY=your_api_key
```

The `.env` file is excluded from Git using `.gitignore`.

The API key is never hardcoded in the Python source code.

## LLM API Connection

A reusable function named `call_llm()` is used for all LLM requests.

The function accepts:

- `system_prompt`
- `user_prompt`
- `temperature`
- `max_tokens`

The API payload contains:

- model
- messages
- temperature
- max_tokens

The messages field contains system and user messages as dictionaries with `role` and `content`.

The request headers contain the Bearer authorization token and JSON content type.

The request is made using `requests.post()`.

If the HTTP response status code is not 200, the function prints the HTTP status code and returns `None`.

For a successful response, the function reads:

```python
response.json()["choices"][0]["message"]["content"]
```

A simple API test was performed using the instruction:

```text
Reply with only the word: hello
```

The API returned a visible response, confirming that the reusable API function was working before the batch-scoring pipeline was executed.

## Exact System Prompt

The exact system prompt used for batch scoring is:

```text
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
```

The system prompt contains exactly one worked input-output example.

The example shows the expected scoring behaviour and the exact structured output format.

## User Prompt Template

The user prompt template used for every collision record is:

```text
Assess this road collision record and return only the required JSON object.

Record:
{JSON_RECORD}
```

`{JSON_RECORD}` is replaced with the JSON representation of the current collision record.

Each of the three records is passed to the LLM using this same prompt structure.

## Scoring Rubric

The road safety scoring rubric defines three risk levels.

### High Risk

A record may be considered high risk when serious or fatal collision severity is present or when multiple casualties occur together with hazardous road, weather, or lighting conditions.

### Medium Risk

A record may be considered medium risk when warning signals such as multiple vehicles, higher speed limits, poor lighting, adverse weather, or adverse road surface conditions are present.

### Low Risk

A record may be considered low risk when few warning signals are present and the driving conditions appear ordinary.

The LLM is instructed to flag high-risk records for review.

Medium-risk records may also be flagged when several warning signals occur together.

## Structured JSON Schema

The expected LLM output contains five required scalar fields:

```json
{
  "risk_tier": "low|medium|high",
  "flag_for_review": true,
  "primary_signal": "string",
  "confidence": "low|medium|high",
  "recommended_action": "string"
}
```

The five required fields are:

- `risk_tier` - string
- `flag_for_review` - boolean
- `primary_signal` - string
- `confidence` - string
- `recommended_action` - string

The schema also prevents additional unexpected properties.

`risk_tier` must be one of:

```text
low
medium
high
```

`confidence` must also be one of:

```text
low
medium
high
```

## Structured Output Validation

After every LLM scoring call, the raw response is first cleaned using:

```python
response.strip()
```

The response is then parsed using:

```python
json.loads()
```

JSON parsing is performed inside a `try-except json.JSONDecodeError` block.

If the response is not valid JSON, the JSON error is printed and a fallback dictionary is returned.

After successful JSON parsing, the dictionary is validated using:

```python
jsonschema.validate()
```

Schema validation is performed inside a `try-except jsonschema.ValidationError` block.

If schema validation fails, the validation error message is printed.

The fallback value is:

```json
{
  "risk_tier": null,
  "flag_for_review": null,
  "primary_signal": null,
  "confidence": null,
  "recommended_action": null
}
```

This prevents malformed or structurally incorrect LLM responses from being silently accepted by the application.

## Three-Record Batch Scoring

Three distinct records were selected from the cleaned road collision dataset.

Each record was converted to a JSON object and sent to the LLM in a separate API request.

For every record, the program prints:

- the input record
- the raw LLM response
- the validated assessment
- the validation outcome

The three-record demonstration is summarized below.

| Input Record | LLM Assessment JSON | Validation Status |
| --- | --- | --- |
| Record 1 | Structured road safety assessment with all five required fields | Pass |
| Record 2 | Structured road safety assessment with all five required fields | Pass |
| Record 3 | Structured road safety assessment with all five required fields | Pass |

All three batch-scoring inputs completed the structured validation pipeline successfully during the final run.

## End-to-End Demonstration Table

| Input | LLM Output | Valid JSON | Pass/Block |
| --- | --- | --- | --- |
| Record 1 | Valid five-field road safety assessment | Pass | Pass |
| Record 2 | Valid five-field road safety assessment | Pass | Pass |
| Record 3 | Valid five-field road safety assessment | Pass | Pass |

The records passed the PII guardrail because the selected road collision fields did not contain an email address or a 10-digit phone number.

The LLM responses were parsed and validated before being accepted.

## Temperature Choice

The main scoring pipeline uses:

```text
temperature = 0
```

Temperature 0 was chosen because this is a structured data task.

The system expects the LLM to follow a fixed scoring rubric and return a predictable JSON structure.

A low temperature near zero produces more deterministic and consistent outputs. This is useful for automated structured-data processing because unexpected variation in wording or output format can cause JSON parsing and validation problems.

## Temperature A/B Comparison

Each of the three test records was processed twice:

- once with temperature 0
- once with temperature 0.7

The comparison is summarized below.

| Input | Output at temp=0 | Output at temp=0.7 | Key difference |
| --- | --- | --- | --- |
| Record 1 | Structured five-field JSON assessment | Structured assessment with possible wording variation | Temperature 0 is more consistent in field wording |
| Record 2 | Deterministic rubric-based assessment | Similar risk decision with potentially varied explanation text | Temperature 0.7 allows broader wording choices |
| Record 3 | Consistent structured assessment | Assessment may use different descriptive language | Higher temperature introduces response variability |

At temperature 0, the model behaves more deterministically because it strongly favours the highest-probability next-token choices.

This produces more stable and predictable output, which is valuable for JSON-based application pipelines.

At temperature 0.7, the model samples from a broader probability distribution. This introduces more variability in generated text.

The overall risk assessment may remain similar, but fields such as `primary_signal` and `recommended_action` can be expressed using different wording.

For this feature, temperature 0 is preferred because structural consistency and repeatability are more important than creative variation.

## PII Guardrail

A regular-expression guardrail is applied before every LLM call.

The guardrail checks for:

- email addresses
- 10-digit phone numbers
- phone numbers formatted using spaces, dots, or hyphens

The guardrail function returns `True` when PII is detected.

If PII is detected, the LLM is not called.

Instead, the program prints:

```text
Input blocked: PII detected.
```

and returns `None`.

## Guardrail Test Results

Two guardrail tests were performed.

| Test Input | Expected Result | Actual Behaviour |
| --- | --- | --- |
| Input containing `test@example.com` | Block | LLM call blocked and PII warning printed |
| Clean road safety input | Pass | Input proceeded to the LLM API |

The email-containing test demonstrated that potentially identifying information is blocked before the external LLM request.

The clean input demonstrated that ordinary non-PII text is allowed to proceed.

## Design Decisions

Track B was selected because road collision records are naturally tabular and can be represented as JSON objects.

The LLM is used as a structured scoring layer rather than as a replacement for the supervised machine learning models developed in Parts 2 and 3.

A strict five-field schema was used to make the LLM output easier for another application to consume.

The system prompt explicitly requests JSON-only output to reduce unnecessary natural-language text.

Temperature 0 was selected to improve consistency.

Schema validation and fallback handling protect the downstream application from malformed LLM output.

The PII regex guardrail is executed before every LLM request so detected email addresses and phone numbers are not intentionally sent to the external LLM API by this application.

## Limitations

The LLM assessment is based on a manually defined road safety scoring rubric.

The output should therefore be interpreted as a rubric-based assessment rather than a statistically calibrated accident-risk probability.

The regex PII guardrail only detects the email and phone patterns specified in the assignment. A production system would require broader privacy detection and data governance controls.

LLM outputs can also vary between model versions and providers. JSON schema validation helps control output structure but does not guarantee that every risk judgment is factually correct.

## Conclusion

Part 4 implements Track B, an LLM-powered tabular road collision batch-scoring feature.

Three cleaned dataset records are formatted as JSON objects and sent to the LLM through a reusable API function.

The solution demonstrates prompt engineering with a defined business rubric and exactly one worked example, deterministic temperature settings, structured JSON parsing, JSON Schema validation, fallback handling, temperature A/B comparison, and a PII guardrail.

The complete Part 4 pipeline ran successfully from top to bottom.
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