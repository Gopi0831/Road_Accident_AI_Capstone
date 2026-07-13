# Part 3 - LLM Road Safety Report Generator

## Overview

In this part of the project, I integrated a large language model API with the road accident project.

The program sends structured accident details to an LLM and receives a road safety analysis in JSON format.

The generated report contains:

- Risk level
- Accident summary
- Main risk factors
- Safety recommendation

## LLM API Integration

The program uses the Groq API.

An HTTP POST request is sent to the chat completions endpoint using the Python `requests` library.

The accident information is included in the JSON request body.

The API response is read as JSON and the model-generated content is parsed into a Python dictionary.

## Structured Output

The model is instructed to return a JSON object with the following fields:

- `risk_level`
- `summary`
- `main_factors`
- `recommendation`

The program checks whether all required fields are available before displaying the report.

This makes the LLM output easier to use in a software application compared with unrestricted text output.

## Sample Accident Data

The sample accident used in the program contains:

- Serious collision severity
- Two vehicles
- Two casualties
- Speed limit of 60
- Rainy weather
- Darkness
- Wet road surface

The LLM classified the accident conditions as high risk and identified rain, darkness and the wet road surface as important factors.

## API Key Security

The Groq API key is not stored directly in the Python source code.

The key is stored locally in a `.env` file using the following environment variable:

```text
GROQ_API_KEY