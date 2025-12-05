from dotenv import load_dotenv
import os
import json
from openai import OpenAI
import requests

load_dotenv()

OPENAI_API_TOKEN = os.getenv("OPENAI_API_TOKEN")

client = OpenAI(api_key=OPENAI_API_TOKEN)

model = "gpt-4.1-mini"


def get_airport_info(airport_code: str):
    url = f"https://api.aviationapi.com/v1/airports"
    response = requests.get(url, params={"apt": airport_code})
    return response.json()


# Describes my function that calls the AviationAPI
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_airport_info",
            "description": "Get information about an airport given its IATA code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "airport_code": {
                        "type": "string",
                        "description": "The IATA code of the airport (e.g., 'JFK' for John F. Kennedy International Airport).",
                    }
                },
                "required": ["airport_code"],
            },
        },
    }
]


def get_response():
    messages = [
        {"role": "system", "content": "Uses tools when needed."},
        {"role": "user", "content": "Tell me about the Traverse City airport."},
    ]

    response = client.chat.completions.create(
        model=model, messages=messages, tools=tools, tool_choice="auto"
    )

    message = response.choices[0].message
    if not message.tool_calls:
        return message.content or "No tool call."

    tool_results = []
    for tc in message.tool_calls:
        if tc.function.name == "get_airport_info":
            code = json.loads(tc.function.arguments)["airport_code"]
            result = get_airport_info(code)
            tool_results.append((tc.id, result))

    return tool_results


response = print(get_response())
