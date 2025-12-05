from dotenv import load_dotenv
import os
import json
from openai import OpenAI

load_dotenv()

OPENAI_API_TOKEN = os.getenv("OPENAI_API_TOKEN")

client = OpenAI(api_key=OPENAI_API_TOKEN)


messages = [
    {
        "role": "user",
        "content": "I love the jacket quality, but shipping took forever and the color was slightly different than the photos.",
    }
]

function_definition = [
    {
        "type": "function",
        "function": {
            "name": "extract_review_data",
            "description": "Extracts structed data from customer reviews.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sentiment": {
                        "type": "string",
                        "description": "overall sentiment e.g. positive/neutral/negative",
                    },
                    "pros": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "list of positive aspects mentioned in the review",
                    },
                    "cons": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "list of negative aspects mentioned in the review",
                    },
                    "product": {
                        "type": "string",
                        "descripiton": "name of the product being reviewed",
                    },
                    "shipping_experience": {
                        "type": "string",
                        "description": "description of the shipping experience",
                    },
                },
                "required": [
                    "sentiment",
                    "pros",
                    "cons",
                    "product",
                    "shipping_experience",
                ],
                "additionalProperties": False,
            },
        },
    }
]


function_definition.append(
    {
        "type": "function",
        "function": {
            "name": "reply_to_review",
            "description": "Generates a polite reply to a customer review.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reply_message": {
                        "type": "string",
                        "description": "a polite and professional reply to the customer's review",
                    }
                },
                "required": ["reply_message"],
                "additionalProperties": False,
            },
        },
    }
)


def get_response(messages, function_definition):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a helpful assistant."}]
        + messages,
        tools=function_definition,
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    msg = response.choices[0].message

    if not msg.tool_calls:
        return "No function call detected in the response."

    # Collect tool call arguments (what model returned for each function)
    out = {}
    for tc in msg.tool_calls:
        fn_name = tc.function.name
        args = json.loads(tc.function.arguments or "{}")
        out[fn_name] = args

    return out


response = get_response(messages, function_definition)
print(response)
