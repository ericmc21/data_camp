from dotenv import load_dotenv
import os
import json
from openai import OpenAI

load_dotenv()

OPENAI_API_TOKEN = os.getenv("OPENAI_API_TOKEN")

client = OpenAI(api_key=OPENAI_API_TOKEN)

model = "gpt-4.1-mini"

# Test messages
messages = [
    {
        "role": "system",
        "content": (
            "You extract structured data from customer product reviews. "
            "Always call the provided tool and fill its JSON arguments from the text. "
            "If a field is not explicitly stated, infer it carefully from context if reasonable; "
            "otherwise use an empty string for product_name/variant and choose sentiment best-fit."
        ),
    },
    {
        "role": "user",
        "content": (
            "Review: I bought the Acme Water Bottle (32oz, blue). "
            "It keeps drinks cold all day and feels premium. Love it!"
        ),
    },
]

function_definition = [
    {
        "type": "function",
        "function": {
            "name": "extract_review_info",
            "description": "Extract for each one the product name, variant, and customer sentiment",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "Name of the product",
                    },
                    "variant": {
                        "type": "string",
                        "description": "Variant of the product",
                    },
                    "sentiment": {
                        "type": "string",
                        "description": "Customer sentiment, positive/negative/neutral",
                        "enum": ["positive", "negative", "neutral"],
                    },
                },
                "required": ["product_name", "variant", "sentiment"],
                "additionalProperties": False,
            },
        },
    }
]

response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=function_definition,
    tool_choice={"type": "function", "function": {"name": "extract_review_info"}},
)

# Print just the JSON args the model produced for the tool call:
args_json = response.choices[0].message.tool_calls[0].function.arguments
print(args_json)

# (Optional) parse into a dict:
print(json.loads(args_json))
