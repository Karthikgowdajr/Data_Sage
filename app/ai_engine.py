import os
import base64
import requests
from pandasai import SmartDataframe
from pandasai.llm.base import LLM


class OpenRouterLLM(LLM):

    @property
    def type(self):
        return "openrouter"

    def call(self, instruction, value=None):

        # Convert PandasAI prompt object to string
        prompt = str(instruction)

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://datasage.ai",
                "X-Title": "Data Sage"
            },
            json={
                "model": "meta-llama/llama-3-8b-instruct",
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        )

        # Check API response
        if response.status_code != 200:
            raise Exception(response.text)

        data = response.json()

        return data["choices"][0]["message"]["content"]


def image_to_base64(path: str):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze(df, query):

    llm = OpenRouterLLM()

    sdf = SmartDataframe(
        df,
        config={
            "llm": llm,
            "save_charts": True,
            "verbose": False
        }
    )

    try:
        result = sdf.chat(query)

    except Exception as e:
        print("FULL ERROR:", e)
        error_message = str(e).lower()

        if "quota" in error_message or "insufficient_quota" in error_message:
            return [{
                "type": "text",
                "value": "⚠️ Data Sage AI usage limit reached. Please try again later."
            }]

        if "rate limit" in error_message or "429" in error_message:
            return [{
                "type": "text",
                "value": "⚠️ Too many requests right now. Please try again in a few minutes."
            }]

        return [{
            "type": "text",
            "value": "⚠️ Something went wrong while analyzing the data."
        }]

    outputs = []

    if isinstance(result, dict) and result.get("type") == "chart":
        chart_path = result.get("value")

        outputs.append({
            "type": "chart",
            "value": image_to_base64(chart_path)
        })

    elif hasattr(result, "to_dict"):
        outputs.append({
            "type": "table",
            "value": result.to_dict()
        })

    else:
        outputs.append({
            "type": "text",
            "value": str(result)
        })

    return outputs