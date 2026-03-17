import os
import base64
import requests
import re
from pandasai import SmartDataframe
from pandasai.llm.base import LLM


class OpenRouterLLM(LLM):

    @property
    def type(self):
        return "openrouter"

    def call(self, instruction, value=None):

        # 🔥 Force STRICT Python code output
        prompt = f"""
You are a Python data analyst.

Return ONLY valid Python code.
- No explanations
- No markdown
- No comments
- Only executable code

Rules:
- Use dataframe name: df
- Use pandas and matplotlib
- If chart requested, 반드시 use plt.show()

Task:
{instruction}
"""

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json",
            },
            json={
                # 🔥 CHANGE MODEL (important)
                "model": "openai/gpt-4o-mini",
                "messages":[
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0
            }
        )

        if response.status_code != 200:
            raise Exception(response.text)

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        # 🔥 Clean unwanted markdown if any
        content = re.sub(r"```.*?```", "", content, flags=re.DOTALL).strip()

        return content


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
            "verbose": False,
            "enable_cache": False,
            "custom_whitelisted_dependencies":["pandas", "matplotlib"]
        },
        description=f"""
The dataframe has the following columns:
{', '.join(df.columns)}

Rules:
- Always generate valid pandas code using dataframe 'df'
- Use exact column names
- For charts:
    - Use matplotlib
    - Always call plt.show()
- Return ONLY Python code
"""
    )

    try:
        result = sdf.chat(query)

    except Exception as e:
        print("FULL ERROR:", e)
        return[{
            "type": "text",
            "value": "⚠️ Error while analyzing data."
        }]

    outputs =[]

    # FIX 1: Check if PandasAI returned a string file path to a saved chart
    if isinstance(result, str) and result.endswith(('.png', '.jpg', '.jpeg')) and os.path.exists(result):
        outputs.append({
            "type": "chart",
            "value": image_to_base64(result)
        })

    # Backup: Just in case older/newer PandasAI returns a dict
    elif isinstance(result, dict) and result.get("type") == "chart":
        chart_path = result.get("value")
        outputs.append({
            "type": "chart",
            "value": image_to_base64(chart_path)
        })

    # Check for Tables/Dataframes
    elif hasattr(result, "to_dict"):
        outputs.append({
            "type": "table",
            "value": result.to_dict()
        })

    # Fallback: Normal text responses
    else:
        outputs.append({
            "type": "text",
            "value": str(result)
        })

    return outputs