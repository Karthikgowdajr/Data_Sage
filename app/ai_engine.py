import os
import pandas as pd
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def analyze(df: pd.DataFrame, query: str):
    preview = df.head(20).to_string()

    prompt = f"""
You are a data analyst.

Dataset preview:
{preview}

User question:
{query}

Answer clearly and accurately.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful data analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return [
        {
            "type": "text",
            "value": response.choices[0].message.content
        }
    ]
