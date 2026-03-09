import streamlit as st
import requests
import os
import base64
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Data Sage AI", layout="centered")

st.title("📊 Data Sage AI")
st.write("Upload a file and ask questions in natural language")

file = st.file_uploader(
    "Upload file",
    type=["csv", "xlsx", "xls", "txt"]
)

question = st.text_input("Ask a question about your data")

if st.button("Analyze"):
    if file is None or question.strip() == "":
        st.error("Please upload a file and enter a question.")
    else:
        with st.spinner("Analyzing with AI..."):
            response = requests.post(
                "https://data-sage-backendd.onrender.com/analyze",
                files={"file": file},
                data={"question": question}
            )

        if response.status_code != 200:
            st.error("Backend error")
            st.write(response.text)
        else:
            results = response.json()["answer"]

            for item in results:
                if item["type"] == "text":
                    st.markdown(item["value"])

                elif item["type"] == "chart":
                    image_bytes = base64.b64decode(item["value"])
                    image = Image.open(BytesIO(image_bytes))
                    st.image(image, caption="Generated Chart", use_container_width=True)

                elif item["type"] == "table":
                    st.dataframe(item["value"])

                else:
                    st.write(item)

