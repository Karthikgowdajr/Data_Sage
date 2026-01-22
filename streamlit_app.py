import streamlit as st
import requests
import os

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
                "http://127.0.0.1:8000/analyze",
                files={"file": file},
                data={"question": question}
            )

        if response.status_code != 200:
            st.error("Backend error")
            st.write(response.text)
        else:
            results = response.json()["answer"]

            # ---------- LAYER 3: SMART RENDERING ----------
            for item in results:

                if item["type"] == "text":
                    st.markdown(item["value"])

                elif item["type"] == "chart":
                    st.image(item["value"], caption="Generated Chart")

                elif item["type"] == "table":
                    st.dataframe(item["value"])

                elif item["type"] == "file":
                    with open(item["value"], "rb") as f:
                        st.download_button(
                            label="Download file",
                            data=f,
                            file_name=os.path.basename(item["value"])
                        )

                else:
                    st.write(item)
