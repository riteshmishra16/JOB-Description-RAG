import streamlit as st
import requests

st.title("Job Description RAG Assistant")

question = st.text_input("Ask your question")

if st.button("Submit"):
    response = requests.post(
        "http://127.0.0.1:8000/ask",
        json={"question": question}
    )

    st.write(response.json()["answer"])
