import os
import streamlit as st
import pandas as pd
import numpy
import json
from dotenv import load_dotenv
from rag import create_vector_store
from google import genai


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)
for model in client.models.list():

    if "generateContent" in model.supported_actions:
        print(model.name)

st.set_page_config(page_title="Finance Dashboard",layout="wide",page_icon="💸")  

def load_transactions(file):
    try:
      df = pd.read_csv(file)
      df.columns = [col.strip() for col in df.columns]
      df["Amount"] = df["Amount"].str.replace("," , "").astype(float)
      df["Date"] = pd.to_datetime(df["Date"])   
      return df
    except Exception as e:
        st.error(f"Error occured while parsing the documents: {str(e)}")
        return None
    

def generate_answer(question, context):

    prompt = f"""
You are a helpful personal finance assistant.

Answer the user's question using ONLY the transaction information
provided in the context.

If the context does not contain enough information to answer,
say that you don't have enough information.

Do not invent transactions or financial information.

Context:
{context}

User question:
{question}

Answer clearly and concisely.
"""

    try:

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:

        return f"Gemini is temporarily unavailable. Please try again. Error: {str(e)}"


def main():

    st.title("Finance Dashboard")

    uploaded_file = st.file_uploader(
        "Upload your transaction file here",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = load_transactions(uploaded_file)

        if df is not None:

            st.dataframe(df)

            with st.spinner("Creating financial knowledge base..."):

                vector_store = create_vector_store(df)

            st.success(
                "Financial data is ready for AI questions!"
            )

            question = st.text_input(
                "🤖 Ask something about your transactions"
            )

            if question:

                results = vector_store.similarity_search(
                    question,
                    k=3
                )
                context = "\n\n".join(
                    result.page_content
                    for result in results
                )

                answer = generate_answer(
                    question,
                    context
                )

                st.subheader("Relevant transactions")
                st.write(answer)

                for result in results:
                    st.write(result.page_content)


main()

def calculate_total(df, keyword):

    matches = df[
        df["Details"]
        .str.contains(keyword, case=False, na=False)
    ]

    debit_transactions = matches[
        matches["Debit/Credit"].str.lower() == "debit"
    ]

    return debit_transactions["Amount"].sum()

   