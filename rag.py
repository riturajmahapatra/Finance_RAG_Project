from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def create_vector_store(df):

    documents = []

    for _, row in df.iterrows():

        text = (
            f"Date: {row['Date']}\n"
            f"Details: {row['Details']}\n"
            f"Amount: {row['Amount']}\n"
            f"Currency: {row['Currency']}\n"
            f"Transaction Type: {row['Debit/Credit']}\n"
            f"Status: {row['Status']}"
        )

        documents.append(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(
        documents,
        embeddings
    )

    return vector_store