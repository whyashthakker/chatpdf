import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import pickle
from dotenv import load_dotenv
import os

# sidebar
with st.sidebar:
    st.write("## AI-Powered PDF Assistant")
    st.markdown(
        "### Query your Compliance, Accounting, and other Business PDFs like never before."
    )
    st.markdown(
        "Our smart PDF assistant uses state-of-the-art AI technology to extract and analyze text from your PDFs. It allows you to ask complex questions and get precise answers based on the information contained in your documents. Ideal for professionals in the field of Compliance, Accounting, Legal, and anyone else who handles data-heavy PDFs in their work."
    )

    add_vertical_space(2)
    st.write("## About AI-Powered PDF Assistant")
    st.markdown(
        "Our platform leverages the power of OpenAI's language model, offering a new level of interaction with your PDFs. We transform static documents into an interactive knowledge base where queries can be made seamlessly. From accounting principles to compliance regulations, our AI-Powered PDF Assistant will provide concise answers extracted from the content of your uploaded files."
    )


def main():
    st.header("OPEN SOURCE CHAT PDF APP.")
    st.write(
        "Get your OPENAI API KEY: https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key"
    )
    OPENAI_API_KEY = st.text_input(
        "Enter your OPEN AI API key", type="password"
    )  # Add this line

    st.write(
        "Your API key is not stored or sent anywhere, entire code is open source, see this: https://github.com/whyashthakker/chatpdf, drop a ⭐️"
    )

    load_dotenv()

    pdf = st.file_uploader("Upload a PDF file", type="pdf")

    if pdf is not None:  # check if a file was uploaded
        pdf_reader = PdfReader(pdf)
        st.write(pdf.name)

        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        # EMBEDDINGS

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pk1"):
            with open(f"{store_name}.pk1", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pk1", "wb") as f:
                pickle.dump(VectorStore, f)

        # ACCEPT USER QUESTIONS / QUERY

        query = st.text_input("Let's ask questions :)")
        st.write(query)

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI(
                model_name="gpt-3.5-turbo",
                max_tokens=100,
                openai_api_key=OPENAI_API_KEY,
            )
            chain = load_qa_chain(llm=llm, chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
                st.write(cb)
                st.write(response)
    else:
        st.write("Please upload a PDF file.")


if __name__ == "__main__":
    main()
