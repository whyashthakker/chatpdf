import streamlit as st
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


def main():
    st.header("Compliance Questions?")
    load_dotenv()

    pdf = st.file_uploader("Upload a PDF file", type="pdf")

    if pdf is not None:  # check if a file was uploaded
        pdf_reader = PdfReader(pdf)
        st.write(pdf.name)
 
        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                                       chunk_overlap=200, 
                                                       length_function = len
                                                       )
        
        chunks = text_splitter.split_text(text = text)

        #st.write(chunks)


    #EMBEDDINGS

        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding = embeddings)
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pk1"):
            with open(f"{store_name}.pk1", "rb") as f:
                VectorStore = pickle.load(f)
            #st.write("VectorStore loaded from file.")
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding = embeddings)
            with open(f"{store_name}.pk1", "wb") as f:
                pickle.dump(VectorStore, f)

            #st.write("VectorStore created and saved to file.")

    #ACCEPT USER QUESTIONS / QUERY

        query = st.text_input("Let's ask questions :)")
        st.write(query)

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI(model_name='gpt-3.5-turbo', max_tokens=100)
            chain = load_qa_chain(llm = llm, chain_type = "stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs,question=query)
                print(cb)
                st.write(response)

            #st.write(docs)


    #st.write(text)

    else:
        st.write("Please upload a PDF file.")

if __name__ == '__main__':
    main()
 