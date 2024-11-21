import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from htmltemp import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page_num in reader.pages:
            text += page_num.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OllamaEmbeddings(model="llama3.2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_retrieval_chain(vector_store):
    llm = OllamaLLM(model="llama3.2")
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever()
    )
    return chain

def handle_userinput(user_question):
    response = st.session_state.qa_chain({'query': user_question})
    st.write(bot_template.replace("{{MSG}}", response['result']), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Compliance BHAI", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    st.header("Compliance BHAI :memo:")

    user_question = st.text_input("Insert query here:")
    if user_question and st.session_state.qa_chain:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and Click Process.", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                # Get PDF text
                raw_text = get_pdf_text(pdf_docs)
                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # Create vector store with embeddings
                vector_store = get_vectorstore(text_chunks)
                # Create retrieval QA chain
                st.session_state.qa_chain = get_retrieval_chain(vector_store)

if __name__ == "__main__":
    main()
