import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_ollama import OllamaEmbeddings, OllamaLLM
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
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
        separator = "\n",
        chunk_size = 2000,
        chunk_overlap = 200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OllamaEmbeddings(model="llama3.2")
    vectorstore = FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorstore

def get_conversation_chain(vector_store):
    # llm = OllamaLLM(model="llama3.2")
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # chain = RetrievalQA.from_chain_type(
    #     llm, 
    #     retriever=vector_store.as_retriever()
    # )
    llm = OllamaLLM(model="llama3.2")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory = memory
    )
    return chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html = True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html = True)
def main():
    load_dotenv()
    st.set_page_config(page_title="Compliance BHAI", page_icon=":books:")
    st.write(css, unsafe_allow_html = True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Compliance BHAI :memo:")

    user_question = st.text_input("Insert query here:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
         st.subheader("Your Documents")
         pdf_docs = st.file_uploader("Upload your PDFs here and Click Process.", accept_multiple_files = True)
         if st.button("Process"):
            with st.spinner("processing"):
            #get pdf text
                raw_text = get_pdf_text(pdf_docs) 
            #get the text chunks   
                text_chunks = get_text_chunks(raw_text) 
            #create vector store with embeddings
                vector_store = get_vectorstore(text_chunks)
            #create converation
                st.session_state.conversation = get_conversation_chain(vector_store)

if __name__ == "__main__":
    main()