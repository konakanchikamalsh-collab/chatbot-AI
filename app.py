import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Document Chatbot", page_icon="📄", layout="wide")
st.title("📄 AI Document Chatbot")
st.markdown("Upload a PDF and ask anything about it!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chain" not in st.session_state:
    st.session_state.chain = None

with st.sidebar:
    st.header("⚙️ Setup")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if st.button("🚀 Process Document", use_container_width=True):
        if not api_key:
            st.error("Please enter your OpenAI API key")
        elif not uploaded_file:
            st.error("Please upload a PDF file")
        else:
            with st.spinner("Processing your document..."):
                pdf_reader = PdfReader(uploaded_file)
                raw_text = ""
                for page in pdf_reader.pages:
                    raw_text += page.extract_text()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = splitter.split_text(raw_text)

                embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                vectorstore = Chroma.from_texts(chunks, embeddings)

                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True
                )
                st.session_state.chain = ConversationalRetrievalChain.from_llm(
                    llm=ChatOpenAI(
                        openai_api_key=api_key,
                        model_name="gpt-3.5-turbo",
                        temperature=0
                    ),
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                    memory=memory
                )
                st.success(f"✅ Document processed! ({len(chunks)} chunks)")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask a question about your document..."):
    if st.session_state.chain is None:
        st.warning("⚠️ Please upload and process a PDF first!")
    else:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chain({"question": prompt})
                answer = response["answer"]
                st.write(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
