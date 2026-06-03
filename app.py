import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq

st.set_page_config(page_title="AI Document Chatbot", page_icon="📄", layout="wide")

api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05) !important;
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.05) !important;
        border: 2px dashed rgba(255,255,255,0.3) !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploader"] * {
        color: white !important;
        background: transparent !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.05) !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: rgba(102, 126, 234, 0.3) !important;
        color: white !important;
        border: 1px solid rgba(102, 126, 234, 0.5) !important;
        border-radius: 8px !important;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    h1, h2, h3, p, span, label, div { color: white !important; }
    hr { border-color: rgba(255,255,255,0.1) !important; }
</style>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False
if "doc_name" not in st.session_state:
    st.session_state.doc_name = ""

with st.sidebar:
    st.markdown("## ⚙️ Setup")
    st.markdown("---")
    uploaded_file = st.file_uploader("📎 Upload PDF", type=["pdf"])

    if st.button("🚀 Process Document", use_container_width=True):
        if not uploaded_file:
            st.error("❌ Upload a PDF first")
        else:
            with st.spinner("🔍 Reading document..."):
                pdf_reader = PdfReader(uploaded_file)
                raw_text = ""
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        raw_text += text + "\n"

                if not raw_text.strip():
                    st.error("❌ Could not extract text. Try a different PDF.")
                else:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=100,
                        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
                    )
                    chunks = splitter.split_text(raw_text)
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)
                    st.session_state.doc_processed = True
                    st.session_state.doc_name = uploaded_file.name
                    st.success(f"✅ Ready! {len(chunks)} chunks")

    st.markdown("---")
    if st.session_state.doc_processed:
        st.markdown(f"📄 **{st.session_state.doc_name}**")
        st.markdown('<span style="color:#34d399">● Document loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#fbbf24">● Waiting for document</span>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

if not st.session_state.doc_processed:
    st.title("📄 AI Document Chatbot")
    st.markdown("Upload any PDF and have a conversation with it")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🧠 **Smart Understanding**\n\nAI reads your entire document instantly")
    with col2:
        st.info("💬 **Natural Chat**\n\nAsk questions in plain English")
    with col3:
        st.info("⚡ **Instant Answers**\n\nNo more scrolling through pages")
    st.markdown("---")
    st.warning("👈 Upload your PDF in the sidebar to get started")

else:
    st.title("📄 AI Document Chatbot")
    st.markdown(f"Chatting with: **{st.session_state.doc_name}**")
    st.markdown("---")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ask anything about your document..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
                context = "\n\n".join([doc.page_content for doc in docs])
                response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {"role": "system", "content": f"Answer questions based on this document:\n\n{context}\n\nBe concise and accurate."},
                        {"role": "user", "content": prompt}
                    ]
                )
                answer = response.choices[0].message.content
                st.write(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
