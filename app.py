import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

st.set_page_config(page_title="AI Document Chatbot", page_icon="📄", layout="wide")

api_key = st.secrets["OPENAI_API_KEY"]

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .main-header p {
        color: rgba(255,255,255,0.5);
        text-align: center;
    }
    .feature-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        color: white;
    }
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05) !important;
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    p, span, label, div { color: rgba(255,255,255,0.85); }
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
                    raw_text += page.extract_text()

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_text(raw_text)
                embeddings = OpenAIEmbeddings(openai_api_key=api_key)
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
    st.markdown("""
    <div class="main-header">
        <h1>📄 AI Document Chatbot</h1>
        <p>Upload any PDF and have a conversation with it</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="feature-card"><div style="font-size:1.8rem">🧠</div><h4>Smart Understanding</h4><p style="font-size:0.8rem;color:rgba(255,255,255,0.5)">AI reads your entire document instantly</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="feature-card"><div style="font-size:1.8rem">💬</div><h4>Natural Chat</h4><p style="font-size:0.8rem;color:rgba(255,255,255,0.5)">Ask questions in plain English</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="feature-card"><div style="font-size:1.8rem">⚡</div><h4>Instant Answers</h4><p style="font-size:0.8rem;color:rgba(255,255,255,0.5)">No more scrolling through pages</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 Upload your PDF in the sidebar to get started")

else:
    st.markdown(f"""
    <div class="main-header">
        <h1>📄 AI Document Chatbot</h1>
        <p>Chatting with: <strong>{st.session_state.doc_name}</strong></p>
    </div>
    """, unsafe_allow_html=True)

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
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": f"Answer questions based on this document:\n\n{context}"},
                        {"role": "user", "content": prompt}
                    ]
                )
                answer = response.choices[0].message.content
                st.write(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
