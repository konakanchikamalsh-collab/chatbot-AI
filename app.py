import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI

st.set_page_config(
    page_title="AI Document Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* Main header */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.5);
        font-size: 1rem;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        background: rgba(52, 211, 153, 0.15);
        border: 1px solid rgba(52, 211, 153, 0.4);
        color: #34d399;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }
    .status-badge-waiting {
        background: rgba(251, 191, 36, 0.15);
        border: 1px solid rgba(251, 191, 36, 0.4);
        color: #fbbf24;
    }

    /* Feature cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 2rem 0;
    }
    .feature-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        color: white;
    }
    .feature-card .icon {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    .feature-card h4 {
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
        color: white;
    }
    .feature-card p {
        font-size: 0.78rem;
        color: rgba(255,255,255,0.5);
        margin: 0;
    }

    /* Chat messages */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    .user-msg {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 18px 18px 4px 18px;
        padding: 0.8rem 1.2rem;
        color: white;
        margin: 0.5rem 0;
        margin-left: 20%;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .bot-msg {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 18px 18px 18px 4px;
        padding: 0.8rem 1.2rem;
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0;
        margin-right: 20%;
    }

    /* Sidebar labels */
    .sidebar-label {
        color: rgba(255,255,255,0.7) !important;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }

    /* Input styling */
    .stTextInput input, .stChatInput textarea {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        color: white !important;
        border-radius: 10px !important;
    }

    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4) !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.05) !important;
        border: 2px dashed rgba(255,255,255,0.2) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }

    /* All text white */
    p, span, label, div {
        color: rgba(255,255,255,0.85);
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 15px !important;
    }

    /* Divider */
    hr {
        border-color: rgba(255,255,255,0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False
if "doc_name" not in st.session_state:
    st.session_state.doc_name = ""

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Setup")
    st.markdown("---")

    api_key = st.text_input(
        "🔑 OpenAI API Key",
        type="password",
        placeholder="sk-..."
    )

    st.markdown("---")
    uploaded_file = st.file_uploader(
        "📎 Upload PDF",
        type=["pdf"],
        help="Max 200MB"
    )

    if st.button("🚀 Process Document", use_container_width=True):
        if not api_key:
            st.error("❌ Enter your OpenAI API key")
        elif not uploaded_file:
            st.error("❌ Upload a PDF first")
        else:
            with st.spinner("🔍 Reading document..."):
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
                st.session_state.vectorstore = Chroma.from_texts(chunks, embeddings)
                st.session_state.api_key = api_key
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

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:rgba(255,255,255,0.3); text-align:center'>
        Built with OpenAI + LangChain<br>
        Powered by RAG Technology
    </div>
    """, unsafe_allow_html=True)

# ── Main Content ──────────────────────────────────────────────
if not st.session_state.doc_processed:
    # Landing state
    st.markdown("""
    <div class="main-header">
        <h1>📄 AI Document Chatbot</h1>
        <p>Upload any PDF and have a conversation with it</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="icon">🧠</div>
            <h4>Smart Understanding</h4>
            <p>AI reads and understands your entire document instantly</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="icon">💬</div>
            <h4>Natural Chat</h4>
            <p>Ask questions in plain English, get clear answers</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="icon">⚡</div>
            <h4>Instant Answers</h4>
            <p>No more scrolling through pages to find what you need</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 Upload your PDF in the sidebar to get started")

else:
    # Chat state
    st.markdown(f"""
    <div class="main-header">
        <h1>📄 AI Document Chatbot</h1>
        <p>Chatting with: <strong>{st.session_state.doc_name}</strong></p>
    </div>
    """, unsafe_allow_html=True)

    # Chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask anything about your document..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
                context = "\n\n".join([doc.page_content for doc in docs])

                client = OpenAI(api_key=st.session_state.api_key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": f"You are a helpful assistant. Answer questions based on this document content:\n\n{context}\n\nBe concise and accurate."},
                        {"role": "user", "content": prompt}
                    ]
                )
                answer = response.choices[0].message.content
                st.write(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
