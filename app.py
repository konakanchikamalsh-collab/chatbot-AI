import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from tavily import TavilyClient
import docx
import openpyxl

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="centered")

groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
tavily_client = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])

st.markdown("""
<style>
    #MainMenu, footer { visibility: hidden; }
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
    [data-testid="stFileUploader"] * { color: white !important; background: transparent !important; }
    [data-testid="stFileUploaderDropzone"] { background: rgba(255,255,255,0.05) !important; }
    [data-testid="stFileUploaderDropzone"] button {
        background: rgba(102,126,234,0.3) !important;
        color: white !important;
        border: 1px solid rgba(102,126,234,0.5) !important;
        border-radius: 8px !important;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.6rem !important;
    }
    .mode-card {
        background: rgba(255,255,255,0.05);
        border: 2px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1rem;
        text-align: center;
        margin: 0.3rem;
    }
    .mode-card.active {
        background: linear-gradient(135deg, rgba(102,126,234,0.3), rgba(118,75,162,0.3));
        border-color: #667eea;
        box-shadow: 0 0 20px rgba(102,126,234,0.3);
    }
    h1, h2, h3, p, span, label, div { color: white !important; }
    hr { border-color: rgba(255,255,255,0.1) !important; }
    @media (max-width: 768px) {
        h1 { font-size: 1.5rem !important; }
        [data-testid="stChatMessage"] { font-size: 0.9rem !important; }
    }
</style>
""", unsafe_allow_html=True)

def read_file(f):
    name = f.name.lower()
    text = ""
    if name.endswith(".pdf"):
        for page in PdfReader(f).pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    elif name.endswith(".docx"):
        d = docx.Document(f)
        for p in d.paragraphs:
            if p.text:
                text += p.text + "\n"
        for table in d.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        text += cell.text.strip() + "\n"
    elif name.endswith(".xlsx"):
        wb = openpyxl.load_workbook(f)
        for sheet in wb.sheetnames:
            ws = wb[sheet]
            text += f"Sheet: {sheet}\n"
            for row in ws.iter_rows(values_only=True):
                r = " | ".join([str(c) for c in row if c is not None])
                if r:
                    text += r + "\n"
    elif name.endswith(".txt"):
        text = f.read().decode("utf-8")
    return text

def web_search(q):
    try:
        res = tavily_client.search(query=q, max_results=5)
        out = ""
        for r in res.get("results", []):
            out += f"Title: {r.get('title','')}\n{r.get('content','')}\nURL: {r.get('url','')}\n\n"
        return out or "Nothing found."
    except Exception as e:
        return f"Search failed: {e}"

def history():
    return [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history[-10:]]

def ask_groq(system, extra_context=""):
    msgs = [{"role": "system", "content": system}] + history()
    return groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=msgs
    ).choices[0].message.content

for key, val in {
    "chat_history": [],
    "doc_text": "",
    "vectorstore": None,
    "doc_processed": False,
    "doc_name": "",
    "mode": "document"
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

with st.sidebar:
    st.markdown("## 🤖 AI Assistant")
    st.markdown("---")
    st.markdown("### Mode")

    c1, c2 = st.columns(2)
    with c1:
        active = "active" if st.session_state.mode == "document" else ""
        st.markdown(f'<div class="mode-card {active}"><div style="font-size:1.8rem">📄</div><div style="font-weight:700;font-size:0.9rem">Document</div><div style="font-size:0.72rem;color:rgba(255,255,255,0.5)">Chat with files</div></div>', unsafe_allow_html=True)
        if st.button("Use", key="d", use_container_width=True):
            st.session_state.mode = "document"
            st.rerun()

    with c2:
        active = "active" if st.session_state.mode == "internet" else ""
        st.markdown(f'<div class="mode-card {active}"><div style="font-size:1.8rem">🌐</div><div style="font-weight:700;font-size:0.9rem">Internet</div><div style="font-size:0.72rem;color:rgba(255,255,255,0.5)">Search the web</div></div>', unsafe_allow_html=True)
        if st.button("Use", key="i", use_container_width=True):
            st.session_state.mode = "internet"
            st.rerun()

    st.markdown("---")

    if st.session_state.mode == "document":
        f = st.file_uploader("📎 Upload file", type=["pdf", "docx", "xlsx", "txt"])
        if st.button("🚀 Process", use_container_width=True):
            if not f:
                st.error("Upload a file first!")
            else:
                with st.spinner("Reading..."):
                    raw = read_file(f)
                    if not raw.strip():
                        st.error("Couldn't read this file.")
                    else:
                        words = len(raw.split())
                        if words <= 4000:
                            st.session_state.doc_text = raw
                            st.session_state.vectorstore = None
                        else:
                            chunks = RecursiveCharacterTextSplitter(
                                chunk_size=800, chunk_overlap=150
                            ).split_text(raw)
                            st.session_state.vectorstore = FAISS.from_texts(
                                chunks,
                                HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                            )
                            st.session_state.doc_text = ""
                        st.session_state.doc_processed = True
                        st.session_state.doc_name = f.name
                        st.success(f"Done! {words} words loaded")

        st.markdown("---")
        if st.session_state.doc_processed:
            st.markdown(f"📄 **{st.session_state.doc_name}**")
            st.markdown('<span style="color:#34d399">● Loaded</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#fbbf24">● No file loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.05);border-radius:12px;padding:1rem;text-align:center">
            <div style="font-size:1.5rem">🌐</div>
            <div style="font-weight:600;margin:0.3rem 0">Web Search Active</div>
            <div style="font-size:0.78rem;color:rgba(255,255,255,0.5)">Ask anything — I'll find the latest answers</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

st.title("🤖 AI Assistant")

if st.session_state.mode == "document" and not st.session_state.doc_processed:
    st.markdown("Upload a file and start chatting with it")
    st.markdown("---")
    cols = st.columns(4)
    for col, icon, label in zip(cols, ["📄","📝","📊","📃"], ["PDF","Word","Excel","Text"]):
        with col:
            st.markdown(f'<div style="background:rgba(255,255,255,0.05);border-radius:10px;padding:0.8rem;text-align:center"><div style="font-size:1.4rem">{icon}</div><div style="font-size:0.8rem;margin-top:0.3rem">{label}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.warning("👈 Upload your file from the sidebar")
elif st.session_state.mode == "document" and st.session_state.doc_processed:
    st.markdown(f"Chatting with **{st.session_state.doc_name}**")
else:
    st.markdown("Ask me anything — searching the web for fresh answers")

st.markdown("---")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

tip = "Ask about your document..." if st.session_state.mode == "document" else "What do you want to know?"

if prompt := st.chat_input(tip):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            if st.session_state.mode == "internet":
                results = web_search(prompt)
                system = f"""You're a helpful assistant with live web access.

Here's what I found online:
{results}

Be direct and conversational. Answer from the search results.
If something is vague like "before year" or "previous one", look at our chat history to figure out what they mean.
Never say you don't have access to recent data. Never send them to other websites. Just answer."""

            elif st.session_state.doc_text:
                system = f"""You're helping someone answer questions based on their document.

Document content:
{st.session_state.doc_text}

Speak in first person, naturally and confidently like you're in a real conversation.
Pull specific details from the document. Don't list things robotically — talk like a human would."""

            elif st.session_state.vectorstore:
                docs = st.session_state.vectorstore.similarity_search(prompt, k=6)
                context = "\n\n".join([d.page_content for d in docs])
                system = f"""You're helping answer questions from a document.

Relevant content:
{context}

Be natural and conversational. Use what's in the document to give a real answer."""

            else:
                system = "You're a helpful assistant. Be friendly and conversational. Remember what we talked about."

            answer = ask_groq(system)
            st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
