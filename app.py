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
    #MainMenu, footer, header { visibility: hidden; }
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
    [data-testid="stSidebar"] { display: none !important; }

    /* Sticky top bar */
    .sticky-top {
        position: sticky;
        top: 0;
        z-index: 999;
        background: linear-gradient(135deg, #0f0c29, #302b63);
        padding: 0.8rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 1rem;
    }

    .stButton button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1rem !important;
        font-size: 0.9rem !important;
        width: 100% !important;
    }

    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.05) !important;
        border: 2px dashed rgba(255,255,255,0.3) !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploader"] * { color: white !important; background: transparent !important; }
    [data-testid="stFileUploaderDropzone"] { background: rgba(255,255,255,0.05) !important; }
    [data-testid="stFileUploaderDropzone"] button {
        background: rgba(102,126,234,0.3) !important;
        color: white !important;
        border: 1px solid rgba(102,126,234,0.5) !important;
        border-radius: 8px !important;
    }

    .mode-btn {
        background: rgba(255,255,255,0.05);
        border: 2px solid rgba(255,255,255,0.15);
        border-radius: 14px;
        padding: 0.8rem;
        text-align: center;
    }
    .mode-btn.active {
        background: linear-gradient(135deg, rgba(102,126,234,0.4), rgba(118,75,162,0.4));
        border-color: #667eea;
        box-shadow: 0 0 15px rgba(102,126,234,0.3);
    }

    .filetype-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 0.6rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .clear-btn button {
        background: rgba(239,68,68,0.2) !important;
        border: 1px solid rgba(239,68,68,0.4) !important;
        color: white !important;
        border-radius: 10px !important;
    }

    [data-testid="stChatMessage"] {
        border-radius: 12px !important;
        margin: 0.3rem 0 !important;
    }

    h1, h2, h3, p, span, label, div { color: white !important; }
    hr { border-color: rgba(255,255,255,0.1) !important; }

    @media (max-width: 480px) {
        h1 { font-size: 1.3rem !important; }
        .mode-btn { padding: 0.6rem !important; }
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
        res = tavily_client.search(query=q, max_results=5, search_depth="advanced")
        out = ""
        for r in res.get("results", []):
            out += f"Title: {r.get('title','')}\n{r.get('content','')}\nURL: {r.get('url','')}\n\n"
        return out or "Nothing found."
    except Exception as e:
        return f"Search failed: {e}"

def history():
    return [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history[-10:]]

def ask_groq(system):
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

# ── Sticky Header ─────────────────────────────────────────────
st.markdown('<div class="sticky-top">', unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;padding:0.3rem 0 0.8rem 0">
    <h1 style="font-size:1.6rem;font-weight:800;background:linear-gradient(90deg,#a78bfa,#60a5fa,#34d399);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0">🤖 AI Assistant</h1>
</div>
""", unsafe_allow_html=True)

# Mode selector in sticky header
c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    active = "active" if st.session_state.mode == "document" else ""
    st.markdown(f'<div class="mode-btn {active}"><span style="font-size:1.2rem">📄</span> <strong>Document</strong></div>', unsafe_allow_html=True)
    if st.button("Select", key="doc_mode", use_container_width=True):
        st.session_state.mode = "document"
        st.rerun()

with c2:
    active = "active" if st.session_state.mode == "internet" else ""
    st.markdown(f'<div class="mode-btn {active}"><span style="font-size:1.2rem">🌐</span> <strong>Internet</strong></div>', unsafe_allow_html=True)
    if st.button("Select", key="web_mode", use_container_width=True):
        st.session_state.mode = "internet"
        st.rerun()

with c3:
    st.markdown('<div style="padding-top:0.5rem"></div>', unsafe_allow_html=True)
    if st.session_state.chat_history:
        st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Document upload ───────────────────────────────────────────
if st.session_state.mode == "document":
    if not st.session_state.doc_processed:
        st.markdown("#### 📎 Upload Your Document")
        f = st.file_uploader("", type=["pdf", "docx", "xlsx", "txt"], label_visibility="collapsed")

        c1, c2, c3, c4 = st.columns(4)
        for col, icon, label in zip([c1,c2,c3,c4], ["📄","📝","📊","📃"], ["PDF","Word","Excel","Text"]):
            with col:
                st.markdown(f'<div class="filetype-card"><div style="font-size:1.2rem">{icon}</div><div style="font-size:0.75rem">{label}</div></div>', unsafe_allow_html=True)

        if st.button("🚀 Process Document", use_container_width=True):
            if not f:
                st.error("Please upload a file first!")
            else:
                with st.spinner("Reading your document..."):
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
                        st.success(f"✅ Ready! {words} words loaded")
                        st.rerun()
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f'<div style="background:rgba(52,211,153,0.1);border:1px solid rgba(52,211,153,0.3);border-radius:10px;padding:0.7rem 1rem"><span style="color:#34d399">●</span> <strong>{st.session_state.doc_name}</strong></div>', unsafe_allow_html=True)
        with col2:
            if st.button("Change", use_container_width=True):
                st.session_state.doc_processed = False
                st.session_state.doc_text = ""
                st.session_state.vectorstore = None
                st.session_state.chat_history = []
                st.rerun()

else:
    st.markdown("""
    <div style="background:rgba(102,126,234,0.1);border:1px solid rgba(102,126,234,0.3);border-radius:12px;padding:0.8rem;text-align:center">
        <span style="font-size:1.2rem">🌐</span> <strong>Web Search Active</strong>
        <div style="font-size:0.8rem;color:rgba(255,255,255,0.5);margin-top:0.2rem">Ask me anything — I'll find real time answers</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Chat ──────────────────────────────────────────────────────
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

Search results:
{results}

When user says things like "before year" or "after year" look at previous messages to understand what topic and year they mean then answer that.
Answer directly from search results. Never mention knowledge cutoff. Never send users to other websites.
Be natural and conversational."""

            elif st.session_state.doc_text:
                system = f"""You're helping someone answer questions based on their document.

Document:
{st.session_state.doc_text}

Speak in first person naturally and confidently. Pull specific details from the document. Talk like a human not a robot."""

            elif st.session_state.vectorstore:
                docs = st.session_state.vectorstore.similarity_search(prompt, k=6)
                context = "\n\n".join([d.page_content for d in docs])
                system = f"""You're helping answer questions from a document.

Content:
{context}

Be natural and conversational. Use the document to give real answers."""

            else:
                system = "You're a helpful assistant. Be friendly and conversational. Remember what we talked about."

            answer = ask_groq(system)
            st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
