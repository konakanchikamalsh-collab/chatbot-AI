import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from tavily import TavilyClient
import docx, openpyxl

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")

groq = Groq(api_key=st.secrets["GROQ_API_KEY"])
tavily = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])

st.markdown("""<style>
#MainMenu, footer, header {visibility:hidden}
.stApp {background:linear-gradient(135deg,#0f0c29,#302b63,#24243e)}
[data-testid="stSidebar"] {background:rgba(255,255,255,0.05)!important;border-right:1px solid rgba(255,255,255,0.1)!important}
.stButton button {background:linear-gradient(135deg,#667eea,#764ba2)!important;color:white!important;border:none!important;border-radius:10px!important;font-weight:600!important;width:100%!important}
[data-testid="stFileUploader"] {background:rgba(255,255,255,0.05)!important;border:2px dashed rgba(255,255,255,0.3)!important;border-radius:10px!important}
[data-testid="stFileUploader"] * {color:white!important;background:transparent!important}
[data-testid="stFileUploaderDropzone"] {background:rgba(255,255,255,0.05)!important}
[data-testid="stFileUploaderDropzone"] button {background:rgba(102,126,234,0.3)!important;color:white!important;border:1px solid rgba(102,126,234,0.5)!important;border-radius:8px!important}
.mcard {background:rgba(255,255,255,0.05);border:2px solid rgba(255,255,255,0.1);border-radius:12px;padding:0.9rem;text-align:center;margin-bottom:4px}
.mcard.on {background:linear-gradient(135deg,rgba(102,126,234,0.35),rgba(118,75,162,0.35));border-color:#667eea;box-shadow:0 0 16px rgba(102,126,234,0.25)}
h1,h2,h3,p,span,label,div {color:white!important}
hr {border-color:rgba(255,255,255,0.1)!important}
@media(max-width:768px){[data-testid="stSidebar"]{display:none!important}}
@media(min-width:769px){.mob{display:none!important}}
</style>""", unsafe_allow_html=True)

def read(f):
    txt, n = "", f.name.lower()
    if n.endswith(".pdf"):
        for pg in PdfReader(f).pages:
            t = pg.extract_text()
            if t: txt += t + "\n"
    elif n.endswith(".docx"):
        d = docx.Document(f)
        for p in d.paragraphs:
            if p.text: txt += p.text + "\n"
        for tbl in d.tables:
            for row in tbl.rows:
                for cell in row.cells:
                    if cell.text.strip(): txt += cell.text.strip() + "\n"
    elif n.endswith(".xlsx"):
        wb = openpyxl.load_workbook(f)
        for s in wb.sheetnames:
            ws = wb[s]
            txt += f"Sheet:{s}\n"
            for row in ws.iter_rows(values_only=True):
                r = " | ".join([str(c) for c in row if c])
                if r: txt += r + "\n"
    elif n.endswith(".txt"):
        txt = f.read().decode("utf-8")
    return txt

def search(q):
    try:
        res = tavily.search(query=q, max_results=5, search_depth="advanced")
        return "\n\n".join(f"Title:{r.get('title','')}\n{r.get('content','')}\nURL:{r.get('url','')}" for r in res.get("results", [])) or "Nothing found."
    except Exception as e:
        return f"Search error: {e}"

def chat_history():
    return [{"role": m["role"], "content": m["content"]} for m in st.session_state.msgs[-10:]]

def ask(system):
    return groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": system}] + chat_history()
    ).choices[0].message.content

def load_doc(f):
    raw = read(f)
    if not raw.strip():
        st.error("Can't read this file"); return
    words = len(raw.split())
    if words <= 4000:
        st.session_state.doc_txt = raw
        st.session_state.vs = None
    else:
        chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150).split_text(raw)
        st.session_state.vs = FAISS.from_texts(chunks, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
        st.session_state.doc_txt = ""
    st.session_state.loaded = True
    st.session_state.fname = f.name
    st.success(f"Done — {words} words loaded")
    st.rerun()

defaults = {"msgs": [], "doc_txt": "", "vs": None, "loaded": False, "fname": "", "mode": "doc"}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

def mode_card(label, icon, key, mode_val, btn_key):
    css = "on" if st.session_state.mode == mode_val else ""
    st.markdown(f'<div class="mcard {css}"><div style="font-size:1.6rem">{icon}</div><div style="font-weight:700;font-size:0.85rem">{label}</div></div>', unsafe_allow_html=True)
    if st.button("Select", key=btn_key, use_container_width=True):
        st.session_state.mode = mode_val
        st.rerun()

def upload_section(key_suffix=""):
    f = st.file_uploader("📎 Upload file", type=["pdf","docx","xlsx","txt"], key=f"up{key_suffix}")
    if st.button("🚀 Process", use_container_width=True, key=f"proc{key_suffix}"):
        if not f: st.error("Upload a file first!")
        else:
            with st.spinner("Reading..."): load_doc(f)

# sidebar
with st.sidebar:
    st.markdown("<div style='text-align:center;padding:1rem 0'><div style='font-size:2rem'>🤖</div><h2 style='background:linear-gradient(90deg,#a78bfa,#60a5fa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0.3rem 0;font-size:1.1rem'>AI Assistant</h2></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Mode**")
    c1, c2 = st.columns(2)
    with c1: mode_card("Document", "📄", "doc", "doc", "d_sb")
    with c2: mode_card("Internet", "🌐", "web", "web", "w_sb")
    st.markdown("---")

    if st.session_state.mode == "doc":
        if not st.session_state.loaded:
            upload_section("_sb")
        else:
            st.markdown(f'<div style="background:rgba(52,211,153,0.1);border:1px solid rgba(52,211,153,0.3);border-radius:8px;padding:0.6rem"><span style="color:#34d399">●</span> <b>{st.session_state.fname}</b></div>', unsafe_allow_html=True)
            if st.button("Change file", use_container_width=True):
                st.session_state.loaded = False
                st.session_state.doc_txt = ""
                st.session_state.vs = None
                st.session_state.msgs = []
                st.rerun()
    else:
        st.markdown('<div style="background:rgba(102,126,234,0.1);border:1px solid rgba(102,126,234,0.3);border-radius:10px;padding:0.8rem;text-align:center"><div>🌐</div><b>Web Search On</b><div style="font-size:0.75rem;color:rgba(255,255,255,0.5)">Ask me anything</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.msgs = []
        st.rerun()

# main
st.markdown("<h1 style='background:linear-gradient(90deg,#a78bfa,#60a5fa,#34d399);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:1.8rem;font-weight:800'>🤖 AI Assistant</h1>", unsafe_allow_html=True)

# mobile controls
st.markdown('<div class="mob">', unsafe_allow_html=True)
mc1, mc2 = st.columns(2)
with mc1: mode_card("Document", "📄", "doc", "doc", "d_mb")
with mc2: mode_card("Internet", "🌐", "web", "web", "w_mb")
st.markdown("---")
if st.session_state.mode == "doc":
    if not st.session_state.loaded:
        upload_section("_mb")
    else:
        st.markdown(f'<div style="background:rgba(52,211,153,0.1);border:1px solid rgba(52,211,153,0.3);border-radius:8px;padding:0.6rem"><span style="color:#34d399">●</span> <b>{st.session_state.fname}</b></div>', unsafe_allow_html=True)
        if st.button("Change file", use_container_width=True, key="chg_mb"):
            st.session_state.loaded = False
            st.session_state.msgs = []
            st.rerun()
if st.session_state.msgs:
    if st.button("🗑️ Clear", use_container_width=True, key="clr_mb"):
        st.session_state.msgs = []
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")

# status
if st.session_state.mode == "doc":
    if st.session_state.loaded:
        st.markdown(f'<div style="background:rgba(52,211,153,0.1);border:1px solid rgba(52,211,153,0.3);border-radius:8px;padding:0.6rem;margin-bottom:0.5rem"><span style="color:#34d399">●</span> Chatting with <b>{st.session_state.fname}</b></div>', unsafe_allow_html=True)
    else:
        st.info("👈 Upload a document to start")
else:
    st.markdown('<div style="background:rgba(102,126,234,0.1);border:1px solid rgba(102,126,234,0.3);border-radius:8px;padding:0.6rem;text-align:center;margin-bottom:0.5rem">🌐 <b>Web Search</b> — ask me anything</div>', unsafe_allow_html=True)

for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        st.write(m["content"])

tip = "Ask about your document..." if st.session_state.mode == "doc" else "What do you want to know?"
if p := st.chat_input(tip):
    st.session_state.msgs.append({"role": "user", "content": p})
    with st.chat_message("user"): st.write(p)
    with st.chat_message("assistant"):
        with st.spinner("..."):
            if st.session_state.mode == "web":
                results = search(p)
                sys = f"""You have live web access. Search results:\n{results}\n
Look at conversation history for context on vague terms like 'before year'.
Answer directly. Never mention knowledge cutoff. Be natural."""
            elif st.session_state.doc_txt:
                sys = f"""Answer from this document:\n{st.session_state.doc_txt}\n
First person, confident, natural. Use real details. No bullet lists unless necessary."""
            elif st.session_state.vs:
                ctx = "\n\n".join(d.page_content for d in st.session_state.vs.similarity_search(p, k=6))
                sys = f"Answer from this content:\n{ctx}\nBe natural and direct."
            else:
                sys = "Helpful friendly assistant. Remember the conversation."
            ans = ask(sys)
            st.write(ans)
            st.session_state.msgs.append({"role": "assistant", "content": ans})
