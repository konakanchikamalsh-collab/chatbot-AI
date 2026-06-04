import streamlit as st
from PyPDF2 import PdfReader
from groq import Groq
from tavily import TavilyClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import docx, openpyxl

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")
g = Groq(api_key=st.secrets["GROQ_API_KEY"])
tv = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])

st.markdown("""<style>
#MainMenu,footer{visibility:hidden}
[data-testid="collapsedControl"]{display:flex!important}
.stApp{background:linear-gradient(135deg,#0f0c29,#302b63,#24243e)}
[data-testid="stSidebar"]{background:rgba(255,255,255,.05)!important;border-right:1px solid rgba(255,255,255,.1)!important}
.stButton button{background:linear-gradient(135deg,#667eea,#764ba2)!important;color:#fff!important;border:none!important;border-radius:10px!important;font-weight:600!important;width:100%!important}
[data-testid="stFileUploader"]{background:rgba(255,255,255,.05)!important;border:2px dashed rgba(255,255,255,.3)!important;border-radius:10px!important}
[data-testid="stFileUploader"] *{color:#fff!important;background:transparent!important}
[data-testid="stFileUploaderDropzone"]{background:rgba(255,255,255,.05)!important}
[data-testid="stFileUploaderDropzone"] button{background:rgba(102,126,234,.3)!important;color:#fff!important;border:1px solid rgba(102,126,234,.5)!important;border-radius:8px!important}
.mc{background:rgba(255,255,255,.05);border:2px solid rgba(255,255,255,.1);border-radius:12px;padding:.9rem;text-align:center;margin-bottom:6px}
.mc.on{background:linear-gradient(135deg,rgba(102,126,234,.35),rgba(118,75,162,.35));border-color:#667eea;box-shadow:0 0 16px rgba(102,126,234,.25)}
.ft{display:inline-block;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.15);border-radius:8px;padding:.25rem .6rem;font-size:.75rem;margin:.15rem}
h1,h2,h3,p,span,label,div{color:#fff!important}
hr{border-color:rgba(255,255,255,.1)!important}
</style>""", unsafe_allow_html=True)

ss = st.session_state
for k,v in dict(msgs=[],doc="",vs=None,ready=False,fname="",mode="doc").items():
    if k not in ss: ss[k] = v

def read_file(f):
    out, name = "", f.name.lower()
    if ".pdf" in name:
        for pg in PdfReader(f).pages:
            t = pg.extract_text()
            if t: out += t+"\n"
    elif ".docx" in name:
        d = docx.Document(f)
        for p in d.paragraphs:
            if p.text: out += p.text+"\n"
        for tb in d.tables:
            for r in tb.rows:
                for c in r.cells:
                    if c.text.strip(): out += c.text.strip()+"\n"
    elif ".xlsx" in name:
        wb = openpyxl.load_workbook(f)
        for sh in wb.sheetnames:
            ws = wb[sh]
            out += f"[{sh}]\n"
            for row in ws.iter_rows(values_only=True):
                line = " | ".join(str(c) for c in row if c is not None)
                if line: out += line+"\n"
    elif ".txt" in name:
        out = f.read().decode()
    return out

def do_search(q):
    try:
        r = tv.search(query=q, max_results=5, search_depth="advanced")
        return "\n\n".join(f"{x['title']}\n{x['content']}\n{x['url']}" for x in r.get("results",[]))
    except: return "search failed"

def llm(prompt):
    hist = [{"role":m["role"],"content":m["content"]} for m in ss.msgs[-10:]]
    return g.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role":"system","content":prompt}]+hist
    ).choices[0].message.content

def ingest(f):
    raw = read_file(f)
    if not raw.strip(): st.error("cant read this"); return
    n = len(raw.split())
    if n <= 4000:
        ss.doc = raw; ss.vs = None
    else:
        chunks = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=150).split_text(raw)
        ss.vs = FAISS.from_texts(chunks, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
        ss.doc = ""
    ss.ready = True
    ss.fname = f.name
    st.success(f"loaded {n} words"); st.rerun()

# sidebar
with st.sidebar:
    st.markdown("<div style='text-align:center;padding:.8rem 0'><div style='font-size:1.8rem'>🤖</div><h2 style='background:linear-gradient(90deg,#a78bfa,#60a5fa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:1.1rem;margin:.2rem 0'>AI Assistant</h2></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Mode**")

    # mode cards as buttons — full width clean
    doc_on = "on" if ss.mode == "doc" else ""
    web_on = "on" if ss.mode == "web" else ""

    st.markdown(f'<div class="mc {doc_on}"><div style="font-size:1.4rem">📄</div><div style="font-weight:700;font-size:.85rem">Document</div><div style="font-size:.65rem;opacity:.5">PDF · Word · Excel · TXT</div></div>', unsafe_allow_html=True)
    if st.button("Switch to Document", key="doc_btn", use_container_width=True):
        ss.mode = "doc"; st.rerun()

    st.markdown(f'<div class="mc {web_on}"><div style="font-size:1.4rem">🌐</div><div style="font-weight:700;font-size:.85rem">Internet</div><div style="font-size:.65rem;opacity:.5">Live web search</div></div>', unsafe_allow_html=True)
    if st.button("Switch to Internet", key="web_btn", use_container_width=True):
        ss.mode = "web"; st.rerun()

    st.markdown("---")

    if ss.mode == "doc":
        if not ss.ready:
            st.markdown('<span class="ft">📄 PDF</span><span class="ft">📝 Word</span><span class="ft">📊 Excel</span><span class="ft">📃 TXT</span>', unsafe_allow_html=True)
            st.markdown("")
            f = st.file_uploader("", type=["pdf","docx","xlsx","txt"], label_visibility="collapsed")
            if st.button("🚀 Process Document", use_container_width=True):
                if not f: st.error("upload a file first")
                else:
                    with st.spinner("reading..."): ingest(f)
        else:
            st.markdown(f'<div style="background:rgba(52,211,153,.1);border:1px solid rgba(52,211,153,.3);border-radius:8px;padding:.6rem"><span style="color:#34d399">●</span> <b>{ss.fname}</b></div>', unsafe_allow_html=True)
            st.markdown("")
            if st.button("Change File", use_container_width=True):
                ss.ready=False; ss.doc=""; ss.vs=None; ss.msgs=[]; st.rerun()
    else:
        st.markdown('<div style="background:rgba(102,126,234,.1);border:1px solid rgba(102,126,234,.3);border-radius:10px;padding:.8rem;text-align:center"><div style="font-size:1.2rem">🌐</div><b>Web Search On</b><br><span style="font-size:.75rem;opacity:.6">Ask me anything</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        ss.msgs=[]; st.rerun()

# main
st.markdown("<h1 style='background:linear-gradient(90deg,#a78bfa,#60a5fa,#34d399);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:1.8rem;font-weight:800;margin:0'>🤖 AI Assistant</h1><p style='color:rgba(255,255,255,.4);font-size:.85rem;margin:.2rem 0 0'>Chat with docs or search the web</p>", unsafe_allow_html=True)
st.markdown("---")

if ss.mode == "doc":
    if ss.ready:
        st.markdown(f'<div style="background:rgba(52,211,153,.1);border:1px solid rgba(52,211,153,.3);border-radius:8px;padding:.5rem 1rem;margin-bottom:.5rem"><span style="color:#34d399">●</span> Chatting with <b>{ss.fname}</b></div>', unsafe_allow_html=True)
    else:
        st.info("👈 Upload a document from the sidebar to start")
else:
    st.markdown('<div style="background:rgba(102,126,234,.1);border:1px solid rgba(102,126,234,.3);border-radius:8px;padding:.5rem 1rem;margin-bottom:.5rem;text-align:center">🌐 <b>Web Search</b> — ask me anything</div>', unsafe_allow_html=True)

for m in ss.msgs:
    with st.chat_message(m["role"]): st.write(m["content"])

tip = "Ask about your document..." if ss.mode=="doc" else "What do you want to know?"
if p := st.chat_input(tip):
    ss.msgs.append({"role":"user","content":p})
    with st.chat_message("user"): st.write(p)
    with st.chat_message("assistant"):
        with st.spinner("..."):
            if ss.mode == "web":
                res = do_search(p)
                sys = f"Live web results:\n{res}\n\nUse chat history for context. Answer directly. Never say knowledge cutoff. Be natural."
            elif ss.doc:
                sys = f"Answer from this document:\n{ss.doc}\n\nFirst person, confident, natural. Real details."
            elif ss.vs:
                ctx = "\n\n".join(d.page_content for d in ss.vs.similarity_search(p,k=6))
                sys = f"Answer from this:\n{ctx}\n\nNatural and direct."
            else:
                sys = "Helpful assistant. Remember the chat."
            ans = llm(sys)
            st.write(ans)
            ss.msgs.append({"role":"assistant","content":ans})
