import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from duckduckgo_search import DDGS
import docx
import openpyxl

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="centered")

api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Mobile friendly base */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        max-width: 100% !important;
    }

    /* Sidebar mobile */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05) !important;
        border-right: 1px solid rgba(255,255,255,0.1);
        min-width: 250px !important;
    }

    /* File uploader */
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

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        width: 100% !important;
        padding: 0.6rem !important;
        font-size: 0.95rem !important;
    }

    /* Chat messages mobile friendly */
    [data-testid="stChatMessage"] {
        padding: 0.5rem !important;
        margin: 0.3rem 0 !important;
        border-radius: 12px !important;
        max-width: 100% !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        border-radius: 20px !important;
    }

    /* Radio buttons */
    .stRadio label {
        color: white !important;
        font-size: 0.9rem !important;
    }

    /* Mode selector */
    .mode-selector {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Text */
    h1, h2, h3, p, span, label, div { color: white !important; }
    hr { border-color: rgba(255,255,255,0.1) !important; }

    /* Mobile adjustments */
    @media (max-width: 768px) {
        .stApp { padding: 0.5rem !important; }
        h1 { font-size: 1.5rem !important; }
        [data-testid="stChatMessage"] { font-size: 0.9rem !important; }
    }
</style>
""", unsafe_allow_html=True)

def extract_text(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text
    elif name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        text = ""
        for para in doc.paragraphs:
            if para.text:
                text += para.text + "\n"
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        text += cell.text.strip() + "\n"
        return text
    elif name.endswith(".xlsx"):
        wb = openpyxl.load_workbook(uploaded_file)
        text = ""
        for sheet in wb.sheetnames:
            ws = wb[sheet]
            text += f"Sheet: {sheet}\n"
            for row in ws.iter_rows(values_only=True):
                row_text = " | ".join([str(cell) for cell in row if cell is not None])
                if row_text:
                    text += row_text + "\n"
        return text
    elif name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    return ""

def search_internet(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            context = ""
            for r in results:
                context += f"Source: {r['href']}\n{r['title']}\n{r['body']}\n\n"
            return context
    except Exception as e:
        return f"Search failed: {str(e)}"

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False
if "doc_name" not in st.session_state:
    st.session_state.doc_name = ""
if "mode" not in st.session_state:
    st.session_state.mode = "Document"

# Sidebar
with st.sidebar:
    st.markdown("## 🤖 AI Assistant")
    st.markdown("---")

    mode = st.radio(
        "Choose Mode",
        ["📄 Document", "🌐 Internet Search"],
        index=0
    )
    st.session_state.mode = mode
    st.markdown("---")

    if "📄 Document" in mode:
        uploaded_file = st.file_uploader(
            "📎 Upload Document",
            type=["pdf", "docx", "xlsx", "txt"]
        )

        if st.button("🚀 Process Document", use_container_width=True):
            if not uploaded_file:
                st.error("❌ Upload a file first")
            else:
                with st.spinner("🔍 Reading document..."):
                    raw_text = extract_text(uploaded_file)
                    if not raw_text.strip():
                        st.error("❌ Could not extract text.")
                    else:
                        word_count = len(raw_text.split())
                        if word_count <= 4000:
                            # Small doc — send full text
                            st.session_state.doc_text = raw_text
                            st.session_state.vectorstore = None
                        else:
                            # Large doc — use chunks
                            splitter = RecursiveCharacterTextSplitter(
                                chunk_size=800,
                                chunk_overlap=150,
                                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
                            )
                            chunks = splitter.split_text(raw_text)
                            embeddings = HuggingFaceEmbeddings(
                                model_name="sentence-transformers/all-MiniLM-L6-v2"
                            )
                            st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)
                            st.session_state.doc_text = ""

                        st.session_state.doc_processed = True
                        st.session_state.doc_name = uploaded_file.name
                        st.success(f"✅ Ready! ({word_count} words)")

        st.markdown("---")
        if st.session_state.doc_processed:
            st.markdown(f"📄 **{st.session_state.doc_name}**")
            st.markdown('<span style="color:#34d399">● Document loaded</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#fbbf24">● No document loaded</span>', unsafe_allow_html=True)

    else:
        st.info("🌐 Ask me anything and I'll search the internet for you!")

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Main content
st.title("🤖 AI Assistant")

if "📄 Document" in mode:
    if not st.session_state.doc_processed:
        st.markdown("Upload any document and chat with it")
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.info("📄 **PDF & Word**\n\nResumes, reports, contracts")
        with col2:
            st.info("📊 **Excel & Text**\n\nSpreadsheets, data files")
        st.warning("👈 Upload your document to get started")
    else:
        st.markdown(f"Chatting with: **{st.session_state.doc_name}**")
else:
    st.markdown("Ask me anything — I'll search the internet!")

st.markdown("---")

# Chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
placeholder = "Ask about your document..." if "📄 Document" in mode else "Ask me anything..."
if prompt := st.chat_input(placeholder):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            if "🌐 Internet" in mode:
                # Internet search mode
                search_results = search_internet(prompt)
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": f"""You are a helpful AI assistant that answers questions using internet search results.

Here are the search results for the question:
{search_results}

Rules:
- Answer clearly and accurately based on search results
- Start with a direct answer
- Add relevant details in a natural conversational way
- Mention sources when relevant
- Keep it concise but complete
- Never make up information not in the search results"""},
                        {"role": "user", "content": prompt}
                    ]
                )

            elif st.session_state.doc_text:
                # Small doc — full text mode
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": f"""You are an expert interview coach helping a candidate answer questions based on their document.

Here is the complete document:
{st.session_state.doc_text}

Rules:
- Answer in first person as the candidate speaking
- Sound confident natural and conversational
- Tell a story not a list
- Use specific details from the document
- Never sound robotic or generic
- Never start with Certainly or Of course"""},
                        {"role": "user", "content": prompt}
                    ]
                )

            elif st.session_state.vectorstore:
                # Large doc — chunks mode
                docs = st.session_state.vectorstore.similarity_search(prompt, k=6)
                context = "\n\n".join([doc.page_content for doc in docs])
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": f"""You are a helpful AI assistant answering questions based on this document:

{context}

Rules:
- Answer clearly and accurately
- Be conversational and natural
- Use specific details from the document
- Never make up information"""},
                        {"role": "user", "content": prompt}
                    ]
                )

            else:
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )

            answer = response.choices[0].message.content
            st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
