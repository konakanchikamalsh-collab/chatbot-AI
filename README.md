# 📄 AI Document Chatbot

Upload any PDF and chat with it using AI. Built with LangChain, OpenAI, and Streamlit.

## 🚀 Features
- Upload any PDF document
- Ask questions in natural language
- AI answers based on document content only (RAG)
- Remembers conversation history
- Clean chat UI

## 🛠️ Tech Stack
- Python
- LangChain (RAG pipeline)
- OpenAI GPT-3.5
- ChromaDB (vector store)
- Streamlit (UI)

## ⚙️ Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/ai-document-chatbot
cd ai-document-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your OpenAI API key
Create a `.env` file:
```
OPENAI_API_KEY=your-key-here
```
Or just enter it in the sidebar when running the app.

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Open browser
Go to `http://localhost:8501`

## 📌 How to Use
1. Enter your OpenAI API key in the sidebar
2. Upload a PDF file
3. Click **Process Document**
4. Start asking questions!

## 🔄 Switching to Azure OpenAI
Replace `ChatOpenAI` and `OpenAIEmbeddings` with `AzureChatOpenAI` and `AzureOpenAIEmbeddings` from `langchain-openai`, and add your Azure credentials.
