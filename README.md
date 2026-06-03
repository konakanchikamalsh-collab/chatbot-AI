AI Document Chatbot
Ever wished you could just talk to a PDF instead of reading through 50 pages? That's exactly what this does.
Upload any PDF, ask questions in plain English, and get answers instantly — no scrolling, no searching, no headaches.
What it does
You upload a document, it reads it, and you can ask anything about it. Built this because I was tired of digging through long documents manually.
Works great for:

Legal contracts
Research papers
Employee handbooks
Any long document you don't want to read fully

Built with
Python, OpenAI GPT-3.5, LangChain, ChromaDB, and Streamlit.
Running it locally
You'll need Python and an OpenAI API key to get started.
**git clone https://github.com/YOUR_USERNAME/ai-document-chatbot
cd ai-document-chatbot
pip install -r requirements.txt
streamlit run app.py**

When the app opens, paste your OpenAI API key in the sidebar, upload a PDF, hit Process Document, and start chatting.
Want to use Azure instead of OpenAI?
Swap ChatOpenAI and OpenAIEmbeddings with AzureChatOpenAI and AzureOpenAIEmbeddings from the langchain-openai package and add your Azure credentials.
