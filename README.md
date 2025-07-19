# PDF-Learning-Assistant
# 📚 AskMyNotes  

> **Chat with your PDF notes!**  
> Upload any PDF and ask questions. This AI-powered Streamlit app uses **LangChain, FAISS & OpenAI GPT** to find answers instantly.  

---

## 🚀 Features  

✅ Upload any PDF notes  
✅ AI finds relevant sections using **FAISS semantic search**  
✅ GPT-powered answers with context  
✅ Simple & clean **Streamlit UI**  
✅ Works like your personal **AI Tutor**  

---

## 🛠️ Tech Stack  

- **Python 3.10+**  
- [Streamlit](https://streamlit.io/) – UI  
- [LangChain](https://www.langchain.com/) – Document processing  
- [FAISS](https://github.com/facebookresearch/faiss) – Vector database  
- [OpenAI GPT](https://platform.openai.com/) – LLM for answering  

---

## 📦 Installation  

1️⃣ **Clone the repo**  
```bash
git clone https://github.com/<your-username>/AskMyNotes.git
cd AskMyNotes

python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

🖼️ How It Works

    Upload your PDF → It extracts all text

    Splits text into small chunks → Uses LangChain text splitter

    Creates FAISS vector embeddings → For semantic search

    Ask a question → Finds matching chunks

    GPT answers your query using the context

📝 Example

    You upload: “AI Basics.pdf”
    You ask: “What is supervised learning?”
    ✅ Bot answers: with relevant text from your PDF.


🔮 Future Improvements

    Support multiple PDFs

    Add conversation history

    Support for different LLMs (Claude, Llama 3)

    Deploy on Streamlit Cloud
🤝 Contributing

    PRs are welcome! Feel free to fork and improve this project.
📜 License

MIT License © 2025 Rathnayaka R.M.C.D
