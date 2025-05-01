
---

# 🎥 YouTube RAG Chatbot

A **multilingual chatbot** that lets you ask questions or get summaries from any YouTube video using its transcript. Supports **Hindi** and **English**! Built with **FastAPI**, **LangChain**, **Groq’s LLaMA models**, and a modern **Streamlit UI**.

---

## 🚀 Features

- ✅ **Ask questions** about any YouTube video  
- ✅ **Summarize** video content  
- ✅ **Supports Hindi and English** transcripts  
- ✅ **RAG pipeline** with LangChain and local embeddings  
- ✅ **Groq-powered LLaMA-4-Scout** for fast, smart answers  
- ✅ **Full chat history** in the frontend  
- ✅ **Elegant Streamlit UI**  
- ✅ **Robust error handling** at every backend step  

---

## 🛠️ Tech Stack

- **Backend:** FastAPI  
- **Frontend:** Streamlit  
- **RAG Pipeline:** LangChain  
- **Embedding Model:** [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)  
- **LLM:** Groq’s LLaMA-4-Scout  
- **Vector DB:** ChromaDB  
- **Transcripts:** youtube-transcript-api  

---

## 📁 Project Structure

```text
rag_youtube_documents/
├── app.py                # FastAPI backend
├── schema.py             # Pydantic models
├── rag_utils.py          # Core RAG utilities
├── frontend.py           # Streamlit frontend
├── template_2.json       # Prompt template for questions
├── model/                # Local embedding model
└── vectorstores/         # Persisted Chroma vector store
```

---

## ⚙️ Setup Instructions

### 1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/youtube-rag-chatbot.git
cd youtube-rag-chatbot
```

---

### 2. **Create & Activate a Virtual Environment**

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

---

### 4. **Download & Save the Embedding Model Locally**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
model.save('./model/model_1')
```
> **Note:** The saved path must match `MODEL_PATH = "./model/model_1/paraphrase-multilingual-MiniLM-L12-v2"` in your code.

---

## ▶️ Running the Application

### **Start FastAPI Backend**

```bash
uvicorn app:app --reload
```

### **Start Streamlit Frontend**

```bash
streamlit run frontend.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔌 API Endpoints

### **POST `/summarize`**

Summarize a YouTube video.

**Request:**
```json
{
  "youtube_url": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```

**Response:**
```json
{
  "video_response": "The video explains the basics of PyTorch, including tensors and autograd."
}
```

---

### **POST `/ask`**

Ask any question about a YouTube video.

**Request:**
```json
{
  "youtube_url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "message": "What is the video about?",
  "chat_history": []
}
```

**Response:**
```json
{
  "response": "The video covers how PyTorch works with tensors and linear regression."
}
```

---

## 💬 Example Interactions

> 👤 **User:** "Summarize the video."  
> 🤖 **Bot:** "This video discusses PyTorch, its tensor operations, autograd, and a demo of linear regression."

> 👤 **User:** "What is autograd in PyTorch?"  
> 🤖 **Bot:** "Autograd is PyTorch’s automatic differentiation engine for neural network training."

---

## 🧯 Error Handling

- ❌ Invalid or missing YouTube URLs
- ❌ No transcript available (disabled by channel)
- ❌ Internet or API failure
- ❌ Model path issues
- ✅ User-friendly error display in the frontend

---

## 🌍 Multilingual Support

Thanks to the multilingual MiniLM model, this chatbot can answer questions from videos in Hindi, English, and many more languages.

---

## 📌 To-Do

- [ ] Deploy the app using Render/Fly.io  
- [ ] Add optional voice input in frontend  
- [ ] Dark/light theme toggle  
- [ ] Add support for subtitle files  

---

## 👨‍💻 Author

**Aditya** — Final Year Computer Science Engineering Student  
📧 subh7282@gmail.com  
[LinkedIn](#) | [GitHub](#)

---

## 📄 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

