# main.py
from dataclasses import field
from .schema import VideoRequest, VideoResponse, ChatRequest, ChatResponse
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import load_prompt
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
import re

load_dotenv()

app = FastAPI()
chat_history = [{"role": "system", "content": "Hi, I am Aditya."}]

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="/home/aditya/Desktop/PRO/model/jina-embeddings-v2-base-en",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Folder where vector data is stored
persist_directory = "./vectorstores"

# Extract video ID from YouTube URL
def extract_video_id(youtube_url: str) -> str:
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, youtube_url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    return match.group(1)


# Load prompt template
prompt = load_prompt("/home/aditya/Desktop/linux/Data science/Langchain/RAG/rag_youtube_documents/template_2.json")

# Load or create vector store from YouTube transcript
def get_or_create_vector_store(video_id: str):
    collection_name = f"transcript_{video_id}"
    vector_path = os.path.join(persist_directory, collection_name)

    if os.path.exists(vector_path):
        return Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id=video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splitted_docs = text_splitter.create_documents([transcript])

        vector_store = Chroma.from_documents(
            documents=splitted_docs,
            embedding=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        return vector_store

    except TranscriptsDisabled:
        raise HTTPException(status_code=404, detail="No captions available for this video.")

# FastAPI route to summarize the YouTube video
@app.post("/summarize", response_model=VideoResponse)
async def summarize_video(request: VideoRequest):
    try:
        video_id = extract_video_id(request.youtube_url)
        vector_store = get_or_create_vector_store(video_id)

        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        def format_doc(retrieved_docs):
            return "\n\n".join(doc.page_content for doc in retrieved_docs)

        from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
        from langchain_core.output_parsers import StrOutputParser

        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_doc),
            'question': RunnablePassthrough()
        })

        llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
        parser = StrOutputParser()
        main_chain = parallel_chain | prompt | llm | parser

        response = main_chain.invoke("Summarize the video")
        return {"video_response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Continuous chat interface using ConversationalRetrievalChain
@app.post("/chat", response_model=ChatResponse)
async def chat_with_video(req: ChatRequest):
    try:
        # Extract video ID from YouTube URL
        video_id = extract_video_id(req.youtube_url)

        # Get the vector store (transcript data)
        vector_store = get_or_create_vector_store(video_id)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        # Format retrieved docs
        def format_doc(retrieved_docs):
            return "\n\n".join(doc.page_content for doc in retrieved_docs)

        # Retrieve relevant context
        context = retriever | RunnableLambda(format_doc)
        docs = context.invoke(req.message)

        # Append user message to memory
        chat_history.append({"role": "user", "content": req.message})

        # Compose full message for model including transcript context
        full_prompt = (
                f"You are a helpful assistant answering questions about a YouTube video.\n\n"
                f"Context:\n{docs}\n\n"
                f"Conversation so far:\n" +
                "\n".join(
                    [f"{m['role'].capitalize()}: {m['content']}" for m in chat_history if m['role'] != 'system']) +
                f"\n\nUser: {req.message}\nAssistant:"
        )

        # Run model
        model = ChatGroq(model="llama3-70b-8192")
        result = model.invoke(
            [{"role": "system", "content": "Answer based on the transcript context and conversation history."},
             {"role": "user", "content": full_prompt}])

        # Append assistant reply
        chat_history.append({"role": "assistant", "content": result.content})

        return {"response": result.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))