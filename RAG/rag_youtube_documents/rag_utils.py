from dotenv import load_dotenv
from langchain_groq import ChatGroq
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import load_prompt
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from fastapi import HTTPException
import os
import re

# Load environment variables
load_dotenv()

# Constants
MODEL_PATH ="/home/aditya/Desktop/linux/Data science/Langchain/RAG/model/model_1"
PROMPT_PATH = "/home/aditya/Desktop/linux/Data science/Langchain/RAG/rag_youtube_documents/template_2.json"
PERSIST_DIR = "/home/aditya/Desktop/linux/Data science/Langchain/RAG/rag_youtube_documents/vectorstores"

# ✅ Fixed: Now returns the model
def embedding_models():
    return HuggingFaceEmbeddings(
        model_name=MODEL_PATH,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def extract_video_id(youtube_url: str) -> str:
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, youtube_url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    return match.group(1)

def load_prompt_template():
    return load_prompt(PROMPT_PATH)

# ✅ Fixed: Added proper function signature and argument usage
def get_or_create_vector_store(video_id: str, embedding_model):
    collection_name = f"transcript_{video_id}"
    vector_path = os.path.join(PERSIST_DIR, collection_name)

    if os.path.exists(vector_path):
        return Chroma(
            collection_name=collection_name,
            persist_directory=PERSIST_DIR,
            embedding_function=embedding_model
        )

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id=video_id, languages=["en","hi"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splitted_docs = text_splitter.create_documents([transcript])

        vector_store = Chroma.from_documents(
            documents=splitted_docs,
            embedding=embedding_model,
            collection_name=collection_name,
            persist_directory=PERSIST_DIR
        )
        return vector_store

    except TranscriptsDisabled:
        raise HTTPException(status_code=404, detail="No captions available for this video.")

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

def build_chain(retriever, prompt):
    llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.2)
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })
    parser = StrOutputParser()
    return parallel_chain | prompt | llm | parser

def chat_chain(retriever, prompt, chat_history=None):
    llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.2)

    messages = []
    if chat_history:
        for human, ai in chat_history:
            messages.append(HumanMessage(content=human))
            messages.append(AIMessage(content=ai))

    memory = ConversationBufferMemory(return_messages=True)
    memory.chat_memory.messages = messages  # ✅ fixed typo: was `message`, should be `messages`

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    parser = StrOutputParser()
    print(parallel_chain)
    return parallel_chain | prompt | llm | parser

