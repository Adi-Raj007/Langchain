# main.py
from dataclasses import field
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import load_prompt
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import os
import re

load_dotenv()
MODEL_PATH = "/home/aditya/Desktop/PRO/model/jina-embeddings-v2-base-en"
PROMPT_PATH = "/home/aditya/Desktop/linux/Data science/Langchain/RAG/rag_youtube_documents/template_2.json"
PERSIST_DIR = "./vectorstores"

def embedding_model():
    embedding_model = HuggingFaceEmbeddings(
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

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text


def build_chain(retriever, prompt_template):
    llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0.2)

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    parser = StrOutputParser()

    return parallel_chain | prompt_template | llm | parser