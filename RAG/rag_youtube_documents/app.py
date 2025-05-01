from streamlit import exception

from .frontend import chat_with_backend
from .rag_utils import extract_video_id, load_prompt_template, embedding_models, get_or_create_vector_store,build_chain, chat_chain
from fastapi import FastAPI,HTTPException
from dotenv import load_dotenv
from .schema import VideoRequest,VideoResponse,ChatRequest,ChatResponse

load_dotenv()


app = FastAPI(
    title="YouTube RAG Chat",
    description="Ask any question about a YouTube video using RAG",
    version="1.0.0"
)
@app.post("/ask", response_model=ChatResponse, summary="Chat with a YouTube video")
async def ask(request: ChatRequest):
    try:
        try:
            video_id = extract_video_id(request.youtube_url)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid YouTube URL: {e}")

        try:
            embedding = embedding_models()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Embedding model error: {e}")

        try:
            prompt = load_prompt_template()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load prompt: {e}")

        try:
            vector_store = get_or_create_vector_store(video_id, embedding)
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Vector store error: {e}")

        try:
            retriever = vector_store.as_retriever(search_kwargs={"k": 2})
            chain = chat_chain(retriever, prompt, chat_history=request.chat_history)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create chat chain: {e}")

        try:
            response = chain.invoke(request.message)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model invocation error: {e}")

        return {"response": response}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

@app.post("/summarize", response_model=VideoResponse, summary="Return summary of the video")
async def summarize(request: VideoRequest):
    try:
        try:
            video_id = extract_video_id(request.youtube_url)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid YouTube URL: {e}")

        try:
            embedding = embedding_models()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load embedding model: {e}")

        try:
            prompt = load_prompt_template()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prompt loading failed: {e}")

        try:
            vector_store = get_or_create_vector_store(video_id, embedding)
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Vector store error: {e}")

        try:
            retriever = vector_store.as_retriever(search_kwargs={"k": 2})
            chain = build_chain(retriever, prompt)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create chain: {e}")

        try:
            result = chain.invoke("summarize the video")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model invocation failed: {e}")

        return {"video_response": result}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")















