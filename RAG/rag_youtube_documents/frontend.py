import streamlit as st
import requests

# Backend URL
BASE_URL = "http://127.0.0.1:8000"

# Page config
st.set_page_config(page_title="YouTube RAG Chatbot", layout="centered")
st.title("🎥 YouTube RAG Chatbot")

# Sidebar for selecting mode
st.sidebar.title("🛠️ Options")
mode = st.sidebar.radio("Select Action", ["Chat with Video", "Summarize Video"])

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_response" not in st.session_state:
    st.session_state.last_response = ""

# Function to call /ask endpoint
def chat_with_backend(youtube_url, message, history):
    try:
        with st.spinner("Generating response..."):
            res = requests.post(f"{BASE_URL}/ask", json={
                "youtube_url": youtube_url,
                "message": message,
                "chat_history": history
            })
            res.raise_for_status()
            return res.json().get("response", "No response received.")
    except requests.exceptions.HTTPError as http_err:
        st.error(f"❌ HTTP error: {http_err}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Unable to connect to backend. Is FastAPI running?")
    except Exception as e:
        st.error(f"⚠️ Unexpected error: {e}")
    return None

# Function to call /summarize endpoint
def summarize_backend(youtube_url):
    try:
        with st.spinner("Summarizing video..."):
            res = requests.post(f"{BASE_URL}/summarize", json={"youtube_url": youtube_url})
            res.raise_for_status()
            return res.json().get("video_response", "")
    except requests.exceptions.HTTPError as http_err:
        st.error(f"❌ HTTP error: {http_err}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Unable to connect to backend. Is FastAPI running?")
    except Exception as e:
        st.error(f"⚠️ Unexpected error: {e}")
    return None

# Chat mode
if mode == "Chat with Video":
    st.subheader("💬 Ask Questions About the YouTube Video")

    youtube_url = st.text_input("Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
    user_input = st.text_input("Your Question", placeholder="e.g., What is this video about?")

    if st.button("Ask"):
        if not youtube_url or not user_input:
            st.warning("Please enter both the YouTube URL and your question.")
        else:
            history = [(h["role"], h["content"]) for h in st.session_state.chat_history]
            response = chat_with_backend(youtube_url, user_input, history)
            if response:
                st.session_state.chat_history.append({"role": "User", "content": user_input})
                st.session_state.chat_history.append({"role": "Bot", "content": response})
                st.session_state.last_response = response
                st.rerun()

    # Latest response
    if st.session_state.last_response:
        with st.expander("🤖 Latest Bot Response", expanded=True):
            st.write(st.session_state.last_response)

    # Full chat history
    with st.expander("📜 Full Chat History"):
        for msg in st.session_state.chat_history:
            st.markdown(f"**{msg['role']}**: {msg['content']}")

# Summarize mode
elif mode == "Summarize Video":
    st.subheader("📽️ Summarize the YouTube Video")

    youtube_url = st.text_input("Enter YouTube Video URL to summarize", placeholder="https://www.youtube.com/watch?v=...")

    if st.button("Summarize"):
        if not youtube_url:
            st.warning("Please enter the YouTube URL.")
        else:
            summary = summarize_backend(youtube_url)
            if summary:
                st.success("✅ Summary:")
                st.write(summary)
