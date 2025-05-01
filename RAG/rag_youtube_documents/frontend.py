import streamlit as st
import requests

API_ENDPOINT = "http://localhost:8000/chat"  # Update if running on different host/port

st.set_page_config(page_title="YouTube Q&A Assistant", layout="centered")

st.title("📺 YouTube Video Assistant")
st.write("Ask questions about any YouTube video with subtitles!")

# Input fields
youtube_url = st.text_input("Enter YouTube Video URL:")
question = st.text_area("Ask your question:")

# Chat history storage in session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Submit button
if st.button("Ask"):
    if not youtube_url or not question:
        st.warning("Please provide both a YouTube URL and a question.")
    else:
        # Prepare request payload
        payload = {
            "youtube_url": youtube_url,
            "message": question,
            "chat_history": st.session_state.chat_history
        }

        try:
            response = requests.post(API_ENDPOINT, json=payload)
            response.raise_for_status()
            result = response.json()["response"]

            # Display and update history
            st.markdown(f"**You:** {question}")
            st.markdown(f"**AI:** {result}")
            st.session_state.chat_history.append((question, result))

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to get response: {e}")
        except KeyError:
            st.error("Unexpected response format from the backend.")

# Option to reset the chat
if st.button("Reset Chat"):
    st.session_state.chat_history = []
    st.success("Chat history cleared.")
