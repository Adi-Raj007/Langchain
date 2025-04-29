import streamlit as st
import requests

# URL of your FastAPI backend
BASE_URL = "http://127.0.0.1:8000"


# Helper function to call the FastAPI summarize endpoint
def summarize_video(youtube_url: str):
    response = requests.post(f"{BASE_URL}/summarize", json={"youtube_url": youtube_url})
    return response.json()


# Helper function to call the FastAPI chat endpoint
def chat_with_video(youtube_url: str, message: str, chat_history: list):
    response = requests.post(f"{BASE_URL}/chat", json={
        "youtube_url": youtube_url,
        "message": message,
        "chat_history": chat_history
    })
    return response.json()


# Streamlit UI
st.title("YouTube Video Summarizer and Chatbot")

# Sidebar for selecting options
option = st.sidebar.selectbox("Select an action", ["Summarize Video", "Chat with Video"])

if option == "Summarize Video":
    st.subheader("Summarize the YouTube Video")

    # YouTube URL input
    youtube_url = st.text_input("Enter YouTube Video URL", "")

    if st.button("Summarize"):
        if youtube_url:
            # Call the summarize_video function
            result = summarize_video(youtube_url)
            st.write("Summary of the video:")
            st.write(result.get("video_response", "Could not fetch summary."))
        else:
            st.error("Please provide a valid YouTube URL.")

elif option == "Chat with Video":
    st.subheader("Chat with the YouTube Video")

    # YouTube URL input
    youtube_url = st.text_input("Enter YouTube Video URL", "")

    if youtube_url:
        # Initialize chat history as an empty list
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # User message input
        user_message = st.text_input("Ask something about the video:")

        if st.button("Send Message"):
            if user_message:
                # Call the chat_with_video function
                result = chat_with_video(youtube_url, user_message, st.session_state.chat_history)

                # Display the response from the chatbot
                st.write("Bot's Response:")
                st.write(result.get("response", "Could not get a response."))

                # Update chat history
                st.session_state.chat_history.append(("User", user_message))
                st.session_state.chat_history.append(("Bot", result.get("response", "No response.")))
            else:
                st.error("Please enter a message to chat.")

        # Chat history display toggle
        show_history = st.checkbox("Show Chat History", value=False)

        if show_history and st.session_state.chat_history:
            st.write("Chat History:")
            for msg in st.session_state.chat_history:
                st.write(f"{msg[0]}: {msg[1]}")
