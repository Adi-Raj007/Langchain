from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st


load_dotenv()

model= ChatGroq(model="deepseek-r1-distill-llama-70b")
st.header("itinerary")
user_input= st.text_input("Enter your prompt")
if st.button("confirm"):
    result=model.invoke(user_input)
    st.write(result.content)


