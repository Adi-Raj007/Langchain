from langchain_groq import ChatGroq
from  dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()

model= ChatGroq(model="deepseek-r1-distill-llama-70b")

st.header("Itinerary")
place_to_visit=st.text_input("Enter a Place where you want an itinerary")
place_from=st.text_input("Enter from where you have to plan an itinerary")
No_of_days=st.number_input("Enter no. of days",min_value=1,step=1,format="%d")
template=load_prompt('Prompts/template.json')
prompt= template.invoke({
    'place_to_visit': place_to_visit,
    'place_from': place_from,
    'No_of_days': No_of_days
})
if st.button("confirm"):
    result=model.invoke(prompt)
    st.write(result.content)