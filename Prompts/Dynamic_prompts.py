from langchain_groq import ChatGroq
from  dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate

load_dotenv()

model= ChatGroq(model="deepseek-r1-distill-llama-70b")

st.header("Itinerary")
place_to_visit=st.text_input("Enter a Place where you want an itinerary")
place_from=st.text_input("Enter from where you have to plan an itinerary")
No_of_days=st.number_input("Enter no. of days",min_value=1,step=1,format="%d")
template=PromptTemplate(
    template="""
    Plan a concise itinerary for a traveler starting from {place_from} to visit {place_to_visit} over {No_of_days} days.
    Include key attractions, local experiences, and a brief recommendation for each day.
    Keep the output short, actionable, and ensure pricing details are included. And make it shorter to 1000 token and finally give itienery  in form of   table . 
    give incomplete message if data is missing.
    """,
    input_variables=["place_to_visit", "place_from", "No_of_days"]

)
prompt= template.invoke({
    'place_to_visit': place_to_visit,
    'place_from': place_from,
    'No_of_days': No_of_days
})
if st.button("confirm"):
    result=model.invoke(prompt)
    st.write(result.content)