from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict,Optional,Literal
load_dotenv()
model=ChatGroq(model="llama3-70b-8192")
class Itinerary (TypedDict):
    Name: str
    Destination: str
    Budget: int
    Days: int
structure_model=model.with_structured_output(Itinerary)
result=structure_model.invoke(" I have to travel Delhi for 5 days in a budget of 5000 plan a Itinerary for this as my name is Aditya")
print(result)
