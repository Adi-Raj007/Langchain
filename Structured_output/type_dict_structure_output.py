from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict,Optional,Literal,Annotated
load_dotenv()
model=ChatGroq(model="llama3-70b-8192")
#This is simple typedict
"""class Itinerary (TypedDict):
    Name: str
    Destination: str
    Budget: int
    Days: int
structure_model=model.with_structured_output(Itinerary)
result=structure_model.invoke(" I have to travel Delhi for 5 days in a budget of 5000 plan a Itinerary for this as my name is Aditya")
print(result)"""

class Itinerary(TypedDict):
    Name: Annotated[Optional[str],"The name of the author or the person who is asking for the itinerary and it is optional as if the name is not there so skip that"]
    Destination: Annotated[list[str],"The place where to visit or explore"]
    Budget: Annotated[Optional[int],"This is the bugdet of the trip"]
    Days: Annotated[Optional[int],"Number of Days of the itinerary "]

structure_model=model.with_structured_output(Itinerary)
result=structure_model.invoke(""""Hi , How are you , I have a plan to travel new country like India ,Japan etc give me a best itinerary for these in my budget so that it does  not cost much and i want to go for a week or two so plan wisely""")
print(result)
