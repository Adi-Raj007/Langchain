from pydantic import BaseModel ,Field
from typing import Optional
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

class Itinerary(BaseModel):
    Name:Optional[str] = Field(default=None,description="The name of the author or the person who is asking for the itinerary and it is optional as if the name is not there so skip that")
    Destination: list[str]=Field(description="The place where to visit or explore")
    Days: Optional[int]=Field(default=None,description="Number of Days of the itinerary")
    Budget: Optional[int]=Field(default=None,description="The budget of the trip ")

model=ChatGroq(model="llama3-70b-8192")
structure_model=model.with_structured_output(Itinerary)
result=structure_model.invoke("""Hi , How are you , I have a plan to travel new country like India ,Japan etc give me a best itinerary for these in my budget so that it does  not cost much and i want to go for a week or two so plan wisely""")
print(result)

"""These are the sample Prompt and their Output so You can adjust your all input and output by adjusting them """