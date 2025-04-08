from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Optional, List
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda

# Load environment variables
load_dotenv()

# Step 1: Input parsing model
class Itinerary(BaseModel):
    place_from: Optional[str] = Field(default=None, description="The place from where to start")
    place_to_visit: List[str] = Field(description="Places to visit")
    no_of_days: Optional[int] = Field(default=None, description="Trip duration in days")
    budget: Optional[int] = Field(default=None, description="Trip budget in INR")

# Step 2: Final output model
class DayPlan(BaseModel):
    day: int
    activity: str
    price_estimate: str
    tip: str

class ItineraryResponse(BaseModel):
    place_from: str
    place_to_visit: str
    no_of_days: int
    days: List[DayPlan]
    budget: str

# Step 3: Initialize model
model = ChatGroq(model="llama3-70b-8192")

# Step 4: Structured model to extract fields from raw user message
structure_model = model.with_structured_output(Itinerary)

# Step 5: User input message
user_input = input( "Plan your trip & give prompt")

# Step 6: Extract structured trip intent
structured_data = structure_model.invoke(user_input)

# Step 7: Prompt Template
itinerary_prompt = PromptTemplate(
    template="""
You are a professional travel planner.

Create a detailed travel itinerary in JSON format for the following request:
- Starting location: {place_from}
- Destination(s): {place_to_visit}
- Trip duration: {no_of_days} days
- Total budget: ₹{budget}

Respond in this structured JSON format:
{{
    "place_from": "...",
    "place_to_visit": "...",
    "no_of_days": ...,
    "budget": "...",
    "days": [
        {{
            "day": 1,
            "activity": "...",
            "price_estimate": "...",
            "tip": "..."
        }},
        ...
    ]
}}

**Constraints**:
- Stick to the total budget of ₹{budget}
- Include local travel, food spots, entry tickets
- Add one tip or suggestion per day to enhance experience
- Be practical and cost-aware
""",
    input_variables=["place_to_visit", "place_from", "no_of_days", "budget"]
)

# Step 8: Chain: Format prompt → LLM → Structured Output
chain = (
    RunnableLambda(lambda x: itinerary_prompt.format(**x)) |
    model.with_structured_output(ItineraryResponse)
)

# Step 9: Run chain with structured input
final_output = chain.invoke(structured_data.model_dump())

# ✅ Final structured output
print(final_output.model_dump_json())