from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()

model=ChatGroq(model="llama3-70b-8192")
parser= JsonOutputParser()
template=PromptTemplate(
    template= """  Plan a concise itinerary for a traveler starting from "{place_from}" to visit "{place_to_visit}" over "{No_of_days}" days.
    Include key attractions, local experiences, and a brief recommendation for each day.
    Keep the output short, actionable, and ensure pricing details are included.{format_instructions} """,
    input_variables=["place_from","place_to_visit","No_of_days"],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)
chain = template|model|parser
result=chain.invoke({'place_from': 'Delhi','place_to_visit':'Ahemdabad','No_of_days':'5'})
print(result)

