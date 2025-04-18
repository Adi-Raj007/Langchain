from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()

model=ChatGroq(model="llama3-70b-8192")

template_1=PromptTemplate(
    template= """  Plan a concise itinerary for a traveler starting from "{place_from}" to visit "{place_to_visit}" over "{No_of_days}" days.
    Include key attractions, local experiences, and a brief recommendation for each day.
    Keep the output short, actionable, and ensure pricing details are included. """,
    input_variables=["place_from","place_to_visit","No_of_days"]
)
template_2=PromptTemplate(
    template="""make it shorter to 1000 token and finally give itinerary  in form of   table . 
    give incomplete message if data is missing.{Name}""",
    input_variables=["name"]

)

Parser=StrOutputParser()
chain_1=template_1|model|Parser
chain_2=template_2|model|Parser
chain=chain_1|chain_2
result=chain.invoke({'place_from': 'Delhi','place_to_visit':'Ahemdabad','No_of_days':'5','name': 'place_to_visit'})