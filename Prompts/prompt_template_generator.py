from langchain_core.prompts import PromptTemplate
template=PromptTemplate(
    template="""
    Plan a concise itinerary for a traveler starting from "{place_from}" to visit "{place_to_visit}" over "{No_of_days}" days.
    Include key attractions, local experiences, and a brief recommendation for each day.
    Keep the output short, actionable, and ensure pricing details are included. And make it shorter to 1000 token and finally give itienery  in form of   table . 
    give incomplete message if data is missing.
    """,
    input_variables=["place_to_visit", "place_from", "No_of_days"]

)
template.save('template.json')