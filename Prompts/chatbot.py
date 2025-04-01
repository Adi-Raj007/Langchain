from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

load_dotenv()

model= ChatGroq(model="llama3-70b-8192")
chat_history=[SystemMessage(content="Hi I am Aditya ")]
while True:
    user_input=input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == "exit":
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("Ai : ", result.content)


print(chat_history)