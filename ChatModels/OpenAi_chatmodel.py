from langchain_openai  import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
model= ChatOpenAI(model='gpt-4o-2024-08-06')
result=model.invoke("What is capital of Delhi ?")
print(result)

