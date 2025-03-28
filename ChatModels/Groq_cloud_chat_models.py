from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()


model= ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0.3)
result=model.invoke("what is capital of Delhi ?")
print(result.content)