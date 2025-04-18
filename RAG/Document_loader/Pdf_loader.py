from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_groq.chat_models import ChatGroq
load_dotenv()
model= ChatGroq(model="deepseek-r1-distill-llama-70b")
loader= PyPDFLoader('/home/aditya/Desktop/linux/Data science/Langchain/RAG/Document_loader/Aditya_resume.pdf')
docs=loader.load()
print(docs[0].page_content)
