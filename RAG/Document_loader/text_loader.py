from langchain_community.document_loaders import TextLoader
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model=ChatGroq(model="gemma2-9b-it")
parser=StrOutputParser()


template=PromptTemplate(
    template=""" make the {file} in 1000 words  :
    """,
    input_variables=["file"]
)


loader=TextLoader('/home/aditya/Desktop/linux/Data science/Langchain/RAG/Document_loader/sample.txt',encoding="utf-8")
docs=loader.load()
chain=template|model|parser
# print(docs[0].page_content)
print(chain.invoke({'file':docs[0].page_content}))
