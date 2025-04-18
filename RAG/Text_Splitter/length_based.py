from langchain.text_splitter import CharacterTextSplitter
import os
from langchain_community.document_loaders import PyPDFLoader

docs_path=("/home/aditya/Desktop/linux/Data science/Langchain/RAG/Document_loader/Aditya_resume.pdf")
print(docs_path)
loader=PyPDFLoader(docs_path)
docs=loader.load()
# print(docs[0].page_content)
splitter= CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=2,
    separator=''
)
splitted_docs=splitter.split_documents(docs)
print(splitted_docs)
