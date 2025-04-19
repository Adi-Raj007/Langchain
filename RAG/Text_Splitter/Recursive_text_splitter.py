from langchain.text_splitter import RecursiveCharacterTextSplitter,Language
from langchain_community.document_loaders import PythonLoader


docs_path="/home/aditya/Desktop/linux/Data science/Langchain/Prompts/Dynamic_prompts.py"
loader=PythonLoader(docs_path)
docs=loader.load()

text_splitter=RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=500,
    chunk_overlap=0,

)
splitted_docs= text_splitter.split_documents(docs)
print(len(splitted_docs))
print(splitted_docs[0].page_content)
