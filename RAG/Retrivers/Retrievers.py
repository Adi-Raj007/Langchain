from langchain_community.retrievers import WikipediaRetriever
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

model=ChatGroq(model="deepseek-r1-distill-llama-70b")

docs="/home/aditya/Desktop/English-short-essay.pdf"
loader=PyPDFLoader(docs)
loaded_docs=loader.load()
# print(loaded_docs[2])

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,



)
splitted_docs=text_splitter.split_documents(loaded_docs)
# print(splitted_docs)

embedding_model = HuggingFaceEmbeddings(
    model_name="/home/aditya/Desktop/PRO/model/jina-embeddings-v2-base-en",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
vector_store=Chroma.from_documents(
    documents=splitted_docs,
    embedding=embedding_model,
    # persist_directory='Retriever_db',
    collection_name='Retriever'

)

retriever=vector_store.as_retriever(search_kwargs={"k": 2})
query = "give detail about Television "
results = retriever.invoke(query)
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)





