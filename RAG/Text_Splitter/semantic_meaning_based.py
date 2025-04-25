from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import  load_dotenv
from langchain_community.document_loaders import PyPDFLoader

from RAG.Text_Splitter.length_based import docs_path, splitted_docs

load_dotenv()

loader=PyPDFLoader(docs_path)
docs=loader.load()
# print(docs[0].page_content)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # or "text-embedding-3-large"
    openai_api_base="https://api.groq.com/openai/v1"
)

# 🔪 Semantic chunking using Groq-powered embeddings
text_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3
)
splitted_docs=text_splitter.split_documents(docs)
print(splitted_docs[0].page_content)


