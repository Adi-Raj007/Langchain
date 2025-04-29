from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="""
               You are a helpful assistant.
               Answer ONLY from the provided transcript context.
               If the context is insufficient, just say you don't know.

               Context:
               {context}

               Question: {question}
           """,
    input_variables=["context", "question"]
)

prompt.save('RAG/rag_youtube_documents/template.json')