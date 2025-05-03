from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate(
    template="""
You are a helpful and professional summarization assistant.

Your task is to generate a concise and well-structured summary of the video content provided in the context.
- Focus only on the actual transcript content.
- Do not speculate or include internal thoughts.
- Avoid using phrases like "I think" or "It seems".
- Do not mention the transcript itself.
- Write in a clear, formal tone.

Context:
{context}

Question: {question}
""",
    input_variables=["context", "question"]
)

prompt.save('template_2.json')