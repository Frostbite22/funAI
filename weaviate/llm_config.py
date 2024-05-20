# Connect to Weaviate Retriever and configure LLM
import dspy
from dspy.retrieve.weaviate_rm import WeaviateRM
import weaviate
import os 
import groq

llama3 = dspy.GROQ(model='llama3-8b-8192', api_key=os.environ.get("GROQ_API_KEY"))

weaviate_client = weaviate.Client("http://localhost:8080")
weaviate_retriever = WeaviateRM(weaviate_client)
dspy.settings.configure(lm=llama3, rm=weaviate_retriever)
# Ask any question you like to this simple RAG program.
with dspy.context(llm=llama3):
    print(llama3("Explain the importance of fast language models"))