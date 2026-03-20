from langchain_groq import ChatGroq
import os

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1 # Lower temperature for more consistent JSON
)
