from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain_community.llms import Ollama

def userIn():
    query = input()
    print(f"This was the input: {query}")

if __name__ == '__main__':
    userIn()