# imports
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_ollama import ChatOllama
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List, Dict, Any, Sequence
import operator

# ollama model declaration
model=ChatOllama(
        model = "llama3.2",
        temperature=0.3
        )

parser = StrOutputParser()

class MusicAssistant:

    def __init__(self):
        self.initial_state = {
            "messages": [], 
            "currTime": "N/A", 
            "userPrefs": {}, 
            "historyLog": [],
            "search_keywords": [],
            "initialCands": {},
            "currNode": "",
            "complete": False
        }

        self.workflow = None
        self.setupWorkflow()

    def setupWorkflow(self):
        class MusicState(TypedDict):
            # input 
            userQuery: str
            currTime: str

            # features
            userPrefs: Dict
            historyLog: List[str]
            search_keywords: Annotated[List[str], operator.add]
            initialCands: List[Dict]
            currNode: str

            # messages and history
            messages: Annotated[List[BaseMessage], add_messages]
            complete: bool

        self.workflow = StateGraph(MusicState)

        # add nodes
        self.workflow.add_node("_userIn", self.userIn)
        self.workflow.add_node("_parseQuery", self.parseQuery)

        # define workflow
        self.workflow.set_entry_point("_userIn")
        self.workflow.add_conditional_edges(
            "_userIn", 
            self.ifContinue,
            {
                "END": END,
                "_parseQuery": "_parseQuery"
            }
        )
        self.workflow.add_edge("_parseQuery", "_userIn")

        # compile
        self.app = self.workflow.compile()

    # take in user query
    def userIn(self, state):
        print("-"*50)
        query = input()

        # used to quit convo
        if query.lower() in ['q', 'quit']:
            return {
                "complete": True, 
                "userQuery": query,
                'messages': [HumanMessage(content=query)]
                }
        
        return {
            'userQuery': query,
            'messages': [HumanMessage(content=query)]
                }

    # get LLM response
    def parseQuery(self, state):
        response = model.invoke(state["userQuery"])
        r = response.content 

        print(f"\n{r}")
        
        return {'messages': [AIMessage(content=r)]}
    
    # determines if convo should continue based on 'complete' state
    def ifContinue(self, state):
        if state.get("complete") == True:
            return "END"
        else:
            return "_parseQuery"


if __name__ == "__main__":
    assistant = MusicAssistant()
    print("Hi, I am your Music Suggestor Assistant! \nPlease enter your request (or press 'q' to quit): ")
    
    # The dictionary passed to invoke must contain ALL fields defined in the state
    current_state = assistant.initial_state 

    while True:
        try:
            # Invoke the graph, passing the initial state. 
            # The _userIn node runs first and asks for input.
            result = assistant.app.invoke(current_state)
            
            # The result is the final state of the execution flow
            current_state = result
            
            # Check for the exit condition set in userIn (if implemented in the calling loop)
            if current_state.get('complete'):
                # print history log of conversation
                messages: List[BaseMessage] = current_state.get('messages', [])
            
                print("\n" + "="*50)
                print("CONVERSATION HISTORY LOG")
                print("="*50)
                
                # Loop through the accumulated messages
                for msg in messages:
                    print(f"[{msg.type.upper(): <6}]: {msg.content.strip()}")
                
                print("="*50 + "\n")


                print("Assistant shutting down.")
                break
                 
        except KeyboardInterrupt:
            print("\nAssistant shutting down via interrupt.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break