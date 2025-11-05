# imports
import os
# suppress logging warnings
os.environ['GLOG_minloglevel'] = '2'
os.environ['GRPC_VERBOSITY'] = 'ERROR'
import sys
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool, StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import spotipy
from google.cloud import secretmanager
from spotipy.oauth2 import SpotifyClientCredentials
from typing import Optional, TypedDict, Annotated, List, Dict, Any, Sequence
import operator

# loading key
load_dotenv()

# check to make sure key is there
if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
    print("FATAL ERROR: Neither GOOGLE_API_KEY nor GEMINI_API_KEY found.")
    sys.exit(1)

SYSTEM_PROMPT = (
    "You are a specialized music search engine. Your ONLY function is to analyze the "
    "user's query and immediately call the 'spotify_search' tool with the extracted "
    "keywords and search type (artist, track, album, tempo, mood, genre, popularity, duration). "
    "If the user asks a question, your response MUST be a tool call to 'spotify_search'."
    "Speak in natural language. Do not use JSON, code blocks, or keys/values."
)

# genAI model declaration
model=ChatGoogleGenerativeAI(
        model = "gemini-2.5-flash",
        temperature=0.5
        )

parser = StrOutputParser()

# Secret Manager and Spotify Setup
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-gcp-project-id") 

def get_secret_value(secret_name):
    """Retrieves the latest version of a secret from Secret Manager."""
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{PROJECT_ID}/secrets/{secret_name}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        # used for debugging IAM issues in Cloud Run
        print(f"Error accessing secret {secret_name}: {e}")
        return None

# AI agent will call this tool
def spotify_search(query: str, search_type: str = 'artist,track') -> str:
    """
    Useful for finding public information on artists, tracks, or albums on Spotify. 
    Input should be a comma-separated list of search types (e.g., 'artist,track') and a query.
    """
    try:
        CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
        CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
        
        if not CLIENT_ID or not CLIENT_SECRET:
            return "Authentication Error: Spotify credentials not loaded from Secret Manager."
        
        # Initialize client using Client Credentials Flow (no user interaction needed)
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET
        ))

        original_query = query
        search_query = query
        
        # Check if the query asks for a genre without using the official filter syntax.
        query_lower = query.lower()
        if 'genre:' not in query_lower and ('songs' in query_lower or 'music' in query_lower):
            # search for tracks tagged with input genre
            keywords = query_lower.replace(' songs', '').replace(' music', '').strip()
            
            # Format the query with the genre filter
            search_query = f'genre:"{keywords}"' 
            search_type = 'track'   
        elif query.count(':') > 0 or 'genre:' in query_lower:
            # The user provided an advanced filter already (e.g., 'genre:metal AND year:2000')
            search_query = query 

        # check if query gives a 'mood' without using proper search syntax
        if 'mood' not in query_lower and ('songs' in query_lower or 'music' in query_lower):
            keywords = query_lower.replace(' songs', '').replace(' music', '').strip()
            search_query = f'mood:"{keywords}"'
            search_type = 'track'
        elif query.count(':') > 0 or 'mood:' in query_lower:
            search_query = query 

        # Perform the search
        results = sp.search(q=query, type=search_type, limit=5)
        
        output = ["Spotify Search Results:"]
        
        # Extract and format results
        if 'tracks' in results and results['tracks']['items']:
            for i, track in enumerate(results['tracks']['items']):
                trackName = track['artists'][0]['name']
                output.append(f"Track {i+1}: '{track['name']}' by {trackName}")

        if 'artists' in results and results['artists']['items']:
            for i, artist in enumerate(results['artists']['items']):
                genres = ', '.join(artist['genres'][:2])
                output.append(f"Artist {i+1}: {artist['name']}, Genres: {genres or 'N/A'}")

        if len(output) == 1:
             return f"No results found for query: {query}"
             
        return "\n".join(output)

    except Exception as e:
        return f"An error occurred during Spotify API call: {e}"

# turn spotify_search into a Tool
spotifySearchToolInstance = StructuredTool.from_function(
    func=spotify_search,
    name="spotify_search",
    description="Useful for finding public information on artists, tracks, or albums on Spotify."
)

# Define the list of tools available to the LLM
tools = [spotifySearchToolInstance]
tools_by_name = {tool.name: tool for tool in tools} 

class MusicState(TypedDict):
            # input 
            userQuery: str

            # features
            search_keywords: Annotated[List[str], operator.add]
            currNode: str

            # messages and history
            messages: Annotated[List[BaseMessage], add_messages]
            convoLog: List[Dict]
            complete: bool
            streamlitPrompt: Optional[str]

class MusicAssistant:
    def __init__(self):
        self.initial_state = {
            "userQuery": "",
            "messages": [], 
            "convoLog": [],
            "currNode": "",
            "complete": False
        }

        # bind tools to model
        self.model = model.with_config({"system_instructions": SYSTEM_PROMPT}).bind_tools(tools)

        self.workflow = None
        self.setupWorkflow()

    def setupWorkflow(self):
        self.workflow = StateGraph(MusicState)

        # add nodes
        self.workflow.add_node("_userIn", self.userIn)
        self.workflow.add_node("_topicRouterStep", self.topicRouterStep)
        self.workflow.add_node("_checkQuery", self.checkQuery)
        self.workflow.add_node("_ifContinue", self.ifContinue)
        self.workflow.add_node("_rejectTurn", self.rejectTurn)
        self.workflow.add_node("_parseQuery", self.parseQuery)
        self.workflow.add_node("_aiOutput", self.aiOutput)
        self.workflow.add_node("_shouldContinueAfterOutput", self.shouldContinueAfterOutput)
        self.workflow.add_node("_routeToolOrResponse", self.routeToolOrResponse)

        # define workflow
        self.workflow.set_entry_point("_userIn")

        # checks if user quit convo
        self.workflow.add_conditional_edges(
            "_userIn",
            self.ifContinue,
            {
                "END": END,
                "CONTINUE": "_topicRouterStep"
            }
        )

        # topic validation
        self.workflow.add_conditional_edges(
            "_topicRouterStep", 
            self.checkQuery,
            {
                "VALID": "_parseQuery",
                "INVALID": "_rejectTurn"
            }
        )

        # continue convo after tool call
        self.workflow.add_conditional_edges(
            "_parseQuery",
            self.routeToolOrResponse,
            {
                "REINVOKE": "_parseQuery",
                "FINAL_RESPONSE": "_aiOutput"
            }
        )
        
        # continue after output
        self.workflow.add_conditional_edges(
            "_aiOutput",
            self.shouldContinueAfterOutput,
            {
                "END": END,
                "CONTINUE": "_userIn"
            }
        )

        self.workflow.add_edge("_rejectTurn", "_userIn")

        # compile
        self.app = self.workflow.compile()

    def draw_mermaid(self):
        """Generate Mermaid diagram of the workflow"""
        try:
            graph = self.app.get_graph() 
            mermaid_output = graph.draw_mermaid()
            print("\nWorkflow Diagram (Mermaid format):")
            print("="*50)
            print(mermaid_output)
            print("="*50)
            print("Copy the above to https://mermaid.live/ to visualize")
            return mermaid_output
        except Exception as e:
            print(f"Error generating diagram: {e}")
            return None

    # take in user query
    def userIn(self, state):
        currConvoLog = state.get('convoLog', [])
        logUpdate = []
        isDuplicate = False
        streamlitPrompt = state.get("streamlitPrompt")
        
        if streamlitPrompt is not None:
            query = streamlitPrompt
        else:
            print("-"*50)
            query = input()

        logEntry = {"speaker": "HUMAN", "content": query.strip()}

        #check to make sure no duplicate lines in the convoLog
        if (len(currConvoLog) == 0) or (currConvoLog[-1].get('content') != logEntry['content']):
            isDuplicate = False
        else:
            isDuplicate = True

        if not isDuplicate:
            currConvoLog.append(logEntry)

        # used to quit convo
        if query.lower() in ['q', 'quit']:
            return {
                "complete": True, 
                "userQuery": query,
                'messages': [HumanMessage(content=query)],
                'convoLog': currConvoLog
                }
        
        
        return {
            'userQuery': query,
            'messages': [HumanMessage(content=query)],
            'convoLog': currConvoLog
                }

    # make sure query is music related
    def checkQuery(self, state):
        musicKeywords = [
            'song', 'track', 'artist', 'band', 'album', 'genre', 'tune', 'playlist', 'spotify', 'tempo',
            'instrument', 'instrumental', 'lyrics', 'beat', 'upbeat', 'downbeat', 'mood', 'jazz', 'rock', 'pop', 'chill',
            'indie', 'folk', 'happy', 'sad', 'angry', 'mad', 'frustrated', 'calm', 'sing', 'sings'
            ]
        
        messages = state.get("messages", [])
        convo = state.get("convoLog", [])

        # ensures messages is not empty
        if not messages:
            return "INVALID"

        query = messages[-1].content
        queryLowCase = query.lower()    

        # context check
        if len(messages)>1:
            prevMessage = messages[-2]

            isAIResponse = (
                prevMessage.type == "ai" and 
                (not hasattr(prevMessage, 'tool_calls') or 
                prevMessage.tool_calls == [] or
                prevMessage.tool_calls is None)
            )
            
            # If the preceding message was a pure AI conversational response, validate this turn.
            if isAIResponse:
                return "VALID"    

        # check for music keywords (for intial or non-follow-ups)
        isMusicTopic = any(word in queryLowCase for word in musicKeywords)
        
        if isMusicTopic:
            return "VALID"
        else:
            return "INVALID"

    # tells user query is invalid
    def rejectTurn(self, state):
        rejection_msg = "I can only help with music-suggestion-related questions. Please re-enter a new query relating to music."
        
        # Get the current message list
        messages = state.get("messages", [])
        convo = state.get("convoLog", [])
        logUpdate = []
            
        # Append the rejection message as an AIMessage
        messages.append(AIMessage(content=rejection_msg))

        logEntry = [{"speaker": "HUMAN", "content": rejection_msg}]

        #check to make sure no duplicate lines in the convoLog
        if (len(convo) == 0) or (convo[-1].get('content') != logEntry[0]['content']):
            logUpdate = [logEntry]
        else:
            logUpdate = []
        
        # Print the rejection message so the user sees it
        if state.get("streamlitPrompt") is None:
            print(f"\n{rejection_msg}")
        
        return {
            "messages": messages,
            "convoLog": logUpdate,
            "complete": False 
        }

    # get LLM response
    def parseQuery(self, state):
        messages = state.get("messages", [])

        response = self.model.invoke(messages) 

        if response.tool_calls:            
            tool_outputs = []
            
            # Loop through all tool calls requested by the model
            for tool_call in response.tool_calls:
                tool_name = tool_call.get('name')
                tool_args = tool_call.get('args', {})
                tool_call_id = tool_call.get('id')
                
                # Check the robust map for the Tool object
                tool_to_execute = tools_by_name.get(tool_name)
                
                if tool_to_execute:
                    try:
                        output = tool_to_execute.invoke(tool_args)
                        
                        # Format the result as a ToolMessage (the 'Observation')
                        tool_outputs.append(
                            ToolMessage(
                                content=output, 
                                tool_call_id=tool_call_id, 
                                name=tool_name
                            )
                        )
                    except Exception as e:
                        # Handle execution error within the tool function (e.g., API failure)
                        error_msg = f"Tool Execution Error ({tool_name}): {e}"
                        print(f"Error: {error_msg}")
                        tool_outputs.append(
                            ToolMessage(
                                content=error_msg, 
                                tool_call_id=tool_call_id, 
                                name=tool_name
                            )
                        )
                else:
                    # Handle the 'Unknown Tool' error (the LLM asked for something not available)
                    error_msg = f"Error: Unknown tool '{tool_name}' requested by LLM."
                    print(error_msg)
                    tool_outputs.append(
                        ToolMessage(
                            content=error_msg, 
                            tool_call_id=tool_call_id, 
                            name=tool_name or "unknown" # Use 'unknown' if name is None
                        )
                    )
            
            finalOutput = ""
            for out in tool_outputs:
                finalOutput += out.content + "\n"

            # Return the updated state
            return {'messages': [response] + tool_outputs}
        
        print("")
        
        return {'messages': [AIMessage(content=response.content)]}
    
    # determines if convo should continue based on 'complete' state
    def ifContinue(self, state):
        if state.get("complete") == True:
            return "END"
        else:
            return "CONTINUE"
        
    # prints AI response
    def aiOutput(self, state):
        messages = state.get("messages", [])
        currConvoLog = state.get("convoLog", [])
        output = ""

        if messages:
            last_message = messages[-1]
            
            # Check if the last message is an AIMessage
            if last_message.type == "ai": 
                # Print the final response to the console
                if isinstance(last_message.content, str):
                    output = last_message.content
                elif isinstance(last_message.content, list):
                    output = last_message.content[0].get('text', '')
                elif isinstance(last_message.content, dict):
                    output = last_message.content.get('text', '')

                if state.get("streamlitPrompt") is None and output:
                    finalOutput = output.strip()
                    print(f"{finalOutput}")
                    # add output to conversational log
                    newEntry = {"speaker": "AI", "content": finalOutput}
                    currConvoLog.append(newEntry)
                
        return {"convoLog": currConvoLog}
    
    # used to route conditional edges
    def topicRouterStep(self, state):
        return state
    
    # re-invoke LLM if tool was last message
    def routeToolOrResponse(self, state):
        messages = state.get("messages", [])
        if messages and messages[-1].type == "tool":
            return "REINVOKE"
        else:
            return "FINAL_RESPONSE"
        
    # should continue for streamlit
    def shouldContinueAfterOutput(self, state):
        if state.get("streamlitPrompt") is not None:
            return "END"
        else:
            return "CONTINUE"

if __name__ == "__main__":
    assistant = MusicAssistant()

    # Check for --graph parameter
    if "--graph" in sys.argv:
        print("🎵 AI Music Suggester - Workflow Graph")
        print("=" * 50)
        assistant.draw_mermaid()
        print("=" * 50)
        print("Graph displayed. Exiting...")
        sys.exit(0)

    print("Hi, I am your Music Suggester Assistant! \nPlease enter your request (or press 'q' to quit): ")
    
    # The dictionary passed to invoke must contain ALL fields defined in the state
    current_state = assistant.initial_state 

    while True:
        try:
            # Invoke the graph, passing the initial state. 
            # The _userIn node runs first and asks for input.
            result = assistant.app.invoke(
                current_state,
                config={"recursion_limit": 50}
            )
            
            # The result is the final state of the execution flow
            current_state = result
            
            # Check for the exit condition set in userIn (if implemented in the calling loop)
            if current_state.get('complete'):
                print("-"*50)
                print("Thank you for using the Music Assistant!")
                print("Assistant shutting down.")
                break
                 
        except KeyboardInterrupt:
            print("\nAssistant shutting down via interrupt.")
            break
        except Exception as e:
            # Handle recursion limit gracefully
            if "Recursion limit" in str(e) or "GraphRecursionError" in str(e):
                print(f"\n Recursion limit reached. The conversation is getting too complex.")
                print("Please start a new conversation by restarting the program.")
                break
            else:
                print(f"An error occurred: {e}")