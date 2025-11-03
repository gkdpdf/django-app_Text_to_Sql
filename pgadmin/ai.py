from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatState(TypedDict):
    messages: List[dict]

graph = StateGraph(ChatState)

def chat_node(state: ChatState):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=state["messages"]
    )
    ai_message = response.choices[0].message.content
    state["messages"].append({"role": "assistant", "content": ai_message})
    return {"messages": state["messages"]}

graph.add_node("chat_node", chat_node)
graph.set_entry_point("chat_node")
graph.set_finish_point("chat_node")

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

def get_ai_response(session_id: str, messages: List[dict]):
    """Invoke LangGraph with persistent thread_id."""
    config = {"configurable": {"thread_id": session_id}}
    result = app.invoke({"messages": messages}, config=config)
    return result["messages"]
