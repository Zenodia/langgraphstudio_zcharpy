from functools import lru_cache
from my_agent.utils.tools import tools
from langgraph.prebuilt import ToolNode
from langchain_nvidia_ai_endpoints import ChatNVIDIA
nvapi_key=os.environ["NVIDIA_API_KEY"]
@lru_cache(maxsize=4)
def _get_model(model_name: str):
    if model_name == "llama3-405b":       

        model = ChatNVIDIA(model="meta/llama-3.1-405b-instruct", nvidia_api_key=nvapi_key, max_tokens=1024)
    elif model_name == "llama3-70b":       

        model = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", nvidia_api_key=nvapi_key, max_tokens=1024)

    
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    model = model.bind_tools(tools)
    return model

# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


system_prompt = """Be a helpful assistant"""

# Define the function that calls the model
def call_model(state, config):
    messages = state["messages"]
    messages = [{"role": "system", "content": system_prompt}] + messages
    model_name = config.get('configurable', {}).get("model_name", "llama3-405b")
    print("current chosen model is : ", model_name)
    model = _get_model(model_name)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define the function to execute tools
tool_node = ToolNode(tools)