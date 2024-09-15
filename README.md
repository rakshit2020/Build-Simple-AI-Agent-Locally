# Build-Simple-AI-Agent Locally

### Building a Simple AI Agent Using LangGraph and Tavily Search Tools

Creating AI agents can be complex, but by leveraging powerful tools like `LangGraph` and `Tavily`, we can break down the process into manageable steps. In this blog, we'll walk through how to build a simple AI agent that uses these tools to look up information dynamically. The code below will act as an interactive assistant that takes user input, processes it using a language model, and uses external tools to fetch relevant data. Let's break down the components step by step to make it easier for beginners.

```
pip install -U langgraph
pip install langchain-ollama
pip install langchain-community
```


## Import all the necessary import
```
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_ollama.chat_models import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
```

---

### **1. Setting Up Tools for Searching**

First, we introduce a search tool, `TavilySearchResults`, which allows our AI to look up information on the web.

```python
tool = TavilySearchResults(max_results=2) #increased number of results
print(type(tool))
print(tool.name)
```

**Explanation:**
- We initialize the `TavilySearchResults` tool and set `max_results=2`, meaning it will return a maximum of two search results per query. This is a simple tool that helps fetch data from a search engine.
- This part of the code allows the AI agent to access external data sources when required.

---

### **2. Defining the Agent's State**

We define an agent's state using a `TypedDict`, which will manage messages between the user and the model.

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
```

**Explanation:**
- The `AgentState` keeps track of a list of messages (interactions between user, system, and tools). Each message can be a user input, system message, or tool response.
- This state helps the agent decide what to do next (e.g., generate a response or call a tool).

---

### **3. Building the AI Agent**

Now we create the `Agent` class, which will handle the logic of the AI.

```python
class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)
```

**Explanation:**
- The `Agent` is constructed with a language model and tools (like `Tavily`). The language model is responsible for generating responses, while the tools perform specific tasks like web searches.
- A `StateGraph` is defined to control the flow of actions. The AI either generates a response (`call_openai`) or invokes a tool (`take_action`), depending on the conversation state.
- The `graph` connects these actions, ensuring that if the agent needs more information (like a search query), it switches to calling tools.

---

### **4. Handling Actions and Responses**

Next, we define the functions that will generate responses and call external tools.

```python
def exists_action(self, state: AgentState):
    result = state['messages'][-1]
    return len(result.tool_calls) > 0

def call_openai(self, state: AgentState):
    messages = state['messages']
    if self.system:
        messages = [SystemMessage(content=self.system)] + messages
    message = self.model.invoke(messages)
    return {'messages': [message]}

def take_action(self, state: AgentState):
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        if not t['name'] in self.tools:
            result = "bad tool name, retry"
        else:
            result = self.tools[t['name']].invoke(t['args'])
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
    return {'messages': results}
```

**Explanation:**
- `exists_action`: Checks if the AI agent needs to perform any action (like calling a tool).
- `call_openai`: Uses the language model to generate responses based on the current conversation. If there’s no action needed, the AI generates a response.
- `take_action`: If a tool needs to be invoked (like making a search query), this function handles it. It calls the correct tool with the right arguments and processes the results.

---

### **5. Creating a Smart Research Assistant**

Finally, we write the prompt for the AI to act as a smart assistant and process user inputs.

```python
prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow-up question, you are allowed to do that!
"""

model = ChatOllama(model="llama3.1", temperature=0)  # Reduce inference cost
abot = Agent(model, [tool], system=prompt)

messages = [HumanMessage(content="my weight is 72 kg and my height is 172 and i am an male what is my BMI")]
result = abot.graph.invoke({"messages": messages})
print(result)
```

**Explanation:**
- This is the entry point for the AI agent. The prompt tells the agent how to behave—it's acting as a research assistant with access to search tools.
- We define a sample message where the user asks the AI to calculate their BMI based on their weight and height.
- The AI processes the request, and if necessary, calls the search tool to look up information or just calculates the BMI.

---

### **Conclusion**

This code example demonstrates how to create a basic AI agent using `LangGraph` and `TavilySearchResults`. The agent can interact with users, generate responses via a language model, and perform external actions (like making search queries) using tools. By breaking down the code step by step, even beginners can follow along and build their own intelligent agent.

This framework can be expanded with more tools and refined prompts, enabling developers to create more sophisticated AI applications. The modular design allows you to integrate any language model or tool as needed.
