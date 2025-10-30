# === GRUNDEINSTELLUNGEN ===

model = "llama3.2:1b" 

#### Task 1: Create a Simple Chain for Summarization
print("\n=== START TASK 1: Simple Chain ===")

from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Load the Ollama model
llm = ChatOllama(model=model)

# ADD HERE YOUR CODE
# Define the prompt template
# Wir verwenden .from_messages für eine klare Struktur
summarization_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize the following text in exactly one sentence."), # <-- LÖSUNG
    ("human", "{text}"),
])

# ADD HERE YOUR CODE
# Create the LLMChain
# Mit dem | (Pipe)-Operator "verketten" wir den Prompt mit dem Modell
summarization_chain = summarization_prompt | llm # <-- LÖSUNG

# Sample text
text = """Over the last decade, deep learning has evolved massively to process and generate unstructured data like text, images, and video. 
These advanced AI models have gained popularity in various industries, and include large language models (LLMs). 
There is currently a significant level of fanfare in both the media and the industry surrounding AI,
and there’s a fair case to be made that Artificial Intelligence (AI), with these advancements,
is about to have a wide-ranging and major impact on businesses, societies, and individuals alike.
This is driven by numerous factors, including advancements in technology, high-profile applications, 
and the potential for transfor- mative impacts across multiple sectors."""

# ADD HERE YOUR CODE
# Invoke the chain (die Kette ausführen)
print("--- Chain Invoke (Antwort auf einmal) ---")
summary = summarization_chain.invoke({"text": text}) # <-- LÖSUNG
print(summary.content)

# Stream the chain output (die Antwort "live" streamen)
print("\n--- Chain Stream (Wort für Wort) ---")
for chunk in summarization_chain.stream({"text": text}):
    print(chunk.content, end="", flush=True)
print("\n=== END TASK 1 ===\n")


#### Task 2: Chain with Tool Usage (Simple Math Tool)
print("\n=== START TASK 2: Chain with Tool ===")

from langchain_core.tools import tool

# ADD HERE YOUR CODE
# Create custom tool
@tool
def multiply(first_int: int, second_int: int) -> int: # <-- LÖSUNG (Argumente + Typ)
    """Multiply two integers.""" # <-- LÖSUNG (WICHTIG: Beschreibung für die AI)
    return first_int * second_int # <-- LÖSUNG (Funktionslogik)


print(f"Tool '{multiply.name}' erstellt mit Beschreibung: '{multiply.description}'")

# Load the Ollama model
llm = ChatOllama(model=model)

# ADD HERE YOUR CODE
# Use bind_tools to pass the definition of our tool in as part of each call to the model
llm_with_tools = llm.bind_tools([multiply]) # <-- LÖSUNG

# When the model invokes the tool, this will show up in the AIMessage.tool_calls attribute of the output
print("\n--- Test: Erkennt das LLM den Tool-Bedarf? ---")
msg = llm_with_tools.invoke("whats 5 times forty two")
print("LLM will folgendes Tool aufrufen:")
print(msg.tool_calls) # Zeigt, dass das LLM verstanden hat, 'multiply' mit 5 und 42 zu rufen

# ADD HERE YOUR CODE
# Create the chain: pass the extracte tool parameters from the input text to the tool
# Kette: 1. LLM erkennt Tool -> 2. Lambda-Funktion holt Argumente -> 3. Tool wird ausgeführt
chain_with_tools = llm_with_tools | (lambda msg: msg.tool_calls[0]['args']) | multiply # <-- LÖSUNG

# Run chain
print("\n--- Chain mit Tool ausführen ---")
result = chain_with_tools.invoke("whats 5 times forty two")
print(f"Ergebnis von '5 times 42' ist: {result}")
print("=== END TASK 2 ===\n")


#### Task 3: Agent with Tool Usage (Two Tools)
print("\n=== START TASK 3: Agent with Tools ===")

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import MessagesPlaceholder # Wichtig für den Agenten

# Load the Ollama model
llm = ChatOllama(model=model)

# ADD HERE YOUR CODE
# Ein Agenten-Prompt braucht Platzhalter für die Eingabe und den "Notizblock" (scratchpad)
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can use tools to calculate."),
    ("human", "{user_input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"), # <-- LÖSUNG
])

# Custom math tools
@tool
def add(first_int: int, second_int: int) -> int:
    "Add two integers."
    return first_int + second_int


@tool
def exponentiate(base: int, exponent: int) -> int:
    "Exponentiate the base to the exponent power."
    return base**exponent


tools = [add, exponentiate]
print(f"Agenten-Tools geladen: {add.name}, {exponentiate.name}")

# ADD HERE YOUR CODE
# Construct the tool calling agent (Das "Gehirn" des Agenten)
agent_with_tools = create_tool_calling_agent(llm, tools, agent_prompt) # <-- LÖSUNG

# ADD HERE YOUR CODE
# Create an agent executor (Der "Manager", der den Agenten laufen lässt)
# verbose=True zeigt uns, was der Agent "denkt"
agent_executor_with_tools = AgentExecutor(
    agent=agent_with_tools, 
    tools=tools, 
    verbose=True # <-- LÖSUNG
) 

print("\n--- Agenten ausführen (3^5 + 12) ---")
agent_executor_with_tools.invoke(
    {
        "user_input": "First take 3 to the power of five and afterwards add 12."
    }
)
print("=== END TASK 3 ===\n")


#### [Optional] Task 4: Enhance Agent with Memory
print("\n=== START TASK 4: Agent with Memory ===")

from langchain.memory import ConversationBufferMemory

# Load the Ollama model
llm = ChatOllama(model=model)

# Define memory object for conversation history
memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True, 
    output_key="output"
)

# ADD HERE YOUR CODE
# Add history placeholder to prompt
agent_prompt_with_memory = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. You use tools and remember the conversation."),
    MessagesPlaceholder(variable_name="chat_history"), # <-- LÖSUNG 
    ("human", "{user_input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
]) # <-- LÖSUNG

# Tools von Task 3 wiederverwenden
# @tool
# def add(first_int: int, second_int: int) -> int: ...
# @tool
# def exponentiate(base: int, exponent: int) -> int: ...
# tools = [add, exponentiate]

# ADD HERE YOUR CODE
# Construct the tool calling agent
agent_with_tools_and_memory = create_tool_calling_agent(
    llm, 
    tools, 
    agent_prompt_with_memory
) # <-- LÖSUNG

# ADD HERE YOUR CODE
# Create an agent executor by passing in the agent and tools
agent_executor_with_tools_and_memory = AgentExecutor(
    agent=agent_with_tools_and_memory,
    tools=tools,
    memory=memory, # <-- LÖSUNG
    verbose=True
)

print("\n--- Agent mit Gedächtnis: 1. Frage ---")
question = "Take 3 to the fifth power then add that 12?"
ai_msg_1 = agent_executor_with_tools_and_memory.invoke({"user_input": question})
print(f"Finale Antwort 1: {ai_msg_1['output']}")

print("\n--- Agent mit Gedächtnis: 2. Frage (nutzt Kontext) ---")
second_question = "Explain how you have calculated the result."
ai_msg_2 = agent_executor_with_tools_and_memory.invoke({"user_input": second_question})
print(f"Finale Antwort 2: {ai_msg_2['output']}")
print("=== END TASK 4 ===\n")