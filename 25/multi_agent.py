# pip install langchain==0.3.24 langchain-google-genai==2.1.3 langchain_community tavily-python streamlit
# Google Gemini API at: https://aistudio.google.com/
# Tavily API Key at: https://app.tavily.com/home

import os
import csv

from my_key import api_key, tavily
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools.tavily_search import TavilySearchResults

os.environ["GOOGLE_API_KEY"] = api_key
os.environ["TAVILY_API_KEY"] = tavily

llm = ChatGoogleGenerativeAI(
    temperature=0.7,
    model="gemini-2.0-flash-001",
    max_tokens=500,
)


def check_inventory(item, quantity, csv_file="shop.csv"):
    try:
        with open(csv_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row["item"].lower() == item.lower():
                    available = int(row["quantity"])
                    if available >= int(quantity):
                        return f"{item.title()} is available. Requested: {quantity}, In stock: {available}."
                    else:
                        return f"Only {available} units of {item} are available. Requested: {quantity}."
        # return f"{item.title()} is not in stock."
    except Exception as e:
        return f"Error checking inventory: {str(e)}"


def inventory_check_natural_language(input_text):
    import re
    match = re.search(r"(\d+)\s+(.+?)\s*(?:are|is)?\s*available", input_text.lower())
    if match:
        quantity = int(match.group(1))
        item = match.group(2).strip()
        return check_inventory(item, quantity)
    else:
        return "Could not parse item and quantity from input."


inventory_tool = Tool(
    name="Inventory Checker",
    func=inventory_check_natural_language,
    description="Use this tool to check the inventory for a specific item and quantity. "
                "Input should be in the format: 'X items are available' or 'X items is available'. "
                "Example: '5 apples are available'."
)


search_tool = TavilySearchResults(max_results=3)
nutrition_tool = Tool(
    name="Nutrition  Info",
    func=lambda query:search_tool.run(f"Nutrition information of {query}"),
    description="Gives the nutrition facts of a given item"
)


conversation_agent = initialize_agent(
    tools=[],
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=ConversationBufferMemory(memory_key="chat_history"),
    verbose=True
)

inventory_agent = initialize_agent(
    tools=[inventory_tool],
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=ConversationBufferMemory(memory_key="chat_history"),
    verbose=True
)

nutrition_agent = initialize_agent(
    tools=[nutrition_tool],
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=ConversationBufferMemory(memory_key="chat_history"),
    verbose=True
)


def run_agents(item:str, quantity:int):
    convo = conversation_agent.run(f"Tell me what an {item} is")
    stock_query = f"Check if {quantity} {item}(s) are available in stock"
    stock = inventory_agent.run(stock_query)
    nutrition = nutrition_agent.run(item)
    return convo, stock, nutrition
