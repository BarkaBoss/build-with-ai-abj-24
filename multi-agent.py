import os
import json
from datetime import datetime
from api_key import api_key, tavily
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
    top_p=0.9,
)

search_tool = TavilySearchResults(max_results=3)
search_tools = [Tool(name="Tavily Search", func=search_tool.run, description="For finding info online.")]

classifier_memory = ConversationBufferMemory(memory_key="chat_history")
classifier = initialize_agent(
    tools=[],
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=classifier_memory,
    verbose=True,
)

evidence_memory = ConversationBufferMemory(memory_key="chat_history")
evidence_agent = initialize_agent(
    tools=search_tools,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=evidence_memory,
    verbose=True,
)

critic_memory = ConversationBufferMemory(memory_key="chat_history")
critic_agent = initialize_agent(
    tools=[],
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=critic_memory,
    verbose=True,
)

def log_to_file(data, log_path="logs.json"):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        **data
    }
    logs = []

    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []

    logs.append(log_entry)
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)
    print("\n📝 Log saved to logs.json")

def spam_detection_pipeline(message):
    print(f"\n🔍 Checking message:\n{message}\n")

    classification_prompt = (
        f"Classify the following message as either SPAM or NOT SPAM. "
        f"Justify your answer briefly.\n\nMessage:\n{message}"
    )
    classification_result = classifier.run(classification_prompt)

    evidence_output = ""
    if "SPAM" in classification_result.upper():
        print("\n🔎 Gathering evidence...")
        evidence_prompt = (
            f"This message was classified as SPAM. Explain why by analyzing it in detail. "
            f"If there are links or suspicious terms, investigate using search.\n\n{message}"
        )
        evidence_output = evidence_agent.run(evidence_prompt)

    print("\n🧐 Critic is reviewing the decision...")
    review_prompt = (
        f"Message:\n{message}\n\nClassification:\n{classification_result}\n\n"
        f"Evidence:\n{evidence_output}\n\n"
        f"Do you agree with this classification? Justify if yes or suggest correction if no."
    )
    critic_output = critic_agent.run(review_prompt)

    log_data = {
        "message": message,
        "classification": classification_result,
        "evidence": evidence_output,
        "critic_feedback": critic_output
    }
    
    log_to_file(log_data)

    print("\n✅ Final Result")
    print("\n--- Classification ---\n", classification_result)
    if evidence_output:
        print("\n--- Evidence ---\n", evidence_output)
    print("\n--- Critic Review ---\n", critic_output)

if __name__ == "__main__":
    test_message = "Hello... Hope you are having a great day"
    #test_message = "Everybody dey play Daily Hammer! N5,000,000 up for grabs! Here's your chance to win BIG. Dial *20202# to subscribe today. T&Cs apply."
    #test_message = "Congratulations! You've won a $1000 Walmart gift card. Click here to claim: http://bit.ly/fakeprize"
    spam_detection_pipeline(test_message)
