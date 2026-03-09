from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from tools.reconstruction_tool import reconstruct_tool


def build_agent():
    llm = ChatOpenAI(model="gpt-5-nano")
    tools = [reconstruct_tool]
    agent = create_react_agent(llm, tools)
    return agent
