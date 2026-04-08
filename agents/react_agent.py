from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from tools.reconstruction_tool import reconstruct_tool
from config import cfg


def build_agent():
    llm = ChatOpenAI(model=cfg["llm_model"]["name"], temperature=0.0)
    tools = [reconstruct_tool]
    agent = create_react_agent(llm, tools)
    return agent
