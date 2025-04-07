import os
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

llm = ChatOpenAI(temperature=0.2, model="gpt-4")

# Tavily search tool
search_tool = TavilySearchResults()

tools = [
    Tool(
        name="Search",
        func=search_tool.run,
        description="Useful for checking if claims in the news are true or false."
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True
)

async def detect_fake_news(article: str):
    prompt = f"""
    This is a news article: {article}

    Use the Search tool to verify the claims.
    Give a response with:
    - A fakeness score from 0 to 100
    - A justification
    - Supporting sources (links)

    Format:
    {{
        "score": <number>,
        "justification": "<explanation>",
        "sources": ["<link1>", "<link2>"]
    }}
    """
    response = agent.run(prompt)
    return response
