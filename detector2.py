import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import Tool, initialize_agent
from langchain.prompts import PromptTemplate

load_dotenv()

llm = ChatOpenAI(temperature=0.2, model="gpt-4")
search_tool = TavilySearchResults()

# 1️⃣ Extract Keywords Agent
def extract_keywords(article: str):
    prompt = f"""
    Extract the top 5-10 key phrases or claims from the following article that need fact-checking.

    Article:
    \"\"\"{article}\"\"\"

    Return a Python list of concise search-worthy phrases.
    """
    response = llm.predict(prompt)
    return eval(response.strip())  # ⚠️ Only use eval in trusted/internal setup

# 2️⃣ Search with Tavily (for each keyword)
def search_facts(keywords):
    search_results = {}
    for keyword in keywords:
        print(f"🔍 Searching Tavily for: {keyword}")
        result = search_tool.run(keyword)
        search_results[keyword] = result
    return search_results

# 3️⃣ Final Analysis Agent
def analyze(article, keywords, search_results):
    prompt_template = PromptTemplate.from_template("""
You are a fake news detection expert. A user submitted an article. You were given the following:

📰 Article:
{article}

🔑 Extracted Claims:
{keywords}

🌐 Search Results:
{search_results}

Your job:
- Score the article’s fakeness (0 = true, 100 = completely fake), lower is the score: higher is the fakeness.
- Give a clear explanation
- List links from the search results that support your judgment

Return a JSON like:
{{
  "score": <int>,
  "justification": "<reason>",
  "sources": ["<link1>", "<link2>", ...]
}}
""")
    final_prompt = prompt_template.format(
        article=article,
        keywords=keywords,
        search_results=search_results
    )
    return llm.predict(final_prompt)

# 4️⃣ Main Function
async def detect_fake_news(article: str):
    keywords = extract_keywords(article)
    print("✅ Keywords:", keywords)

    search_results = search_facts(keywords)
    print("🌐 Search Done.")

    judgment = analyze(article, keywords, search_results)
    return judgment
