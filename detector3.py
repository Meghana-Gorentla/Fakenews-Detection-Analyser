import os
import json
from dotenv import load_dotenv
from serpapi import GoogleSearch
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()

llm = ChatOpenAI(temperature=0.2, model="gpt-4")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# 1️⃣ Extract Keywords Agent
def extract_keywords(article: str):
    prompt = f"""
    Extract the top 5-10 key phrases or claims from the following article that need fact-checking.

    Article:
    \"\"\"{article}\"\"\"

    Return a Python list of concise search-worthy phrases.
    """
    response = llm.predict(prompt)
    return eval(response.strip())

# 2️⃣ Search with SerpAPI
def search_facts(keywords):
    search_results = {}
    for keyword in keywords:
        print(f"🔍 Searching SerpAPI for: {keyword}")
        params = {
            "q": keyword,
            "api_key": SERPAPI_API_KEY,
            "num": 5
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        links = []
        if "organic_results" in results:
            for r in results["organic_results"]:
                links.append(r.get("link"))
        search_results[keyword] = links
    return search_results

# 3️⃣ Final Analysis Agent
def analyze(article, keywords, search_results):
    prompt_template = PromptTemplate.from_template("""
You are a fake news detection expert. A user submitted an article. You were given the following:

📰 Article:
{article}

🔑 Extracted Claims:
{keywords}

🌐 Search Results (Top Links from Google):
{search_results}

Your job:
- Score the article’s fakeness (0 = completely fake, 100 = fully credible)
- Justify your reasoning based on the search results
- Include links that support your explanation

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
        search_results=json.dumps(search_results, indent=2)
    )
    return llm.predict(final_prompt)

# 4️⃣ Main Entry Function
async def detect_fake_news(article: str):
    keywords = extract_keywords(article)
    print("✅ Keywords:", keywords)

    search_results = search_facts(keywords)
    print("🌐 Search Done.")

    judgment = analyze(article, keywords, search_results)
    return judgment
