# crawler_agent.py

import os
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Limit crawling
visited = set()
MAX_DEPTH = 2
MAX_PAGES = 20

def fetch_html(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    try:
        print(f"[FETCHING] {url}")
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.text
        else:
            print(f"[ERROR] HTTP {response.status_code} → {url}")
            return ""
    except Exception as e:
        print(f"[ERROR] {url} → {e}")
        return ""

def extract_clean_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup.find_all(['script', 'style']):
        tag.decompose()
    body = soup.find('body')
    if not body:
        return ""
    text = body.get_text(separator=' ', strip=True)
    return text

def is_same_domain(base_url, target_url):
    return urlparse(base_url).netloc == urlparse(target_url).netloc

def get_links(base_url, html):
    soup = BeautifulSoup(html, 'html.parser')
    return {
        urljoin(base_url, a['href'])
        for a in soup.find_all('a', href=True)
        if is_same_domain(base_url, urljoin(base_url, a['href']))
    }

def try_answering_page(page_text, query):
    llm = ChatOpenAI(temperature=0.5)
    prompt = f"""
You are a helpful assistant. A user is asking a question.

User's question:
"{query}"

Below is a webpage's content:
\"\"\"
{page_text}
\"\"\"

If this content contains enough information to answer the question, answer it directly. 
If not, reply with: "Not enough info."
"""
    response = llm([HumanMessage(content=prompt)])
    return response.content.strip()

def crawl_and_answer(base_url, query, depth=0):
    if base_url in visited or depth > MAX_DEPTH or len(visited) >= MAX_PAGES:
        return None

    visited.add(base_url)
    html = fetch_html(base_url)
    if not html:
        return None

    text = extract_clean_text(html)
    if not text:
        return None

    answer = try_answering_page(text, query)
    print(f"[CHECK] {base_url} → {answer[:60]}...")

    if answer.lower().startswith("not enough"):
        for link in get_links(base_url, html):
            result = crawl_and_answer(link, query, depth + 1)
            if result:
                return result
        return None
    else:
        return answer

def run_direct_qa(base_url, query):
    visited.clear()
    return crawl_and_answer(base_url, query) or "No relevant information found."

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "vv"


    # Set the test website and query
    base_url = "https://www.rediff.com/"
    query = "Any news related to Cricket?"
    
    result = run_direct_qa(base_url, query)
    print("\n[ANSWER]")
    print(result)
