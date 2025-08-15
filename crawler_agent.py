# crawler_agent.py

import os
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.schema import HumanMessage
from playwright.sync_api import sync_playwright
from playwright.sync_api import sync_playwright

# Crawler settings
visited = set()
MAX_DEPTH = 2
MAX_PAGES = 20
HEADLESS = True

# ----------------- PLAYWRIGHT HTML FETCH -----------------
def fetch_rendered_html(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        page = browser.new_page()
        print(f"[FETCHING] {url}")
        try:
            page.goto(url, timeout=15000)
            html = page.content()
        except Exception as e:
            print(f"[ERROR] Could not fetch {url}: {e}")
            html = ""
        browser.close()
        return html



def fetch_rendered_html_new(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # try visible first
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/114.0.0.0 Safari/537.36",
            locale='en-US',
            viewport={"width": 1280, "height": 800},
        )
        page = context.new_page()

        try:
            print(f"[FETCHING] {url}")
            page.goto(url, timeout=15000)
            page.wait_for_load_state('networkidle')
            html = page.content()
        except Exception as e:
            print(f"[ERROR] {url} → {e}")
            html = ""
        browser.close()
        return html

# ----------------- CLEAN TEXT -----------------
def extract_clean_text(html):
    soup = BeautifulSoup(html, 'html.parser')

    # Remove noisy elements
    for tag in soup.find_all(['script', 'style']):
        tag.decompose()

    body = soup.find('body')
    if not body:
        return ""

    text = body.get_text(separator=' ', strip=True)
    print(f"\n[TEXT PREVIEW]\n{text[:500]}...\n")
    return text

# ----------------- URL HELPERS -----------------
def is_same_domain(base_url, target_url):
    return urlparse(base_url).netloc == urlparse(target_url).netloc

def get_links(base_url, html):
    soup = BeautifulSoup(html, 'html.parser')
    return {
        urljoin(base_url, a['href'])
        for a in soup.find_all('a', href=True)
        if is_same_domain(base_url, urljoin(base_url, a['href']))
    }

# ----------------- CRAWL AND FILTER -----------------
def crawl_and_collect(base_url, query, depth=0):
    if base_url in visited or depth > MAX_DEPTH or len(visited) >= MAX_PAGES:
        return []

    html = fetch_rendered_html(base_url)
    if not html:
        return []

    visited.add(base_url)
    page_text = extract_clean_text(html)
    if not page_text:
        return []

    # Ask LLM if page is relevant
    llm = ChatOpenAI(temperature=0)
    prompt = f"""
You are helping a web crawler decide whether this webpage is relevant to the user's question.

Query: "{query}"

Webpage preview:
\"\"\"
{page_text[:2000]}
\"\"\"

If this page could help answer the question, respond with "yes". Otherwise, respond with "no".
"""
    decision = llm([HumanMessage(content=prompt)])
    print(f"[DECISION] {base_url} → {decision.content.strip()}")

    documents = []
    if 'yes' in decision.content.lower():
        documents.append(Document(page_content=page_text, metadata={"source": base_url}))

    for link in get_links(base_url, html):
        documents.extend(crawl_and_collect(link, query, depth + 1))

    return documents

# ----------------- VECTOR DB + QA -----------------
def build_vector_store(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="./chroma_db")
    vectorstore.persist()
    return vectorstore

def answer_query(query, vectorstore):
    llm = ChatOpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return qa_chain.run(query)

# ----------------- MAIN WORKFLOW -----------------
def run_agentic_workflow(base_url, query):
    visited.clear()
    docs = crawl_and_collect(base_url, query)
    if not docs:
        return "No relevant content found."
    vectorstore = build_vector_store(docs)
    return answer_query(query, vectorstore)

# ----------------- DIRECT DEBUG RUN -----------------
if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = ""

    base_url = "https://www.cricbuzz.com/"
    query = "What is India's first innings score in ENG vs IND Test match?"

    print(f"Running query on {base_url}")
    result = run_agentic_workflow(base_url, query)

    print("\n[ANSWER]")
    print(result)
