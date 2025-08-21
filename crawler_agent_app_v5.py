# Enhancement in v7
# Adds Streamlit table rendering in chat output
# Detects markdown/table-like outputs and shows as grid

import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import streamlit as st
import pandas as pd
from openai import OpenAI
from typing import Optional

# ---- API setup ----
os.environ["OPENAI_API_KEY"] = ""

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("No OpenAI API key found. Set OPENAI_API_KEY env var.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ---- Web scraping helpers ----
def fetch_page(url: str) -> str:
    try:
        resp = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (compatible; WebChatBot/1.0)"}
        )
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""


def extract_content(html: str):
    soup = BeautifulSoup(html, "html.parser")
    # Remove unwanted elements
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript", "iframe", "svg"]):
        tag.decompose()

    headings = [h.get_text(" ", strip=True) for h in soup.find_all(["h1", "h2", "h3"])]
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]

    # Extract tables
    tables = []
    for table in soup.find_all("table"):
        rows = []
        for row in table.find_all("tr"):
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
            if cells:
                rows.append(" | ".join(cells))
        if rows:
            tables.append("\n".join(rows))

    links = []
    for a in soup.find_all("a", href=True):
        link = a["href"].strip()
        if not link or link.startswith("#"):
            continue
        links.append(link)

    return {"headings": headings, "paragraphs": paragraphs, "tables": tables, "links": links}


def crawl_website(base_url: str, max_pages: int = 5, max_depth: int = 1):
    visited = set()
    to_visit = [(base_url, 0)]
    results = []

    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc

    while to_visit and len(results) < max_pages:
        url, depth = to_visit.pop(0)
        if url in visited or depth > max_depth:
            continue

        visited.add(url)
        html = fetch_page(url)
        if not html:
            continue

        content = extract_content(html)
        results.append({"url": url, "content": content})

        # Queue internal links
        for link in content["links"]:
            absolute = urljoin(url, link)
            parsed_link = urlparse(absolute)
            if parsed_link.netloc == base_domain and absolute not in visited:
                to_visit.append((absolute, depth + 1))

    return results


# ---- LLM helpers ----
def select_relevant_chunks(query: str, content: dict, max_chunks: int = 5):
    text_blocks = content["headings"] + content["paragraphs"] + content.get("tables", [])
    if not text_blocks:
        return []

    chunks = []
    chunk_size = 5
    for i in range(0, len(text_blocks), chunk_size):
        chunk = "\n".join(text_blocks[i:i + chunk_size])
        chunks.append(chunk)

    candidate_text = "\n\n".join([f"[Chunk {i+1}]\n{c}" for i, c in enumerate(chunks[:20])])

    prompt = f"""
The user asked:

"{query}"

Here are text chunks from the webpage:

{candidate_text}

Which {max_chunks} chunks are most relevant to the query?
Return only their numbers as a comma-separated list (e.g., 1,3,5).
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    choice = resp.choices[0].message.content.strip()
    chosen_ids = [int(c) for c in choice.replace(" ", "").split(",") if c.isdigit()]
    return [chunks[i-1] for i in chosen_ids if 0 < i <= len(chunks)]


def summarize_crawled_content(query: str, crawled_pages: list) -> str:
    relevant_chunks = []
    for page in crawled_pages:
        relevant_chunks.extend(select_relevant_chunks(query, page["content"]))
    if not relevant_chunks:
        return "No relevant content found on the crawled pages."

    joined_text = "\n".join(relevant_chunks)
    prompt = f"""
The user is asking:

"{query}"

Relevant content from the site:
{joined_text}

➡️ Summarize the information that answers the user's question.
➡️ If it’s a list (e.g., job postings), return it as bullet points or a table-like format.
➡️ Keep the summary concise and focused.
If nothing relevant is found, reply: "No relevant content found."
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


def summarize_article(query: str, url: str) -> str:
    html = fetch_page(url)
    content = extract_content(html)
    relevant_chunks = select_relevant_chunks(query, content)
    if not relevant_chunks:
        return "No content available in this article."

    text = "\n".join(relevant_chunks)
    prompt = f"""
Summarize the following article in relation to the user's query:

Query: "{query}"

Article Content:
{text}

Provide a concise summary (3–5 sentences).
If it’s structured data (jobs, scores, tables), present it in a neat list or table-style summary.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


def choose_relevant_link(query: str, follow_up: str, crawled_pages: list, base_url: str) -> Optional[str]:
    candidates = []
    for page in crawled_pages:
        content = page["content"]
        for i, heading in enumerate(content["headings"][:10]):
            if i < len(content["links"]):
                link = content["links"][i]
                if not link.startswith("http"):
                    link = urljoin(page["url"], link)
                candidates.append(f"{heading} -> {link}")

    if not candidates:
        return None

    prompt = f"""
The user asked a follow-up question:

"{follow_up}"

Here are some article/page options (title -> link):

{chr(10).join(candidates[:40])}

Which single link best matches the follow-up?
Only return the link (nothing else).
If none are relevant, return "NONE".
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    choice = resp.choices[0].message.content.strip()
    return None if choice.upper() == "NONE" else choice


# ---- Utility: render assistant messages with table parsing ----
def render_assistant_message(msg: str):
    # Detect table-like text (rows with '|' separators)
    lines = msg.splitlines()
    if any("|" in line for line in lines):
        try:
            data = [line.split("|") for line in lines if "|" in line]
            df = pd.DataFrame(data[1:], columns=[c.strip() for c in data[0]])
            st.chat_message("assistant").table(df)
            return
        except Exception:
            pass
    # fallback normal text
    st.chat_message("assistant").write(msg)


# ---- Streamlit Chat UI ----
st.set_page_config(page_title="Web Chat Crawler v7", layout="wide")
st.title("Web Chat Crawler (v7)")

if "history" not in st.session_state:
    st.session_state.history = []
if "crawled" not in st.session_state:
    st.session_state.crawled = None
if "base_url" not in st.session_state:
    st.session_state.base_url = None
if "query" not in st.session_state:
    st.session_state.query = None

with st.sidebar:
    st.header("Setup")
    base_url = st.text_input("Website URL", value="https://www.espncricinfo.com")
    query = st.text_input("Initial Query", value="Any news related to cricket?")
    max_pages = st.slider("Max Pages", 1, 10, 3)
    max_depth = st.slider("Max Depth", 1, 3, 1)

    if st.button("Fetch & Summarize"):
        st.session_state.base_url = base_url
        st.session_state.query = query
        crawled = crawl_website(base_url, max_pages=max_pages, max_depth=max_depth)
        st.session_state.crawled = crawled
        if crawled:
            summary = summarize_crawled_content(query, crawled)
            st.session_state.history.append(("assistant", summary))
        else:
            st.session_state.history.append(("assistant", "Could not crawl site."))

# Display chat
for role, msg in st.session_state.history:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        render_assistant_message(msg)

if st.session_state.crawled:
    user_input = st.chat_input("Ask a follow-up question...")
    if user_input:
        st.session_state.history.append(("user", user_input))
        full_query = f"{st.session_state.query} -> {user_input}"
        match = choose_relevant_link(full_query, user_input, st.session_state.crawled, st.session_state.base_url)
        if match:
            detail = summarize_article(full_query, match)
            st.session_state.history.append(("assistant", detail))
        else:
            st.session_state.history.append(("assistant", "No relevant link found."))
        st.rerun()
