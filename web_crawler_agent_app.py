import os
import requests
from bs4 import BeautifulSoup
import streamlit as st
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
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        return f"Error fetching {url}: {e}"


def extract_content(html: str):
    soup = BeautifulSoup(html, "html.parser")
    headings = [h.get_text(" ", strip=True) for h in soup.find_all(["h1", "h2", "h3"])]
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    links = [a["href"] for a in soup.find_all("a", href=True)]
    return {"headings": headings, "paragraphs": paragraphs, "links": links}


# ---- LLM helpers ----
def summarize_content(query: str, content: dict) -> str:
    text_blocks = content["headings"] + content["paragraphs"]
    if not text_blocks:
        return "No content found on the page."

    joined_text = "\n".join(text_blocks[:50])
    prompt = f"""
The user is asking:

"{query}"

Here is some content extracted from a webpage:

{joined_text}

From this content, extract only the parts that are relevant to the query.
Then, write a clear and concise summary (3–5 sentences).
If multiple relevant items exist, list them with short summaries.
If nothing is relevant, say: "No relevant content found."
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


def summarize_article(query: str, url: str) -> str:
    html = fetch_page(url)
    content = extract_content(html)
    text = " ".join(content["paragraphs"][:30])

    if not text:
        return "No content available in this article."

    prompt = f"""
Summarize the following article in relation to the user's query:

Query: "{query}"

Article Content:
{text}

Provide a concise summary (3–5 sentences).
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


def choose_relevant_link(query: str, follow_up: str, content: dict, base_url: str) -> Optional[str]:
    if not content["headings"] or not content["links"]:
        return None

    candidates = []
    for i, heading in enumerate(content["headings"][:20]):
        if i < len(content["links"]):
            link = content["links"][i]
            if not link.startswith("http"):
                if base_url.endswith("/"):
                    link = base_url[:-1] + link
                else:
                    link = base_url + link
            candidates.append(f"{heading} -> {link}")

    if not candidates:
        return None

    prompt = f"""
The user asked a follow-up question:

"{follow_up}"

Here are some article options (title -> link):

{chr(10).join(candidates)}

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


# ---- Streamlit Chat UI ----
st.set_page_config(page_title="Web Chat Crawler", page_icon="", layout="wide")
st.title("Web Chat Crawler")

if "history" not in st.session_state:
    st.session_state.history = []
if "content" not in st.session_state:
    st.session_state.content = None
if "base_url" not in st.session_state:
    st.session_state.base_url = None
if "query" not in st.session_state:
    st.session_state.query = None

# Sidebar for base setup
with st.sidebar:
    st.header("Setup")
    base_url = st.text_input("Website URL", value="https://www.espncricinfo.com")
    query = st.text_input("Initial Query", value="Any news related to cricket?")
    if st.button("Fetch & Summarize"):
        html = fetch_page(base_url)
        st.session_state.base_url = base_url
        st.session_state.query = query
        if html:
            content = extract_content(html)
            st.session_state.content = content
            summary = summarize_content(query, content)
            st.session_state.history.append(("assistant", summary))
        else:
            st.session_state.history.append(("assistant", "Could not fetch base page."))


# Display conversation
for role, msg in st.session_state.history:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)

# Input box for chat
if st.session_state.content:
    user_input = st.chat_input("Ask a follow-up question...")
    if user_input:
        st.session_state.history.append(("user", user_input))
        match = choose_relevant_link(
            st.session_state.query, user_input, st.session_state.content, st.session_state.base_url
        )
        if match:
            detail = summarize_article(st.session_state.query, match)
            st.session_state.history.append(("assistant", detail))
        else:
            st.session_state.history.append(("assistant", "No relevant link found."))
        st.rerun()

