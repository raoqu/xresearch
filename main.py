import nest_asyncio
nest_asyncio.apply()

import asyncio
import aiohttp
import gradio as gr
import json

# ---------------------------
# Configuration Constants
# ---------------------------
OPENROUTER_API_KEY = "..." # Replace with your OpenRouter API key
SERPAPI_API_KEY = "..." # Replace with your SERPAPI API key
JINA_API_KEY = "..." # Replace with your Jina API key

OPENROUTER_URL = "http://127.0.0.1:1234/v1/chat/completions"
SERPAPI_URL = "https://google.serper.dev/search"
JINA_BASE_URL = "https://r.jina.ai/"

DEFAULT_MODEL = "anthropic/claude-3.5-haiku"

# -------------------------------
# Asynchronous Helper Functions
# -------------------------------

async def call_openrouter_async(session, messages, model=DEFAULT_MODEL):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "X-Title": "OpenDeepResearcher, by Matt Shumer",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages
    }
    try:
        async with session.post(OPENROUTER_URL, headers=headers, json=payload) as resp:
            if resp.status == 200:
                result = await resp.json()
                try:
                    return result['choices'][0]['message']['content']
                except (KeyError, IndexError):
                    print("Unexpected OpenRouter response structure:", result)
                    return None
            else:
                text = await resp.text()
                print(f"OpenRouter API error: {resp.status} - {text}")
                return None
    except Exception as e:
        print("Error calling OpenRouter:", e)
        return None

async def generate_search_queries_async(session, user_query):
    prompt = (
        "You are an expert research assistant. Given the user's query, generate up to four distinct, "
        "precise search queries that would help gather complete information on the topic. "
        "Return only a Python list of strings, for example: ['query1', 'query2', 'query3']."
    )
    messages = [
        {"role": "system", "content": "You are a helpful and precise research assistant."},
        {"role": "user", "content": f"User Query: {user_query}\n\n{prompt}"}
    ]
    response = await call_openrouter_async(session, messages)
    if response:
        try:
            search_queries = eval(response)
            if isinstance(search_queries, list):
                return search_queries
            else:
                print("LLM did not return a list. Response:", response)
                return []
        except Exception as e:
            print("Error parsing search queries:", e, "\nResponse:", response)
            return []
    return []

async def perform_search_async(session, query):
    params = {
        "q": query
    }
    headers = {
        'X-API-KEY': SERPAPI_API_KEY, 
        'Content-Type': 'application/json'
    }
    try:
        async with session.get(SERPAPI_URL, params=params, headers=headers) as resp:
            if resp.status == 200:
                results = await resp.json()
                if "organic" in results:
                    links = [item.get("link") for item in results["organic"] if "link" in item]
                    return links
                else:
                    print("No organic results in SERPAPI response.")
                    return []
            else:
                text = await resp.text()
                print(f"SERPAPI error: {resp.status} - {text}")
                return []
    except Exception as e:
        print("Error performing SERPAPI search:", e)
        return []

async def fetch_webpage_text_async(session, url):
    full_url = f"{JINA_BASE_URL}{url}"
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}"
    }
    try:
        async with session.get(full_url, headers=headers) as resp:
            if resp.status == 200:
                return await resp.text()
            else:
                text = await resp.text()
                print(f"Jina fetch error for {url}: {resp.status} - {text}")
                return ""
    except Exception as e:
        print("Error fetching webpage text with Jina:", e)
        return ""

async def is_page_useful_async(session, user_query, page_text):
    prompt = (
        "You are a critical research evaluator. Given the user's query and the content of a webpage, "
        "determine if the webpage contains information that is useful for addressing the query. "
        "Respond with exactly one word: 'Yes' if the page is useful, or 'No' if it is not. Do not include any extra text."
    )
    messages = [
        {"role": "system", "content": "You are a strict and concise evaluator of research relevance."},
        {"role": "user", "content": f"User Query: {user_query}\n\nWebpage Content (first 20000 characters):\n{page_text[:20000]}\n\n{prompt}"}
    ]
    response = await call_openrouter_async(session, messages)
    if response:
        answer = response.strip()
        if answer in ["Yes", "No"]:
            return answer
        else:
            if "Yes" in answer:
                return "Yes"
            elif "No" in answer:
                return "No"
    return "No"

async def extract_relevant_context_async(session, user_query, search_query, page_text):
    prompt = (
        "You are an expert information extractor. Given the user's query, the search query that led to this page, "
        "and the webpage content, extract all pieces of information that are useful for answering the user's query. "
        "Return only the relevant context as plain text without extra commentary."
    )
    messages = [
        {"role": "system", "content": "You are an expert in extracting and summarizing relevant information."},
        {"role": "user", "content": f"User Query: {user_query}\nSearch Query: {search_query}\n\nWebpage Content (first 20000 characters):\n{page_text[:20000]}\n\n{prompt}"}
    ]
    response = await call_openrouter_async(session, messages)
    if response:
        return response.strip()
    return ""

async def get_new_search_queries_async(session, user_query, previous_search_queries, all_contexts):
    context_combined = "\n".join(all_contexts)
    prompt = (
        "You are an analytical research assistant. Based on the original query, the search queries performed so far, "
        "and the extracted contexts from webpages, decide if further research is needed. "
        "If further research is needed, provide up to four new search queries as a Python list (for example, "
        "['new query1', 'new query2']). If you believe no further research is needed, respond with exactly <done>."
        "\nOutput only a Python list or the token <done> without any extra text."
    )
    messages = [
        {"role": "system", "content": "You are a systematic research planner."},
        {"role": "user", "content": f"User Query: {user_query}\nPrevious Search Queries: {previous_search_queries}\n\nExtracted Relevant Contexts:\n{context_combined}\n\n{prompt}"}
    ]
    response = await call_openrouter_async(session, messages)
    if response:
        cleaned = response.strip()
        if cleaned == "<done>":
            return "<done>"
        try:
            new_queries = eval(cleaned)
            if isinstance(new_queries, list):
                return new_queries
            else:
                print("LLM did not return a list for new search queries. Response:", response)
                return []
        except Exception as e:
            print("Error parsing new search queries:", e, "\nResponse:", response)
            return []
    return []

async def generate_final_report_async(session, user_query, all_contexts):
    context_combined = "\n".join(all_contexts)
    prompt = (
        "You are an expert researcher and report writer. Based on the gathered contexts below and the original query, "
        "write a complete, well-structured, and detailed report that addresses the query thoroughly. "
        "Include all useful insights and conclusions without extra commentary."
    )
    messages = [
        {"role": "system", "content": "You are a skilled report writer."},
        {"role": "user", "content": f"User Query: {user_query}\n\nGathered Relevant Contexts:\n{context_combined}\n\n{prompt}"}
    ]
    report = await call_openrouter_async(session, messages)
    return report

async def process_link(session, link, user_query, search_query, log):
    log.append(f"Fetching content from: {link}")
    page_text = await fetch_webpage_text_async(session, link)
    if not page_text:
        log.append(f"Failed to fetch content from: {link}")
        return None
    usefulness = await is_page_useful_async(session, user_query, page_text)
    log.append(f"Page usefulness for {link}: {usefulness}")
    if usefulness == "Yes":
        context = await extract_relevant_context_async(session, user_query, search_query, page_text)
        if context:
            log.append(f"Extracted context from {link} (first 200 chars): {context[:200]}")
            return context
    return None

# -----------------------------
# Main Asynchronous Routine
# -----------------------------

async def async_research(user_query, iteration_limit):
    aggregated_contexts = []
    all_search_queries = []
    log_messages = []  # List to store intermediate steps
    iteration = 0

    async with aiohttp.ClientSession() as session:
        log_messages.append("Generating initial search queries...")
        new_search_queries = await generate_search_queries_async(session, user_query)
        if not new_search_queries:
            log_messages.append("No search queries were generated by the LLM. Exiting.")
            return "No search queries were generated by the LLM. Exiting.", "\n".join(log_messages)
        all_search_queries.extend(new_search_queries)
        log_messages.append(f"Initial search queries: {new_search_queries}")

        while iteration < iteration_limit:
            log_messages.append(f"\n=== Iteration {iteration + 1} ===")
            iteration_contexts = []
            search_tasks = [perform_search_async(session, query) for query in new_search_queries]
            search_results = await asyncio.gather(*search_tasks)
            unique_links = {}
            for idx, links in enumerate(search_results):
                query_used = new_search_queries[idx]
                for link in links:
                    if link not in unique_links:
                        unique_links[link] = query_used

            log_messages.append(f"Aggregated {len(unique_links)} unique links from this iteration.")
            link_tasks = [
                process_link(session, link, user_query, unique_links[link], log_messages)
                for link in unique_links
            ]
            link_results = await asyncio.gather(*link_tasks)
            for res in link_results:
                if res:
                    iteration_contexts.append(res)

            if iteration_contexts:
                aggregated_contexts.extend(iteration_contexts)
                log_messages.append(f"Found {len(iteration_contexts)} useful contexts in this iteration.")
            else:
                log_messages.append("No useful contexts were found in this iteration.")

            new_search_queries = await get_new_search_queries_async(session, user_query, all_search_queries, aggregated_contexts)
            if new_search_queries == "<done>":
                log_messages.append("LLM indicated that no further research is needed.")
                break
            elif new_search_queries:
                log_messages.append(f"LLM provided new search queries: {new_search_queries}")
                all_search_queries.extend(new_search_queries)
            else:
                log_messages.append("LLM did not provide any new search queries. Ending the loop.")
                break

            iteration += 1

        log_messages.append("\nGenerating final report...")
        final_report = await generate_final_report_async(session, user_query, aggregated_contexts)
        return final_report, "\n".join(log_messages)

def run_research(user_query, iteration_limit=10):
    return asyncio.run(async_research(user_query, iteration_limit))

# -----------------------------
# Gradio UI Setup
# -----------------------------

def gradio_run(user_query, iteration_limit):
    try:
        final_report, logs = run_research(user_query, int(iteration_limit))
        return final_report, logs
    except Exception as e:
        return f"An error occurred: {e}", ""

iface = gr.Interface(
    fn=gradio_run,
    inputs=[
        gr.Textbox(lines=2, label="Research Query/Topic"),
        gr.Number(value=10, label="Max Iterations")
    ],
    outputs=[
        gr.Textbox(label="Final Report"),
        gr.Textbox(label="Intermediate Steps Log")
    ],
    title="Research Assistant",
    description="Enter your query and a maximum iteration count to generate a report. The log will show the steps taken."
)

iface.launch()