## Deep Research学习

本仓库代码来源于 [OpenDeepResearch](https://github.com/mshumer/OpenDeepResearcher)，建议先阅读原仓库的 README

代码目前为止只做了一些小小的<b>改动</b>：
1. 修改了 SERPAPI_URL 的API调用地址和API请求/应答处理（原仓库的URL API调用在我实测的时候并不能跑成功）
2. 将原仓库默认的 OpenRouter 调用 Claud 模型改成了调用本地的模型（可以使用ollama或LMStudio在本机加载模型），也可以通过修改 OPENROUTER_URL 和 DEFAULT_MODEL 切换使用别的LLM模型

### 部署依赖
除了安装必要的pip包 `pip install nest_asyncio gradio aiohttp` 以外，主要依赖：
1. [Serper](https://serper.dev) API
2. [Jina](https://jina.ai) API - 后续尝试是否能用firecrawl替代
3. 大模型 API（本地<默认>或网络API）

### 代码分析与提示词
由于代码很短，可以直接让AI帮我们解读工作原理，核心在于5个提示词，分别用于：

这段代码实现了一个研究助手，它使用大型语言模型（LLM）和搜索引擎来自动进行研究并生成报告。代码中定义了几个关键的 LLM 提示词，每个提示词都负责一个特定的任务。下面我们来详细解读每个提示词的作用：

1. **`generate_search_queries_async` 函数中的提示词:**

   ```python
   prompt = (
       "You are an expert research assistant. Given the user's query, generate up to four distinct, "
       "precise search queries that would help gather complete information on the topic. "
       "Return only a Python list of strings, for example: ['query1', 'query2', 'query3']."
   )
   ```

   **作用:** 这个提示词用于**生成初始的搜索查询**。
   - **角色设定:**  将 LLM 设定为 "专家研究助手"，强调其在研究方面的专业能力。
   - **任务描述:**  明确要求 LLM 基于用户输入的查询，生成最多四个不同的、精确的搜索查询。
   - **输出格式要求:**  强调输出必须是 Python 字符串列表，例如 `['query1', 'query2', 'query3']`。这确保了代码可以方便地解析和使用 LLM 生成的搜索查询。
   - **目的:**  这是研究过程的第一步，目的是将用户提出的宽泛研究主题转化为具体的、可执行的搜索关键词，以便从搜索引擎中获取相关信息。

2. **`is_page_useful_async` 函数中的提示词:**

   ```python
   prompt = (
       "You are a critical research evaluator. Given the user's query and the content of a webpage, "
       "determine if the webpage contains information that is useful for addressing the query. "
       "Respond with exactly one word: 'Yes' if the page is useful, or 'No' if it is not. Do not include any extra text."
   )
   ```

   **作用:** 这个提示词用于**判断网页内容是否对回答用户查询有用**。
   - **角色设定:** 将 LLM 设定为 "严格的研究评估者"，强调其评估的严谨性。
   - **输入信息:**  提供用户查询和网页内容（截取前 20000 字符）。
   - **任务描述:**  要求 LLM 判断网页内容是否包含有助于解决用户查询的信息。
   - **输出格式要求:**  强制 LLM 仅输出一个词 "Yes" 或 "No"，并且不包含任何额外的文本。这使得代码可以简单地判断网页的有用性。
   - **目的:**  在抓取了网页内容之后，需要过滤掉无关的网页，只保留包含有价值信息的网页，从而提高后续信息提取的效率和准确性。

3. **`extract_relevant_context_async` 函数中的提示词:**

   ```python
   prompt = (
       "You are an expert information extractor. Given the user's query, the search query that led to this page, "
       "and the webpage content, extract all pieces of information that are useful for answering the user's query. "
       "Return only the relevant context as plain text without extra commentary."
   )
   ```

   **作用:** 这个提示词用于**从有用的网页中提取与用户查询相关的上下文信息**。
   - **角色设定:** 将 LLM 设定为 "专家信息提取器"，强调其信息提取的专业性。
   - **输入信息:** 提供用户查询、导致该网页的搜索查询，以及网页内容（截取前 20000 字符）。
   - **任务描述:**  要求 LLM 从网页内容中提取所有对回答用户查询有用的信息片段。
   - **输出格式要求:**  要求仅返回相关的上下文信息，以纯文本形式，不包含额外的评论或解释。
   - **目的:**  在确认网页有用后，需要进一步从网页中提取出具体的信息，这些信息将作为构建最终报告的素材。同时，提供搜索查询可以帮助 LLM 更好地理解网页的上下文和相关性。

4. **`get_new_search_queries_async` 函数中的提示词:**

   ```python
   prompt = (
       "You are an analytical research assistant. Based on the original query, the search queries performed so far, "
       "and the extracted contexts from webpages, decide if further research is needed. "
       "If further research is needed, provide up to four new search queries as a Python list (for example, "
       "['new query1', 'new query2']). If you believe no further research is needed, respond with exactly <done>."
       "\nOutput only a Python list or the token <done> without any extra text."
   )
   ```

   **作用:** 这个提示词用于**判断是否需要进行进一步的研究，并生成新的搜索查询（如果需要）**。
   - **角色设定:** 将 LLM 设定为 "分析型研究助手"，强调其分析和决策能力。
   - **输入信息:** 提供用户查询、之前执行的搜索查询列表，以及从网页中提取的上下文信息。
   - **任务描述:**  要求 LLM 基于以上信息，判断是否需要进行更深入的研究。
   - **输出格式要求:**
     - 如果需要进一步研究，则返回新的搜索查询列表 (Python list)。
     - 如果不需要进一步研究，则返回文本 `<done>`。
     - 强调输出只能是列表或 `<done>`，不包含任何额外文本。
   - **目的:**  这是一个迭代研究过程的关键步骤。它让 LLM 基于已有的信息来决定是否需要扩大搜索范围，以及如何扩大搜索范围。通过生成新的搜索查询，可以使研究更加深入和全面。当 LLM 判断信息已经足够或无法再通过搜索获取更多有价值的信息时，它会指示研究过程结束。

5. **`generate_final_report_async` 函数中的提示词:**

   ```python
   prompt = (
       "You are an expert researcher and report writer. Based on the gathered contexts below and the original query, "
       "write a complete, well-structured, and detailed report that addresses the query thoroughly. "
       "Include all useful insights and conclusions without extra commentary."
   )
   ```

   **作用:** 这个提示词用于**基于收集到的上下文信息生成最终的研究报告**。
   - **角色设定:** 将 LLM 设定为 "专家研究员和报告撰写者"，强调其报告撰写能力。
   - **输入信息:** 提供用户查询和所有收集到的相关上下文信息。
   - **任务描述:**  要求 LLM 基于这些信息，撰写一份完整、结构良好、详细的报告，充分解答用户查询。
   - **输出内容要求:**  报告需要包含所有有用的见解和结论，但不包含额外的评论性内容。
   - **目的:**  这是研究过程的最后一步，目的是将所有收集到的零散信息整合起来，形成一份结构化、易于理解的报告，直接回答用户的原始查询。

**总结:**

这段代码中的 LLM 提示词被精心设计，每个提示词都专注于一个特定的研究子任务，并明确了 LLM 的角色、任务描述、输入信息和输出格式要求。这种分解任务的方式使得整个研究流程更加模块化和可控，也更好地利用了 LLM 在不同方面的能力：

- **生成搜索查询**:  利用 LLM 的创意和语言理解能力。
- **评估网页有用性**:  利用 LLM 的判断和理解能力。
- **提取相关信息**:  利用 LLM 的信息抽取和总结能力。
- **决定研究方向和迭代**:  利用 LLM 的分析和决策能力。
- **撰写最终报告**:  利用 LLM 的文本生成和组织能力。

通过这些精心设计的提示词，代码实现了一个相对智能和自动化的研究助手。