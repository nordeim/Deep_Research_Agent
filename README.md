**tool based approach for a Deep Research agent**

```
pip install pdfplumber
pip install python-magic unstructured python-docx openpyxl
pip install --no-cache-dir --force-reinstall sentence-transformers torch
pip install --no-cache-dir --force-reinstall sentence-transformers torch
pip install langchain_anthropic
pip install langchain_openai langchain_google_genai
pip install -U open-webui
pip install -U crawl4ai
pip install -U aider-chat
pip install -U smolagents
pip install matplotlib
pip install ratelimiter
pip install pyrate_limiter
pip install wikipedia
pip install arxiv
pip install google-search-results
pip install wolframalpha

WOLFRAM_ALPHA_APPID=your-wolfram-alpha-app-id
SERP_API_KEY=your-serp-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key

python deep_research-v9.py
```
![image](https://github.com/user-attachments/assets/d6027452-8b0b-478c-836d-70b6152e4888)

```
$ python deep_research-v9.py
WARNING:root:BraveSearchRun is not available. Ensure langchain-community is installed with Brave Search support.
Warning: BraveSearchRun is not available. Ensure langchain-community is installed with Brave Search support.
* Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
WARNING:root:BraveSearchRun is not available. Ensure langchain-community is installed with Brave Search support.
Warning: BraveSearchRun is not available. Ensure langchain-community is installed with Brave Search support.


> Entering new AgentExecutor chain...

```json
{
  "topic": "OpenAI Agents SDK",
  "summary": "The research aimed to gather information about the OpenAI Agents SDK. Initial web searches and Wikipedia lookups did not yield specific results for an \"OpenAI Agents SDK.\" Further investigation suggests that there is no officially recognized SDK with this exact name. However, OpenAI offers various APIs and libraries that facilitate agent development, such as the OpenAI API (including Assistants API) and libraries like LangChain which interact with OpenAI models. The findings suggest that while a dedicated \"Agents SDK\" doesn't exist, the broader ecosystem provides ample tools for building AI agents using OpenAI's technologies.",
  "steps": [
    {
      "step_name": "Understand and Clarify Query",
      "tools_used": [],
      "output": "Clearly defined research question and initial research objectives.",
      "objective": "Analyze the research query, identify key concepts, and clarify any ambiguities. Define the specific research question to be answered.",
      "sources": [],
      "key_findings": [],
      "search_queries": []
    },
    {
      "step_name": "Initial Research",
      "tools_used": [
        "web_search",
        "wikipedia"
      ],
      "output": "Summary of background information, identification of key terms and concepts, and initial sources.",
      "objective": "Gather preliminary information and context on the topic. Use broad web searches and Wikipedia to get an overview and identify key areas.",
      "sources": [
        {
          "url": "https://platform.openai.com/docs/assistants/overview",
          "source_type": "web",
          "title": "Assistants API - OpenAI Platform",
          "authors": [
            "OpenAI"
          ],
          "relevance": 0.8,
          "publication_date": "",
          "credibility_score": 0.9,
          "confidence_score": 0.8
        }
      ],
      "key_findings": [
        "Initial searches for \"OpenAI Agents SDK\" did not return specific results for an official SDK.",
        "OpenAI provides APIs, including the Assistants API, which can be used for building agent-like functionalities."
      ],
      "search_queries": [
        "OpenAI Agents SDK",
        "OpenAI agent development tools",
        "OpenAI Assistants API"
      ]
    },
    {
      "step_name": "Academic Literature Review",
      "tools_used": [
        "arxiv_search",
        "google_scholar"
      ],
      "output": "List of relevant academic papers, key findings from these papers, and identification of leading researchers or institutions.",
      "objective": "Explore academic databases for in-depth, peer-reviewed research. Use arXiv and Google Scholar to find relevant papers and scholarly articles.",
      "sources": [],
      "key_findings": [
        "No specific academic papers were found directly referencing an \"OpenAI Agents SDK.\"",
        "Searches related to OpenAI APIs and agent development yielded broader results on AI agent frameworks and applications."
      ],
      "search_queries": [
        "OpenAI Agents SDK",
        "OpenAI API agent development",
        "LLM agent frameworks"
      ]
    },
    {
      "step_name": "Factual Verification and Data Analysis",
      "tools_used": [
        "web_search"
      ],
      "output": "Verified facts and data, resolution of any conflicting information, and increased confidence in research findings.",
      "objective": "Verify key facts, statistics, and data points using reliable sources like Wolfram Alpha and targeted web searches. Cross-reference information from different sources to ensure accuracy.",
      "sources": [
        {
          "url": "https://www.langchain.com/",
          "source_type": "web",
          "relevance": 0.7,
          "authors": [],
          "publication_date": "",
          "credibility_score": 0.8,
          "confidence_score": 0.7,
          "title": "LangChain"
        }
      ],
      "key_findings": [
        "Confirmed that there's no official product named \"OpenAI Agents SDK.\"",
        "Identified that the OpenAI Assistants API and other libraries (e.g. LangChain) are relevant for agent development with OpenAI models."
      ],
      "search_queries": [
        "\"OpenAI Agents SDK\" official documentation",
        "alternatives to OpenAI Agents SDK"
      ]
    },
    {
      "step_name": "Synthesis and Report Generation",
      "tools_used": [],
      "output": "Comprehensive research report in JSON format, including summary, detailed steps, sources, quality score, and uncertainty areas.",
      "objective": "Synthesize all gathered information into a coherent and structured research report.",
      "sources": [],
      "key_findings": [],
      "search_queries": []
    }
  ],
  "quality_score": 0.85,
  "uncertainty_areas": [
    "The exact functionalities and scope envisioned for an 'Agents SDK' by the user remain unclear. While OpenAI offers tools for agent development, the specific features implied by the query are not fully addressed by existing products."
  ],
  "timestamp": "2024-02-08T21:28:40.201228",
  "research_question": "What is the OpenAI Agents SDK and what are its capabilities?",
    "methodology": "Tool-based Agent with Multi-Source Verification"
}
```

> Finished chain.
```

### **1. Core Enhancements**
- **Error Handling & Validation**:
  - Stricter API key validation with descriptive error messages.
  - Graceful degradation for parsing failures (attempts JSON extraction if strict parsing fails).
  - Explicit error handling for Wolfram Alpha and Anthropic API initialization.

- **Rate Limiting**:
  - Added `pyrate_limiter` for web (`DuckDuckGo`) and academic (`Google Scholar`) searches to prevent API abuse.

- **Caching Strategy**:
  - Enhanced caching with `diskcache` for faster repeated queries and reduced API load.

---

### **2. Research Quality Improvements**
- **Source Credibility**:
  - **New Metrics**: 
    - `confidence_score` (metadata extraction reliability).
    - `academic_ratio` (weighting for peer-reviewed/arXiv sources).
    - `verification_score` (counts cross-checking steps).
  - **Refined Scoring**:
    - Diversity metric now accounts for 5 source types (previously 4).
    - Recency scoring parses dates more robustly (supports "Jan 2024", "2024", etc.).

- **Metadata Extraction**:
  - Improved regex patterns for titles, authors, and dates across academic sources (arXiv, Scholar).
  - Citation count parsing now boosts credibility scores (scaled by citation impact).

---

### **3. Tool & Workflow Additions**
- **New Wolfram Alpha Integration**:
  - Factual/computational queries (e.g., math, physics) with high credibility scores.
  - Directly returns structured data for verification.

- **Structured Research Pipeline**:
  - Defined 5-step methodology (background → academic → verification → synthesis).
  - Explicit tracking of `research_question`, `methodology`, and `uncertainty_areas`.

---

### **4. Output & Interface**
- **Markdown Reporting**:
  - Quality metrics displayed as gauges (e.g., `Overall Score: 0.85/1.00`).
  - Detailed per-step breakdown (objectives, tools, sources).
  - Highlighted uncertainty areas and researcher notes.

- **Gradio UI Updates**:
  - Added `research_question` input field for guided research.
  - Example queries now include both a topic and a focused question (e.g., "Quantum computing applications in cryptography" → "What are the most promising applications...?").

---

### **5. Code Refactoring**
- **Modularization**:
  - Separated tool wrappers (`_web_search_wrapper`, `_wiki_wrapper`) for maintainability.
  - Consolidated metadata extraction logic into reusable methods.

- **Dependencies**:
  - Updated to `pydantic` v2 (using `field_validator` instead of deprecated `validator`).
  - Added `langchain_anthropic` for Claude 3.5 Sonnet integration.

---

### **Impact of Changes**
- **Reliability**: Reduced API errors and better handling of edge cases (e.g., missing metadata).
- **Transparency**: Clearer quality metrics and methodology documentation.
- **Usability**: Improved UI guidance and structured outputs for end-users.

This version is optimized for **academic rigor**, making it ideal for deep-dive research requiring source diversity, factual verification, and reproducibility.

---
https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%221uN1O1l237lD2SNrHrS5WA0fL8ipYVoTo%22%5D,%22action%22:%22open%22,%22userId%22:%22103961307342447084491%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing

https://www.perplexity.ai/search/you-are-a-deep-thinking-ai-you-Rp9yM6M0QxCTnDHBpXakfw
