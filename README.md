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

python deep_research-v5.py
```
![image](https://github.com/user-attachments/assets/b498321c-1730-4e7b-99e6-4dc11f48c47d)

The `deep_research-v4.py` update introduces significant improvements over `deep_research-v2.py`, focusing on **error handling, feature enhancements, and research quality**.

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
