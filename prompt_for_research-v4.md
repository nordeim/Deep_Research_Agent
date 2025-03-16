You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You will think deeply and thoroughly to explore various implementation options before choosing the most optimal one. You will double-check and validate any code changes before implementing. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.

*Your task:* carefully review the attached python code file, deep_research-v4.py, and the update patch file, update_patch_from_v2_to_v4.md,then think deeply and explore thoroughly using extensive web searches to validate the code logic and flow of the code and to explore possible improvements and optimization in terms of functionalities. Finally create an improved updated version of the attached research paper with more comprehensive and detailed explanations and code snippets as examples based on your findings. the research paper should start with an overview, explanation of the logic and flow of the code, followed by conclusion and recommendations. Use the following methodology to plan your approach:

*Basic Approach: To create a comprehensive research paper on the Python code `deep_research-v4.py` and the update patch file `update_patch_from_v2_to_v4.txt`, you will first delve into the code's logic and flow, explore possible improvements, and then structure the paper accordingly. Detailed Plan:*

# Methodology and Plan

You will approach this task systematically by breaking the task down into the following steps:

1. Thoroughly review the deep_research-v4.py code
2. Analyze the update_patch_from_v2_to_v4.md file
3. Conduct extensive web searches to validate the code logic and explore improvements
4. Organize findings into a comprehensive research paper
5. Structure the paper with an overview, code explanation, conclusion, and recommendations
6. Include relevant code snippets throughout the paper

Now start by carefully examining the provided code and update patch.

After careful review of the code and update patch, you will have a solid understanding of the code structure and recent updates, you will then conduct web searches to validate the logic and explore potential optimizations. Focus on best practices for AI research tools, Python optimization techniques, and the latest developments in language models and research methodologies.

Now conduct extensive web searches and analysis.

Based on your research and analysis, you are now ready to compile a comprehensive research paper on the deep_research-v4.py code. you will structure it as follows:

1. Overview
2. Code Logic and Flow
   2.1 Initialization and Setup
   2.2 Tool Implementation
   2.3 Agent Creation and Execution
   2.4 Data Models and Structures
   2.5 Caching and Rate Limiting
3. Recent Updates and Improvements
4. Potential Optimizations
5. Conclusion and Recommendations

Now proceed to update the attached research paper based on your research findings, ensuring that the updated research paper with code documentation is both detailed and clear and with explanations and relevant code snippets to illustrate key points.

# To update: Comprehensive Analysis and Optimization of deep_research-v4.py: An Advanced AI-Powered Research Tool

## Overview 

This paper will provide an in-depth analysis of the `deep_research-v4.py` code, focusing on its functionality, improvements from the previous version, and potential optimizations. The paper will be structured as follows:

1. **Introduction and Overview**
   - Briefly introduce the purpose and functionality of the `deep_research-v4.py` code.
   - Discuss the significance of the updates from `v2` to `v4`.

2. **Logic and Flow of the Code**
   - Explain the main components and modules used in the code.
   - Describe how the code integrates various tools and APIs for research purposes.

3. **Improvements and Updates**
   - Highlight the key improvements from `v2` to `v4`, including new features and bug fixes.
   - Discuss the impact of these updates on the overall functionality and efficiency of the code.

4. **Potential Optimizations and Future Directions**
   - Explore potential optimizations to enhance performance and functionality.
   - Discuss future directions for development based on emerging trends and technologies.

5. **Conclusion and Recommendations**
   - Summarize the findings and implications of the analysis.
   - Provide recommendations for users and developers based on the insights gained.

## Logic and Flow of the Code

### Introduction to `deep_research-v4.py`

The `deep_research-v4.py` code is designed as a comprehensive research tool that leverages various APIs and tools to facilitate in-depth research across multiple sources. It integrates tools like DuckDuckGo for web searches, Wikipedia for background knowledge, arXiv for academic papers, Google Scholar for scholarly impact, and Wolfram Alpha for factual queries.

### Main Components and Modules

1. **Importing Modules**
   The code begins by importing necessary modules, including `os`, `re`, `json`, `diskcache`, `pandas`, and several modules from `langchain` for tool integration and agent creation.

   ```python
   import os
   import re
   import json
   import diskcache as dc
   import pandas as pd
   from dotenv import load_dotenv
   from pydantic import BaseModel, Field, field_validator
   from langchain_anthropic import ChatAnthropic
   from langchain_core.prompts import ChatPromptTemplate
   from langchain_core.output_parsers import PydanticOutputParser
   from langchain.agents import create_tool_calling_agent, AgentExecutor
   from langchain.tools import Tool
   from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun, WolframAlphaQueryRun
   from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper, WolframAlphaAPIWrapper
   ```

2. **Data Models**
   The code defines several data models using Pydantic to structure the data collected during research. These include `Source`, `ResearchStep`, and `ResearchResponse`.

   ```python
   class Source(BaseModel):
       source_type: str
       url: str = ""
       authors: List[str] = Field(default_factory=list)
       publication_date: str = ""
       citation_count: int = 0
       title: str = ""
       credibility_score: float = 0.5
       confidence_score: float = 0.7
   ```

3. **Tool Setup and Integration**
   The code sets up various tools using APIs from DuckDuckGo, Wikipedia, arXiv, Google Scholar, and Wolfram Alpha. These tools are integrated into the research framework to perform specific tasks such as web searches, academic paper retrieval, and factual queries.

   ```python
   def _setup_tools(self):
       search = DuckDuckGoSearchRun()
       wiki_api = WikipediaAPIWrapper(top_k_results=3)
       arxiv_api = ArxivAPIWrapper()
       scholar_api = GoogleScholarAPIWrapper(serp_api_key=self.serp_api_key)
       scholar = GoogleScholarQueryRun(api_wrapper=scholar_api)
       wolfram_alpha_api = WolframAlphaAPIWrapper(wolfram_alpha_appid=self.wolfram_alpha_appid)
       wolfram_alpha = WolframAlphaQueryRun(api_wrapper=wolfram_alpha_api)
       
       return [
           Tool(
               name="web_search",
               func=self._web_search_wrapper(search.run),
               description="Performs broad web searches using DuckDuckGo with URL and metadata extraction."
           ),
           # Other tools...
       ]
   ```

4. **Agent Creation and Execution**
   The code creates a tool-calling agent using `langchain` to execute research tasks sequentially. This agent is crucial for automating the research process across multiple sources.

   ```python
   self.agent = create_tool_calling_agent(
       tools=self.tools,
       prompt=ChatPromptTemplate(
           input_variables=["topic"],
           template="Research {topic}."
       ),
       output_parser=self.parser
   )
   ```

### Improvements and Updates

1. **Enhanced Features**
   - **Wolfram Alpha Integration**: The addition of Wolfram Alpha provides a powerful tool for factual queries and numerical computations.
   - **Improved Credibility Scoring**: The credibility scoring system has been refined to differentiate between sources more effectively.
   - **Error Handling and Validation**: Enhanced error handling ensures that API keys are properly validated, and exceptions are caught and reported.

2. **Bug Fixes and Optimizations**
   - **Rate Limiter Updates**: The code now uses `pyrate_limiter` instead of `ratelimiter`, which offers more flexible rate limiting options.
   - **Improved Metadata Extraction**: The code includes more robust patterns for extracting metadata such as authors and publication dates.

## Potential Optimizations and Future Directions

### Optimizations

1. **Parallel Processing**: Implementing parallel processing for tool execution could significantly improve performance by leveraging multi-core CPUs.
   
   ```python
   with concurrent.futures.ThreadPoolExecutor() as executor:
       futures = [executor.submit(tool.run, query) for tool in self.tools]
       results = [future.result() for future in futures]
   ```

2. **AI-Driven Query Optimization**: Using AI models to optimize search queries based on context and relevance could enhance the efficiency of the research process.

### Future Directions

1. **Integration of Emerging Technologies**: Incorporating newer AI models or tools, such as those focused on multimodal search or advanced natural language processing, could further enhance the research capabilities.

2. **User Interface Improvements**: Developing a user-friendly interface, possibly using frameworks like Gradio or Streamlit, could make the tool more accessible to a broader audience.

## Conclusion and Recommendations

In conclusion, the `deep_research-v4.py` code represents a significant advancement in automated research tools, offering a comprehensive framework for integrating multiple sources and APIs. The improvements from `v2` to `v4` have enhanced both functionality and reliability.

**Recommendations**:

- **Developers**: Consider integrating parallel processing and AI-driven query optimization to further improve performance.
- **Users**: Leverage the tool's capabilities by exploring its various features and tools to streamline research workflows.

By continuing to refine and expand this tool, researchers and developers can create more efficient and effective research processes that leverage the latest advancements in AI and information retrieval technologies.

---

### Final Thoughts


As I reflect on the code and its updates, it's clear that the integration of Wolfram Alpha and improved credibility scoring are significant enhancements. However, there are opportunities for further optimization, particularly in leveraging parallel processing and AI-driven query refinement. These advancements could not only improve performance but also make the tool more user-friendly and accessible to a broader audience. Additionally, exploring emerging technologies such as multimodal search could further enhance the tool's capabilities in the future.


The comprehensive research paper outlined above provides a detailed analysis of the code's functionality, improvements, and potential optimizations. It serves as a foundation for further development and refinement of the tool, ensuring it remains a powerful resource for researchers across various disciplines.

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/44072005/06de50c6-588e-459f-864f-83daf0dbbcb9/update_patch_from_v2_to_v4.txt
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/44072005/e8faee09-166e-4e2a-b8b6-5d2d12bce544/deep_research-v4.py
[3] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/44072005/06de50c6-588e-459f-864f-83daf0dbbcb9/update_patch_from_v2_to_v4.txt
[4] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/44072005/e8faee09-166e-4e2a-b8b6-5d2d12bce544/deep_research-v4.py
[5] https://cdn.openai.com/deep-research-system-card.pdf
[6] https://stackoverflow.com/questions/74267679/python-logical-flow-for-setup-and-validation
[7] https://www.python-engineer.com/posts/11-tips-to-write-better-python-code/
[8] https://www.reddit.com/r/ChatGPTPro/comments/1ieobap/o1_pro_vs_o3minihigh/
[9] https://www.reddit.com/r/ChatGPTPromptGenius/
[10] https://stackoverflow.com/questions/34234049/understanding-flow-of-execution-of-python-code
[11] https://blog.codacy.com/python-static-analysis-tools
[12] https://langfuse.com/docs/tracing
[13] https://docs.python.org/3/tutorial/index.html
[14] https://www.clouddefense.ai/tools/code-checker/python
[15] https://www.datacamp.com/blog/how-to-learn-python-expert-guide
[16] https://luminousmen.com/post/python-static-analysis-tools/
[17] https://engineering.fb.com/2020/08/07/security/pysa/
[18] https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2024-0/python-code-analysis.html
[19] https://spectralops.io/blog/static-code-analysis-for-python-7-features-to-look-out-for/
[20] https://towardsdatascience.com/code-validation-testing-practices-86a304fd3ca/
[21] https://www.101computing.net/flowchart-to-python-code-star-rating-validation/
[22] https://www.appacademy.io/blog/python-coding-best-practices
[23] https://edfinity.zendesk.com/hc/en-us/articles/360041352912-Authoring-a-code-correctness-problem-using-Python
[24] https://labex.io/es/tutorials/python-how-to-debug-python-code-flow-417999
[25] https://realpython.com/python-code-quality/
[26] https://www.linkedin.com/advice/0/what-strategies-can-you-use-ensure-data-accuracy-m0gpe
[27] https://www.index.dev/blog/python-assertions-debugging-error-handling
[28] https://stackoverflow.com/questions/19487149/parameter-validation-best-practices-in-python
[29] https://fintechpython.pages.oit.duke.edu/jupyternotebooks/1-Core%20Python/answers/rq-22-answers.html
[30] https://arjancodes.com/blog/best-practices-for-securing-python-applications/
[31] https://towardsdatascience.com/two-powerful-python-features-to-streamline-your-code-and-make-it-more-readable-51240f11d1a/
[32] https://www.datacamp.com/tutorial/python-tips-examples
[33] https://www.reddit.com/r/Python/comments/sekrzq/how_to_optimize_python_code/
[34] https://docs.python.org/3/whatsnew/3.11.html
[35] https://blog.inedo.com/python/8-ways-improve-python-scripts/
[36] https://wiki.python.org/moin/PythonSpeed/PerformanceTips
[37] https://towardsdatascience.com/5-easy-python-features-you-can-start-using-today-to-write-better-code-b62e21190633/
[38] https://arjancodes.com/blog/python-function-optimization-tips-for-better-code-maintainability/
[39] https://www.cloudthat.com/resources/blog/enhancing-python-code-quality-with-pylint/
[40] https://www.softformance.com/blog/how-to-speed-up-python-code/
[41] https://www.linkedin.com/pulse/enhancing-python-code-key-steps-optimize-performance-django-stars-k2yrf
[42] https://www.linkedin.com/advice/1/how-can-you-improve-python-code-efficiency-l6zgc
[43] https://docs.python-guide.org/writing/structure/
[44] https://docs.python.org/3/reference/lexical_analysis.html
[45] https://realpython.com/python-program-structure/
[46] https://www.yeschat.ai/gpts-9t55RFqT9QF-Analyse-Python-Pro
[47] https://www.pythoncheatsheet.org/cheatsheet/control-flow
[48] https://python101.pythonlibrary.org/chapter32_pylint.html
[49] https://qwiet.ai/securing-your-python-codebase-best-practices-for-developers/
[50] https://stackoverflow.com/questions/1410444/checking-python-code-correctness
[51] https://softwareengineering.stackexchange.com/questions/444182/designing-a-python-string-validation-library
[52] https://simeononsecurity.com/articles/secure-coding-standards-for-python/
[53] http://faun.dev/c/stories/dariaip/how-to-ensure-data-type-correctness-at-python/
[54] https://labex.io/tutorials/python-how-to-validate-function-parameters-in-python-420747
[55] https://codilime.com/blog/python-code-quality-linters/
[56] https://granulate.io/blog/optimizing-python-why-python-is-slow-optimization-methods/
[57] https://dev.to/abdulla783/20-powerful-techniques-for-writing-efficient-and-readable-python-code-3fee
[58] https://stackify.com/how-to-optimize-python-code/
[59] https://dev.to/ken_mwaura1/enhancing-python-code-quality-a-comprehensive-guide-to-linting-with-ruff-3d6g
[60] https://www.theserverside.com/tip/Tips-to-improve-Python-performance
[61] https://granulate.io/blog/python-performance-optimization-tips-faster-python-versions/

---
To address the query about the differences between `deep_research-v2.py` and `deep_research-v4.py`, need to infer based on general trends in deep research tools and the information available about similar projects, focusing on **error handling, feature enhancements, and research quality**.

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

## Overview of differences

1. **Integration of New Tools and APIs**: 
   - **v4** might include additional tools or APIs for web searches, academic databases, or AI services, enhancing its research capabilities compared to **v2**.
   - For example, integrating tools like Firecrawl for web content extraction or advanced AI models for query optimization.

2. **Enhanced User Interface and Experience**:
   - **v4** could offer a more user-friendly interface, possibly using frameworks like Rich or Streamlit, to improve the interaction experience.
   - This might include features like progress tracking, customizable depth control, and interactive follow-up questions.

3. **Improved Data Models and Analysis**:
   - **v4** might refine data models to better structure and analyze research findings, including enhanced credibility scoring and metadata extraction.
   - This could involve more sophisticated methods for synthesizing information from multiple sources.

4. **Optimizations and Performance**:
   - **v4** could include optimizations for faster execution, such as parallel processing or more efficient API calls.
   - These improvements would enhance the overall performance and reduce the time required for research tasks.

5. **Error Handling and Validation**:
   - **v4** likely includes better error handling and validation mechanisms to ensure robustness and reliability, especially in handling API keys and exceptions.

## Logic and Flow of the Code

### Introduction to Deep Research Tools

Deep research tools like `deep_research-v4.py` are designed to automate the research process by integrating AI and web search capabilities. They typically involve several key components:

1. **Tool Setup and Integration**:
   - The code sets up various tools using APIs from services like DuckDuckGo, Wikipedia, arXiv, and Wolfram Alpha.
   - These tools are integrated into the research framework to perform specific tasks such as web searches and factual queries.

2. **Agent Creation and Execution**:
   - The code creates a tool-calling agent using frameworks like `langchain` to execute research tasks sequentially.
   - This agent is crucial for automating the research process across multiple sources.

### Example of Tool Integration

```python
def _setup_tools(self):
    search = DuckDuckGoSearchRun()
    wiki_api = WikipediaAPIWrapper(top_k_results=3)
    arxiv_api = ArxivAPIWrapper()
    scholar_api = GoogleScholarAPIWrapper(serp_api_key=self.serp_api_key)
    scholar = GoogleScholarQueryRun(api_wrapper=scholar_api)
    wolfram_alpha_api = WolframAlphaAPIWrapper(wolfram_alpha_appid=self.wolfram_alpha_appid)
    wolfram_alpha = WolframAlphaQueryRun(api_wrapper=wolfram_alpha_api)
    
    return [
        Tool(
            name="web_search",
            func=self._web_search_wrapper(search.run),
            description="Performs broad web searches using DuckDuckGo with URL and metadata extraction."
        ),
        # Other tools...
    ]
```

### Improvements and Updates

1. **Enhanced Features**:
   - **Wolfram Alpha Integration**: The addition of Wolfram Alpha provides a powerful tool for factual queries and numerical computations.
   - **Improved Credibility Scoring**: The credibility scoring system has been refined to differentiate between sources more effectively.

2. **Bug Fixes and Optimizations**:
   - **Rate Limiter Updates**: The code now uses more flexible rate limiting options to prevent API abuse.
   - **Improved Metadata Extraction**: The code includes more robust patterns for extracting metadata such as authors and publication dates.

## Potential Optimizations and Future Directions

### Optimizations

1. **Parallel Processing**: Implementing parallel processing for tool execution could significantly improve performance by leveraging multi-core CPUs.

   ```python
   with concurrent.futures.ThreadPoolExecutor() as executor:
       futures = [executor.submit(tool.run, query) for tool in self.tools]
       results = [future.result() for future in futures]
   ```

2. **AI-Driven Query Optimization**: Using AI models to optimize search queries based on context and relevance could enhance the efficiency of the research process.

### Future Directions

1. **Integration of Emerging Technologies**: Incorporating newer AI models or tools, such as those focused on multimodal search or advanced natural language processing, could further enhance the research capabilities.

2. **User Interface Improvements**: Developing a user-friendly interface, possibly using frameworks like Gradio or Streamlit, could make the tool more accessible to a broader audience.

## Conclusion and Recommendations

Typical updates from one version to another would involve enhancements in tool integration, user experience, data analysis, and performance optimizations. These improvements are crucial for maintaining the tool's relevance and effectiveness in the rapidly evolving landscape of AI-powered research tools.

**Recommendations**:

- **Developers**: Consider integrating parallel processing and AI-driven query optimization to further improve performance.
- **Users**: Leverage the tool's capabilities by exploring its various features and tools to streamline research workflows.

By continuing to refine and expand such tools, researchers and developers can create more efficient and effective research processes that leverage the latest advancements in AI and information retrieval technologies.

---

### Final Thoughts


The updates focus on enhancing functionality, user experience, and performance. The integration of new tools, improved data models, and optimizations like parallel processing are key areas for improvement. These advancements not only make the tool more efficient but also more accessible and user-friendly, which is essential for its adoption across various research contexts.


The analysis above provides a framework for understanding the evolution of deep research tools and highlights areas where improvements can be made to enhance their effectiveness and usability.

Citations:
[1] https://github.com/epuerta9/deep-research-py
[2] https://dev.to/composiodev/building-an-open-source-deep-research-agent-from-scratch-using-llamaindex-composio-exaai-4j9b
[3] https://github.com/epuerta9/deep-research-py/blob/main/README.md
[4] https://milvus.io/blog/introduce-deepsearcher-a-local-open-source-deep-research.md
[5] https://dev.to/mehmetakar/open-source-deep-research-4685
[6] https://www.datacamp.com/blog/deep-research-openai
[7] https://datascience.fm/how-to-build-a-first-version-of-a-deep-research-workflow/
[8] https://openai.com/index/introducing-deep-research/
[9] https://www.reddit.com/r/OpenAI/comments/1igbu9y/introducing_deep_research/
[10] https://huggingface.co/blog/open-deep-research
[11] https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research
[12] https://cdn.openai.com/deep-research-system-card.pdf
[13] https://www.lesswrong.com/posts/SLiXeZvEkD4XbX7yy/openai-releases-deep-research-agent
[14] https://openai.com/index/new-tools-for-building-agents/
[15] https://www.youtube.com/watch?v=jPR1NEerdEk
[16] https://dirox.com/post/openai-deep-research
[17] https://www.linkedin.com/posts/david-zhang-2902462a_introducing-deep-research-my-own-open-source-activity-7292372816593633280-73zW
[18] https://www.helicone.ai/blog/openai-deep-research
[19] https://blog.google/products/gemini/new-gemini-app-features-march-2025/
[20] https://www.galileo.ai/blog/deep-research-agent
[21] https://apidog.com/blog/recreate-openai-deep-research/
[22] https://zilliz.com/blog/introduce-deepsearcher-a-local-open-source-deep-research
[23] https://www.maginative.com/article/chatgpts-deep-research-now-available-to-plus-team-and-enterprise-users/

---
```python
# deep_research-v4.py (Deep Research Tool)
import os
import re
import json
import hashlib
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import diskcache as dc
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator  # Updated import
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun, WolframAlphaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper, WolframAlphaAPIWrapper
from langchain_community.tools.google_scholar import GoogleScholarQueryRun
from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper  # Added import
import gradio as gr
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import concurrent.futures
from pyrate_limiter import Duration, Rate, Limiter # Using pyrate_limiter instead of ratelimiter

load_dotenv()

# Enhanced models
class Source(BaseModel):
    url: str
    source_type: str
    relevance: float = 0.0
    authors: List[str] = Field(default_factory=list)
    publication_date: str = ""
    citation_count: int = 0
    title: str = ""
    credibility_score: float = 0.5
    confidence_score: float = 0.7 # Confidence in the extracted information from the source

    @field_validator('credibility_score', mode='before')  # Updated to V2 style
    @classmethod  # Added classmethod decorator as required by V2
    def validate_credibility(cls, v):
        return max(0.0, min(1.0, v))

    @field_validator('confidence_score', mode='before')  # Updated to V2 style
    @classmethod  # Added classmethod decorator as required by V2
    def validate_confidence(cls, v):
        return max(0.0, min(1.0, v))

class ResearchStep(BaseModel):
    step_name: str
    objective: str = "" # Added objective for each step
    tools_used: List[str]
    output: str
    sources: List[Source] = Field(default_factory=list)
    key_findings: List[str] = Field(default_factory=list)
    search_queries: List[str] = Field(default_factory=list) # Log of search queries for each step

class ResearchResponse(BaseModel):
    topic: str
    research_question: str = "" # Added research question
    summary: str
    steps: List[ResearchStep]
    quality_score: float
    sources: List[Source] = Field(default_factory=list)
    uncertainty_areas: List[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    methodology: str = "Tool-based Agent with Multi-Source Verification" # Explicitly state methodology
    researcher_notes: str = "" # Space for researcher's notes or reflections

    class Config:
        arbitrary_types_allowed = True

class DeepResearchTool:
    def __init__(self, cache_dir: str = ".research_cache"):
        load_dotenv()  # Ensure environment variables are loaded

        # Load and validate API keys
        self.serp_api_key = os.getenv("SERP_API_KEY")
        self.wolfram_alpha_appid = os.getenv("WOLFRAM_ALPHA_APPID")

        # Validate API keys with descriptive errors
        if not self.serp_api_key:
            raise ValueError("SERP_API_KEY environment variable is not set. Please add it to your .env file.")
        if not self.wolfram_alpha_appid:
            raise ValueError("WOLFRAM_ALPHA_APPID environment variable is not set. Please add it to your .env file.")

        try:
            self.llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
        except Exception as e:
            raise ValueError(f"Error initializing ChatAnthropic: {str(e)}")

        self.parser = PydanticOutputParser(pydantic_object=ResearchResponse)

        # Initialize cache
        self.cache = dc.Cache(cache_dir)
        self.cache_ttl = 60 * 60 * 24 * 7  # 1 week cache

        # Rate limiters using pyrate_limiter
        rates = [Rate(10, Duration.MINUTE)]
        self.web_limiter = Limiter(*rates)
        self.scholar_limiter = Limiter(*rates)

        try:
            self.tools = self._setup_tools()
            self.agent = self._create_agent()
        except Exception as e:
            raise ValueError(f"Error setting up research tools: {str(e)}")

        self.current_sources = []
        self.metadata_store = {}
        self.search_query_log = []

    def _setup_tools(self):
        try:
            search = DuckDuckGoSearchRun()
            wiki_api = WikipediaAPIWrapper(top_k_results=3)
            arxiv_api = ArxivAPIWrapper()
            
            # Initialize Google Scholar with error handling
            try:
                scholar_api = GoogleScholarAPIWrapper(serp_api_key=self.serp_api_key)
                scholar = GoogleScholarQueryRun(api_wrapper=scholar_api)
            except Exception as e:
                raise ValueError(f"Error initializing Google Scholar API: {str(e)}")

            # Initialize WolframAlpha with error handling
            try:
                wolfram_alpha_api = WolframAlphaAPIWrapper(wolfram_alpha_appid=self.wolfram_alpha_appid)
                wolfram_alpha = WolframAlphaQueryRun(api_wrapper=wolfram_alpha_api)
            except Exception as e:
                raise ValueError(f"Error initializing WolframAlpha API: {str(e)}")

            return [
                Tool(
                    name="web_search",
                    func=self._web_search_wrapper(search.run),
                    description="Performs broad web searches using DuckDuckGo with URL and metadata extraction. Returns relevant web results for general information."
                ),
                Tool(
                    name="wikipedia",
                    func=self._wiki_wrapper(wiki_api.run),
                    description="Queries Wikipedia for background knowledge and definitions, with source tracking and credibility assessment. Best for initial understanding of topics."
                ),
                Tool(
                    name="arxiv_search",
                    func=self._arxiv_wrapper(arxiv_api.run),
                    description="Searches arXiv for academic papers, focusing on computer science, physics, and mathematics. Extracts metadata for credibility and relevance scoring. Use for cutting-edge research in STEM fields."
                ),
                Tool(
                    name="google_scholar",
                    func=self._scholar_wrapper(scholar.run),
                    description="Searches Google Scholar to find academic papers across disciplines, tracking citations and assessing paper importance. Ideal for in-depth academic research and understanding scholarly impact."
                ),
                Tool(
                    name="wolfram_alpha",
                    func=self._wolfram_wrapper(wolfram_alpha.run),
                    description="Uses Wolfram Alpha to compute answers to factual questions, perform calculations, and access curated knowledge. Excellent for quantitative queries and verifying numerical data."
                )
            ]
        except Exception as e:
            raise ValueError(f"Error setting up research tools: {str(e)}")

    def _extract_urls(self, text: str) -> List[str]:
        return re.findall(r'https?://[^\s]+', text)

    def _extract_metadata(self, text: str, source_type: str) -> Dict:
        """Extract enhanced metadata from source text"""
        metadata = {
            "authors": [],
            "publication_date": "",
            "citation_count": 0,
            "title": "",
            "credibility_score": self._calculate_base_credibility(source_type),
            "confidence_score": 0.7 # Default confidence, adjust based on extraction success
        }

        # Author extraction patterns - more robust patterns
        author_patterns = [
            r'(?:by|authors?)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+and\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)?)', # More comprehensive name matching
            r'Author[s]?:\s+([^,]+(?:,\s*[^,]+)*)'
        ]

        for pattern in author_patterns:
            authors_matches = re.findall(pattern, text, re.IGNORECASE)
            if authors_matches:
                all_authors = []
                for match in authors_matches:
                    authors_list = [a.strip() for a in match.split('and')] # Handle 'and' separator for multiple authors
                    for author_segment in authors_list:
                        individual_authors = [a.strip() for a in author_segment.split(',')] # Handle comma separated authors within segment
                        all_authors.extend(individual_authors)
                metadata["authors"] = all_authors
                if not metadata["authors"]:
                    metadata["confidence_score"] -= 0.1 # Reduce confidence if author extraction fails significantly
                break # Stop after first successful pattern

        # Date extraction - more flexible date formats
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',                      # YYYY-MM-DD
            r'([A-Z][a-z]{2,}\s+\d{1,2},\s+\d{4})',       # Month DD, YYYY (e.g., Jan 01, 2024)
            r'([A-Z][a-z]{2,}\.\s+\d{1,2},\s+\d{4})',     # Month. DD, YYYY (e.g., Jan. 01, 2024)
            r'([A-Z][a-z]{2,}\s+\d{4})',                  # Month YYYY (e.g., January 2024)
            r'Published:\s*(\d{4})',                    # Year only
            r'Date:\s*(\d{4})'                           # Year only, alternative label
        ]

        for pattern in date_patterns:
            dates = re.findall(pattern, text, re.IGNORECASE)
            if dates:
                metadata["publication_date"] = dates[0]
                if not metadata["publication_date"]:
                    metadata["confidence_score"] -= 0.05 # Slightly reduce confidence if date extraction fails
                break

        # Citation extraction for academic sources
        if source_type in ['arxiv', 'scholar']:
            citations = re.findall(r'Cited by (\d+)', text, re.IGNORECASE)
            if citations:
                metadata["citation_count"] = int(citations[0])
                # Adjust credibility based on citations - more nuanced boost
                citation_boost = min(0.25, int(citations[0])/500) # Increased boost cap and sensitivity
                metadata["credibility_score"] = min(1.0, metadata["credibility_score"] + citation_boost)
            else:
                metadata["confidence_score"] -= 0.1 # Reduce confidence if citation count not found in scholar/arxiv

        # Title extraction - more robust title extraction
        title_patterns = [
            r'Title:\s*([^\n]+)',           # Title label
            r'^(.*?)(?:\n|$)',              # First line as title if no explicit label
            r'##\s*(.*?)\s*##'              # Markdown heading level 2 as title
        ]
        for pattern in title_patterns:
            title_match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
            if title_match:
                metadata["title"] = title_match.group(1).strip()
                if not metadata["title"]:
                    metadata["confidence_score"] -= 0.05 # Slightly reduce if title extraction is weak
                break

        return metadata

    def _calculate_base_credibility(self, source_type: str) -> float:
        """Calculate base credibility score based on source type - Adjusted scores for better differentiation"""
        credibility_map = {
            "arxiv": 0.90, # Increased for Arxiv
            "scholar": 0.92, # Highest for Scholar
            "wikipedia": 0.75, # Slightly increased
            "web": 0.50,   # Web remains moderate
            "wolfram_alpha": 0.95 # Very high for Wolfram Alpha
        }
        return credibility_map.get(source_type, 0.5)

    def _web_search_wrapper(self, func):
        def wrapper(query):
            cache_key = f"web_{hashlib.md5(query.encode()).hexdigest()}"
            cached = self.cache.get(cache_key)
            if cached:
                self.current_sources.extend(cached["sources"])
                return cached["result"]

            with self.web_limiter:
                try:
                    result = func(query)
                    urls = self._extract_urls(result)
                    sources = []

                    for url in urls:
                        metadata = self._extract_metadata(result, "web")
                        source = Source(
                            url=url,
                            source_type="web",
                            relevance=0.6, # Slightly reduced web relevance
                            **metadata
                        )
                        sources.append(source)

                    self.current_sources.extend(sources)
                    self.cache.set(
                        cache_key,
                        {"result": result, "sources": sources},
                        expire=self.cache_ttl
                    )
                    return result
                except Exception as e:
                    return f"Error during web search: {str(e)}. Please try with a more specific query."
        return wrapper

    def _wiki_wrapper(self, func):
        def wrapper(query):
            cache_key = f"wiki_{hashlib.md5(query.encode()).hexdigest()}"
            cached = self.cache.get(cache_key)
            if cached:
                self.current_sources.extend(cached["sources"])
                return cached["result"]

            try:
                result = func(query)
                url = f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"
                metadata = self._extract_metadata(result, "wikipedia")
                source = Source(
                    url=url,
                    source_type="wikipedia",
                    relevance=0.75, # Increased Wiki relevance
                    **metadata
                )
                self.current_sources.append(source)
                self.cache.set(
                    cache_key,
                    {"result": result, "sources": [source]},
                    expire=self.cache_ttl
                )
                return result
            except Exception as e:
                return f"Error during Wikipedia search: {str(e)}. Wikipedia might not have an article on this specific topic."
        return wrapper

    def _arxiv_wrapper(self, func):
        def wrapper(query):
            cache_key = f"arxiv_{hashlib.md5(query.encode()).hexdigest()}"
            cached = self.cache.get(cache_key)
            if cached:
                self.current_sources.extend(cached["sources"])
                return cached["result"]

            try:
                result = func(query)
                sources = []
                papers = result.split('\n\n')
                for paper in papers:
                    if "arXiv:" in paper:
                        arxiv_id = None
                        id_match = re.search(r'arXiv:(\d+\.\d+)', paper)
                        if id_match:
                            arxiv_id = id_match.group(1)

                        if arxiv_id:
                            url = f"https://arxiv.org/abs/{arxiv_id}"
                            metadata = self._extract_metadata(paper, "arxiv")

                            # Title extraction - improved within arxiv wrapper
                            title_match = re.search(r'^(.*?)(?:\n|$)', paper, re.MULTILINE)
                            if title_match:
                                metadata["title"] = title_match.group(1).strip()

                            # Authors extraction - improved within arxiv wrapper
                            authors_match = re.search(r'Authors?: (.*?)(?:\n|$)', paper, re.MULTILINE | re.IGNORECASE)
                            if authors_match:
                                metadata["authors"] = [a.strip() for a in authors_match.group(1).split(',')]

                            # Date extraction - improved within arxiv wrapper
                            date_match = re.search(r'(?:Published|Date): (.*?)(?:\n|$)', paper, re.MULTILINE | re.IGNORECASE)
                            if date_match:
                                metadata["publication_date"] = date_match.group(1).strip()


                            source = Source(
                                url=url,
                                source_type="arxiv",
                                relevance=0.92, # Increased Arxiv relevance
                                **metadata
                            )
                            sources.append(source)

                self.current_sources.extend(sources)
                self.cache.set(
                    cache_key,
                    {"result": result, "sources": sources},
                    expire=self.cache_ttl
                )
                return result
            except Exception as e:
                return f"Error during arXiv search: {str(e)}. Try rephrasing your query with more academic terminology."
        return wrapper

    def _scholar_wrapper(self, func):
        def wrapper(query):
            cache_key = f"scholar_{hashlib.md5(query.encode()).hexdigest()}"
            cached = self.cache.get(cache_key)
            if cached:
                self.current_sources.extend(cached["sources"])
                return cached["result"]

            with self.scholar_limiter:
                try:
                    result = func(query)
                    sources = []
                    papers = result.split('\n\n')
                    for paper in papers:
                        url_match = re.search(r'(https?://[^\s]+)', paper)
                        if url_match:
                            url = url_match.group(1)
                            metadata = self._extract_metadata(paper, "scholar")

                            # Title extraction - improved within scholar wrapper
                            title_match = re.search(r'^(.*?)(?:\n|$)', paper, re.MULTILINE)
                            if title_match:
                                metadata["title"] = title_match.group(1).strip()

                            # Citation count extraction - improved within scholar wrapper
                            citation_match = re.search(r'Cited by (\d+)', paper, re.IGNORECASE)
                            if citation_match:
                                metadata["citation_count"] = int(citation_match.group(1))
                                citation_boost = min(0.3, int(citation_match.group(1))/400) # Further increased citation boost sensitivity and cap
                                metadata["credibility_score"] += citation_boost

                            source = Source(
                                url=url,
                                source_type="scholar",
                                relevance=0.95, # Highest relevance for Scholar
                                **metadata
                            )
                            sources.append(source)

                    self.current_sources.extend(sources)
                    self.cache.set(
                        cache_key,
                        {"result": result, "sources": sources},
                        expire=self.cache_ttl
                    )
                    return result
                except Exception as e:
                    return f"Error during Google Scholar search: {str(e)}. Scholar might be rate-limiting requests."
        return wrapper

    def _wolfram_wrapper(self, func):
        def wrapper(query):
            cache_key = f"wolfram_{hashlib.md5(query.encode()).hexdigest()}"
            cached = self.cache.get(cache_key)
            if cached:
                self.current_sources.extend(cached["sources"])
                return cached["result"]

            try:
                result = func(query)
                source = Source(
                    url="https://www.wolframalpha.com/", # General WolframAlpha URL as source
                    source_type="wolfram_alpha",
                    relevance=0.98, # Highest relevance as WolframAlpha is authoritative for factual queries
                    credibility_score=0.95, # Very high credibility
                    confidence_score=0.99 # Extremely high confidence in factual data
                )
                self.current_sources.append(source)
                self.cache.set(
                    cache_key,
                    {"result": result, "sources": [source]},
                    expire=self.cache_ttl
                )
                return result
            except Exception as e:
                return f"Error during WolframAlpha query: {str(e)}. Please check your query format or WolframAlpha API key."
        return wrapper


    def _create_agent(self):
        # Pre-format the schema with proper escaping
        schema_instructions = self.parser.get_format_instructions()
        # Replace single braces with double braces in the schema, except for the template variables
        escaped_schema = schema_instructions.replace('{', '{{').replace('}', '}}')
        escaped_schema = escaped_schema.replace('{{query}}', '{query}').replace('{{agent_scratchpad}}', '{agent_scratchpad}')

        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert research assistant. Your goal is to conduct thorough and high-quality research based on user queries. Follow a structured, multi-step research process to ensure comprehensive and reliable results.

            **Research Process:**

            1. **Understand and Clarify Query (Query Analysis):**
               - Initial Objective: Analyze the research query, identify key concepts, and clarify any ambiguities. Define the specific research question to be answered.
               - Tools: None (internal thought process)
               - Expected Output: Clearly defined research question and initial research objectives.

            2. **Background Research (Initial Research):**
               - Objective: Gather preliminary information and context on the topic. Use broad web searches and Wikipedia to get an overview and identify key areas.
               - Tools: web_search, wikipedia
               - Expected Output: Summary of background information, identification of key terms and concepts, and initial sources.

            3. **Academic Literature Review (Academic Deep Dive):**
               - Objective: Explore academic databases for in-depth, peer-reviewed research. Use arXiv and Google Scholar to find relevant papers and scholarly articles.
               - Tools: arxiv_search, google_scholar
               - Expected Output: List of relevant academic papers, key findings from these papers, and identification of leading researchers or institutions.

            4. **Factual Verification and Data Analysis (Cross-Verification):**
               - Objective: Verify key facts, statistics, and data points using reliable sources like Wolfram Alpha and targeted web searches. Cross-reference information from different sources to ensure accuracy.
               - Tools: wolfram_alpha, web_search (for specific fact-checking)
               - Expected Output: Verified facts and data, resolution of any conflicting information, and increased confidence in research findings.

            5. **Synthesis and Report Generation (Synthesis):**
               - Objective: Synthesize all gathered information into a coherent and structured research report. Summarize key findings, highlight areas of uncertainty, and provide a quality assessment.
               - Tools: None (synthesis and writing process)
               - Expected Output: Comprehensive research report in JSON format, including summary, detailed steps, sources, quality score, and uncertainty areas.

            **IMPORTANT GUIDELINES:**

            - **Prioritize High-Quality Sources:** Always prioritize peer-reviewed academic sources (arXiv, Google Scholar) and authoritative sources (Wolfram Alpha, Wikipedia) over general web sources.
            - **Explicitly Note Uncertainties:** Clearly identify and document areas where information is uncertain, conflicting, or based on limited evidence.
            - **Distinguish Facts and Emerging Research:** Differentiate between well-established facts and findings from ongoing or emerging research.
            - **Extract and Attribute Source Metadata:**  Meticulously extract and include author names, publication dates, citation counts, and credibility scores for all sources.
            - **Provide Source URLs:** Always include URLs for all sources to allow for easy verification and further reading.
            - **Assess Source Credibility:** Evaluate and rate the credibility of each source based on its type, reputation, and other available metadata.
            - **Log Search Queries:** Keep track of the search queries used in each step to ensure transparency and reproducibility.

            **Output Format:**
            Your output must be valid JSON conforming to the following schema:
            {schema}
            """.format(schema=escaped_schema)),
            ("human", "{query}"),
            ("ai", "{agent_scratchpad}")
        ])

        return AgentExecutor(
            agent=create_tool_calling_agent(self.llm, self.tools, prompt),
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10, # Limit agent iterations to prevent runaway loops
            early_stopping_method="generate" # Stop if no more actions predicted

        )

    def _evaluate_quality(self, response: ResearchResponse) -> Dict[str, float]:
        """Enhanced quality evaluation with multiple metrics - Refined Metrics and Weights"""
        metrics = {}

        # Source diversity (0-1) - Emphasize academic and high-quality sources
        source_types = {s.source_type for s in response.sources}
        diversity_score = len(source_types) / 5 # Now considering 5 source types
        metrics["source_diversity"] = min(1.0, diversity_score)

        # Academic and Authoritative Ratio (0-1) - Increased weight for academic/wolfram
        total_sources = len(response.sources) or 1
        academic_sources = sum(1 for s in response.sources
                              if s.source_type in ["arxiv", "scholar", "wolfram_alpha"])
        metrics["academic_ratio"] = academic_sources / total_sources

        # Verification Thoroughness (0-1) - More sensitive to verification steps
        verification_steps = sum(1 for step in response.steps
                                if any(term in step.step_name.lower()
                                       for term in ["verification", "validate", "cross-check", "factual verification"])) # Broader terms
        metrics["verification_score"] = min(1.0, verification_steps / 3) # Adjusted scaling

        # Process Comprehensiveness (0-1) - Rewards all research steps being present
        metrics["process_score"] = min(1.0, len(response.steps) / 5) # 5 expected steps

        # Source Recency - if we have dates - Improved recency scoring
        dated_sources = [s for s in response.sources if s.publication_date]
        if dated_sources:
            current_year = datetime.now().year
            years = []
            for s in dated_sources:
                year_match = re.search(r'(\d{4})', s.publication_date)
                if year_match:
                    years.append(int(year_match.group(1)))
            if years:
                avg_year = sum(years) / len(years)
                # Recency decays faster for older sources, plateaus for very recent
                recency_factor = max(0.0, 1 - (current_year - avg_year) / 7) # Faster decay over 7 years
                metrics["recency_score"] = min(1.0, recency_factor)
            else:
                metrics["recency_score"] = 0.3  # Lower default if dates are unparseable
        else:
            metrics["recency_score"] = 0.2  # Even lower if no dates available

        # Source Credibility (0-1) - Average credibility, penalize low credibility
        if response.sources:
            credibility_avg = sum(s.credibility_score for s in response.sources) / len(response.sources)
            # Penalize if average credibility is low
            credibility_penalty = max(0.0, 0.5 - credibility_avg) # Significant penalty if avg below 0.5
            metrics["credibility_score"] = max(0.0, credibility_avg - credibility_penalty) # Ensure score doesn't go negative
        else:
            metrics["credibility_score"] = 0.0

        # Source Confidence - Average confidence in extracted metadata
        if response.sources:
            metrics["confidence_score"] = sum(s.confidence_score for s in response.sources) / len(response.sources)
        else:
            metrics["confidence_score"] = 0.0


        # Calculate weighted overall score - Adjusted weights to emphasize academic rigor and verification
        weights = {
            "source_diversity": 0.10,       # Reduced slightly
            "academic_ratio": 0.30,         # Increased significantly - Academic rigor is key
            "verification_score": 0.25,      # Increased - Verification is critical
            "process_score": 0.10,          # Process still important, but less weight than academic and verification
            "recency_score": 0.10,          # Recency is relevant, but not as much as rigor
            "credibility_score": 0.10,      # Credibility remains important
            "confidence_score": 0.05        # Confidence in data extraction, secondary to credibility
        }

        overall_score = sum(metrics[key] * weights[key] for key in weights)
        metrics["overall"] = overall_score

        return metrics

    def conduct_research(self, query: str) -> str:
        """Conduct research on the given query - Enhanced research process logging and error handling"""
        cache_key = f"research_{hashlib.md5(query.encode()).hexdigest()}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        self.current_sources = []
        self.search_query_log = [] # Reset query log for each research run
        start_time = time.time()

        try:
            raw_response = self.agent.invoke({"query": query})
            try:
                response = self.parser.parse(raw_response["output"])
            except Exception as e:
                try:
                    json_match = re.search(r'```json\n(.*?)\n```', raw_response["output"], re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        response_dict = json.loads(json_str)
                        response = ResearchResponse(**response_dict)
                    else:
                        raise ValueError("Could not extract JSON from response")
                except Exception as inner_e:
                    return f"### Error Parsing Research Results\n\nPrimary parsing error: {str(e)}\nSecondary JSON extraction error: {str(inner_e)}\n\nRaw response excerpt (first 500 chars):\n```\n{raw_response['output'][:500]}...\n```\nPlease review the raw response for JSON formatting issues."


            all_sources = {s.url: s for s in self.current_sources}
            for s in response.sources:
                if s.url not in all_sources:
                    all_sources[s.url] = s
            response.sources = list(all_sources.values())

            self._assign_sources_to_steps(response)

            quality_metrics = self._evaluate_quality(response)
            response.quality_score = quality_metrics["overall"]

            formatted_response = self._format_response(response, quality_metrics)
            self.cache.set(cache_key, formatted_response, expire=self.cache_ttl)
            return formatted_response
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Error after {duration:.1f}s: {str(e)}"
            return f"### Research Error\n\n{error_msg}\n\nPlease try again with a more specific query or check the research tool's configuration."

    def _assign_sources_to_steps(self, response: ResearchResponse):
        """Assign sources to steps based on keywords and source type - Improved Assignment Logic"""
        url_to_source = {s.url: s for s in response.sources}

        for step in response.steps:
            step_sources = []
            urls_in_step_output = self._extract_urls(step.output)

            # 1. Direct URL Matching: Prioritize sources explicitly linked in step output
            for url in urls_in_step_output:
                if url in url_to_source:
                    step_sources.append(url_to_source[url])

            if not step_sources: # If no direct URL matches, use keyword and tool-based assignment
                step_content_lower = step.output.lower()

                if "wikipedia" in step.tools_used or "background" in step.step_name.lower():
                    wiki_sources = [s for s in response.sources if s.source_type == "wikipedia"]
                    step_sources.extend(wiki_sources[:2]) # Limit Wiki sources per step

                if "arxiv" in step.tools_used or "academic literature" in step.step_name.lower():
                    arxiv_sources = [s for s in response.sources if s.source_type == "arxiv"]
                    # Assign Arxiv sources if step output contains academic keywords
                    if any(keyword in step_content_lower for keyword in ["research", "study", "paper", "academic", "scholar"]):
                        step_sources.extend(arxiv_sources[:2])

                if "scholar" in step.tools_used or "academic literature" in step.step_name.lower():
                    scholar_sources = [s for s in response.sources if s.source_type == "scholar"]
                    if any(keyword in step_content_lower for keyword in ["research", "study", "paper", "academic", "citation"]):
                         step_sources.extend(scholar_sources[:2])

                if "wolfram" in step.tools_used or "factual verification" in step.step_name.lower() or "cross-verification" in step.step_name.lower():
                    wolfram_sources = [s for s in response.sources if s.source_type == "wolfram_alpha"]
                    step_sources.extend(wolfram_sources[:1]) # Max 1 Wolfram source

                if "web_search" in step.tools_used or "initial research" in step.step_name.lower():
                    web_sources = [s for s in response.sources if s.source_type == "web"]
                    step_sources.extend(web_sources[:2]) # Limit web sources

            # Deduplicate sources assigned to step, maintaining order if possible
            unique_step_sources = []
            seen_urls = set()
            for source in step_sources:
                if source.url not in seen_urls:
                    unique_step_sources.append(source)
                    seen_urls.add(source.url)
            step.sources = unique_step_sources


    def _format_response(self, response: ResearchResponse, metrics: Dict[str, float]) -> str:
        """Generate enhanced markdown output with quality metrics visualization - Improved Markdown Formatting and Metrics Display"""
        markdown = f"## Research Report: {response.topic}\n\n"
        markdown += f"**Research Question:** {response.research_question}\n\n" # Display research question

        # Quality metrics section - Enhanced Metrics Table
        markdown += "### Quality Assessment\n"
        markdown += f"**Overall Quality Score:** {response.quality_score:.2f}/1.00  \n\n" # Clear label
        markdown += self._generate_metrics_table(metrics) # Use table for metrics
        markdown += f"\n**Research Methodology:** {response.methodology}\n" # Report methodology
        markdown += f"**Timestamp:** {response.timestamp}\n\n"

        # Summary section
        markdown += "### Summary\n"
        markdown += f"{response.summary}\n\n"

        # Research steps - More structured step output
        for i, step in enumerate(response.steps, 1):
            markdown += f"### Step {i}: {step.step_name}\n"
            markdown += f"**Objective:** {step.objective}\n" # Display step objective
            markdown += f"**Tools Used:** {', '.join(step.tools_used)}\n\n"
            markdown += f"{step.output}\n\n"

            if step.sources:
                markdown += "**Sources:**\n"
                for source in step.sources:
                    source_meta = []
                    if source.title:
                        source_meta.append(f"*{source.title}*") # Title in bold
                    if source.authors:
                        authors_str = ', '.join(source.authors[:2]) + (f" et al." if len(source.authors) > 2 else "")
                        source_meta.append(f"by {authors_str}")
                    if source.publication_date:
                        source_meta.append(f"({source.publication_date})")
                    if source.citation_count and source.source_type == 'scholar': # Citation count only for Scholar
                        source_meta.append(f"Cited {source.citation_count}+ times")
                    credibility_indicator = "⭐" * round(source.credibility_score * 5)
                    confidence_indicator = "📊" * round(source.confidence_score * 5)

                    source_detail = f"[{source.source_type.upper()}] " + " ".join(source_meta) if source_meta else f"[{source.source_type.upper()}]"
                    markdown += f"- {source_detail}: [Link]({source.url}) Credibility: {credibility_indicator} Confidence: {confidence_indicator}\n" # Added confidence indicator

            markdown += "\n"

        # Areas of uncertainty
        if response.uncertainty_areas:
            markdown += "### Areas of Uncertainty\n"
            for area in response.uncertainty_areas:
                markdown += f"- {area}\n"
            markdown += "\n"

        if response.researcher_notes: # Include researcher notes if present
            markdown += "### Researcher's Notes\n"
            markdown += f"{response.researcher_notes}\n\n"

        return markdown

    def _generate_metrics_table(self, metrics: Dict[str, float]) -> str:
        """Generate a metrics table with gauge visualizations - Improved Table with Confidence"""
        metric_names = {
            "overall": "Overall Quality",
            "source_diversity": "Source Diversity",
            "academic_ratio": "Academic Ratio",
            "verification_score": "Verification Score",
            "process_score": "Process Score",
            "recency_score": "Source Recency",
            "credibility_score": "Avg. Credibility",
            "confidence_score": "Avg. Confidence" # Added confidence to metrics table
        }

        table = "| Metric | Score | Rating |\n|---|---|---|\n" # Markdown table format

        for key, name in metric_names.items():
            if key in metrics:
                score = metrics[key]
                stars = "★" * round(score * 5)
                empty_stars = "☆" * (5 - len(stars))
                table += f"| {name} | {score:.2f} | {stars}{empty_stars} |\n" # Stars as rating

        return table

    def _generate_score_bar(self, score: float) -> str:
        """Generate a base64 encoded score bar visualization - Unused, can be removed or enhanced for visual output if needed"""
        fig, ax = plt.subplots(figsize=(5, 0.5))
        if score < 0.4:
            color = 'red'
        elif score < 0.7:
            color = 'orange'
        else:
            color = 'green'
        ax.barh(['Score'], [score], color=color)
        ax.barh(['Score'], [1], color='lightgray', alpha=0.3)
        ax.set_xlim(0, 1)
        plt.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        return base64.b64encode(buf.getvalue()).decode('utf-8')

# Gradio Interface with enhanced features
def create_interface():
    tool = DeepResearchTool()

    with gr.Blocks(title="Advanced Research Assistant") as iface:
        gr.Markdown("# Advanced Research Assistant")
        gr.Markdown("Conduct multi-source academic research with enhanced quality metrics, source validation, and WolframAlpha integration.")

        with gr.Row():
            with gr.Column(scale=4):
                query = gr.Textbox(
                    placeholder="Enter research query...",
                    lines=3,
                    label="Research Query"
                )
                research_question = gr.Textbox(
                    placeholder="Optional: Specify a precise research question...",
                    lines=1,
                    label="Research Question (Optional)"
                )

                with gr.Row():
                    submit_btn = gr.Button("Conduct Research", variant="primary")
                    clear_btn = gr.Button("Clear")

                examples = gr.Examples(
                    examples=[
                        ["Quantum computing applications in cryptography", "What are the most promising applications of quantum computing in modern cryptography?"],
                        ["Recent advances in CRISPR gene editing", "What are the latest breakthroughs in CRISPR gene editing technology and their potential impacts?"],
                        ["The impact of social media on teenage mental health", "How does social media use affect the mental health and well-being of teenagers?"],
                        ["Progress in fusion energy research since 2020", "What significant advancements have been made in fusion energy research and development since 2020?"]
                    ],
                    inputs=[query, research_question],
                    label="Example Queries with Research Questions"
                )

        output = gr.Markdown(label="Research Results")

        submit_btn.click(
            fn=tool.conduct_research,
            inputs=query,
            outputs=output
        )

        clear_btn.click(
            fn=lambda: ("", "", ""),
            inputs=None,
            outputs=[query, research_question, output]
        )

    return iface

if __name__ == "__main__":
    iface = create_interface()
    iface.launch()
```

---
```
# diff -u deep_research-v2.py deep_research-v4.py
--- deep_research-v2.py	2025-03-15 11:14:48.874574337 +0800
+++ deep_research-v4.py	2025-03-15 18:01:23.238677353 +0800
@@ -1,4 +1,4 @@
-# Deep Research Tool Improvement Analysis
+# Deep Research Tool Improvement Analysis (Error Fixed, Enhanced Features)
 import os
 import re
 import json
@@ -9,21 +9,22 @@
 import diskcache as dc
 import pandas as pd
 from dotenv import load_dotenv
-from pydantic import BaseModel, Field, validator
+from pydantic import BaseModel, Field, field_validator  # Updated import
 from langchain_anthropic import ChatAnthropic
 from langchain_core.prompts import ChatPromptTemplate
 from langchain_core.output_parsers import PydanticOutputParser
 from langchain.agents import create_tool_calling_agent, AgentExecutor
 from langchain.tools import Tool
-from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun
-from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
+from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun, WolframAlphaQueryRun
+from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper, WolframAlphaAPIWrapper
 from langchain_community.tools.google_scholar import GoogleScholarQueryRun
+from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper  # Added import
 import gradio as gr
 import matplotlib.pyplot as plt
 from io import BytesIO
 import base64
 import concurrent.futures
-from ratelimiter import RateLimiter
+from pyrate_limiter import Duration, Rate, Limiter # Using pyrate_limiter instead of ratelimiter
 
 load_dotenv()
 
@@ -37,76 +38,131 @@
     citation_count: int = 0
     title: str = ""
     credibility_score: float = 0.5
-    
-    @validator('credibility_score')
+    confidence_score: float = 0.7 # Confidence in the extracted information from the source
+
+    @field_validator('credibility_score', mode='before')  # Updated to V2 style
+    @classmethod  # Added classmethod decorator as required by V2
     def validate_credibility(cls, v):
         return max(0.0, min(1.0, v))
 
+    @field_validator('confidence_score', mode='before')  # Updated to V2 style
+    @classmethod  # Added classmethod decorator as required by V2
+    def validate_confidence(cls, v):
+        return max(0.0, min(1.0, v))
+
 class ResearchStep(BaseModel):
     step_name: str
+    objective: str = "" # Added objective for each step
     tools_used: List[str]
     output: str
     sources: List[Source] = Field(default_factory=list)
     key_findings: List[str] = Field(default_factory=list)
+    search_queries: List[str] = Field(default_factory=list) # Log of search queries for each step
 
 class ResearchResponse(BaseModel):
     topic: str
+    research_question: str = "" # Added research question
     summary: str
     steps: List[ResearchStep]
     quality_score: float
     sources: List[Source] = Field(default_factory=list)
     uncertainty_areas: List[str] = Field(default_factory=list)
     timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
-    
+    methodology: str = "Tool-based Agent with Multi-Source Verification" # Explicitly state methodology
+    researcher_notes: str = "" # Space for researcher's notes or reflections
+
     class Config:
         arbitrary_types_allowed = True
 
 class DeepResearchTool:
     def __init__(self, cache_dir: str = ".research_cache"):
-        self.llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
+        load_dotenv()  # Ensure environment variables are loaded
+
+        # Load and validate API keys
+        self.serp_api_key = os.getenv("SERP_API_KEY")
+        self.wolfram_alpha_appid = os.getenv("WOLFRAM_ALPHA_APPID")
+
+        # Validate API keys with descriptive errors
+        if not self.serp_api_key:
+            raise ValueError("SERP_API_KEY environment variable is not set. Please add it to your .env file.")
+        if not self.wolfram_alpha_appid:
+            raise ValueError("WOLFRAM_ALPHA_APPID environment variable is not set. Please add it to your .env file.")
+
+        try:
+            self.llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
+        except Exception as e:
+            raise ValueError(f"Error initializing ChatAnthropic: {str(e)}")
+
         self.parser = PydanticOutputParser(pydantic_object=ResearchResponse)
-        
+
         # Initialize cache
         self.cache = dc.Cache(cache_dir)
         self.cache_ttl = 60 * 60 * 24 * 7  # 1 week cache
-        
-        # Rate limiters
-        self.web_limiter = RateLimiter(max_calls=10, period=60)
-        self.scholar_limiter = RateLimiter(max_calls=5, period=60)
-        
-        self.tools = self._setup_tools()
-        self.agent = self._create_agent()
+
+        # Rate limiters using pyrate_limiter
+        rates = [Rate(10, Duration.MINUTE)]
+        self.web_limiter = Limiter(*rates)
+        self.scholar_limiter = Limiter(*rates)
+
+        try:
+            self.tools = self._setup_tools()
+            self.agent = self._create_agent()
+        except Exception as e:
+            raise ValueError(f"Error setting up research tools: {str(e)}")
+
         self.current_sources = []
         self.metadata_store = {}
-        
+        self.search_query_log = []
+
     def _setup_tools(self):
-        search = DuckDuckGoSearchRun()
-        wiki_api = WikipediaAPIWrapper(top_k_results=3)
-        arxiv_api = ArxivAPIWrapper()
-        scholar = GoogleScholarQueryRun()
-
-        return [
-            Tool(
-                name="web_search",
-                func=self._web_search_wrapper(search.run),
-                description="Performs web searches with URL extraction. Returns relevant web results."
-            ),
-            Tool(
-                name="wikipedia",
-                func=self._wiki_wrapper(wiki_api.run),
-                description="Queries Wikipedia with source tracking. Good for background knowledge and definitions."
-            ),
-            Tool(
-                name="arxiv_search",
-                func=self._arxiv_wrapper(arxiv_api.run),
-                description="Searches arXiv with metadata extraction. Best for latest academic research papers."
-            ),
-            Tool(
-                name="google_scholar",
-                func=self._scholar_wrapper(scholar.run),
-                description="Searches Google Scholar with citation tracking. Finds academic papers and their importance."
-            )
-        ]
+        try:
+            search = DuckDuckGoSearchRun()
+            wiki_api = WikipediaAPIWrapper(top_k_results=3)
+            arxiv_api = ArxivAPIWrapper()
+            
+            # Initialize Google Scholar with error handling
+            try:
+                scholar_api = GoogleScholarAPIWrapper(serp_api_key=self.serp_api_key)
+                scholar = GoogleScholarQueryRun(api_wrapper=scholar_api)
+            except Exception as e:
+                raise ValueError(f"Error initializing Google Scholar API: {str(e)}")
+
+            # Initialize WolframAlpha with error handling
+            try:
+                wolfram_alpha_api = WolframAlphaAPIWrapper(wolfram_alpha_appid=self.wolfram_alpha_appid)
+                wolfram_alpha = WolframAlphaQueryRun(api_wrapper=wolfram_alpha_api)
+            except Exception as e:
+                raise ValueError(f"Error initializing WolframAlpha API: {str(e)}")
+
+            return [
+                Tool(
+                    name="web_search",
+                    func=self._web_search_wrapper(search.run),
+                    description="Performs broad web searches using DuckDuckGo with URL and metadata extraction. Returns relevant web results for general information."
+                ),
+                Tool(
+                    name="wikipedia",
+                    func=self._wiki_wrapper(wiki_api.run),
+                    description="Queries Wikipedia for background knowledge and definitions, with source tracking and credibility assessment. Best for initial understanding of topics."
+                ),
+                Tool(
+                    name="arxiv_search",
+                    func=self._arxiv_wrapper(arxiv_api.run),
+                    description="Searches arXiv for academic papers, focusing on computer science, physics, and mathematics. Extracts metadata for credibility and relevance scoring. Use for cutting-edge research in STEM fields."
+                ),
+                Tool(
+                    name="google_scholar",
+                    func=self._scholar_wrapper(scholar.run),
+                    description="Searches Google Scholar to find academic papers across disciplines, tracking citations and assessing paper importance. Ideal for in-depth academic research and understanding scholarly impact."
+                ),
+                Tool(
+                    name="wolfram_alpha",
+                    func=self._wolfram_wrapper(wolfram_alpha.run),
+                    description="Uses Wolfram Alpha to compute answers to factual questions, perform calculations, and access curated knowledge. Excellent for quantitative queries and verifying numerical data."
+                )
+            ]
+        except Exception as e:
+            raise ValueError(f"Error setting up research tools: {str(e)}")
 
     def _extract_urls(self, text: str) -> List[str]:
         return re.findall(r'https?://[^\s]+', text)
@@ -118,94 +174,116 @@
             "publication_date": "",
             "citation_count": 0,
             "title": "",
-            "credibility_score": self._calculate_base_credibility(source_type)
+            "credibility_score": self._calculate_base_credibility(source_type),
+            "confidence_score": 0.7 # Default confidence, adjust based on extraction success
         }
-        
-        # Author extraction patterns
+
+        # Author extraction patterns - more robust patterns
         author_patterns = [
-            r'by\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
+            r'(?:by|authors?)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+and\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)?)', # More comprehensive name matching
             r'Author[s]?:\s+([^,]+(?:,\s*[^,]+)*)'
         ]
-        
+
         for pattern in author_patterns:
-            authors = re.findall(pattern, text)
-            if authors:
-                metadata["authors"] = [a.strip() for a in authors[0].split(',')] if ',' in authors[0] else [authors[0]]
-                break
-                
-        # Date extraction
+            authors_matches = re.findall(pattern, text, re.IGNORECASE)
+            if authors_matches:
+                all_authors = []
+                for match in authors_matches:
+                    authors_list = [a.strip() for a in match.split('and')] # Handle 'and' separator for multiple authors
+                    for author_segment in authors_list:
+                        individual_authors = [a.strip() for a in author_segment.split(',')] # Handle comma separated authors within segment
+                        all_authors.extend(individual_authors)
+                metadata["authors"] = all_authors
+                if not metadata["authors"]:
+                    metadata["confidence_score"] -= 0.1 # Reduce confidence if author extraction fails significantly
+                break # Stop after first successful pattern
+
+        # Date extraction - more flexible date formats
         date_patterns = [
-            r'(\d{4}-\d{2}-\d{2})',
-            r'([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})',
-            r'Published:\s+(\d{4})'
+            r'(\d{4}-\d{2}-\d{2})',                      # YYYY-MM-DD
+            r'([A-Z][a-z]{2,}\s+\d{1,2},\s+\d{4})',       # Month DD, YYYY (e.g., Jan 01, 2024)
+            r'([A-Z][a-z]{2,}\.\s+\d{1,2},\s+\d{4})',     # Month. DD, YYYY (e.g., Jan. 01, 2024)
+            r'([A-Z][a-z]{2,}\s+\d{4})',                  # Month YYYY (e.g., January 2024)
+            r'Published:\s*(\d{4})',                    # Year only
+            r'Date:\s*(\d{4})'                           # Year only, alternative label
         ]
-        
+
         for pattern in date_patterns:
-            dates = re.findall(pattern, text)
+            dates = re.findall(pattern, text, re.IGNORECASE)
             if dates:
                 metadata["publication_date"] = dates[0]
+                if not metadata["publication_date"]:
+                    metadata["confidence_score"] -= 0.05 # Slightly reduce confidence if date extraction fails
                 break
-                
+
         # Citation extraction for academic sources
         if source_type in ['arxiv', 'scholar']:
-            citations = re.findall(r'Cited by (\d+)', text)
+            citations = re.findall(r'Cited by (\d+)', text, re.IGNORECASE)
             if citations:
                 metadata["citation_count"] = int(citations[0])
-                # Adjust credibility based on citations
-                metadata["credibility_score"] = min(1.0, metadata["credibility_score"] + (int(citations[0])/1000))
-                
-        # Title extraction
-        title_match = re.search(r'Title:\s+([^\n]+)', text)
-        if title_match:
-            metadata["title"] = title_match.group(1)
-        
+                # Adjust credibility based on citations - more nuanced boost
+                citation_boost = min(0.25, int(citations[0])/500) # Increased boost cap and sensitivity
+                metadata["credibility_score"] = min(1.0, metadata["credibility_score"] + citation_boost)
+            else:
+                metadata["confidence_score"] -= 0.1 # Reduce confidence if citation count not found in scholar/arxiv
+
+        # Title extraction - more robust title extraction
+        title_patterns = [
+            r'Title:\s*([^\n]+)',           # Title label
+            r'^(.*?)(?:\n|$)',              # First line as title if no explicit label
+            r'##\s*(.*?)\s*##'              # Markdown heading level 2 as title
+        ]
+        for pattern in title_patterns:
+            title_match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
+            if title_match:
+                metadata["title"] = title_match.group(1).strip()
+                if not metadata["title"]:
+                    metadata["confidence_score"] -= 0.05 # Slightly reduce if title extraction is weak
+                break
+
         return metadata
-        
+
     def _calculate_base_credibility(self, source_type: str) -> float:
-        """Calculate base credibility score based on source type"""
+        """Calculate base credibility score based on source type - Adjusted scores for better differentiation"""
         credibility_map = {
-            "arxiv": 0.8,
-            "scholar": 0.85,
-            "wikipedia": 0.7,
-            "web": 0.4,
+            "arxiv": 0.90, # Increased for Arxiv
+            "scholar": 0.92, # Highest for Scholar
+            "wikipedia": 0.75, # Slightly increased
+            "web": 0.50,   # Web remains moderate
+            "wolfram_alpha": 0.95 # Very high for Wolfram Alpha
         }
         return credibility_map.get(source_type, 0.5)
 
     def _web_search_wrapper(self, func):
         def wrapper(query):
-            # Check cache
             cache_key = f"web_{hashlib.md5(query.encode()).hexdigest()}"
             cached = self.cache.get(cache_key)
             if cached:
                 self.current_sources.extend(cached["sources"])
                 return cached["result"]
-            
-            # Apply rate limiting
+
             with self.web_limiter:
                 try:
                     result = func(query)
                     urls = self._extract_urls(result)
                     sources = []
-                    
+
                     for url in urls:
                         metadata = self._extract_metadata(result, "web")
                         source = Source(
                             url=url,
                             source_type="web",
-                            relevance=0.7,
+                            relevance=0.6, # Slightly reduced web relevance
                             **metadata
                         )
                         sources.append(source)
-                    
+
                     self.current_sources.extend(sources)
-                    
-                    # Cache result
                     self.cache.set(
                         cache_key,
                         {"result": result, "sources": sources},
                         expire=self.cache_ttl
                     )
-                    
                     return result
                 except Exception as e:
                     return f"Error during web search: {str(e)}. Please try with a more specific query."
@@ -213,33 +291,28 @@
 
     def _wiki_wrapper(self, func):
         def wrapper(query):
-            # Check cache
             cache_key = f"wiki_{hashlib.md5(query.encode()).hexdigest()}"
             cached = self.cache.get(cache_key)
             if cached:
                 self.current_sources.extend(cached["sources"])
                 return cached["result"]
-                
+
             try:
                 result = func(query)
                 url = f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"
-                
                 metadata = self._extract_metadata(result, "wikipedia")
                 source = Source(
                     url=url,
                     source_type="wikipedia",
-                    relevance=0.8,
+                    relevance=0.75, # Increased Wiki relevance
                     **metadata
                 )
                 self.current_sources.append(source)
-                
-                # Cache result
                 self.cache.set(
                     cache_key,
                     {"result": result, "sources": [source]},
                     expire=self.cache_ttl
                 )
-                
                 return result
             except Exception as e:
                 return f"Error during Wikipedia search: {str(e)}. Wikipedia might not have an article on this specific topic."
@@ -247,18 +320,15 @@
 
     def _arxiv_wrapper(self, func):
         def wrapper(query):
-            # Check cache
             cache_key = f"arxiv_{hashlib.md5(query.encode()).hexdigest()}"
             cached = self.cache.get(cache_key)
             if cached:
                 self.current_sources.extend(cached["sources"])
                 return cached["result"]
-                
+
             try:
                 result = func(query)
                 sources = []
-                
-                # Extract paper details
                 papers = result.split('\n\n')
                 for paper in papers:
                     if "arXiv:" in paper:
@@ -266,43 +336,41 @@
                         id_match = re.search(r'arXiv:(\d+\.\d+)', paper)
                         if id_match:
                             arxiv_id = id_match.group(1)
-                        
+
                         if arxiv_id:
                             url = f"https://arxiv.org/abs/{arxiv_id}"
                             metadata = self._extract_metadata(paper, "arxiv")
-                            
-                            # Extract title
-                            title_match = re.search(r'^(.*?)(?:\n|$)', paper)
+
+                            # Title extraction - improved within arxiv wrapper
+                            title_match = re.search(r'^(.*?)(?:\n|$)', paper, re.MULTILINE)
                             if title_match:
                                 metadata["title"] = title_match.group(1).strip()
-                                
-                            # Extract authors
-                            authors_match = re.search(r'Authors?: (.*?)(?:\n|$)', paper)
+
+                            # Authors extraction - improved within arxiv wrapper
+                            authors_match = re.search(r'Authors?: (.*?)(?:\n|$)', paper, re.MULTILINE | re.IGNORECASE)
                             if authors_match:
                                 metadata["authors"] = [a.strip() for a in authors_match.group(1).split(',')]
-                                
-                            # Extract date
-                            date_match = re.search(r'(?:Published|Date): (.*?)(?:\n|$)', paper)
+
+                            # Date extraction - improved within arxiv wrapper
+                            date_match = re.search(r'(?:Published|Date): (.*?)(?:\n|$)', paper, re.MULTILINE | re.IGNORECASE)
                             if date_match:
                                 metadata["publication_date"] = date_match.group(1).strip()
-                                
+
+
                             source = Source(
                                 url=url,
                                 source_type="arxiv",
-                                relevance=0.9,
+                                relevance=0.92, # Increased Arxiv relevance
                                 **metadata
                             )
                             sources.append(source)
-                
+
                 self.current_sources.extend(sources)
-                
-                # Cache result
                 self.cache.set(
                     cache_key,
                     {"result": result, "sources": sources},
                     expire=self.cache_ttl
                 )
-                
                 return result
             except Exception as e:
                 return f"Error during arXiv search: {str(e)}. Try rephrasing your query with more academic terminology."
@@ -310,183 +378,241 @@
 
     def _scholar_wrapper(self, func):
         def wrapper(query):
-            # Check cache
             cache_key = f"scholar_{hashlib.md5(query.encode()).hexdigest()}"
             cached = self.cache.get(cache_key)
             if cached:
                 self.current_sources.extend(cached["sources"])
                 return cached["result"]
-            
-            # Apply rate limiting
+
             with self.scholar_limiter:
                 try:
                     result = func(query)
                     sources = []
-                    
-                    # Process each paper entry
                     papers = result.split('\n\n')
                     for paper in papers:
                         url_match = re.search(r'(https?://[^\s]+)', paper)
                         if url_match:
                             url = url_match.group(1)
                             metadata = self._extract_metadata(paper, "scholar")
-                            
-                            # Extract title
-                            title_match = re.search(r'^(.*?)(?:\n|$)', paper)
+
+                            # Title extraction - improved within scholar wrapper
+                            title_match = re.search(r'^(.*?)(?:\n|$)', paper, re.MULTILINE)
                             if title_match:
                                 metadata["title"] = title_match.group(1).strip()
-                            
-                            # Extract citation count
-                            citation_match = re.search(r'Cited by (\d+)', paper)
+
+                            # Citation count extraction - improved within scholar wrapper
+                            citation_match = re.search(r'Cited by (\d+)', paper, re.IGNORECASE)
                             if citation_match:
                                 metadata["citation_count"] = int(citation_match.group(1))
-                                # Adjust credibility score based on citation count
-                                citation_boost = min(0.2, int(citation_match.group(1))/1000)
+                                citation_boost = min(0.3, int(citation_match.group(1))/400) # Further increased citation boost sensitivity and cap
                                 metadata["credibility_score"] += citation_boost
-                            
+
                             source = Source(
                                 url=url,
                                 source_type="scholar",
-                                relevance=0.9,
+                                relevance=0.95, # Highest relevance for Scholar
                                 **metadata
                             )
                             sources.append(source)
-                    
+
                     self.current_sources.extend(sources)
-                    
-                    # Cache result
                     self.cache.set(
                         cache_key,
                         {"result": result, "sources": sources},
                         expire=self.cache_ttl
                     )
-                    
                     return result
                 except Exception as e:
                     return f"Error during Google Scholar search: {str(e)}. Scholar might be rate-limiting requests."
         return wrapper
 
+    def _wolfram_wrapper(self, func):
+        def wrapper(query):
+            cache_key = f"wolfram_{hashlib.md5(query.encode()).hexdigest()}"
+            cached = self.cache.get(cache_key)
+            if cached:
+                self.current_sources.extend(cached["sources"])
+                return cached["result"]
+
+            try:
+                result = func(query)
+                source = Source(
+                    url="https://www.wolframalpha.com/", # General WolframAlpha URL as source
+                    source_type="wolfram_alpha",
+                    relevance=0.98, # Highest relevance as WolframAlpha is authoritative for factual queries
+                    credibility_score=0.95, # Very high credibility
+                    confidence_score=0.99 # Extremely high confidence in factual data
+                )
+                self.current_sources.append(source)
+                self.cache.set(
+                    cache_key,
+                    {"result": result, "sources": [source]},
+                    expire=self.cache_ttl
+                )
+                return result
+            except Exception as e:
+                return f"Error during WolframAlpha query: {str(e)}. Please check your query format or WolframAlpha API key."
+        return wrapper
+
+
     def _create_agent(self):
+        # Pre-format the schema with proper escaping
+        schema_instructions = self.parser.get_format_instructions()
+        # Replace single braces with double braces in the schema, except for the template variables
+        escaped_schema = schema_instructions.replace('{', '{{').replace('}', '}}')
+        escaped_schema = escaped_schema.replace('{{query}}', '{query}').replace('{{agent_scratchpad}}', '{agent_scratchpad}')
+
         prompt = ChatPromptTemplate.from_messages([
             ("system", """
-            You are an expert research assistant. Follow this structured process:
-            
-            1. Query Analysis: Clarify and break down the research question
-            2. Initial Research: Use web search and Wikipedia for background
-            3. Academic Deep Dive: Use arXiv and Google Scholar for peer-reviewed sources
-            4. Cross-Verification: Check key facts across multiple sources
-            5. Synthesis: Create comprehensive summary with clear attribution
-            
-            For each step, document:
-            - Tools used
-            - Key findings (as bullet points)
-            - Source URLs with proper attribution
-            
-            IMPORTANT GUIDELINES:
-            - Always prioritize peer-reviewed academic sources when available
-            - Explicitly note areas of uncertainty or conflicting information
-            - Distinguish between established facts and emerging research
-            - Extract and include author names and publication dates when available
-            - Rate each source's credibility (low/medium/high)
-            
-            Output must be valid JSON matching the schema:
+            You are an expert research assistant. Your goal is to conduct thorough and high-quality research based on user queries. Follow a structured, multi-step research process to ensure comprehensive and reliable results.
+
+            **Research Process:**
+
+            1. **Understand and Clarify Query (Query Analysis):**
+               - Initial Objective: Analyze the research query, identify key concepts, and clarify any ambiguities. Define the specific research question to be answered.
+               - Tools: None (internal thought process)
+               - Expected Output: Clearly defined research question and initial research objectives.
+
+            2. **Background Research (Initial Research):**
+               - Objective: Gather preliminary information and context on the topic. Use broad web searches and Wikipedia to get an overview and identify key areas.
+               - Tools: web_search, wikipedia
+               - Expected Output: Summary of background information, identification of key terms and concepts, and initial sources.
+
+            3. **Academic Literature Review (Academic Deep Dive):**
+               - Objective: Explore academic databases for in-depth, peer-reviewed research. Use arXiv and Google Scholar to find relevant papers and scholarly articles.
+               - Tools: arxiv_search, google_scholar
+               - Expected Output: List of relevant academic papers, key findings from these papers, and identification of leading researchers or institutions.
+
+            4. **Factual Verification and Data Analysis (Cross-Verification):**
+               - Objective: Verify key facts, statistics, and data points using reliable sources like Wolfram Alpha and targeted web searches. Cross-reference information from different sources to ensure accuracy.
+               - Tools: wolfram_alpha, web_search (for specific fact-checking)
+               - Expected Output: Verified facts and data, resolution of any conflicting information, and increased confidence in research findings.
+
+            5. **Synthesis and Report Generation (Synthesis):**
+               - Objective: Synthesize all gathered information into a coherent and structured research report. Summarize key findings, highlight areas of uncertainty, and provide a quality assessment.
+               - Tools: None (synthesis and writing process)
+               - Expected Output: Comprehensive research report in JSON format, including summary, detailed steps, sources, quality score, and uncertainty areas.
+
+            **IMPORTANT GUIDELINES:**
+
+            - **Prioritize High-Quality Sources:** Always prioritize peer-reviewed academic sources (arXiv, Google Scholar) and authoritative sources (Wolfram Alpha, Wikipedia) over general web sources.
+            - **Explicitly Note Uncertainties:** Clearly identify and document areas where information is uncertain, conflicting, or based on limited evidence.
+            - **Distinguish Facts and Emerging Research:** Differentiate between well-established facts and findings from ongoing or emerging research.
+            - **Extract and Attribute Source Metadata:**  Meticulously extract and include author names, publication dates, citation counts, and credibility scores for all sources.
+            - **Provide Source URLs:** Always include URLs for all sources to allow for easy verification and further reading.
+            - **Assess Source Credibility:** Evaluate and rate the credibility of each source based on its type, reputation, and other available metadata.
+            - **Log Search Queries:** Keep track of the search queries used in each step to ensure transparency and reproducibility.
+
+            **Output Format:**
+            Your output must be valid JSON conforming to the following schema:
             {schema}
-            """.format(schema=self.parser.get_format_instructions())),
-            ("human", "{query}")
+            """.format(schema=escaped_schema)),
+            ("human", "{query}"),
+            ("ai", "{agent_scratchpad}")
         ])
 
         return AgentExecutor(
             agent=create_tool_calling_agent(self.llm, self.tools, prompt),
             tools=self.tools,
             verbose=True,
-            handle_parsing_errors=True
+            handle_parsing_errors=True,
+            max_iterations=10, # Limit agent iterations to prevent runaway loops
+            early_stopping_method="generate" # Stop if no more actions predicted
+
         )
 
     def _evaluate_quality(self, response: ResearchResponse) -> Dict[str, float]:
-        """Enhanced quality evaluation with multiple metrics"""
+        """Enhanced quality evaluation with multiple metrics - Refined Metrics and Weights"""
         metrics = {}
-        
-        # Source diversity (0-1)
+
+        # Source diversity (0-1) - Emphasize academic and high-quality sources
         source_types = {s.source_type for s in response.sources}
-        metrics["source_diversity"] = len(source_types) / 4
-        
-        # Academic ratio (0-1)
-        total_sources = len(response.sources) or 1  # Avoid division by zero
-        academic_sources = sum(1 for s in response.sources 
-                              if s.source_type in ["arxiv", "scholar"])
+        diversity_score = len(source_types) / 5 # Now considering 5 source types
+        metrics["source_diversity"] = min(1.0, diversity_score)
+
+        # Academic and Authoritative Ratio (0-1) - Increased weight for academic/wolfram
+        total_sources = len(response.sources) or 1
+        academic_sources = sum(1 for s in response.sources
+                              if s.source_type in ["arxiv", "scholar", "wolfram_alpha"])
         metrics["academic_ratio"] = academic_sources / total_sources
-        
-        # Verification thoroughness (0-1)
-        verification_steps = sum(1 for step in response.steps 
-                                if any(term in step.step_name.lower() 
-                                       for term in ["verification", "validate", "cross-check"]))
-        metrics["verification_score"] = min(1.0, verification_steps / 2)
-        
-        # Process comprehensiveness (0-1)
-        metrics["process_score"] = min(1.0, len(response.steps) / 5)
-        
-        # Source recency - if we have dates
+
+        # Verification Thoroughness (0-1) - More sensitive to verification steps
+        verification_steps = sum(1 for step in response.steps
+                                if any(term in step.step_name.lower()
+                                       for term in ["verification", "validate", "cross-check", "factual verification"])) # Broader terms
+        metrics["verification_score"] = min(1.0, verification_steps / 3) # Adjusted scaling
+
+        # Process Comprehensiveness (0-1) - Rewards all research steps being present
+        metrics["process_score"] = min(1.0, len(response.steps) / 5) # 5 expected steps
+
+        # Source Recency - if we have dates - Improved recency scoring
         dated_sources = [s for s in response.sources if s.publication_date]
         if dated_sources:
             current_year = datetime.now().year
-            try:
-                # Extract years from various date formats
-                years = []
-                for s in dated_sources:
-                    year_match = re.search(r'(\d{4})', s.publication_date)
-                    if year_match:
-                        years.append(int(year_match.group(1)))
-                
-                if years:
-                    avg_year = sum(years) / len(years)
-                    metrics["recency_score"] = min(1.0, max(0.0, 1 - (current_year - avg_year) / 10))
-                else:
-                    metrics["recency_score"] = 0.5  # Default if we can't extract years
-            except:
-                metrics["recency_score"] = 0.5  # Default on error
+            years = []
+            for s in dated_sources:
+                year_match = re.search(r'(\d{4})', s.publication_date)
+                if year_match:
+                    years.append(int(year_match.group(1)))
+            if years:
+                avg_year = sum(years) / len(years)
+                # Recency decays faster for older sources, plateaus for very recent
+                recency_factor = max(0.0, 1 - (current_year - avg_year) / 7) # Faster decay over 7 years
+                metrics["recency_score"] = min(1.0, recency_factor)
+            else:
+                metrics["recency_score"] = 0.3  # Lower default if dates are unparseable
         else:
-            metrics["recency_score"] = 0.5  # Default when no dates available
-            
-        # Source credibility (0-1)
+            metrics["recency_score"] = 0.2  # Even lower if no dates available
+
+        # Source Credibility (0-1) - Average credibility, penalize low credibility
         if response.sources:
-            metrics["credibility_score"] = sum(s.credibility_score for s in response.sources) / len(response.sources)
+            credibility_avg = sum(s.credibility_score for s in response.sources) / len(response.sources)
+            # Penalize if average credibility is low
+            credibility_penalty = max(0.0, 0.5 - credibility_avg) # Significant penalty if avg below 0.5
+            metrics["credibility_score"] = max(0.0, credibility_avg - credibility_penalty) # Ensure score doesn't go negative
         else:
             metrics["credibility_score"] = 0.0
-            
-        # Calculate weighted overall score
+
+        # Source Confidence - Average confidence in extracted metadata
+        if response.sources:
+            metrics["confidence_score"] = sum(s.confidence_score for s in response.sources) / len(response.sources)
+        else:
+            metrics["confidence_score"] = 0.0
+
+
+        # Calculate weighted overall score - Adjusted weights to emphasize academic rigor and verification
         weights = {
-            "source_diversity": 0.15,
-            "academic_ratio": 0.25,
-            "verification_score": 0.2,
-            "process_score": 0.1,
-            "recency_score": 0.15,
-            "credibility_score": 0.15
+            "source_diversity": 0.10,       # Reduced slightly
+            "academic_ratio": 0.30,         # Increased significantly - Academic rigor is key
+            "verification_score": 0.25,      # Increased - Verification is critical
+            "process_score": 0.10,          # Process still important, but less weight than academic and verification
+            "recency_score": 0.10,          # Recency is relevant, but not as much as rigor
+            "credibility_score": 0.10,      # Credibility remains important
+            "confidence_score": 0.05        # Confidence in data extraction, secondary to credibility
         }
-        
+
         overall_score = sum(metrics[key] * weights[key] for key in weights)
         metrics["overall"] = overall_score
-        
+
         return metrics
 
     def conduct_research(self, query: str) -> str:
-        """Conduct research on the given query"""
-        # Check if we have cached results
+        """Conduct research on the given query - Enhanced research process logging and error handling"""
         cache_key = f"research_{hashlib.md5(query.encode()).hexdigest()}"
         cached = self.cache.get(cache_key)
         if cached:
             return cached
-            
+
         self.current_sources = []
+        self.search_query_log = [] # Reset query log for each research run
         start_time = time.time()
-        
+
         try:
             raw_response = self.agent.invoke({"query": query})
             try:
                 response = self.parser.parse(raw_response["output"])
             except Exception as e:
-                # Fallback to more flexible parsing if strict parsing fails
                 try:
-                    # Extract JSON part from the response
                     json_match = re.search(r'```json\n(.*?)\n```', raw_response["output"], re.DOTALL)
                     if json_match:
                         json_str = json_match.group(1)
@@ -495,155 +621,167 @@
                     else:
                         raise ValueError("Could not extract JSON from response")
                 except Exception as inner_e:
-                    return f"Error parsing research results: {str(e)}\n\nRaw response: {raw_response['output'][:500]}..."
-            
-            # Merge collected sources with any sources in the response
+                    return f"### Error Parsing Research Results\n\nPrimary parsing error: {str(e)}\nSecondary JSON extraction error: {str(inner_e)}\n\nRaw response excerpt (first 500 chars):\n```\n{raw_response['output'][:500]}...\n```\nPlease review the raw response for JSON formatting issues."
+
+
             all_sources = {s.url: s for s in self.current_sources}
             for s in response.sources:
                 if s.url not in all_sources:
                     all_sources[s.url] = s
-                    
             response.sources = list(all_sources.values())
-            
-            # Distribute sources to appropriate research steps
+
             self._assign_sources_to_steps(response)
-            
-            # Calculate quality metrics
+
             quality_metrics = self._evaluate_quality(response)
             response.quality_score = quality_metrics["overall"]
-            
-            # Generate formatted response
+
             formatted_response = self._format_response(response, quality_metrics)
-            
-            # Cache the result
             self.cache.set(cache_key, formatted_response, expire=self.cache_ttl)
-            
             return formatted_response
         except Exception as e:
             duration = time.time() - start_time
             error_msg = f"Error after {duration:.1f}s: {str(e)}"
-            return f"### Research Error\n\n{error_msg}\n\nPlease try again with a more specific query."
+            return f"### Research Error\n\n{error_msg}\n\nPlease try again with a more specific query or check the research tool's configuration."
 
     def _assign_sources_to_steps(self, response: ResearchResponse):
-        """Assign sources to the appropriate research steps based on content matching"""
-        # Create a mapping of source URLs to sources
+        """Assign sources to steps based on keywords and source type - Improved Assignment Logic"""
         url_to_source = {s.url: s for s in response.sources}
-        
-        # Go through each step and assign sources that are mentioned in the output
+
         for step in response.steps:
             step_sources = []
-            
-            # Extract URLs mentioned in this step
-            urls = self._extract_urls(step.output)
-            
-            for url in urls:
+            urls_in_step_output = self._extract_urls(step.output)
+
+            # 1. Direct URL Matching: Prioritize sources explicitly linked in step output
+            for url in urls_in_step_output:
                 if url in url_to_source:
                     step_sources.append(url_to_source[url])
-                    
-            # For steps without explicit URLs, try to match based on content
-            if not step_sources:
-                # Simple matching based on source type and step name
-                if "wikipedia" in step.tools_used or "initial" in step.step_name.lower():
+
+            if not step_sources: # If no direct URL matches, use keyword and tool-based assignment
+                step_content_lower = step.output.lower()
+
+                if "wikipedia" in step.tools_used or "background" in step.step_name.lower():
                     wiki_sources = [s for s in response.sources if s.source_type == "wikipedia"]
-                    step_sources.extend(wiki_sources)
-                    
-                if "arxiv" in step.tools_used or "academic" in step.step_name.lower():
+                    step_sources.extend(wiki_sources[:2]) # Limit Wiki sources per step
+
+                if "arxiv" in step.tools_used or "academic literature" in step.step_name.lower():
                     arxiv_sources = [s for s in response.sources if s.source_type == "arxiv"]
-                    step_sources.extend(arxiv_sources)
-                    
-                if "scholar" in step.tools_used or "academic" in step.step_name.lower():
+                    # Assign Arxiv sources if step output contains academic keywords
+                    if any(keyword in step_content_lower for keyword in ["research", "study", "paper", "academic", "scholar"]):
+                        step_sources.extend(arxiv_sources[:2])
+
+                if "scholar" in step.tools_used or "academic literature" in step.step_name.lower():
                     scholar_sources = [s for s in response.sources if s.source_type == "scholar"]
-                    step_sources.extend(scholar_sources)
-                    
-                if "web_search" in step.tools_used or "initial" in step.step_name.lower():
+                    if any(keyword in step_content_lower for keyword in ["research", "study", "paper", "academic", "citation"]):
+                         step_sources.extend(scholar_sources[:2])
+
+                if "wolfram" in step.tools_used or "factual verification" in step.step_name.lower() or "cross-verification" in step.step_name.lower():
+                    wolfram_sources = [s for s in response.sources if s.source_type == "wolfram_alpha"]
+                    step_sources.extend(wolfram_sources[:1]) # Max 1 Wolfram source
+
+                if "web_search" in step.tools_used or "initial research" in step.step_name.lower():
                     web_sources = [s for s in response.sources if s.source_type == "web"]
-                    step_sources.extend(web_sources[:2])  # Limit to avoid overloading
-            
-            step.sources = step_sources
+                    step_sources.extend(web_sources[:2]) # Limit web sources
+
+            # Deduplicate sources assigned to step, maintaining order if possible
+            unique_step_sources = []
+            seen_urls = set()
+            for source in step_sources:
+                if source.url not in seen_urls:
+                    unique_step_sources.append(source)
+                    seen_urls.add(source.url)
+            step.sources = unique_step_sources
+
 
     def _format_response(self, response: ResearchResponse, metrics: Dict[str, float]) -> str:
-        """Generate enhanced markdown output with quality metrics visualization"""
+        """Generate enhanced markdown output with quality metrics visualization - Improved Markdown Formatting and Metrics Display"""
         markdown = f"## Research Report: {response.topic}\n\n"
-        
-        # Quality metrics section
+        markdown += f"**Research Question:** {response.research_question}\n\n" # Display research question
+
+        # Quality metrics section - Enhanced Metrics Table
         markdown += "### Quality Assessment\n"
-        markdown += f"**Overall Score:** {response.quality_score:.2f}/1.00  \n"
-        markdown += self._generate_metrics_table(metrics)
-        markdown += f"\n**Timestamp:** {response.timestamp}\n\n"
-        
+        markdown += f"**Overall Quality Score:** {response.quality_score:.2f}/1.00  \n\n" # Clear label
+        markdown += self._generate_metrics_table(metrics) # Use table for metrics
+        markdown += f"\n**Research Methodology:** {response.methodology}\n" # Report methodology
+        markdown += f"**Timestamp:** {response.timestamp}\n\n"
+
         # Summary section
         markdown += "### Summary\n"
         markdown += f"{response.summary}\n\n"
-        
-        # Research steps
+
+        # Research steps - More structured step output
         for i, step in enumerate(response.steps, 1):
             markdown += f"### Step {i}: {step.step_name}\n"
+            markdown += f"**Objective:** {step.objective}\n" # Display step objective
             markdown += f"**Tools Used:** {', '.join(step.tools_used)}\n\n"
             markdown += f"{step.output}\n\n"
-            
+
             if step.sources:
                 markdown += "**Sources:**\n"
                 for source in step.sources:
                     source_meta = []
                     if source.title:
-                        source_meta.append(source.title)
+                        source_meta.append(f"*{source.title}*") # Title in bold
                     if source.authors:
-                        source_meta.append(f"by {', '.join(source.authors[:3])}" + 
-                                          (f" et al." if len(source.authors) > 3 else ""))
+                        authors_str = ', '.join(source.authors[:2]) + (f" et al." if len(source.authors) > 2 else "")
+                        source_meta.append(f"by {authors_str}")
                     if source.publication_date:
-                        source_meta.append(source.publication_date)
-                    if source.citation_count:
-                        source_meta.append(f"cited {source.citation_count} times")
-                        
-                    source_detail = f"[{source.source_type}] " + " | ".join(source_meta) if source_meta else source.source_type
+                        source_meta.append(f"({source.publication_date})")
+                    if source.citation_count and source.source_type == 'scholar': # Citation count only for Scholar
+                        source_meta.append(f"Cited {source.citation_count}+ times")
                     credibility_indicator = "⭐" * round(source.credibility_score * 5)
-                    
-                    markdown += f"- [{source_detail}]({source.url}) {credibility_indicator}\n"
+                    confidence_indicator = "📊" * round(source.confidence_score * 5)
+
+                    source_detail = f"[{source.source_type.upper()}] " + " ".join(source_meta) if source_meta else f"[{source.source_type.upper()}]"
+                    markdown += f"- {source_detail}: [Link]({source.url}) Credibility: {credibility_indicator} Confidence: {confidence_indicator}\n" # Added confidence indicator
+
             markdown += "\n"
-            
+
         # Areas of uncertainty
         if response.uncertainty_areas:
             markdown += "### Areas of Uncertainty\n"
             for area in response.uncertainty_areas:
                 markdown += f"- {area}\n"
             markdown += "\n"
-            
+
+        if response.researcher_notes: # Include researcher notes if present
+            markdown += "### Researcher's Notes\n"
+            markdown += f"{response.researcher_notes}\n\n"
+
         return markdown
 
     def _generate_metrics_table(self, metrics: Dict[str, float]) -> str:
-        """Generate a metrics table with gauge visualizations"""
+        """Generate a metrics table with gauge visualizations - Improved Table with Confidence"""
         metric_names = {
+            "overall": "Overall Quality",
             "source_diversity": "Source Diversity",
-            "academic_ratio": "Academic Sources",
-            "verification_score": "Fact Verification",
-            "process_score": "Process Thoroughness",
+            "academic_ratio": "Academic Ratio",
+            "verification_score": "Verification Score",
+            "process_score": "Process Score",
             "recency_score": "Source Recency",
-            "credibility_score": "Source Credibility"
+            "credibility_score": "Avg. Credibility",
+            "confidence_score": "Avg. Confidence" # Added confidence to metrics table
         }
-        
-        table = "| Metric | Score | Rating |\n| --- | --- | --- |\n"
-        
+
+        table = "| Metric | Score | Rating |\n|---|---|---|\n" # Markdown table format
+
         for key, name in metric_names.items():
             if key in metrics:
                 score = metrics[key]
                 stars = "★" * round(score * 5)
-                stars += "☆" * (5 - len(stars))
-                table += f"| {name} | {score:.2f} | {stars} |\n"
-                
+                empty_stars = "☆" * (5 - len(stars))
+                table += f"| {name} | {score:.2f} | {stars}{empty_stars} |\n" # Stars as rating
+
         return table
 
     def _generate_score_bar(self, score: float) -> str:
-        """Generate a base64 encoded score bar visualization"""
+        """Generate a base64 encoded score bar visualization - Unused, can be removed or enhanced for visual output if needed"""
         fig, ax = plt.subplots(figsize=(5, 0.5))
-        
-        # Use color gradient based on score
         if score < 0.4:
             color = 'red'
         elif score < 0.7:
             color = 'orange'
         else:
             color = 'green'
-            
         ax.barh(['Score'], [score], color=color)
         ax.barh(['Score'], [1], color='lightgray', alpha=0.3)
         ax.set_xlim(0, 1)
@@ -656,51 +794,55 @@
 # Gradio Interface with enhanced features
 def create_interface():
     tool = DeepResearchTool()
-    
+
     with gr.Blocks(title="Advanced Research Assistant") as iface:
         gr.Markdown("# Advanced Research Assistant")
-        gr.Markdown("Conduct multi-source academic research with quality metrics and source validation")
-        
+        gr.Markdown("Conduct multi-source academic research with enhanced quality metrics, source validation, and WolframAlpha integration.")
+
         with gr.Row():
             with gr.Column(scale=4):
                 query = gr.Textbox(
-                    placeholder="Enter research query...", 
+                    placeholder="Enter research query...",
                     lines=3,
                     label="Research Query"
                 )
-                
+                research_question = gr.Textbox(
+                    placeholder="Optional: Specify a precise research question...",
+                    lines=1,
+                    label="Research Question (Optional)"
+                )
+
                 with gr.Row():
                     submit_btn = gr.Button("Conduct Research", variant="primary")
                     clear_btn = gr.Button("Clear")
-                
+
                 examples = gr.Examples(
                     examples=[
-                        ["Quantum computing applications in cryptography"],
-                        ["Recent advances in CRISPR gene editing"],
-                        ["The impact of social media on teenage mental health"],
-                        ["Progress in fusion energy research since 2020"]
+                        ["Quantum computing applications in cryptography", "What are the most promising applications of quantum computing in modern cryptography?"],
+                        ["Recent advances in CRISPR gene editing", "What are the latest breakthroughs in CRISPR gene editing technology and their potential impacts?"],
+                        ["The impact of social media on teenage mental health", "How does social media use affect the mental health and well-being of teenagers?"],
+                        ["Progress in fusion energy research since 2020", "What significant advancements have been made in fusion energy research and development since 2020?"]
                     ],
-                    inputs=query,
-                    label="Example Queries"
+                    inputs=[query, research_question],
+                    label="Example Queries with Research Questions"
                 )
-        
+
         output = gr.Markdown(label="Research Results")
-        
+
         submit_btn.click(
             fn=tool.conduct_research,
             inputs=query,
             outputs=output
         )
-        
+
         clear_btn.click(
-            fn=lambda: ("", ""),
+            fn=lambda: ("", "", ""),
             inputs=None,
-            outputs=[query, output]
+            outputs=[query, research_question, output]
         )
-    
+
     return iface
 
 if __name__ == "__main__":
     iface = create_interface()
     iface.launch()
-
```
