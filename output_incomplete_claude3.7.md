<think>
Let me carefully review the information provided and develop a comprehensive plan for implementing an improved Deep Research Tool.

First, I notice that the prompt is cut off mid-way through the system prompt in the sample code. I'll need to infer what the complete system prompt might have been based on the context and purpose of the tool.

I also notice that I wasn't provided with the complete code files that were mentioned in the prompt (`sample_agent_code_use_as_reference.py`, `tools.py`, and `sample deep research tool with improved design.py`). I only see a partial implementation of what appears to be the `sample deep research tool with improved design.py` file.

Given these constraints, I'll need to:
1. Complete the implementation based on what I can infer from the partial code
2. Enhance it based on best practices for deep research tools
3. Create a fully working Python code with a Gradio web UI
4. Write a comprehensive research paper

Let me start by completing the implementation of the `DeepResearchTool` class based on what I can infer from the partial code.

The partial code shows:
- A `DeepResearchTool` class initialized with Claude 3.5 Sonnet
- A Pydantic model for structuring research responses
- Several tools already set up (web search, Wikipedia, arXiv, Google Scholar, fact checking, saving research)
- The beginning of a system prompt for the agent

I'll need to complete:
- The system prompt
- The `_fact_check` method
- The `_save_research` method
- The `conduct_research` method
- The Gradio UI implementation

Let me now think about best practices for deep research tools based on current knowledge:

1. **Multi-step reasoning**: The tool should break down complex research questions into manageable steps, following a logical progression.

2. **Source diversity and evaluation**: Using multiple sources and evaluating their credibility is crucial for comprehensive research.

3. **Iterative refinement**: The research process should allow for refining queries based on initial findings.

4. **Transparency**: The tool should clearly document the sources used and reasoning steps taken.

5. **Fact-checking mechanisms**: Incorporating explicit verification steps improves accuracy.

6. **User control and guidance**: Allowing users to guide the research direction improves relevance.

7. **Structured output**: Presenting findings in a well-organized format improves usability.

8. **Knowledge synthesis**: Going beyond information retrieval to synthesize insights.

9. **Error handling and uncertainty communication**: The tool should gracefully handle failures and communicate limitations.

10. **Customizable depth**: Allowing users to specify how deep the research should go.

Now, let me implement a complete, enhanced version of the Deep Research Tool.
</think>

# Improved Deep Research Tool Implementation

After reviewing the provided code snippet, I'll develop an enhanced Deep Research Tool that incorporates best practices for conducting comprehensive research. Let me first complete the implementation and then create the Gradio web UI.

## Complete Implementation of the Deep Research Tool

```python
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools.google_scholar import GoogleScholarQueryRun
from datetime import datetime
import gradio as gr
import json
import re
import logging
import time
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class ResearchStep(BaseModel):
    step_name: str = Field(description="Name of the research step")
    description: str = Field(description="Detailed description of the step and its objectives")
    tools_used: List[str] = Field(description="List of tools used in this step")
    output: str = Field(description="Output or findings from this step")
    sources: List[str] = Field(default_factory=list, description="Sources used in this step, including URLs or tool names")

class ResearchResponse(BaseModel):
    topic: str = Field(description="The main topic of the research")
    summary: str = Field(description="A concise summary of the research findings")
    detailed_steps: List[ResearchStep] = Field(description="List of research steps taken")
    sources: List[str] = Field(default_factory=list, description="Aggregated list of all sources used throughout the research")
    quality_score: float = Field(..., ge=0, le=1, description="Overall quality score of the research output")
    timestamp: str = Field(description="Timestamp of when the research was completed")

class DeepResearchTool:
    def __init__(self, model_name="claude-3-5-sonnet-20241022", research_depth="medium"):
        """
        Initialize the Deep Research Tool with specified model and research depth.
        
        Args:
            model_name: The model to use for research (default: claude-3-5-sonnet)
            research_depth: Depth of research - "basic", "medium", or "comprehensive"
        """
        self.llm = ChatAnthropic(model=model_name)
        self.parser = PydanticOutputParser(pydantic_object=ResearchResponse)
        self.research_depth = research_depth
        self.tools = self._setup_tools()
        self.agent = self._setup_agent()
        self.research_cache = {}  # Cache for storing research results
        
    def _setup_tools(self):
        """Set up the research tools with enhanced capabilities"""
        search = DuckDuckGoSearchRun()
        wiki_api = WikipediaAPIWrapper(top_k_results=3)
        arxiv_api = ArxivAPIWrapper(top_k_results=5, sort_by="relevance")
        scholar = GoogleScholarQueryRun()

        tools = [
            Tool(
                name="web_search",
                func=self._enhanced_web_search,
                description="Search the web for general information using DuckDuckGo. Good for broad queries and initial information gathering. Returns search results with source URLs when available."
            ),
            Tool(
                name="wikipedia",
                func=self._enhanced_wiki_search,
                description="Query Wikipedia for factual information and background context. Useful for quick facts and encyclopedic knowledge. Returns content with source links."
            ),
            Tool(
                name="arxiv_search",
                func=self._enhanced_arxiv_search,
                description="Search arXiv for academic papers and preprints in physics, mathematics, computer science, and related fields. Ideal for finding cutting-edge research. Returns paper summaries with links."
            ),
            Tool(
                name="google_scholar_search",
                func=scholar.run,
                description="Search Google Scholar for academic literature across various disciplines. Best for comprehensive scholarly research and finding peer-reviewed articles."
            ),
            Tool(
                name="fact_check_statement",
                func=self._fact_check,
                description="Verify the accuracy of a statement by cross-referencing information from multiple reliable sources. Crucial for ensuring information accuracy."
            ),
            Tool(
                name="synthesize_information",
                func=self._synthesize_information,
                description="Synthesize information from multiple sources into a coherent narrative or analysis. Useful for creating summaries or identifying patterns across sources."
            ),
            Tool(
                name="save_research_output",
                func=self._save_research,
                description="Save the research output to a text file for future reference and documentation. Useful for archiving and sharing research findings."
            )
        ]
        
        self.tool_map = {tool.name: tool for tool in tools}
        return tools
    
    def _enhanced_web_search(self, query: str) -> str:
        """Enhanced web search that attempts to extract source URLs from results"""
        try:
            search = DuckDuckGoSearchRun()
            results = search.run(query)
            
            # Extract URLs from the results when possible using regex
            urls = re.findall(r'https?://[^\s]+
