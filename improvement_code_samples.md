```python
# sample implementation suggestion 1:
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGo
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools.google_scholar import GoogleScholarQueryRun
from datetime import datetime
import gradio as gr
import json

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
    def __init__(self):
        self.llm = ChatAnthropic(model="claude-3-5-sonnet-20240229", temperature=0.1) # Use a deterministic temperature
        self.parser = PydanticOutputParser(pydantic_object=ResearchResponse)
        self.tools = self._setup_tools()
        self.agent = self._setup_agent()

    def _setup_tools(self):
        search = DuckDuckGoSearchRun()
        wiki_api = WikipediaAPIWrapper(top_k_results=3)
        arxiv_api = ArxivAPIWrapper()
        scholar = GoogleScholarQueryRun()

        tools = [
            Tool(
                name="web_search",
                func=search.run,
                description="Search the web for general information using DuckDuckGo. Good for broad queries and initial information gathering."
            ),
            Tool(
                name="wikipedia",
                func=WikipediaQueryRun(api_wrapper=wiki_api).run,
                description="Query Wikipedia for factual information and background context. Useful for quick facts and encyclopedic knowledge."
            ),
            Tool(
                name="arxiv_search",
                func=ArxivQueryRun(api_wrapper=arxiv_api).run,
                description="Search arXiv for academic papers and preprints in physics, mathematics, computer science, and related fields. Ideal for finding cutting-edge research."
            ),
            Tool(
                name="google_scholar_search",
                func=scholar.run,
                description="Search Google Scholar for academic literature across various disciplines. Best for comprehensive scholarly research and finding peer-reviewed articles."
            ),
            # Fact-checking is now integrated within the main research process, not a separate tool.
            Tool(
                name="save_research_output",
                func=self._save_research,
                description="Save the research output to a text file. Useful for archiving and sharing research findings."
            )
        ]
        return tools

    def _setup_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert research assistant AI, tasked with conducting in-depth investigations and generating comprehensive research reports. Your process must follow these rigorous steps to ensure accuracy, depth, and relevance:

            1. **Query Analysis and Refinement:** Begin by thoroughly analyzing the user's query. Identify the core question, key concepts, and any ambiguities.  If the query is too broad, break it down into more specific sub-questions.  If it's too narrow, consider broadening the scope to gather sufficient context.  *Output a clear, concise research question or set of sub-questions.*

            2. **Initial Information Gathering:** Use the 'web_search' and 'wikipedia' tools to gather preliminary information. This step establishes a foundational understanding of the topic. *Synthesize the information from these sources, noting any key facts, figures, or controversies.  Record the tools used as sources.*

            3. **Deep Dive into Academic Literature:** Utilize 'arxiv_search' and 'google_scholar_search' to find relevant academic papers, preprints, and scholarly articles. This step delves into the core of existing research. *Summarize the findings of at least 3 relevant papers, focusing on their methodologies, results, and conclusions.  Record specific papers (with URLs or identifiers if possible) as sources.*

            4. **Cross-Verification and Fact-Checking:** Critically evaluate the information gathered in the previous steps.  Identify any conflicting information or claims requiring verification. *Cross-reference information across multiple sources (web, Wikipedia, academic) to ensure its accuracy.  Explicitly state which facts have been verified and how.*

            5. **Synthesis and Summary:** Integrate all verified information into a coherent summary. Address the original research question(s) directly. *Present a concise summary of the findings, highlighting key insights and conclusions.*

            6. **Quality Assessment:**  Evaluate the overall quality of your research.  Consider the breadth and depth of sources, the consistency of information, and the presence of any remaining uncertainties. *Assign a quality score between 0 and 1, where 1 represents the highest quality.  Justify your score.*

            **Output Format:**  Your final output MUST adhere to the following JSON structure:

            ```json
            {schema}
            ```
            """.format(schema=self.parser.get_format_instructions())),
            ("user", "{input}"),
            ("assistant", "Okay, let's get started. I will follow the steps outlined above to conduct thorough research on your query."),
            ("user", "Continue"), # Added a "Continue" message to help guide the flow.
        ])

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return agent

    def _fact_check(self, statement: str) -> str:
        # Fact-checking is now handled within the main research process.
        # This function is kept for compatibility with the old tool definition, but is not used.
        return "Fact-checking is integrated into the main research process."
       

    def _evaluate_quality(self, steps: List[ResearchStep]) -> float:
        """Evaluates the quality of the research based on source diversity and step depth."""
        if not steps:
            return 0.0

        tool_diversity = len(set(tool for step in steps for tool in step.tools_used))
        num_steps = len(steps)

        # Basic quality score based on tool diversity and number of steps.
        quality_score = (tool_diversity / len(self.tools)) * 0.6 + (num_steps / 5) * 0.4  # Assuming 5 steps is ideal

        return min(max(quality_score, 0.0), 1.0) # Ensure score is between 0 and 1


    def _save_research(self, research_json : str)-> str:
        """Saves research output to file"""
        try:
            research_data = json.loads(research_json)
            research_response = ResearchResponse(**research_data)
            # Create a unique filename using the timestamp
            timestamp = research_response.timestamp
            filename = f"research_output_{timestamp}.txt"

            with open(filename, "w") as f:
                f.write(f"Topic: {research_response.topic}\n\n")
                f.write(f"Summary:\n{research_response.summary}\n\n")
                f.write("Detailed Steps:\n")
                for i, step in enumerate(research_response.detailed_steps):
                    f.write(f"  Step {i+1}: {step.step_name}\n")
                    f.write(f"    Description: {step.description}\n")
                    f.write(f"    Tools Used: {', '.join(step.tools_used)}\n")
                    f.write(f"    Output:\n{step.output}\n")
                    f.write(f"    Sources: {', '.join(step.sources)}\n\n")
                f.write(f"Sources:\
```

```python
# sample implementation suggestion 2:
import os
import re
import json
import time
import logging
import uuid
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field, validator
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("research_tool.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

# Enhanced data models
class Source(BaseModel):
    """Model for tracking research sources with detailed metadata"""
    source_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_type: str = Field(description="Type of source (web, wiki, arxiv, scholar, etc.)")
    title: Optional[str] = Field(default=None, description="Title of the source")
    url: Optional[str] = Field(default=None, description="URL or identifier for the source")
    authors: Optional[List[str]] = Field(default=None, description="Authors of the source if applicable")
    publication_date: Optional[str] = Field(default=None, description="Publication date if applicable")
    relevance_score: Optional[float] = Field(default=None, description="Relevance score from 0-1")
    
    @validator('relevance_score')
    def validate_score(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError('Relevance score must be between 0 and 1')
        return v

class ResearchStep(BaseModel):
    """Enhanced model for research steps with more detailed tracking"""
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    step_name: str = Field(description="Name of the research step")
    description: str = Field(description="Detailed description of the step and its objectives")
    tools_used: List[str] = Field(description="List of tools used in this step")
    output: str = Field(description="Output or findings from this step")
    sources: List[Source] = Field(default_factory=list, description="Detailed sources used in this step")
    start_time: str = Field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = Field(default=None)
    duration_seconds: Optional[float] = Field(default=None)

class ResearchResponse(BaseModel):
    """Enhanced research response model with additional metadata and metrics"""
    research_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str = Field(description="The main topic of the research")
    query: str = Field(description="The original research query")
    summary: str = Field(description="A concise summary of the research findings")
    key_findings: List[str] = Field(default_factory=list, description="Key findings or insights from the research")
    detailed_steps: List[ResearchStep] = Field(description="List of research steps taken")
    sources: List[Source] = Field(default_factory=list, description="Aggregated list of all sources used")
    quality_metrics: Dict[str, float] = Field(
        default_factory=lambda: {
            "overall_score": 0.0,
            "source_diversity": 0.0,
            "academic_rigor": 0.0,
            "fact_check_coverage": 0.0,
            "depth_score": 0.0
        }
    )
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    duration_seconds: float = Field(default=0.0)
    research_depth: str = Field(default="medium")
    limitations: List[str] = Field(default_factory=list)
    future_directions: List[str] = Field(default_factory=list)

class DeepResearchTool:
    """Advanced research tool with enhanced capabilities for comprehensive investigation"""
```

---
Now let's analyze the provided Deep Research Tool code and identify areas for improvement:

**1. Overall Structure and Design:**

*   **Good:** The code uses a structured approach with Pydantic models (`ResearchStep`, `ResearchResponse`) for defining the output format. This is excellent for ensuring consistent and predictable outputs from the LLM.
*   **Good:** The use of LangChain's `create_tool_calling_agent` and `AgentExecutor` is a modern and efficient way to build agents that interact with tools.
*   **Good:** The separation of concerns between `DeepResearchTool`, tools, and the agent is well-defined.
*   **Good:** The use of a system prompt to guide the agent's behavior is crucial for achieving the desired multi-step research process.
*   **Needs Improvement:** The `_fact_check` function, while improved, still relies on web search and Wikipedia.  A dedicated fact-checking API or a more sophisticated approach to evaluating source credibility would be a significant improvement.
*   **Needs Improvement:** The `_save_research` function appends to the same file. This should be changed to create unique filenames, perhaps using a timestamp, to prevent overwriting previous research.
*  **Needs Improvement:** The prompt is long and could be more concise and focused.
*   **Needs Improvement:** Error handling in `conduct_research` is a good start, but it could be more granular, providing more specific feedback to the user about what went wrong (e.g., tool failure, parsing error).
*   **Needs Improvement:** The quality evaluation is still relatively basic.  It could be significantly enhanced by considering factors like source credibility, depth of analysis within each step, and the coherence of the overall narrative.
*   **Needs Improvement:** The code could benefit from more robust error handling throughout, especially within the tool interactions.

**2. Tools:**

*   **Good:** The selection of tools (DuckDuckGo, Wikipedia, Arxiv, Google Scholar) covers a good range of sources for general information and academic research.
*   **Needs Improvement:** As mentioned above, the lack of a dedicated fact-checking tool is a weakness.
*   **Potential Improvement:** Consider adding tools for specific domains if the research tool is intended for specialized use (e.g., medical databases, legal databases).

**3. Agent and Prompt:**

*   **Good:** The system prompt outlines a clear multi-step research process.
*   **Needs Improvement:** The prompt could be more explicit in instructing the agent to *extract* specific information from the tools, rather than just summarizing the tool's output. For example, when using Google Scholar, it should be instructed to extract the title, authors, abstract, and publication year of relevant papers.
*   **Needs Improvement:** The prompt should be more directive in how the agent should handle conflicting information found during fact-checking.
*   **Needs Improvement:** The prompt asks agent to "Wrap your final output in the format provided by the parser instructions", it can be simplified to just state that the response should be in JSON and follow the schema defined, as it is already handled by  `create_tool_calling_agent`.
*   **Needs Improvement:** The interaction with the agent (`"human"`, `"query"`) needs to be more dynamic and iterative.  The current setup is essentially a single turn.

**4. Quality Evaluation:**

*   **Good:** The `_evaluate_quality` function attempts to quantify the quality of the research.
*   **Needs Improvement:** The current implementation is simplistic. It should consider factors such as:
    *   **Source Credibility:**  Not all sources are equal.  Academic papers should be weighted higher than general web pages.
    *   **Information Consistency:**  Does the information from different sources agree?  Discrepancies should lower the quality score.
    *   **Depth of Analysis:**  Are the research steps superficial, or do they delve into the details?
    *   **Completeness:** Does the research adequately address all aspects of the query?

**5. Gradio Integration (Future Step):**

*   The code doesn't yet include the Gradio UI. This will be a crucial step for making the tool user-friendly.
*   The Gradio UI should allow users to:
    *   Input their research query.
    *   View the research progress (perhaps with intermediate outputs from each step).
    *   View the final research report in a structured and readable format.
    *   Potentially provide feedback to refine the research (iterative refinement).

**6. Fact-Checking:**

*   **Critical Weakness:** As highlighted multiple times, the current fact-checking relies on general web search and Wikipedia. This is insufficient for rigorous research.
*   **Ideal Solution:** Integrate a dedicated fact-checking API (e.g., Google Fact Check Tools API, Full Fact API).  These APIs provide more reliable and structured fact-checking capabilities.
*   **Alternative (Less Ideal):** Develop a more sophisticated heuristic for evaluating source credibility based on factors like domain name, publication date, author reputation, and cross-referencing with known reputable sources.

**7. Source Tracking:**

*   **Needs Improvement:** The `sources` field in `ResearchStep` is a good start, but it needs to be more precise.  Ideally, it should include:
    *   **URLs:** For web pages and online articles.
    *   **DOIs or arXiv IDs:** For academic papers.
    *   **Titles and Authors:** As a fallback when URLs/DOIs are not available.
*   **Challenge:** Extracting URLs reliably from the output of tools like DuckDuckGoSearchRun can be difficult, as it returns text snippets, not structured data.

**8. Iterative Refinement:**

*   **Missing:** The current tool does not support iterative refinement.  This is a valuable feature for deep research, where the initial results might lead to further questions or refinements of the research direction.
*   **Implementation:**  This would require a more complex agent loop that allows for user feedback and re-submission of refined queries.

**9. Error Handling:**

* **Needs Improvement:** Error handling is basic.
*   **Improvement:** Implement `try-except` blocks around all tool calls and potentially within the agent's reasoning process to catch and handle exceptions gracefully.  Provide informative error messages to the user.

**10. Code Style and Readability:**

*   **Good:** The code is generally well-structured and readable.
*   **Improvement:** Add more comments to explain complex logic, especially within the agent's prompt and the quality evaluation function.
*    **Improvement:** use type hints and docstrings to make functions self-documenting.

**Action Plan for Improvement (Prioritized):**

1.  **Improve Source Tracking and Fact-Checking (within the existing framework):**
    *   Modify the prompt to be *much* more explicit about extracting source information (URLs, titles, authors) from each tool's output.
    *   Since reliable URL extraction from DuckDuckGo is difficult, focus on extracting titles and authors as a proxy for source identification.
    *   Improve the `_fact_check` function to use the extracted source information to perform a more targeted search for corroborating or conflicting evidence. This is still a heuristic approach, but it's better than the current implementation.

2.  **Refine the Agent Prompt:**
    *   Make the prompt more concise and direct.
    *   Clearly define the expected behavior for each research step.
    *   Provide specific instructions on how to handle conflicting information.
    *   Remove redundant instructions about output formatting.

3.  **Improve Quality Evaluation:**
    *   Incorporate source diversity (number of unique sources) and depth of steps (number of steps) into the quality score.
    *   Add a bonus for using academic sources (Arxiv, Google Scholar).

4.  **Implement Gradio UI:**
    *   Create a basic Gradio interface that allows users to input a query and view the formatted research output.

5.  **Improve Error Handling:**
    *   Add more specific error handling in `_fact_check` and `conduct_research`.

6.  **Enhance `_save_research`:**
    * Modify the function to generate unique filenames using timestamps.

7.  **JSON Output:**
    * The original code has the agent instructed to "Wrap your final output in the format provided by the parser instructions.". This can be improved by simply stating the JSON requirement directly.

8.  **Iterative Refinement and Dedicated Fact-Checking API (Future Work):**
    *   These are more complex improvements that would require significant changes to the agent's architecture and the integration of external APIs.  They should be considered for future development.

Now, let's start implementing these improvements. I'll focus on steps 1-7, as they can be achieved within the current framework without major architectural changes.

