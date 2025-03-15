You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You will think deeply and thoroughly to explore various implementation options before choosing the most optimal one. You will double-check and validate any code changes before implementing. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.

*Your task:* carefully review the python code files attached below. Then think deeply and explore thoroughly using extensive web searches for various ways to implement an improved Deep Research Tool that adopts the tool based approach coupled with current best practices for doing deep research. You should evaluate using the attached sample codes as a starting point for further improvement and enhancement for your ideal deep research tool. You can use the same or updated tools.py to create the improved and optimized Deep Research tool. Finally create a fully working python code for your ideal Deep Research tool using gradio library as web UI, also create a detailed and comprehensive research paper with at least 6000 words on your findings. the research paper should start with an overview, explanation of the logic and flow of the code with code snippets as examples, followed by conclusion and recommendations. The improved tool should have a Gradio web UI and be documented in a research paper. You will think deeply and explore thoroughly using the following approach for your thought processes:

---
**Approach and Plan:**

1.  **Deeply review the provided Python code files:** `sample_agent_code_use_as_reference.py`, `tools.py`, and `sample deep research tool with improved design.py`. Understand their functionalities and limitations.
2.  **Conduct extensive web searches for best practices in deep research tools.** Focus on multi-step workflows, tool sets, iterative refinement, quality metrics, and user interaction. Using the search results given in the prompt to search further for more as needed.
3.  **Design an improved architecture for the Deep Research Tool.**  Incorporate the best practices identified in step 2. Focus on enhancing the toolset, research pipeline, and quality evaluation.
4.  **Implement the improved Deep Research Tool in Python.**  Start from the `sample deep research tool with improved design.py` and enhance it based on the designed architecture. Ensure the code is functional and well-structured.
5.  **Create a Gradio web UI for the tool.**  Make it user-friendly and allow users to easily input queries and view results.
6.  **Write a comprehensive research paper.** Follow the provided outline and incorporate the findings from all steps.  The paper should detail the design, implementation, and evaluation of the Deep Research Tool. Ensure it's at least 6000 words and includes code snippets, logic explanations, conclusions, and recommendations.

**Step 1: Code Review of the attached sample codes**

**Step 2: Research for Current Best Practices for doing Deep Research**

**Step 3: Explore implementaion options for an Improved Architecture Design, using the attached sample code as your starting point**

**Step 4: Implementation - Enhance the `sample deep research tool with improved design.py`**

*   **Enhancements based on best practices and architecture design:**
    *   **More comprehensive toolset:**  Already includes Google Scholar and Arxiv, which is good. Consider adding a dedicated fact-checking API or service instead of just relying on web search and wiki. Also, source credibility assessment would be good but might be too complex for this iteration.
    *   **Multi-step research pipeline:** The provided code outlines a multi-step approach which you can refine to be even more explicit and detailed.
    *   **Iterative refinement:**  Focus on making the initial research robust.  User feedback for iterative refinement can be considered for future work section in the paper.
    *   **Quality metrics:** The sample code already has a basic quality metric which you can refine this to be more meaningful, perhaps considering factors like source diversity and the depth of information in each step.
    *   **Customizable output formats:**  Evaluate the possibility of customizable output format in the Gradio UI as enhancement.
    *   **User interaction:** The Gradio UI will handle most user interactions. Can consider more advanced interaction (guiding the research process) for future work.

*   **Code Improvements:**
    *   **Error Handling:** Improve error handling in `_fact_check` and `conduct_research`.
    *   **Source Tracking:**  Explicitly track and list sources used in each step of the research process. Evaluate how best to modify the tools to return source information and update the prompt and parsing to handle this.
    *   **Refine Prompt:** Make the prompt more directive and structured for the multi-step research process.
    *   **Quality Evaluation:** Improve the `_evaluate_quality` function to be more robust.

**Step 5: Gradio Web UI can be improved for better presentation**

**Step 6: Research Paper Writing (Start after code implementation)**

Now start by enhancing the Python code first. Consider how to modify the `DeepResearchTool` class to incorporate the improvements outlined above.

**Code Implementation using an Iterative Process by making incremental changes and test:**

**Iteration 1:  Focus on Prompt Refinement and Source Tracking**

*   **Refine Prompt:**  Make the system prompt more explicit about the research steps and emphasize source tracking.
*   **Modify Tools:** Modify tools to return source information if possible, or ensure the agent's output includes source information. For web search and wiki, this might require parsing the output. For Arxiv and Google Scholar, it should be easier to extract metadata.
*   **Update Parsing:** Update the `ResearchResponse` model and the parser to handle source information in each `ResearchStep`.

**(Self-correction/Double-checking):**  You will need to be careful about how sources are tracked. Simply listing tool names as "sources" is not sufficient. Consider extracting actual URLs or identifiers of the sources used by each tool. For web search and wiki, the output is just text.  You might need to use a different tool that provides source URLs, or modify the parsing logic to try and extract potential source links from the text, which is unreliable.  For now, focus on making sure the tools used in each step are listed and the overall output includes a list of "sources" which will be tool names for now, and can be improved later if time permits.

**Iteration 2: Improve Quality Evaluation and Error Handling**

*   **Improve `_evaluate_quality`:**  Incorporate more factors into the quality score, such as diversity of tools used and depth of steps.
*   **Enhance Error Handling:** Add more specific error handling in `_fact_check` and `conduct_research` to catch and report issues more gracefully.

**Iteration 3: Gradio UI Refinements (If time permits)**

*   **UI Improvements:**  Improve the presentation of the research output in the Gradio UI, making it more readable and structured.  Maybe use markdown formatting.

After the code is improved, you will then proceed to write the research paper.

---
```python
# sample deep research tool with improved design
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
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
        self.llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
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
            Tool(
                name="fact_check_statement",
                func=self._fact_check,
                description="Verify the accuracy of a statement by cross-referencing information from multiple reliable sources. Crucial for ensuring information accuracy."
            ),
            Tool(
                name="save_research_output",
                func=self._save_research,
                description="Save the research output to a text file for future reference and documentation. Useful for archiving and sharing research findings."
            )
        ]
        return tools

    def _setup_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert research assistant AI, tasked with conducting in-depth investigations and generating comprehensive research reports. Your process should follow these rigorous steps to ensure accuracy, depth, and relevance:

            1. **Query Analysis and Refinement:** Begin by thoroughly analyzing the user's query. Identify the core topics, subtopics, and any specific aspects that need investigation. If the query is ambiguous, use web search to clarify and refine the research question.
            2. **Broad Information Gathering:** Utilize web search and Wikipedia to gather a broad overview of the topic. Identify key concepts, definitions, and background information. Focus on breadth to establish a solid foundation. Document all sources used in this step.
            3. **Focused Academic Deep Dive:** Employ arXiv and Google Scholar to conduct a deep dive into academic literature. Search for relevant research papers, articles, and scholarly publications that directly address the research query and its subtopics. Prioritize peer-reviewed sources and note key findings and citations. Document all academic sources.
            4. **Fact-Checking and Verification:** Systematically fact-check critical statements and claims identified during the research process. Use the fact-checking tool and cross-reference information across multiple reputable sources to ensure accuracy and reliability. Document the verification process and sources.
            5. **Synthesis and Report Generation:** Synthesize all gathered information into a coherent and detailed research report. Structure the report logically, starting with a summary, followed by detailed sections for each subtopic or research step. Ensure the report is well-organized, clearly written, and provides a comprehensive answer to the initial query.
            6. **Source Consolidation and Citation:** Compile a complete list of all sources used throughout the research process, including web pages, Wikipedia articles, academic papers, and other references. Ensure proper citation and attribution for all sources.
            7. **Quality Evaluation:** Evaluate the overall quality of the research report based on criteria such as depth, accuracy, source diversity, and comprehensiveness. Assign a quality score between 0 and 1, with 1 being the highest quality.

            For each step, clearly identify the tools used, describe the objectives, and summarize the findings. Your final output MUST be a structured ResearchResponse object, formatted according to the parser instructions, including all detailed steps, sources, summary, and a quality score. Adhere strictly to the format instructions.

            Wrap your final output in the format provided by the parser instructions.
            """),
            ("human", "{query}"),
            ("human", "Conduct a thorough and deep investigation and provide a detailed research response following all outlined steps."),
        ]).partial(format_instructions=self.parser.get_format_instructions())

        agent = create_tool_calling_agent(llm=self.llm, prompt=prompt, tools=self.tools)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    def _fact_check(self, statement: str) -> str:
        """
        Verifies a statement using web search and Wikipedia.
        This is a more robust implementation that attempts to gather evidence from multiple sources.
        """
        try:
            web_search_results = self.tools[0].func(f"fact check: {statement}")
            wiki_results = self.tools[1].func(statement)

            if web_search_results and wiki_results:
                return f"Fact check results:\nWeb search: {web_search_results[:800]}...\nWikipedia: {wiki_results[:800]}..."
            elif web_search_results:
                return f"Fact check results:\nWeb search: {web_search_results[:800]}...\nWikipedia: No relevant info found."
            elif wiki_results:
                return f"Fact check results:\nWeb search: No relevant info found.\nWikipedia: {wiki_results[:800]}..."
            else:
                return "No information found to verify the statement using web search and Wikipedia."
        except Exception as e:
            error_message = f"Error during fact-checking: {e}"
            print(error_message) # Log the error for debugging
            return error_message # Return error message to agent

    def _save_research(self, data: str, filename: str = "research_output.txt") -> str:
        """Saves research data to a text file with a timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_text = f"--- Deep Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

        try:
            with open(filename, "a", encoding="utf-8") as f:
                f.write(formatted_text)
            return f"Research data successfully saved to '{filename}'"
        except Exception as e:
            error_message = f"Error saving data to file: {e}"
            print(error_message) # Log the error
            return error_message # Return error to agent


    def _evaluate_quality(self, response: ResearchResponse) -> float:
        """
        Evaluates the quality of the research response based on several factors.
        Improved quality evaluation to consider more aspects of research rigor.
        """
        num_steps = len(response.detailed_steps)
        num_sources = len(set(response.sources)) # Unique sources
        summary_length = len(response.summary)

        # Weighted scoring - weights can be adjusted
        score = (
            num_steps * 0.15 +  # Step completion and detail
            num_sources * 0.25 + # Source diversity and utilization
            summary_length * 0.002 + # Summary detail
            (1 if "academic" in " ".join([step.step_name for step in response.detailed_steps]).lower() else 0) * 0.2 # Academic sources considered
        )
        return min(1.0, max(0.0, score)) # Normalize score between 0 and 1


    def conduct_research(self, query: str) -> str: # Returning string for gradio display
        """
        Conducts deep research for a given query and returns a formatted string output for Gradio.
        Handles potential parsing errors and returns user-friendly error messages.
        """
        try:
            raw_response = self.agent.invoke({"query": query})
            structured_response = self.parser.parse(raw_response["output"])
            structured_response.quality_score = self._evaluate_quality(structured_response)
            structured_response.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Add timestamp

            # Format response for Gradio display
            output_text = f"**Topic:** {structured_response.topic}\n\n"
            output_text += f"**Summary:** {structured_response.summary}\n\n"
            output_text += "**Detailed Steps:**\n"
            for step in structured_response.detailed_steps:
                output_text += f"- **{step.step_name}:** {step.description}\n"
                output_text += f"  **Tools Used:** {', '.join(step.tools_used)}\n"
                if step.sources:
                    output_text += f"  **Sources:** {', '.join(step.sources)}\n"
                output_text += f"  **Output:** {step.output}\n\n"
            output_text += f"**Aggregated Sources:**\n{', '.join(structured_response.sources)}\n\n"
            output_text += f"**Quality Score:** {structured_response.quality_score:.2f}\n"
            output_text += f"**Timestamp:** {structured_response.timestamp}\n"

            return output_text

        except Exception as e:
            error_message = f"Error processing research request: {e}"
            print(error_message) # Log error for debugging
            return f"An error occurred while processing your request. Please check the logs for details. Error: {e}"


# Gradio UI
deep_research_tool = DeepResearchTool()

def research_interface(query):
    """Interface function for Gradio to interact with DeepResearchTool."""
    return deep_research_tool.conduct_research(query)

iface = gr.Interface(
    fn=research_interface,
    inputs=gr.Textbox(lines=3, placeholder="Enter your deep research query here..."),
    outputs=gr.Markdown(), # Use Markdown output for better formatting
    title="✨ Enhanced Deep Research Tool ✨",
    description="Conduct in-depth research on complex topics using AI-driven multi-step analysis. This tool leverages a suite of specialized tools including web search, academic databases, and fact-checking to provide comprehensive and reliable research reports.",
    theme=gr.themes.Soft(), # Optional theme for a nicer look
)

if __name__ == "__main__":
    iface.launch(share=False) # share=False for local use, set to True to create a shareable link
```

---
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

