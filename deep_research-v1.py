import os
import re
from typing import List
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools.google_scholar import GoogleScholarQueryRun
import gradio as gr

load_dotenv()

class Source(BaseModel):
    url: str
    source_type: str
    relevance: float = 0.0
    authors: List[str] = Field(default_factory=list)
    publication_date: str = ""

class ResearchStep(BaseModel):
    step_name: str
    tools_used: List[str]
    output: str
    sources: List[Source] = Field(default_factory=list)

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    steps: List[ResearchStep]
    quality_score: float
    sources: List[Source] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class DeepResearchTool:
    def __init__(self):
        self.llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
        self.parser = PydanticOutputParser(pydantic_object=ResearchResponse)
        self.tools = self._setup_tools()
        self.agent = self._create_agent()
        self.current_sources = []

    def _setup_tools(self):
        search = DuckDuckGoSearchRun()
        wiki_api = WikipediaAPIWrapper(top_k_results=3)
        arxiv_api = ArxivAPIWrapper()
        scholar = GoogleScholarQueryRun()

        return [
            Tool(
                name="web_search",
                func=self._web_search_wrapper(search.run),
                description="Performs web searches with URL extraction"
            ),
            Tool(
                name="wikipedia",
                func=self._wiki_wrapper(wiki_api.run),
                description="Queries Wikipedia with source tracking"
            ),
            Tool(
                name="arxiv_search",
                func=self._arxiv_wrapper(arxiv_api.run),
                description="Searches arXiv with metadata extraction"
            ),
            Tool(
                name="google_scholar",
                func=self._scholar_wrapper(scholar.run),
                description="Searches Google Scholar with citation tracking"
            )
        ]

    def _extract_urls(self, text: str) -> List[str]:
        return re.findall(r'https?://[^\s]+', text)

    def _web_search_wrapper(self, func):
        def wrapper(query):
            result = func(query)
            urls = self._extract_urls(result)
            self.current_sources.extend([
                Source(url=url, source_type="web", relevance=0.7) 
                for url in urls
            ])
            return result
        return wrapper

    def _wiki_wrapper(self, func):
        def wrapper(query):
            result = func(query)
            url = f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"
            self.current_sources.append(
                Source(url=url, source_type="wikipedia", relevance=0.8)
            )
            return result
        return wrapper

    def _arxiv_wrapper(self, func):
        def wrapper(query):
            result = func(query)
            urls = []
            for entry in result.split('\n'):
                if "arXiv:" in entry:
                    arxiv_id = entry.split(":")[1].strip()
                    url = f"https://arxiv.org/abs/{arxiv_id}"
                    urls.append(url)
            self.current_sources.extend([
                Source(url=url, source_type="arxiv", relevance=0.9) 
                for url in urls
            ])
            return result
        return wrapper

    def _scholar_wrapper(self, func):
        def wrapper(query):
            result = func(query)
            urls = self._extract_urls(result)
            self.current_sources.extend([
                Source(url=url, source_type="scholar", relevance=0.9) 
                for url in urls
            ])
            return result
        return wrapper

    def _create_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert research assistant. Follow this structured process:
            
            1. Query Analysis: Clarify and break down the research question
            2. Initial Research: Use web search and Wikipedia
            3. Academic Deep Dive: Use arXiv and Google Scholar
            4. Cross-Verification: Check key facts across sources
            5. Synthesis: Create comprehensive summary
            
            For each step, document:
            - Tools used
            - Key findings
            - Source URLs
            
            Output must be valid JSON matching the schema:
            {schema}
            """.format(schema=self.parser.get_format_instructions())),
            ("human", "{query}")
        ])

        return AgentExecutor(
            agent=create_tool_calling_agent(self.llm, self.tools, prompt),
            tools=self.tools,
            verbose=True
        )

    def _evaluate_quality(self, response: ResearchResponse) -> float:
        source_types = len({s.source_type for s in response.sources})
        academic_sources = sum(1 for s in response.sources 
                              if s.source_type in ["arxiv", "scholar"])
        verification_steps = sum(1 for step in response.steps 
                                if "verification" in step.step_name.lower())
        
        return (
            (source_types / 4) * 0.3 +
            (academic_sources / len(response.sources)) * 0.3 +
            (verification_steps / len(response.steps)) * 0.2 +
            (len(response.steps) >= 5) * 0.2
        )

    def conduct_research(self, query: str) -> str:
        self.current_sources = []
        try:
            raw_response = self.agent.invoke({"query": query})
            response = self.parser.parse(raw_response["output"])
            response.sources = self.current_sources
            response.quality_score = self._evaluate_quality(response)
            return self._format_response(response)
        except Exception as e:
            return f"Error: {str(e)}"

    def _format_response(self, response: ResearchResponse) -> str:
        markdown = f"## Research Report: {response.topic}\n\n"
        markdown += f"**Quality Score:** {response.quality_score:.2f}/1.00  "
        markdown += f"![Quality](data:image/png;base64,{self._generate_score_bar(response.quality_score)})\n"
        markdown += f"**Timestamp:** {response.timestamp}\n\n"
        
        for step in response.steps:
            markdown += f"### {step.step_name}\n"
            markdown += f"**Tools:** {', '.join(step.tools_used)}\n"
            markdown += f"**Output:** {step.output}\n"
            markdown += "**Sources:**\n"
            for source in step.sources:
                markdown += f"- [{source.source_type}]({source.url})\n"
                
        return markdown

    def _generate_score_bar(self, score: float) -> str:
        import base64
        from io import BytesIO
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(5, 0.5))
        ax.barh(['Score'], [score], color='green')
        ax.set_xlim(0, 1)
        plt.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        return base64.b64encode(buf.getvalue()).decode('utf-8')

# Gradio Interface
tool = DeepResearchTool()

def research_interface(query):
    return tool.conduct_research(query)

iface = gr.Interface(
    fn=research_interface,
    inputs=gr.Textbox(placeholder="Enter research query...", lines=3),
    outputs=gr.Markdown(),
    title="Advanced Research Assistant",
    description="Conduct multi-source academic research with quality metrics",
    examples=[
        ["Quantum computing applications in cryptography"],
        ["Recent advances in CRISPR gene editing"]
    ]
)

if __name__ == "__main__":
    iface.launch()
  
