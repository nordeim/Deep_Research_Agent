# Deep Research Tool Improvement Analysis

Looking at the provided Deep Research tool, I see several areas that could be enhanced. After carefully examining the code, I'll implement an optimized version focusing on improved source metadata extraction, better quality evaluation, and research caching.

Here's the improved implementation:

```python
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
from pydantic import BaseModel, Field, validator
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools.google_scholar import GoogleScholarQueryRun
import gradio as gr
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import concurrent.futures
from ratelimiter import RateLimiter

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
    
    @validator('credibility_score')
    def validate_credibility(cls, v):
        return max(0.0, min(1.0, v))

class ResearchStep(BaseModel):
    step_name: str
    tools_used: List[str]
    output: str
    sources: List[Source] = Field(default_factory=list)
    key_findings: List[str] = Field(default_factory=list)

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    steps: List[ResearchStep]
    quality_score: float
    sources: List[Source] = Field(default_factory=list)
    uncertainty_areas: List[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    class Config:
        arbitrary_types_allowed = True

class DeepResearchTool:
    def __init__(self, cache_dir: str = ".research_cache"):
        self.llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
        self.parser = PydanticOutputParser(pydantic_object=ResearchResponse)
        
        # Initialize cache
        self.cache = dc.Cache(cache_dir)
        self.cache_ttl = 60 * 60 * 24 * 7  # 1 week cache
        
        # Rate limiters
        self.web_limiter = RateLimiter(max_calls=10, period=60)
        self.scholar_limiter = RateLimiter(max_calls=5, period=60)
        
        self.tools = self._setup_tools()
        self.agent = self._create_agent()
        self.current_sources = []
        self.metadata_store = {}
        
    def _setup_tools(self):
        search = DuckDuckGoSearchRun()
        wiki_api = WikipediaAPIWrapper(top_k_results=3)
        arxiv_api = ArxivAPIWrapper()
        scholar = GoogleScholarQueryRun()

        return [
            Tool(
                name="web_search",
                func=self._web_search_wrapper(search.run),
                description="Performs web searches with URL extraction. Returns relevant web results."
            ),
            Tool(
                name="wikipedia",
                func=self._wiki_wrapper(wiki_api.run),
                description="Queries Wikipedia with source tracking. Good for background knowledge and definitions."
            ),
            Tool(
                name="arxiv_search",
                func=self._arxiv_wrapper(arxiv_api.run),
                description="Searches arXiv with metadata extraction. Best for latest academic research papers."
            ),
            Tool(
                name="google_scholar",
                func=self._scholar_wrapper(scholar.run),
                description="Searches Google Scholar with citation tracking. Finds academic papers and their importance."
            )
        ]

    def _extract_urls(self, text: str) -> List[str]:
        return re.findall(r'https?://[^\s]+', text)

    def _extract_metadata(self, text: str, source_type: str) -> Dict:
        """Extract enhanced metadata from source text"""
        metadata = {
            "authors": [],
            "publication_date": "",
            "citation_count": 0,
            "title": "",
            "credibility_score": self._calculate_base_credibility(source_type)
        }
        
        # Author extraction patterns
        author_patterns = [
            r'by\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'Author[s]?:\s+([^,]+(?:,\s*[^,]+)*)'
        ]
        
        for pattern in author_patterns:
            authors = re.findall(pattern, text)
            if authors:
                metadata["authors"] = [a.strip() for a in authors[0].split(',')] if ',' in authors[0] else [authors[0]]
                break
                
        # Date extraction
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',
            r'([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})',
            r'Published:\s+(\d{4})'
        ]
        
        for pattern in date_patterns:
            dates = re.findall(pattern, text)
            if dates:
                metadata["publication_date"] = dates[0]
                break
                
        # Citation extraction for academic sources
        if source_type in ['arxiv', 'scholar']:
            citations = re.findall(r'Cited by (\d+)', text)
            if citations:
                metadata["citation_count"] = int(citations[0])
                # Adjust credibility based on citations
                metadata["credibility_score"] = min(1.0, metadata["credibility_score"] + (int(citations[0])/1000))
                
        # Title extraction
        title_match = re.search(r'Title:\s+([^\n]+)', text)
        if title_match:
            metadata["title"] = title_match.group(1)
        
        return metadata
        
    def _calculate_base_credibility(self, source_type: str) -> float:
        """Calculate base credibility score based on source type"""
        credibility_map = {
            "arxiv": 0.8,
            "scholar": 0.85,
            "wikipedia": 0.7,
            "web": 0.4,
        }
        return credibility_map.get(source_type, 0.5)

    def _web_search_wrapper(self, func):
        def wrapper(query):
            # Check cache
            cache_key = f"web_{hashlib.md5(query.encode()).hexdigest()}"
            cached = self.cache.get(cache_key)
            if cached:
                self.current_sources.extend(cached["sources"])
                return cached["result"]
            
            # Apply rate limiting
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
                            relevance=0.7,
                            **metadata
                        )
                        sources.append(source)
                    
                    self.current_sources.extend(sources)
                    
                    # Cache result
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
            # Check cache
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
                    relevance=0.8,
                    **metadata
                )
                self.current_sources.append(source)
                
                # Cache result
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
            # Check cache
            cache_key = f"arxiv_{hashlib.md5(query.encode()).hexdigest()}"
            cached = self.cache.get(cache_key)
            if cached:
                self.current_sources.extend(cached["sources"])
                return cached["result"]
                
            try:
                result = func(query)
                sources = []
                
                # Extract paper details
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
                            
                            # Extract title
                            title_match = re.search(r'^(.*?)(?:\n|$)', paper)
                            if title_match:
                                metadata["title"] = title_match.group(1).strip()
                                
                            # Extract authors
                            authors_match = re.search(r'Authors?: (.*?)(?:\n|$)', paper)
                            if authors_match:
                                metadata["authors"] = [a.strip() for a in authors_match.group(1).split(',')]
                                
                            # Extract date
                            date_match = re.search(r'(?:Published|Date): (.*?)(?:\n|$)', paper)
                            if date_match:
                                metadata["publication_date"] = date_match.group(1).strip()
                                
                            source = Source(
                                url=url,
                                source_type="arxiv",
                                relevance=0.9,
                                **metadata
                            )
                            sources.append(source)
                
                self.current_sources.extend(sources)
                
                # Cache result
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
            # Check cache
            cache_key = f"scholar_{hashlib.md5(query.encode()).hexdigest()}"
            cached = self.cache.get(cache_key)
            if cached:
                self.current_sources.extend(cached["sources"])
                return cached["result"]
            
            # Apply rate limiting
            with self.scholar_limiter:
                try:
                    result = func(query)
                    sources = []
                    
                    # Process each paper entry
                    papers = result.split('\n\n')
                    for paper in papers:
                        url_match = re.search(r'(https?://[^\s]+)', paper)
                        if url_match:
                            url = url_match.group(1)
                            metadata = self._extract_metadata(paper, "scholar")
                            
                            # Extract title
                            title_match = re.search(r'^(.*?)(?:\n|$)', paper)
                            if title_match:
                                metadata["title"] = title_match.group(1).strip()
                            
                            # Extract citation count
                            citation_match = re.search(r'Cited by (\d+)', paper)
                            if citation_match:
                                metadata["citation_count"] = int(citation_match.group(1))
                                # Adjust credibility score based on citation count
                                citation_boost = min(0.2, int(citation_match.group(1))/1000)
                                metadata["credibility_score"] += citation_boost
                            
                            source = Source(
                                url=url,
                                source_type="scholar",
                                relevance=0.9,
                                **metadata
                            )
                            sources.append(source)
                    
                    self.current_sources.extend(sources)
                    
                    # Cache result
                    self.cache.set(
                        cache_key,
                        {"result": result, "sources": sources},
                        expire=self.cache_ttl
                    )
                    
                    return result
                except Exception as e:
                    return f"Error during Google Scholar search: {str(e)}. Scholar might be rate-limiting requests."
        return wrapper

    def _create_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert research assistant. Follow this structured process:
            
            1. Query Analysis: Clarify and break down the research question
            2. Initial Research: Use web search and Wikipedia for background
            3. Academic Deep Dive: Use arXiv and Google Scholar for peer-reviewed sources
            4. Cross-Verification: Check key facts across multiple sources
            5. Synthesis: Create comprehensive summary with clear attribution
            
            For each step, document:
            - Tools used
            - Key findings (as bullet points)
            - Source URLs with proper attribution
            
            IMPORTANT GUIDELINES:
            - Always prioritize peer-reviewed academic sources when available
            - Explicitly note areas of uncertainty or conflicting information
            - Distinguish between established facts and emerging research
            - Extract and include author names and publication dates when available
            - Rate each source's credibility (low/medium/high)
            
            Output must be valid JSON matching the schema:
            {schema}
            """.format(schema=self.parser.get_format_instructions())),
            ("human", "{query}")
        ])

        return AgentExecutor(
            agent=create_tool_calling_agent(self.llm, self.tools, prompt),
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

    def _evaluate_quality(self, response: ResearchResponse) -> Dict[str, float]:
        """Enhanced quality evaluation with multiple metrics"""
        metrics = {}
        
        # Source diversity (0-1)
        source_types = {s.source_type for s in response.sources}
        metrics["source_diversity"] = len(source_types) / 4
        
        # Academic ratio (0-1)
        total_sources = len(response.sources) or 1  # Avoid division by zero
        academic_sources = sum(1 for s in response.sources 
                              if s.source_type in ["arxiv", "scholar"])
        metrics["academic_ratio"] = academic_sources / total_sources
        
        # Verification thoroughness (0-1)
        verification_steps = sum(1 for step in response.steps 
                                if any(term in step.step_name.lower() 
                                       for term in ["verification", "validate", "cross-check"]))
        metrics["verification_score"] = min(1.0, verification_steps / 2)
        
        # Process comprehensiveness (0-1)
        metrics["process_score"] = min(1.0, len(response.steps) / 5)
        
        # Source recency - if we have dates
        dated_sources = [s for s in response.sources if s.publication_date]
        if dated_sources:
            current_year = datetime.now().year
            try:
                # Extract years from various date formats
                years = []
                for s in dated_sources:
                    year_match = re.search(r'(\d{4})', s.publication_date)
                    if year_match:
                        years.append(int(year_match.group(1)))
                
                if years:
                    avg_year = sum(years) / len(years)
                    metrics["recency_score"] = min(1.0, max(0.0, 1 - (current_year - avg_year) / 10))
                else:
                    metrics["recency_score"] = 0.5  # Default if we can't extract years
            except:
                metrics["recency_score"] = 0.5  # Default on error
        else:
            metrics["recency_score"] = 0.5  # Default when no dates available
            
        # Source credibility (0-1)
        if response.sources:
            metrics["credibility_score"] = sum(s.credibility_score for s in response.sources) / len(response.sources)
        else:
            metrics["credibility_score"] = 0.0
            
        # Calculate weighted overall score
        weights = {
            "source_diversity": 0.15,
            "academic_ratio": 0.25,
            "verification_score": 0.2,
            "process_score": 0.1,
            "recency_score": 0.15,
            "credibility_score": 0.15
        }
        
        overall_score = sum(metrics[key] * weights[key] for key in weights)
        metrics["overall"] = overall_score
        
        return metrics

    def conduct_research(self, query: str) -> str:
        """Conduct research on the given query"""
        # Check if we have cached results
        cache_key = f"research_{hashlib.md5(query.encode()).hexdigest()}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
            
        self.current_sources = []
        start_time = time.time()
        
        try:
            raw_response = self.agent.invoke({"query": query})
            try:
                response = self.parser.parse(raw_response["output"])
            except Exception as e:
                # Fallback to more flexible parsing if strict parsing fails
                try:
                    # Extract JSON part from the response
                    json_match = re.search(r'```json\n(.*?)\n```', raw_response["output"], re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        response_dict = json.loads(json_str)
                        response = ResearchResponse(**response_dict)
                    else:
                        raise ValueError("Could not extract JSON from response")
                except Exception as inner_e:
                    return f"Error parsing research results: {str(e)}\n\nRaw response: {raw_response['output'][:500]}..."
            
            # Merge collected sources with any sources in the response
            all_sources = {s.url: s for s in self.current_sources}
            for s in response.sources:
                if s.url not in all_sources:
                    all_sources[s.url] = s
                    
            response.sources = list(all_sources.values())
            
            # Distribute sources to appropriate research steps
            self._assign_sources_to_steps(response)
            
            # Calculate quality metrics
            quality_metrics = self._evaluate_quality(response)
            response.quality_score = quality_metrics["overall"]
            
            # Generate formatted response
            formatted_response = self._format_response(response, quality_metrics)
            
            # Cache the result
            self.cache.set(cache_key, formatted_response, expire=self.cache_ttl)
            
            return formatted_response
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Error after {duration:.1f}s: {str(e)}"
            return f"### Research Error\n\n{error_msg}\n\nPlease try again with a more specific query."

    def _assign_sources_to_steps(self, response: ResearchResponse):
        """Assign sources to the appropriate research steps based on content matching"""
        # Create a mapping of source URLs to sources
        url_to_source = {s.url: s for s in response.sources}
        
        # Go through each step and assign sources that are mentioned in the output
        for step in response.steps:
            step_sources = []
            
            # Extract URLs mentioned in this step
            urls = self._extract_urls(step.output)
            
            for url in urls:
                if url in url_to_source:
                    step_sources.append(url_to_source[url])
                    
            # For steps without explicit URLs, try to match based on content
            if not step_sources:
                # Simple matching based on source type and step name
                if "wikipedia" in step.tools_used or "initial" in step.step_name.lower():
                    wiki_sources = [s for s in response.sources if s.source_type == "wikipedia"]
                    step_sources.extend(wiki_sources)
                    
                if "arxiv" in step.tools_used or "academic" in step.step_name.lower():
                    arxiv_sources = [s for s in response.sources if s.source_type == "arxiv"]
                    step_sources.extend(arxiv_sources)
                    
                if "scholar" in step.tools_used or "academic" in step.step_name.lower():
                    scholar_sources = [s for s in response.sources if s.source_type == "scholar"]
                    step_sources.extend(scholar_sources)
                    
                if "web_search" in step.tools_used or "initial" in step.step_name.lower():
                    web_sources = [s for s in response.sources if s.source_type == "web"]
                    step_sources.extend(web_sources[:2])  # Limit to avoid overloading
            
            step.sources = step_sources

    def _format_response(self, response: ResearchResponse, metrics: Dict[str, float]) -> str:
        """Generate enhanced markdown output with quality metrics visualization"""
        markdown = f"## Research Report: {response.topic}\n\n"
        
        # Quality metrics section
        markdown += "### Quality Assessment\n"
        markdown += f"**Overall Score:** {response.quality_score:.2f}/1.00  \n"
        markdown += self._generate_metrics_table(metrics)
        markdown += f"\n**Timestamp:** {response.timestamp}\n\n"
        
        # Summary section
        markdown += "### Summary\n"
        markdown += f"{response.summary}\n\n"
        
        # Research steps
        for i, step in enumerate(response.steps, 1):
            markdown += f"### Step {i}: {step.step_name}\n"
            markdown += f"**Tools Used:** {', '.join(step.tools_used)}\n\n"
            markdown += f"{step.output}\n\n"
            
            if step.sources:
                markdown += "**Sources:**\n"
                for source in step.sources:
                    source_meta = []
                    if source.title:
                        source_meta.append(source.title)
                    if source.authors:
                        source_meta.append(f"by {', '.join(source.authors[:3])}" + 
                                          (f" et al." if len(source.authors) > 3 else ""))
                    if source.publication_date:
                        source_meta.append(source.publication_date)
                    if source.citation_count:
                        source_meta.append(f"cited {source.citation_count} times")
                        
                    source_detail = f"[{source.source_type}] " + " | ".join(source_meta) if source_meta else source.source_type
                    credibility_indicator = "⭐" * round(source.credibility_score * 5)
                    
                    markdown += f"- [{source_detail}]({source.url}) {credibility_indicator}\n"
            markdown += "\n"
            
        # Areas of uncertainty
        if response.uncertainty_areas:
            markdown += "### Areas of Uncertainty\n"
            for area in response.uncertainty_areas:
                markdown += f"- {area}\n"
            markdown += "\n"
            
        return markdown

    def _generate_metrics_table(self, metrics: Dict[str, float]) -> str:
        """Generate a metrics table with gauge visualizations"""
        metric_names = {
            "source_diversity": "Source Diversity",
            "academic_ratio": "Academic Sources",
            "verification_score": "Fact Verification",
            "process_score": "Process Thoroughness",
            "recency_score": "Source Recency",
            "credibility_score": "Source Credibility"
        }
        
        table = "| Metric | Score | Rating |\n| --- | --- | --- |\n"
        
        for key, name in metric_names.items():
            if key in metrics:
                score = metrics[key]
                stars = "★" * round(score * 5)
                stars += "☆" * (5 - len(stars))
                table += f"| {name} | {score:.2f} | {stars} |\n"
                
        return table

    def _generate_score_bar(self, score: float) -> str:
        """Generate a base64 encoded score bar visualization"""
        fig, ax = plt.subplots(figsize=(5, 0.5))
        
        # Use color gradient based on score
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
        gr.Markdown("Conduct multi-source academic research with quality metrics and source validation")
        
        with gr.Row():
            with gr.Column(scale=4):
                query = gr.Textbox(
                    placeholder="Enter research query...", 
                    lines=3,
                    label="Research Query"
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Conduct Research", variant="primary")
                    clear_btn = gr.Button("Clear")
                
                examples = gr.Examples(
                    examples=[
                        ["Quantum computing applications in cryptography"],
                        ["Recent advances in CRISPR gene editing"],
                        ["The impact of social media on teenage mental health"],
                        ["Progress in fusion energy research since 2020"]
                    ],
                    inputs=query,
                    label="Example Queries"
                )
        
        output = gr.Markdown(label="Research Results")
        
        submit_btn.click(
            fn=tool.conduct_research,
            inputs=query,
            outputs=output
        )
        
        clear_btn.click(
            fn=lambda: ("", ""),
            inputs=None,
            outputs=[query, output]
        )
    
    return iface

if __name__ == "__main__":
    iface = create_interface()
    iface.launch()
```

Would you like me to explain or break down the code improvements?
