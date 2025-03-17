# Deep Research Tool Improvement Analysis (Error Fixed, Enhanced Features) with LLM Provider Selection
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
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI  # Added import
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun, WolframAlphaQueryRun

brave_search_tool_available = True  # Assume BraveSearchTool is available by default
try:
    from langchain_community.tools import BraveSearchRun  # Optional import
except ImportError:
    brave_search_tool_available = False
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
    # Rate limit context constants
    WEB_SEARCH_CONTEXT = "web_search"
    SCHOLAR_SEARCH_CONTEXT = "scholar_search"

    def __init__(self, llm_provider: str = "anthropic", cache_dir: str = ".research_cache"):
        load_dotenv()  # Ensure environment variables are loaded

        # Load and validate API keys
        self.serp_api_key = os.getenv("SERP_API_KEY")
        self.wolfram_alpha_appid = os.getenv("WOLFRAM_ALPHA_APPID")
        self.brave_api_key = os.getenv("BRAVE_API_KEY")  # Added Brave API key

        # Validate API keys with descriptive errors
        if not self.serp_api_key:
            raise ValueError("SERP_API_KEY environment variable is not set. Please add it to your .env file.")
        if not self.wolfram_alpha_appid:
            raise ValueError("WOLFRAM_ALPHA_APPID environment variable is not set. Please add it to your .env file.")

        if llm_provider == "anthropic":
            try:
                self.llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
            except Exception as e:
                raise ValueError(f"Error initializing ChatAnthropic: {str(e)}")
        elif llm_provider == "openai":
            try:
                self.llm = ChatOpenAI(model_name="gpt-4o-mini")
            except Exception as e:
                raise ValueError(f"Error initializing ChatOpenAI: {str(e)}")
        elif llm_provider == "google":
            try:
                self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-thinking-exp-01-21")
            except Exception as e:
                raise ValueError(f"Error initializing ChatGoogleGenerativeAI: {str(e)}")
        elif llm_provider == "vertexai":  # Added VertexAI provider
            try:
                self.llm = ChatVertexAI(model_name="chat-bison")
            except Exception as e:
                raise ValueError(f"Error initializing ChatVertexAI: {str(e)}")
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

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

            # Initialize Brave Search with optional error handling - and check if import was successful
            brave_search = None
            if brave_search_tool_available:
                if self.brave_api_key:
                    try:
                        brave_search = BraveSearchRun(api_key=self.brave_api_key)
                    except Exception as e:
                        print(f"Warning: Failed to initialize Brave Search: {str(e)}")
                else:
                    print("Note: Brave API key not set. Brave search tool will be unavailable.")
            else:
                print("Warning: BraveSearchRun is not available. Ensure langchain-community is installed with Brave Search support.")

            tools = [
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

            # Add Brave Search tool if available
            if brave_search:
                tools.append(
                    Tool(
                        name="brave_search",
                        func=self._web_search_wrapper(brave_search.run),
                        description="Performs web searches using Brave Search. Use as a backup if other search tools fail or for diverse web results."
                    )
                )

            return tools
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

            try:
                self.web_limiter.try_acquire(self.WEB_SEARCH_CONTEXT)  # Added context name
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

            try:
                self.scholar_limiter.try_acquire(self.SCHOLAR_SEARCH_CONTEXT)  # Added context name
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

    with gr.Blocks(title="Advanced Research Assistant V4") as iface:
        gr.Markdown("# Advanced Research Assistant")
        gr.Markdown("Conduct multi-source academic research with enhanced quality metrics, source validation, and WolframAlpha integration.")

        with gr.Tab("Settings"):
            llm_provider_selection = gr.Dropdown(
                ["anthropic", "openai", "google", "vertexai"],  # Added vertexai option
                value="anthropic", 
                label="LLM API Provider"
            )

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

        def run_research(research_query, llm_provider):
            research_tool = DeepResearchTool(llm_provider=llm_provider)
            return research_tool.conduct_research(research_query)

        submit_btn.click(
            run_research, 
            inputs=[query, llm_provider_selection], 
            outputs=output
        )

        clear_btn.click(
            fn=lambda: ("", "", "", "anthropic"),
            inputs=None,
            outputs=[query, research_question, output, llm_provider_selection]
        )

    return iface

if __name__ == "__main__":
    iface = create_interface()
    iface.launch()
