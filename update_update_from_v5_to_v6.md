# diff -u deep_research-v5.py deep_research-v6.py
```
--- deep_research-v5.py	2025-03-16 11:29:32.042343791 +0800
+++ deep_research-v6.py	2025-03-17 07:45:52.129352722 +0800
@@ -77,6 +77,10 @@
         arbitrary_types_allowed = True
 
 class DeepResearchTool:
+    # Rate limit context constants
+    WEB_SEARCH_CONTEXT = "web_search"
+    SCHOLAR_SEARCH_CONTEXT = "scholar_search"
+
     def __init__(self, llm_provider: str = "anthropic", cache_dir: str = ".research_cache"):
         load_dotenv()  # Ensure environment variables are loaded
 
@@ -277,31 +281,31 @@
                 self.current_sources.extend(cached["sources"])
                 return cached["result"]
 
-            with self.web_limiter:
-                try:
-                    result = func(query)
-                    urls = self._extract_urls(result)
-                    sources = []
-
-                    for url in urls:
-                        metadata = self._extract_metadata(result, "web")
-                        source = Source(
-                            url=url,
-                            source_type="web",
-                            relevance=0.6, # Slightly reduced web relevance
-                            **metadata
-                        )
-                        sources.append(source)
+            try:
+                self.web_limiter.try_acquire(self.WEB_SEARCH_CONTEXT)  # Added context name
+                result = func(query)
+                urls = self._extract_urls(result)
+                sources = []
 
-                    self.current_sources.extend(sources)
-                    self.cache.set(
-                        cache_key,
-                        {"result": result, "sources": sources},
-                        expire=self.cache_ttl
+                for url in urls:
+                    metadata = self._extract_metadata(result, "web")
+                    source = Source(
+                        url=url,
+                        source_type="web",
+                        relevance=0.6, # Slightly reduced web relevance
+                        **metadata
                     )
-                    return result
-                except Exception as e:
-                    return f"Error during web search: {str(e)}. Please try with a more specific query."
+                    sources.append(source)
+
+                self.current_sources.extend(sources)
+                self.cache.set(
+                    cache_key,
+                    {"result": result, "sources": sources},
+                    expire=self.cache_ttl
+                )
+                return result
+            except Exception as e:
+                return f"Error during web search: {str(e)}. Please try with a more specific query."
         return wrapper
 
     def _wiki_wrapper(self, func):
@@ -399,46 +403,46 @@
                 self.current_sources.extend(cached["sources"])
                 return cached["result"]
 
-            with self.scholar_limiter:
-                try:
-                    result = func(query)
-                    sources = []
-                    papers = result.split('\n\n')
-                    for paper in papers:
-                        url_match = re.search(r'(https?://[^\s]+)', paper)
-                        if url_match:
-                            url = url_match.group(1)
-                            metadata = self._extract_metadata(paper, "scholar")
-
-                            # Title extraction - improved within scholar wrapper
-                            title_match = re.search(r'^(.*?)(?:\n|$)', paper, re.MULTILINE)
-                            if title_match:
-                                metadata["title"] = title_match.group(1).strip()
-
-                            # Citation count extraction - improved within scholar wrapper
-                            citation_match = re.search(r'Cited by (\d+)', paper, re.IGNORECASE)
-                            if citation_match:
-                                metadata["citation_count"] = int(citation_match.group(1))
-                                citation_boost = min(0.3, int(citation_match.group(1))/400) # Further increased citation boost sensitivity and cap
-                                metadata["credibility_score"] += citation_boost
+            try:
+                self.scholar_limiter.try_acquire(self.SCHOLAR_SEARCH_CONTEXT)  # Added context name
+                result = func(query)
+                sources = []
+                papers = result.split('\n\n')
+                for paper in papers:
+                    url_match = re.search(r'(https?://[^\s]+)', paper)
+                    if url_match:
+                        url = url_match.group(1)
+                        metadata = self._extract_metadata(paper, "scholar")
+
+                        # Title extraction - improved within scholar wrapper
+                        title_match = re.search(r'^(.*?)(?:\n|$)', paper, re.MULTILINE)
+                        if title_match:
+                            metadata["title"] = title_match.group(1).strip()
+
+                        # Citation count extraction - improved within scholar wrapper
+                        citation_match = re.search(r'Cited by (\d+)', paper, re.IGNORECASE)
+                        if citation_match:
+                            metadata["citation_count"] = int(citation_match.group(1))
+                            citation_boost = min(0.3, int(citation_match.group(1))/400) # Further increased citation boost sensitivity and cap
+                            metadata["credibility_score"] += citation_boost
 
-                            source = Source(
-                                url=url,
-                                source_type="scholar",
-                                relevance=0.95, # Highest relevance for Scholar
-                                **metadata
-                            )
-                            sources.append(source)
+                        source = Source(
+                            url=url,
+                            source_type="scholar",
+                            relevance=0.95, # Highest relevance for Scholar
+                            **metadata
+                        )
+                        sources.append(source)
 
-                    self.current_sources.extend(sources)
-                    self.cache.set(
-                        cache_key,
-                        {"result": result, "sources": sources},
-                        expire=self.cache_ttl
-                    )
-                    return result
-                except Exception as e:
-                    return f"Error during Google Scholar search: {str(e)}. Scholar might be rate-limiting requests."
+                self.current_sources.extend(sources)
+                self.cache.set(
+                    cache_key,
+                    {"result": result, "sources": sources},
+                    expire=self.cache_ttl
+                )
+                return result
+            except Exception as e:
+                return f"Error during Google Scholar search: {str(e)}. Scholar might be rate-limiting requests."
         return wrapper
 
     def _wolfram_wrapper(self, func):
```
