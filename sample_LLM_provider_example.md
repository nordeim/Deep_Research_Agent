```
The user wants a Python library, similar to LangChain's OpenAI integration, that can be used in a simple chat application to interact with OpenAI-compatible APIs like OpenRouter. The library should support tool-based agents and allow customization of parameters like system/user prompts, base URL, API key, temperature, and max tokens.

Based on extensive web search results, PydanticAI seems like the most promising candidate. Investigating PydanticAI:

The OpenRouter documentation (https://openrouter.ai/docs/community/frameworks) explicitly mentions using PydanticAI with OpenRouter through its OpenAI-compatible interface. The provided example shows how to initialize an OpenAIModel with the OpenRouter base URL and API key and then use an Agent to run a query. This looks very promising.

Let's verify if PydanticAI allows setting system and user prompts, temperature, and max tokens. A quick search for "PydanticAI OpenAIModel parameters" should provide this information.

After reviewing the PydanticAI documentation (https://pydantic-ai.readthedocs.io/en/latest/models/openai.html), it appears that the OpenAIModel class accepts parameters like model_name, base_url, api_key, temperature, max_tokens, and system_prompt. This aligns perfectly with the user's requirements.

Therefore, PydanticAI seems to be the most suitable library for this task. It provides a high-level interface for interacting with OpenAI-compatible APIs, including OpenRouter, and allows the necessary parameter customization.

Now create a sample Python code that uses PydanticAI to build a simple chat application. The code will:

- Import the necessary classes from PydanticAI.
- Define a function that takes the required parameters as input.
- Inside the function, initialize an OpenAIModel with the provided parameters.
- Create an Agent instance with the initialized model.
- Use the agent's run method to get a response from the LLM, passing the user prompt.
- Return the LLM's response.
```

---
```python
# sample example code of how to call this function with different parameters.
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

def chat_with_llm(system_prompt: str, user_prompt: str, base_url: str, api_key: str, temperature: float = 0.7, max_tokens: int = 8192):
"""
Interacts with an OpenAI-compatible LLM using PydanticAI.

Args:
    system_prompt: The system prompt to guide the LLM.
    user_prompt: The user's input message.
    base_url: The base URL of the OpenAI-compatible API provider (e.g., "https://openrouter.ai/api/v1").
    api_key: The API key for the provider.
    temperature: The sampling temperature (default: 0.7).
    max_tokens: The maximum number of tokens to generate (default: 8192).

Returns:
    str: The LLM's response.
"""
try:
    model = OpenAIModel(
        model_name="gpt-4o-mini",  # You can change this to other compatible models
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )
    agent = Agent(model)
    response = agent.run(user_prompt)
    return response
except Exception as e:
    return f"An error occurred: {e}"
if name == "main":
# Example usage with OpenRouter
openrouter_base_url = "https://openrouter.ai/api/v1"
openrouter_api_key = "YOUR_OPENROUTER_API_KEY"  # Replace with your actual API key

system_prompt = "You are a helpful and friendly AI assistant."
user_prompt = "Tell me a joke."

response = chat_with_llm(
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    base_url=openrouter_base_url,
    api_key=openrouter_api_key,
    temperature=0.8,
    max_tokens=150,
)
print(f"System Prompt: {system_prompt}")
print(f"User Prompt: {user_prompt}")
print(f"LLM Response: {response}")

print("\n--- Another example with different parameters ---")

system_prompt_2 = "You are a concise and professional assistant."
user_prompt_2 = "Summarize the main points of the last US presidential election."

response_2 = chat_with_llm(
    system_prompt=system_prompt_2,
    user_prompt=user_prompt_2,
    base_url=openrouter_base_url,
    api_key=openrouter_api_key,
    temperature=0.5,
    max_tokens=300,
)
print(f"System Prompt: {system_prompt_2}")
print(f"User Prompt: {user_prompt_2}")
print(f"LLM Response: {response_2}")
```
