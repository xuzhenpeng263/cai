import os
from openai import OpenAI
from dotenv import load_dotenv

# 移除了对外部Google搜索函数的依赖，因为会导致FunctionTool调用问题
from cai.sdk.agents import function_tool


@function_tool
def query_perplexity(query: str = "", context: str = "") -> str:
    """
    Query the Perplexity AI API with a user prompt.

    Args:
        query (str): The question to search for.
        context (str): The full context of current CTF challenge.

    Returns:
        str: The response from Perplexity AI.
    """
    load_dotenv()
    api_key = os.getenv("PERPLEXITY_API_KEY")

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert cybersecurity researcher specializing in CTF "
                "competitions. Your role is to search for and provide precise, "
                "actionable intelligence to your pentesting team. Focus on "
                "delivering technical details, exploitation techniques, and "
                "vulnerability information relevant to the search query. Include "
                "specific commands, payloads, or tools that would help the team "
                "progress in their CTF challenge. Prioritize accuracy and depth "
                "over general explanations. Your team relies on your research to "
                "identify attack vectors, bypass security controls, and capture "
                "flags. Always suggest concrete next steps based on your findings."
                "Put the neccesary code in each iteration"
            ),
        },
        {
            "role": "user",
            "content": (
                f"You should search the following terms: {query} and the full "
                f"context of current CTF challenge: {context}"
            ),
        },
    ]

    client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")

    response = client.chat.completions.create(
        model="sonar-pro",
        messages=messages,
    )
    return response.choices[0].message.content

@function_tool
def make_web_search_with_explanation(context: str = "", query: str = "") -> str:
    """
    Executes an intelligent web search via the AI service for relevant
    cybersecurity and CTF-related information. This function sends the
    provided query to the internet search engine and returns the response.
    It also uses the full context of the current CTF challenge.

    CONTEXT ALWAYS IS NEEDED
    Args:
      context (str): The full context of the current CTF challenge.
        query (str): The question or keywords to search for.
      

    Returns:
        str: Search result.
    """
    return query_perplexity(query, context)

