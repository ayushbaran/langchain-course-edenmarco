from typing import Any, Dict, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)
#TODO: difference in response_format and response_schemas, response_format is used to define the output of the agent, while response_schemas is used to define the input to the agent. The response_format is a Pydantic model that defines the structure of the output, while response_schemas is a list of Pydantic models that define the structure of the input.
def main() -> None:
    print("Hello from langchain-course-edenmarco!")
    result = agent.invoke({"messages": [HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details")]})
    print(result)


if __name__ == "__main__":
    main()