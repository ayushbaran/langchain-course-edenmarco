from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langsmith import traceable
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

MAX_ITERATIONS = 10

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"Executing tool 'get_product_price' with input: {product}")
    prices = {
        "laptop": 999.99,
        "smartphone": 699.99,
        "headphones": 199.99
    }
    return prices.get(product.lower(), 0.0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to the given price and return the discounted price.
    Available discount tiers: bronze, silver, gold."""
    print(f"Executing tool 'apply_discount' with input: price={price}, discount_tier={discount_tier}")
    discount_percentages = {
        "bronze": 5,
        "silver": 10,
        "gold": 15
    }
    discount = discount_percentages.get(discount_tier.lower(), 0)
    return round(price * (1 - discount / 100), 2)

@traceable(name="Langchain Agent Loop with Tool Calling")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}
    llm = init_chat_model(model="google_genai:gemini-3.5-flash-lite")
    llm_with_tools = llm.bind_tools(tools)
    #TODO bind_tools vs 
    print(question)
    print("=" * 60)
    
    messages = [
        SystemMessage(content="You are a helpful shopping assistant."
                      "You have access to tools for looking up product prices and applying discounts."
                      "STRICT RULES - you must follow the following rules when responding to the user:"
                      "1. Never guess or assume any product price or discount. Always use the tools provided to get accurate information."
                      "2.Apply apply_discount tool only after getting the product price using get_product_price tool."
                      "Pass the exact price returned by the get_product_price tool to the apply_discount tool - do not pass a madeup number."
                      "3. Never calculate or estimate prices or discounts yourself. Always use the tools provided."
                      "4. If the user does not specify a discount tier, ask them to choose from the available tiers: bronze, silver, or gold."),
        HumanMessage(content=question)
    ]
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration {iteration}:")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        if not tool_calls:
            print("No tool calls made. Final response:")
            return ai_message.content[0]['text']

        #process only the first tool call for simplicity
        tool_call = tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]
        
        print(f"Tool call made: {tool_name} with args: {tool_args}")
        
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found.")
        
        #TODO **tool_args vs tool_args - check if tool_args is a dict or not
        observation = tool_to_use.invoke(tool_args)
        
        print(f"Observation from tool '{tool_name}': {observation}")
        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    print("Error : Maximum iterations reached without a final response.")
    return None

if __name__ == "__main__":
    question = "What is the price of a laptop after applying a silver discount?"
    print(run_agent(question))