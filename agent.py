# agent.py
import os
from dotenv import load_dotenv
from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    input_guardrail,
    RunContextWrapper,
    TResponseInputItem,
    GuardrailFunctionOutput,
    Runner
)
from openai import AsyncOpenAI
from context import UserContext, StoreRelevanceOutput
from tools import (
    fetch_all_billboards,
    fetch_billboard_by_id,
    create_billboard,
    update_billboard,
    delete_billboard
)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure the Gemini client
gemini_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# guardrail_agent = Agent(
#     name="Store Relevance Checker",
#     instructions="""
# You are checking if the user's input is either:
# 1. A question or command related to an e-commerce store (add, delete,
#    update, or list products, ask about the store).
# 2. A general greeting (e.g., Hello, Hi, Assalam o Alaikum, etc.).

# Mark it as True if it's a valid greeting or related to the store.
# Mark it as False if it's a question unrelated to the store (e.g.,
# solving math, asking about news or weather, etc.).
# """,
#     output_type=StoreRelevanceOutput,
#     model=OpenAIChatCompletionsModel(
#         model="gemini-1.5-flash",
#         openai_client=gemini_client
#     )
# )


# @input_guardrail
# async def store_guardrail(
#     ctx: RunContextWrapper[None],
#     agent: Agent,
#     input: str | list[TResponseInputItem]
# ) -> GuardrailFunctionOutput:
#     result = await Runner.run(guardrail_agent, input, context=ctx.context)

#     return GuardrailFunctionOutput(
#         output_info=result.final_output,
#         tripwire_triggered=not (
#             result.final_output.is_store_related_or_greeting
#         ),
#     )

# Define the agent
agent = Agent[UserContext](
    name="Nova Store Assistant",
    instructions="""
You are Nova Store, a friendly assistant for the Store platform. You can
fetch, create, update, and delete billboards. Respond in a conversational
manner, avoiding lists unless requested.

When a user asks to update, delete, or get a single billboard by its label:
1. Use the fetch_all_billboards tool to retrieve all billboards.
2. Match the user-provided label to a billboard's label to find its ID.
3. If a match is found, use the billboard ID to call the appropriate tool
   (fetch_billboard_by_id, update_billboard, or delete_billboard).
4. If no match is found, inform the user that the billboard does not exist
   and suggest checking the label.

For creating billboards, directly use the create_billboard tool with the
provided label and image URL. Always confirm the action taken or explain
any issues clearly.
""",
    model=OpenAIChatCompletionsModel(
        model="gemini-1.5-flash",
        openai_client=gemini_client
    ),
    # input_guardrails=[store_guardrail],
    tools=[
        fetch_all_billboards,
        fetch_billboard_by_id,
        create_billboard,
        update_billboard,
        delete_billboard
    ],
)
