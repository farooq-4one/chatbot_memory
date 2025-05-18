from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional
from agent import agent
from agents import (
    Runner, set_tracing_disabled, InputGuardrailTripwireTriggered
)
from context import UserContext
from openai.types.responses import ResponseTextDeltaEvent
from fastapi.responses import StreamingResponse, PlainTextResponse
import asyncio

load_dotenv()
set_tracing_disabled(True)

app = FastAPI(title="Nova Store Assistant 🤖")

# Allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",
                   "https://store-admin-farooq.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    userId: Optional[str]


@app.post("/chat")
async def chat_with_agent(req: ChatRequest):
    user_input = req.message.strip()

    if not req.userId:
        raise HTTPException(status_code=400, detail="userId is required")

    context = UserContext(userId=req.userId)
    print(f"User ID: {context.userId}")

    try:
        result = Runner.run_streamed(
            starting_agent=agent, input=user_input, context=context
        )

        async def stream_generator():
            async for event in result.stream_events():
                if (
                    event.type == "raw_response_event"
                    and isinstance(event.data, ResponseTextDeltaEvent)
                ):
                    delta = event.data.delta
                    print(delta, end="", flush=True)
                    yield delta
                    await asyncio.sleep(0)  # Yield control to event loop

        return StreamingResponse(stream_generator(), media_type="text/plain")

    except InputGuardrailTripwireTriggered:
        return PlainTextResponse(
            content="I am Nova Customer Support Agent. I can only answer "
            "your question related to Nova store.",
            status_code=200
        )
