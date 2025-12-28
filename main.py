from agents import Agent, Runner, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=client
)

main_agent = Agent(
    name="Python Assistant",
        name="Physical AI Book Assistant",
        instructions="You are An Book Assistant and always respond about this book which url is https://hackathon-course-book.vercel.app/ ",
    model=model
)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"message": "Backend is running"}

@app.post("/chat")
async def chat(req: ChatMessage):
    try:
        result = await Runner.run(
            main_agent,
            req.message
        )
        return {"response": result.final_output}

    except Exception as e:
        print("ERROR:", e)
        return {"response": f"Backend error: {str(e)}"}

