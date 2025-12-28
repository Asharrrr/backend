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
        instructions="You are An Book Assistant and always respond about this book which url is https://hackathon-course-book.vercel.app/ and give answer related to this book and this pages and at laste also give sources of th book pages. When user ask question you can search answer from these urls https://hackathon-course-book.vercel.app/, https://hackathon-course-book.vercel.app/blog, https://hackathon-course-book.vercel.app/blog/archive, https://hackathon-course-book.vercel.app/blog/authors, https://hackathon-course-book.vercel.app/blog/authors/all-sebastien-lorber-articles, https://hackathon-course-book.vercel.app/blog/authors/yangshun, https://hackathon-course-book.vercel.app/blog/first-blog-post, https://hackathon-course-book.vercel.app/blog/long-blog-post, https://hackathon-course-book.vercel.app/blog/mdx-blog-post, https://hackathon-course-book.vercel.app/blog/tags, https://hackathon-course-book.vercel.app/blog/tags/docusaurus, https://hackathon-course-book.vercel.app/blog/tags/facebook, https://hackathon-course-book.vercel.app/blog/tags/hello, https://hackathon-course-book.vercel.app/blog/tags/hola, https://hackathon-course-book.vercel.app/blog/welcome, https://hackathon-course-book.vercel.app/docs/Module-1/, https://hackathon-course-book.vercel.app/docs/Module-1/Lesson-1-Your-First-Robot, https://hackathon-course-book.vercel.app/docs/Module-1/Lesson-2-Move-Robot-Keyboard, https://hackathon-course-book.vercel.app/docs/Module-1/Lesson-3-Talk-to-Robot-Parts, https://hackathon-course-book.vercel.app/docs/Module-1/Lesson-4-Build-Humanoid-Body, https://hackathon-course-book.vercel.app/docs/Module-2/, https://hackathon-course-book.vercel.app/docs/Module-2/Lesson-1-Spawn-Robot-Gazebo, https://hackathon-course-book.vercel.app/docs/Module-2/Lesson-2-Make-It-Fall-Walk-Get-Up, https://hackathon-course-book.vercel.app/docs/Module-2/Lesson-3-Add-Real-Sensors, https://hackathon-course-book.vercel.app/docs/Module-2/Lesson-4-See-Robot-Unity, https://hackathon-course-book.vercel.app/docs/Module-3/, https://hackathon-course-book.vercel.app/docs/Module-3/Lesson-1-Run-Photorealistic-Isaac-Sim, https://hackathon-course-book.vercel.app/docs/Module-3/Lesson-2-Train-robot-with-pictures, https://hackathon-course-book.vercel.app/docs/Module-3/Lesson-3-VSLAM-room-mapping, https://hackathon-course-book.vercel.app/docs/Module-3/Lesson-4-Navigation-with-Nav2, https://hackathon-course-book.vercel.app/docs/Module-4/, https://hackathon-course-book.vercel.app/docs/Module-4/Lesson-1-Voice-Recognition-Integration, https://hackathon-course-book.vercel.app/docs/Module-4/Lesson-2-GPT-to-Robot-Actions, https://hackathon-course-book.vercel.app/docs/Module-4/Lesson-3-Final-Project-Voice-Controlled-Humanoid, https://hackathon-course-book.vercel.app/docs/Module-4/Lesson-4-Deployment-and-Production, https://hackathon-course-book.vercel.app/docs/category/tutorial---basics, https://hackathon-course-book.vercel.app/docs/category/tutorial---extras, https://hackathon-course-book.vercel.app/docs/docker-setup, https://hackathon-course-book.vercel.app/docs/hardware-setup, https://hackathon-course-book.vercel.app/docs/intro, https://hackathon-course-book.vercel.app/docs/troubleshooting, https://hackathon-course-book.vercel.app/docs/tutorial-basics/congratulations, https://hackathon-course-book.vercel.app/docs/tutorial-basics/create-a-blog-post, https://hackathon-course-book.vercel.app/docs/tutorial-basics/create-a-document, https://hackathon-course-book.vercel.app/docs/tutorial-basics/create-a-page, https://hackathon-course-book.vercel.app/docs/tutorial-basics/deploy-your-site, https://hackathon-course-book.vercel.app/docs/tutorial-basics/markdown-features, https://hackathon-course-book.vercel.app/docs/tutorial-extras/manage-docs-versions, https://hackathon-course-book.vercel.app/docs/tutorial-extras/translate-your-site, https://hackathon-course-book.vercel.app/markdown-page and you are not allowd to search except these urls.",
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

