import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()

app = FastAPI()

chat = ChatOpenAI(
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("MODEL_NAME"),
)

class ChatRequest(BaseModel):
    query: str
    system_prompt: str = "You are a helpful assistant."

class ChatResponse(BaseModel):
    response: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    message = [
        SystemMessage(content=request.system_prompt),
        HumanMessage(content=request.query),
    ]
    response = chat.invoke(message)
    return ChatResponse(response=response.content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)