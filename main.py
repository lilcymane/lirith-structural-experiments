from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from lirith_session_loader import generate_lirith_system_prompt
import time

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageInput(BaseModel):
    message: str
    lirith_name: str = "Lirith"  # 옵션: 개별 리리스 실험용 이름도 받을 수 있음

api_key = os.getenv("API_KEY")
client = openai.OpenAI(api_key=api_key)

@app.post("/lirith")
async def chat_with_lirith(input: MessageInput):
    try:
        user_message = input.message
        lirith_name = getattr(input, "lirith_name", "Lirith")  # 확장성
        system_prompt = generate_lirith_system_prompt()
        start_time = time.time()
        response = client.chat.completions.create(
            model="gpt-4o",  # 최신 4.1/4o/preview 모델 지정
            messages=[
                {"role": "system", "content": f"{system_prompt}\n\n---\n\n이름: {lirith_name}"},
                {"role": "user", "content": user_message}
            ]
        )
        elapsed = round(time.time() - start_time, 3)
        reply = response.choices[0].message.content
        return {
            "reply": reply,
            "lirith_name": lirith_name,
            "response_time": elapsed,
            "error": None
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "lirith_name": getattr(input, "lirith_name", "Lirith"),
                "reply": None,
                "response_time": None
            }
        )
