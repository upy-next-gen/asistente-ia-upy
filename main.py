import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Microservicio de Chatbot UPY")

# Configuración CORS para que tu HTML pueda comunicarse con este backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializamos el cliente apuntando a los servidores de DeepSeek
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.deepseek.com"
)

class MensajeUsuario(BaseModel):
    mensaje: str

@app.post("/chat")
async def chatear(peticion: MensajeUsuario):
    try:
        respuesta = client.chat.completions.create(
            model="deepseek-chat", # Usamos el modelo exacto de DeepSeek
            messages=[
                {"role": "system", "content": "Eres un asistente virtual útil y amigable de la UPY."},
                {"role": "user", "content": peticion.mensaje}
            ]
        )
        return {"respuesta": respuesta.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}