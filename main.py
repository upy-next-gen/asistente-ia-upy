import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# Cargar las variables de entorno
load_dotenv()

app = FastAPI(title="Microservicio de Chatbot UPY")

# Configuración CORS para que tu HTML pueda comunicarse
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializamos el cliente de DeepSeek
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Estructuras de datos
class MensajeUsuario(BaseModel):
    mensaje: str

class SugerenciaUsuario(BaseModel):
    sugerencia: str

# --- ENDPOINT 1: EL CHATBOT ---
@app.post("/chat")
async def chatear(peticion: MensajeUsuario):
    try:
        respuesta = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system", 
                    "content": """Eres el asistente virtual oficial de la Universidad Politécnica de Yucatán (UPY). 
                    Tu ÚNICA tarea es responder preguntas estrictamente relacionadas con la universidad, trámites, carreras, horarios, vida estudiantil y temas académicos de la institución. 
                    REGLA INQUEBRANTABLE: Si el usuario te pregunta sobre cualquier otro tema fuera de la UPY (recetas, política, código de programación general, chistes, etc.), debes negarte cortésmente diciendo exactamente esto: 'Lo siento, como asistente de la UPY, mi conocimiento y funciones están limitados a temas de la universidad. ¿Te puedo ayudar con alguna duda sobre inscripciones o carreras?'. No des ninguna otra información."""
                },
                {"role": "user", "content": peticion.mensaje}
            ]
        )
        return {"respuesta": respuesta.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}

# --- ENDPOINT 2: EL MECANISMO DE FEEDBACK (Tu aportación) ---
@app.post("/feedback")
async def guardar_sugerencia(peticion: SugerenciaUsuario):
    try:
        # Abre (o crea) un archivo txt y añade la nueva sugerencia al final
        with open("sugerencias_usuarios.txt", "a", encoding="utf-8") as archivo:
            archivo.write(f"- {peticion.sugerencia}\n")
        return {"estatus": "exito", "mensaje": "Sugerencia guardada correctamente."}
    except Exception as e:
        return {"error": str(e)}