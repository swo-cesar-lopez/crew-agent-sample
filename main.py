import os

# Desactivar completamente el uso de embeddings en crewai
os.environ["CREWAI_DISABLE_EMBEDDINGS"] = "true"

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional
import uvicorn

# Cargar variables de entorno
load_dotenv()
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

model_name="llama3-70b-8192"

os.environ["OPENAI_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = model_name
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

# Groq
from langchain_groq import ChatGroq

# CrewAI
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, tool

# Configurar FastAPI
app = FastAPI(title="CrewAI Content Generator")

# Definir el modelo de solicitud
class ContentRequest(BaseModel):
    topic: str
    verbose: Optional[bool] = True

# Configuración existente del LLM
groq_llm = ChatGroq(
    temperature=0, 
    model_name=model_name
)

search_tool = SerperDevTool()

@tool("Spelling Checker")
def spelling_checker(text: str) -> str:
    """Useful when you finish a article and want to check for spelling mistakes"""
    
    return text + " Checked!!!"

# Manteniendo los agentes originales exactamente como estaban
planner = Agent(
    role="Content Planner",
    goal="Plan engaging and factually accurate content on {topic}",
    backstory="You're working on planning a blog article "
              "about the topic: {topic}."
              "You collect information that helps the "
              "audience learn something "
              "and make informed decisions. "
              "Your work is the basis for "
              "the Content Writer to write an article on this topic.",
    allow_delegation=False,
    verbose=True
)

writer = Agent(
    role="Content Writer",
    goal="Write insightful and factually accurate "
         "opinion piece about the topic: {topic}",
    backstory="You're working on a writing "
              "a new opinion piece about the topic: {topic}. "
              "You base your writing on the work of "
              "the Content Planner, who provides an outline "
              "and relevant context about the topic. "
              "You follow the main objectives and "
              "direction of the outline, "
              "as provide by the Content Planner. "
              "You also provide objective and impartial insights "
              "and back them up with information "
              "provide by the Content Planner. "
              "You acknowledge in your opinion piece "
              "when your statements are opinions "
              "as opposed to objective statements.",
    allow_delegation=False,
    verbose=True
)

editor = Agent(
    role="Editor",
    goal="Edit a given blog post to align with "
         "the writing style of the organization. ",
    backstory="You are an editor who receives a blog post "
              "from the Content Writer. "
              "Your goal is to review the blog post "
              "to ensure that it follows journalistic best practices,"
              "provides balanced viewpoints "
              "when providing opinions or assertions, "
              "and also avoids major controversial topics "
              "or opinions when possible.",
    allow_delegation=False,
    verbose=True
)

# Manteniendo las tareas originales exactamente como estaban
plan = Task(
    description=(
        "1. Prioritize the latest trends, key players, "
            "and noteworthy news on {topic}.\n"
        "2. Identify the target audience, considering "
            "their interests and pain points.\n"
        "3. Develop a detailed content outline including "
            "an introduction, key points, and a call to action.\n"
        "4. Include SEO keywords and relevant data or sources."
    ),
    expected_output="A comprehensive content plan document "
        "with an outline, audience analysis, "
        "SEO keywords, and resources.",
    agent=planner,
)

write = Task(
    description=(
        "1. Use the content plan to craft a compelling "
            "blog post on {topic}.\n"
        "2. Incorporate SEO keywords naturally.\n"
        "3. Sections/Subtitles are properly named "
            "in an engaging manner.\n"
        "4. Ensure the post is structured with an "
            "engaging introduction, insightful body, "
            "and a summarizing conclusion.\n"
        "5. Proofread for grammatical errors and "
            "alignment with the brand's voice.\n"
    ),
    expected_output="A well-written blog post "
        "in markdown format, ready for publication, "
        "each section should have 2 or 3 paragraphs.",
    agent=writer,
)

edit = Task(
    description=("Proofread the given blog post for "
                 "grammatical errors and "
                 "alignment with the brand's voice."),
    expected_output="A well-written blog post in markdown format, "
                    "ready for publication, "
                    "each section should have 2 or 3 paragraphs.",
    agent=editor
)

# Endpoint para generar contenido (síncrono - espera la respuesta completa)
@app.post("/generate-content/")
def generate_content(request: ContentRequest):
    """
    Genera contenido basado en el tema proporcionado.
    Esta llamada es síncrona y esperará hasta que el contenido esté completo.
    """
    # Crear y configurar el crew con los agentes y tareas
    crew = Crew(
        agents=[planner, writer, editor],
        tasks=[plan, write, edit],
        verbose=2 if request.verbose else 0
    )
    
    # Ejecutar el proceso y esperar el resultado'?
    result = crew.kickoff(inputs={'topic': request.topic})
    
    # Devolver el resultado directamente
    return {"content": result, "topic": request.topic}

# Endpoint simple para verificar que el servidor está funcionando
@app.get("/")
def read_root():
    return {"status": "online", "message": "CrewAI Content Generator API"}

# Punto de entrada para ejecutar la aplicación
# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)