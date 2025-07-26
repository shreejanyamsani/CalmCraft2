import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()  # Loads .env file locally

class Config:
    # 🧠 API Keys and Models
    GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    GROQ_MODEL = "qwen/qwen3-32b"

    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api/generate")
    GRANITE_MODEL = os.getenv("GRANITE_MODEL", "ibm/granite-3-3-8b-instruct")

    # 🧪 Watson Configuration
    WATSON_API_KEY = os.getenv("WATSON_API_KEY") or st.secrets.get("WATSON_API_KEY")
    WATSON_URL = os.getenv("WATSON_URL", "https://eu-de.ml.cloud.ibm.com")
    WATSON_PROJECT_ID = os.getenv("WATSON_PROJECT_ID") or st.secrets.get("WATSON_PROJECT_ID")

    # 🛢️ MongoDB Configuration — supports both local and cloud
    MONGODB_URI = (
        os.getenv("MONGODB_URI") or 
        st.secrets.get("MONGODB_URI") or 
        "mongodb://localhost:27017/"
    )
    DATABASE_NAME = os.getenv("DATABASE_NAME") or st.secrets.get("DATABASE_NAME", "wellness_platform")

    # 📁 Collection Names
    USERS_COLLECTION = "users"
    CONVERSATIONS_COLLECTION = "conversations"
    TASKS_COLLECTION = "tasks"
    REWARDS_COLLECTION = "rewards"

    # 💰 Task Reward Values
    TASK_REWARDS = {
        "meditation": 15,
        "exercise": 20,
        "sleep_schedule": 10,
        "social_connection": 12,
        "journaling": 8,
        "breathing_exercise": 10,
        "nature_walk": 15,
        "healthy_meal": 12,
        "screen_break": 5,
        "gratitude_practice": 8,
        "professional_help": 25
    }
