import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    OLLAMA_BASE_URL = "http://localhost:11434/api/generate"
    GRANITE_MODEL = "granite3.3:8b"
    # Watson Configuration
    WATSON_API_KEY = "fPuHCEVgnNbPf5A2fSDXA9kDwppGdrUye7Fmq3DTe9vv"
    WATSON_URL = "https://eu-de.ml.cloud.ibm.com"
    WATSON_PROJECT_ID = "71f0101e-62d9-4532-bab8-d784ce8fb5c3"
    
    # AI Models - Updated to use Watson Granite model
    GROQ_MODEL = "qwen/qwen3-32b"
    GRANITE_MODEL = "ibm/granite-3-3-8b-instruct"
    
    # MongoDB Configuration
    MONGODB_URI = "mongodb://localhost:27017/"
    DATABASE_NAME = "wellness_platform"
    
    # Collections
    USERS_COLLECTION = "users"
    CONVERSATIONS_COLLECTION = "conversations"
    TASKS_COLLECTION = "tasks"
    REWARDS_COLLECTION = "rewards"
    
    # Task Reward Values
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