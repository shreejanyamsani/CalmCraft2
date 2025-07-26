"""
Setup script for the Wellness Platform

Run this script to initialize the system:
1. Install dependencies
2. Set up MongoDB
3. Pull Ollama models
4. Initialize database collections
"""

import subprocess
import sys
import os
import json
from pymongo import MongoClient

def install_requirements():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ Dependencies installed successfully!")

def setup_mongodb():
    """Set up MongoDB connection and collections"""
    print("🍃 Setting up MongoDB...")
    try:
        client = MongoClient("mongodb://localhost:27017/")
        db = client["wellness_platform"]
        
        # Create collections
        collections = ["users", "activity_logs", "suggestions", "rewards"]
        for collection in collections:
            if collection not in db.list_collection_names():
                db.create_collection(collection)
                print(f"✅ Created collection: {collection}")
        
        print("✅ MongoDB setup complete!")
        return True
    except Exception as e:
        print(f"❌ MongoDB setup failed: {e}")
        return False

def setup_ollama():
    """Pull required Ollama models"""
    print("🦙 Setting up Ollama models...")
    try:
        # Check if Ollama is running
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Ollama is not running. Please start Ollama first.")
            return False
        
        # Pull granite model
        print("📥 Pulling granite-code model...")
        subprocess.run(["ollama", "pull", "granite-code"], check=True)
        print("✅ Granite model ready!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ollama setup failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ Ollama not found. Please install Ollama first.")
        return False

def create_env_file():
    """Create .env file with default configuration"""
    print("📝 Creating environment configuration...")
    env_content = """
# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017/

# API Keys (update with your actual keys)
GROQ_API_KEY=your_groq_api_key_here

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434/api/generate
"""
    
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write(env_content.strip())
        print("✅ Created .env file")
    else:
        print("ℹ️ .env file already exists")

def create_sample_user():
    """Create a sample user profile"""
    print("👤 Creating sample user profile...")
    sample_profile = {
        "user_id": "demo_user",
        "Age": 30,
        "Gender": "Male",
        "Occupation": "Engineering",
        "Country": "USA",
        "Consultation_History": "No",
        "Stress_Level": "Medium",
        "Sleep_Hours": 7.5,
        "Work_Hours": 40,
        "Physical_Activity_Hours": 3.0,
        "Social_Media_Usage": 2.0,
        "Diet_Quality": "Average",
        "Smoking_Habit": "Non-Smoker",
        "Alcohol_Consumption": "Social Drinker",
        "Medication_Usage": "No",
        "coins": 50,
        "total_coins_earned": 50
    }
    
    try:
        client = MongoClient("mongodb://localhost:27017/")
        db = client["wellness_platform"]
        
        # Insert sample user if doesn't exist
        if not db.users.find_one({"user_id": "demo_user"}):
            db.users.insert_one(sample_profile)
            print("✅ Sample user created!")
        else:
            print("ℹ️ Sample user already exists")
    except Exception as e:
        print(f"❌ Failed to create sample user: {e}")

def main():
    """Main setup function"""
    print("🌟 Welcome to Wellness Platform Setup!")
    print("=" * 50)
    
    # Install dependencies
    install_requirements()
    print()
    
    # Create environment file
    create_env_file()
    print()
    
    # Setup MongoDB
    if setup_mongodb():
        create_sample_user()
    print()
    
    # Setup Ollama
    setup_ollama()
    print()
    
    print("🎉 Setup Complete!")
    print("=" * 50)
    print("Next steps:")
    print("1. Update your GROQ_API_KEY in the .env file")
    print("2. Make sure MongoDB is running (mongod)")
    print("3. Make sure Ollama is running (ollama serve)")
    print("4. Run the application: streamlit run main_app.py")
    print()
    print("🚀 Happy wellness tracking!")

if __name__ == "__main__":
    main()
