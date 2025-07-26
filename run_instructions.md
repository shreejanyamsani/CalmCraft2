# 🌟 Wellness Platform - Setup & Run Instructions

## Prerequisites
- Python 3.9+
- MongoDB
- Ollama
- Internet connection (for Groq API)

## Quick Start

### 1. Clone/Download the project
```bash
# Save all the code files in a directory called 'wellness_platform'
mkdir wellness_platform
cd wellness_platform
```

### 2. Install Dependencies
```bash
pip install streamlit pymongo requests pandas numpy plotly python-dotenv scikit-learn schedule
```

### 3. Set up MongoDB
```bash
# Install MongoDB if not already installed
# Start MongoDB service
mongod
```

### 4. Set up Ollama
```bash
# Install Ollama if not already installed
# Start Ollama service
ollama serve

# In another terminal, pull the required model
ollama pull granite-code
```

### 5. Configure API Keys
Edit the `config.py` file and add your Groq API key:
```python
GROQ_API_KEY = "your_actual_groq_api_key_here"
```

### 6. Run the Setup Script
```bash
python setup.py
```

### 7. Start the Application
```bash
streamlit run main_app.py
```

### 8. Access the Application
Open your browser and go to: `http://localhost:8501`

## Features Overview

### 📊 Real-Time Dashboard
- Live fitness data simulation
- Health metrics visualization
- Automatic reward system
- Progress tracking

### 🤖 AI Insights
- Groq LLM generates personalized health insights
- Granite LLM reviews for safety and accuracy
- Evidence-based recommendations
- Risk factor analysis

### 💬 Interactive Chat
- Ask health-related questions
- Get personalized advice
- Context-aware responses
- Quick question buttons

### 📈 Health Analytics
- 7-day trend analysis
- Weekly summaries
- Reward history
- Progress visualization

### 🎁 Reward System
- Earn coins for healthy behaviors
- Real-time achievement tracking
- Multiple reward categories
- Gamified wellness experience

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   Data          │    │   MongoDB       │
│   Frontend      │────│   Simulator     │────│   Database      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │
         │              ┌─────────────────┐
         │              │   Reward        │
         │              │   System        │
         │              └─────────────────┘
         │
┌─────────────────┐    ┌─────────────────┐
│   Groq Agent    │    │   Granite Agent │
│   (Insights)    │────│   (Safety)      │
└─────────────────┘    └─────────────────┘
```

## Troubleshooting

### Common Issues:

1. **MongoDB Connection Error**
   - Make sure MongoDB is running: `mongod`
   - Check connection string in config.py

2. **Ollama Model Not Found**
   - Ensure Ollama is running: `ollama serve`
   - Pull the model: `ollama pull granite-code`

3. **Groq API Error**
   - Verify your API key in config.py
   - Check internet connection
   - Ensure API key has sufficient credits

4. **Streamlit Not Loading**
   - Check if port 8501 is available
   - Try: `streamlit run main_app.py --server.port 8502`

### Performance Tips:
- Close unused browser tabs
- Restart the application if it becomes slow
- Monitor MongoDB and Ollama resource usage

## Development Notes

### Code Structure:
- `config.py` - Configuration and settings
- `database_manager.py` - MongoDB operations
- `data_simulator.py` - Real-time data simulation
- `groq_agent.py` - Groq LLM integration
- `granite_agent.py` - Ollama/Granite integration
- `reward_system.py` - Gamification logic
- `main_app.py` - Streamlit application
- `setup.py` - Setup and initialization

### Adding New Features:
1. Add configuration to `config.py`
2. Implement business logic in appropriate module
3. Add database operations to `database_manager.py`
4. Update UI in `main_app.py`

## Docker Deployment (Optional)

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access application at http://localhost:8501
```

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all prerequisites are installed
3. Ensure all services are running
4. Check logs for error messages

Happy wellness tracking! 🌟