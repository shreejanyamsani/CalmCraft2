import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import html
import re

warnings.filterwarnings('ignore')

# Import our modules
from config import Config
from database_manager import DatabaseManager
from groq_agent import GroqAgent
from granite_agent import GraniteAgent
from granite_chat import GraniteChatAgent  # New import for chat functionality
from reward_system import RewardSystem
from summarisation import create_summarizer

def clean_health_ai_response(raw_response):
    """
    Health-specific AI response cleaner with medical disclaimer handling.
    """
    if not raw_response or not isinstance(raw_response, str):
        return None, {}
    
    try:
        response = raw_response.strip()
        
        # Remove HTML tags
        response = re.sub(r'<[^>]+>', '', response)
        response = html.unescape(response)
        
        # Clean up formatting
        response = re.sub(r'\n{3,}', '\n\n', response)
        response = re.sub(r' {2,}', ' ', response)
        
        # Remove system prefixes
        response = re.sub(r'^(Assistant|AI|Bot|Health Coach):\s*', '', response, flags=re.IGNORECASE)
        
        # Ensure medical disclaimer is properly formatted if present
        if "consult" in response.lower() and "doctor" in response.lower():
            response = re.sub(
                r'(.*consult.*doctor.*)',
                r'**Important:** \1',
                response,
                flags=re.IGNORECASE
            )
        
        # Clean up any JSON-like formatting
        response = re.sub(r'^\{.*?\}$', '', response, flags=re.DOTALL)
        response = re.sub(r'"[^"]*":\s*', '', response)
        response = re.sub(r'[{}"\[\],]', '', response)
        
        response = response.strip()
        
        if not response:
            return None, {"error": "Response became empty after cleaning"}
        
        metadata = {
            "original_length": len(raw_response),
            "cleaned_length": len(response),
            "has_disclaimer": "consult" in response.lower() and "doctor" in response.lower(),
            "response_type": "health_advice"
        }
        
        return response, metadata
        
    except Exception as e:
        return None, {"error": f"Error cleaning health response: {str(e)}"}


# Page configuration
st.set_page_config(
    page_title="🌟 Dynamic Wellness Platform",
    page_icon="🌟",
    layout="wide"
)

# Custom CSS with improved styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .task-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .completed-task {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .pending-task {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #2d3436;
        margin: 0.5rem 0;
        border-left: 5px solid #e17055;
    }
    .reward-notification {
        background: linear-gradient(135deg, #fd79a8 0%, #fdcb6e 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .ai-analysis {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-left: 5px solid #2196f3;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #000000;
        font-size: 16px;
        line-height: 1.6;
    }
    .risk-indicator {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 5px solid;
        color: #000000;
        font-weight: bold;
    }
    .risk-low { border-color: #28a745; }
    .risk-medium { border-color: #ffc107; }
    .risk-high { border-color: #dc3545; }
    
    .wellness-tips {
        background: linear-gradient(135deg, #e8f5e8, #f0f8f0);
        border-left: 5px solid #28a745;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        color: #000000;
        font-size: 16px;
        line-height: 1.8;
    }
    .progress-summary {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border-left: 5px solid #ffc107;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        color: #000000;
        font-size: 16px;
        line-height: 1.6;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .bullet-list {
        font-size: 16px;
        line-height: 1.8;
        color: #000000;
        margin: 0;
        padding-left: 0;
    }
    .bullet-list li {
        margin-bottom: 8px;
        list-style-type: none;
        color: #000000;
    }
    /* Chat message styling */
    .chat-message {
        color: white !important;
        background: transparent;
    }
    .chat-user-message {
        color: white !important;
        background: rgba(255, 255, 255, 0.1);
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
    .chat-assistant-message {
        color: white !important;
        background: rgba(255, 255, 255, 0.05);
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
    .stChatMessage {
        color: white !important;
    }
    .stChatMessage p {
        color: white !important;
    }
    .stChatMessage div {
        color: white !important;
    }
    .granite-chat-indicator {
        background: linear-gradient(135deg, #4a4a4a 0%, #6c6c6c 100%);
        border-left: 5px solid #2196f3;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        color: white;
        font-size: 14px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize services
@st.cache_resource
def initialize_services():
    config = Config()
    db_manager = DatabaseManager(config)
    groq_agent = GroqAgent(config)
    granite_agent = GraniteAgent(config)
    granite_chat_agent = GraniteChatAgent(config)  # Add GraniteChatAgent
    reward_system = RewardSystem(config, db_manager)
    summarizer = create_summarizer(config)
    return config, db_manager, groq_agent, granite_agent, granite_chat_agent, reward_system, summarizer

def calculate_health_scores(user_profile):
    """Calculate strict health factor scores based on user profile"""
    scores = {}
    
    # Sleep Quality Score (stricter evaluation)
    sleep_hours = user_profile.get('Sleep_Hours', 7)
    sleep_quality = user_profile.get('Sleep_Quality', 'Fair')
    if sleep_hours >= 7 and sleep_hours <= 9 and sleep_quality in ['Excellent', 'Good']:
        scores['Sleep Quality'] = 9
    elif sleep_hours >= 6 and sleep_hours <= 10 and sleep_quality == 'Good':
        scores['Sleep Quality'] = 7
    elif sleep_hours >= 6 and sleep_hours <= 10 and sleep_quality == 'Fair':
        scores['Sleep Quality'] = 5
    else:
        scores['Sleep Quality'] = 3
    
    # Stress Level Score (stricter)
    stress_level = user_profile.get('Stress_Level', 'Medium')
    anxiety_freq = user_profile.get('Anxiety_Frequency', 'Sometimes')
    if stress_level == 'Low' and anxiety_freq in ['Never', 'Rarely']:
        scores['Stress Management'] = 9
    elif stress_level == 'Low' and anxiety_freq == 'Sometimes':
        scores['Stress Management'] = 7
    elif stress_level == 'Medium' and anxiety_freq in ['Never', 'Rarely']:
        scores['Stress Management'] = 6
    elif stress_level == 'Medium' and anxiety_freq == 'Sometimes':
        scores['Stress Management'] = 4
    else:
        scores['Stress Management'] = 2
    
    # Work-Life Balance Score (stricter)
    work_hours = user_profile.get('Work_Hours', 40)
    energy_level = user_profile.get('Energy_Level', 'Medium')
    if work_hours <= 40 and energy_level in ['Very High', 'High']:
        scores['Work-Life Balance'] = 9
    elif work_hours <= 45 and energy_level in ['High', 'Medium']:
        scores['Work-Life Balance'] = 7
    elif work_hours <= 50 and energy_level == 'Medium':
        scores['Work-Life Balance'] = 5
    elif work_hours <= 55:
        scores['Work-Life Balance'] = 3
    else:
        scores['Work-Life Balance'] = 1
    
    # Physical Activity Score (stricter)
    activity_hours = user_profile.get('Physical_Activity_Hours', 3)
    if activity_hours >= 5:
        scores['Physical Activity'] = 9
    elif activity_hours >= 3:
        scores['Physical Activity'] = 7
    elif activity_hours >= 1.5:
        scores['Physical Activity'] = 5
    elif activity_hours >= 0.5:
        scores['Physical Activity'] = 3
    else:
        scores['Physical Activity'] = 1
    
    # Diet & Lifestyle Score (new, stricter)
    diet = user_profile.get('Diet', 'Average')
    smoking = user_profile.get('Smoking', 'Non-Smoker')
    alcohol = user_profile.get('Alcohol_Consumption', 'Rarely')
    
    diet_score = 9 if diet == 'Healthy' else 5 if diet == 'Average' else 2
    smoking_penalty = 0 if smoking == 'Non-Smoker' else -2 if smoking == 'Occasional Smoker' else -4
    alcohol_penalty = 0 if alcohol in ['Never', 'Rarely'] else -1 if alcohol == 'Occasionally' else -3
    
    scores['Diet & Lifestyle'] = max(1, diet_score + smoking_penalty + alcohol_penalty)
    
    # Social Media & Digital Wellness (new)
    social_media_hours = user_profile.get('Social_Media_Hours', 3)
    if social_media_hours <= 1:
        scores['Digital Wellness'] = 9
    elif social_media_hours <= 2:
        scores['Digital Wellness'] = 7
    elif social_media_hours <= 4:
        scores['Digital Wellness'] = 5
    elif social_media_hours <= 6:
        scores['Digital Wellness'] = 3
    else:
        scores['Digital Wellness'] = 1
    
    return scores

def calculate_risk_level(user_profile):
    """Calculate overall risk level based on health scores"""
    health_scores = calculate_health_scores(user_profile)
    
    # Calculate weighted average of health scores
    # Lower scores indicate higher risk, so we need to invert the scale
    total_score = sum(health_scores.values())
    max_possible_score = len(health_scores) * 9  # Maximum score per factor is 9
    
    # Convert to risk scale (1-10, where 10 is highest risk)
    # If average score is 9, risk should be 1
    # If average score is 1, risk should be 10
    average_score = total_score / len(health_scores)
    risk_level = max(1, min(10, 11 - average_score))
    
    # Apply additional risk factors
    additional_risk = 0
    
    # High work hours increase risk
    if user_profile.get('Work_Hours', 40) > 60:
        additional_risk += 1
    
    # Poor mood significantly increases risk
    mood = user_profile.get('Mood', 'Neutral')
    if mood in ['Sad', 'Very Sad']:
        additional_risk += 2
    elif mood == 'Neutral':
        additional_risk += 0.5
    
    # High stress with frequent anxiety increases risk
    if (user_profile.get('Stress_Level') == 'High' and 
        user_profile.get('Anxiety_Frequency') in ['Often', 'Always']):
        additional_risk += 1.5
    
    # Very low sleep hours increase risk
    if user_profile.get('Sleep_Hours', 7) < 5:
        additional_risk += 1
    
    # Current medication might indicate existing health issues
    if user_profile.get('Medication') == 'Yes':
        additional_risk += 0.5
    
    final_risk = min(10, risk_level + additional_risk)
    return round(final_risk)

def get_risk_indicator(risk_level):
    """Return risk indicator HTML"""
    if risk_level <= 3:
        risk_class = "risk-low"
        risk_text = "LOW RISK"
        risk_description = "Your wellness indicators look good"
    elif risk_level <= 6:
        risk_class = "risk-medium"
        risk_text = "MODERATE RISK"
        risk_description = "Some areas need attention"
    else:
        risk_class = "risk-high"
        risk_text = "HIGH RISK"
        risk_description = "Important to address wellness concerns"
    
    return f"""
    <div class="risk-indicator {risk_class}">
        <h4>{risk_text} ({risk_level}/10)</h4>
        <p>{risk_description}</p>
    </div>
    """

def collect_user_profile():
    """Collect user profile data using the same form as the mental health app"""
    st.header("👤 Your Health Profile")
    st.markdown("*Please fill out your information to get started with personalized wellness coaching*")
    
    with st.form("user_profile_form"):
        st.subheader("📋 Personal Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            gender = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Prefer not to say"])
            occupation = st.selectbox("Occupation", ["Engineering", "Healthcare", "Education", "IT", "Finance", "Sales", "Other"])
            country = st.selectbox("Country", ["USA", "Canada", "UK", "Germany", "Australia", "India", "Other"])
        
        with col2:
            consultation = st.selectbox("Previous Mental Health Consultation", ["Yes", "No"])
            medication = st.selectbox("Currently Taking Medication", ["Yes", "No"])
            diet = st.selectbox("Diet Quality", ["Healthy", "Average", "Unhealthy"])
            smoking = st.selectbox("Smoking Habit", ["Non-Smoker", "Occasional Smoker", "Regular Smoker", "Heavy Smoker"])
        
        st.subheader("📊 Lifestyle Factors")
        
        col3, col4 = st.columns(2)
        
        with col3:
            sleep_hours = st.slider("Average Sleep Hours per Night", 4, 12, 7)
            work_hours = st.slider("Work Hours per Week", 20, 80, 40)
            social_media_hours = st.slider("Social Media Hours per Day", 0, 12, 3)
            
        with col4:
            physical_activity = st.slider("Physical Activity Hours per Week", 0, 20, 3)
            stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])
            alcohol_consumption = st.selectbox("Alcohol Consumption", ["Never", "Rarely", "Occasionally", "Regularly"])
        
        st.subheader("💭 Mental Health & Mood")
        
        col5, col6 = st.columns(2)
        
        with col5:
            mood = st.selectbox("General Mood", ["Very Happy", "Happy", "Neutral", "Sad", "Very Sad"])
            anxiety_frequency = st.selectbox("Anxiety Frequency", ["Never", "Rarely", "Sometimes", "Often", "Always"])
            
        with col6:
            sleep_quality = st.selectbox("Sleep Quality", ["Excellent", "Good", "Fair", "Poor"])
            energy_level = st.selectbox("Energy Level", ["Very High", "High", "Medium", "Low", "Very Low"])
        
        submitted = st.form_submit_button("🚀 Start My Wellness Journey", type="primary", use_container_width=True)
        
        if submitted:
            # Create user profile dictionary
            user_profile = {
                "user_id": f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "Age": age,
                "Gender": gender,
                "Occupation": occupation,
                "Country": country,
                "Mental_Health_Consultation": consultation,
                "Medication": medication,
                "Diet": diet,
                "Smoking": smoking,
                "Sleep_Hours": sleep_hours,
                "Work_Hours": work_hours,
                "Social_Media_Hours": social_media_hours,
                "Physical_Activity_Hours": physical_activity,
                "Stress_Level": stress_level,
                "Alcohol_Consumption": alcohol_consumption,
                "Mood": mood,
                "Anxiety_Frequency": anxiety_frequency,
                "Sleep_Quality": sleep_quality,
                "Energy_Level": energy_level,
                "created_at": datetime.now()
            }
            
            return user_profile
    
    return None

def display_user_dashboard(user_profile, db_manager, groq_agent, granite_agent, granite_chat_agent, reward_system, summarizer):
    """Display the main user dashboard with all features using the summarizer"""
    config = Config()
    user_id = user_profile['user_id']
    
    # Header with user info
    st.markdown('<h1 class="main-title">🌟 Your Personal Wellness Dashboard</h1>', unsafe_allow_html=True)
    st.markdown(f"*Welcome back! Here's your personalized wellness experience.*")
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    reward_summary = reward_system.get_reward_summary(user_id)
    
    with col1:
        st.metric("💰 Total Coins", reward_summary['total_coins'])
    with col2:
        st.metric("⚡ Coins Earned", reward_summary['total_earned'])
    with col3:
        st.metric("⏳ Pending Tasks", reward_summary['pending_tasks'])
    with col4:
        st.metric("🏆 Tasks Done", reward_summary['completed_tasks'])
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["🤖 AI Health Coach", "📋 My Tasks", "💬 Ask Questions", "📊 My Progress"])
    
    with tab1:
        st.header("🤖 AI Health Analysis & Coaching")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🔍 Get AI Health Analysis", type="primary", use_container_width=True):
                with st.spinner("🤖 Analyzing your health profile..."):
                    # Calculate risk level based on health scores
                    risk_level = calculate_risk_level(user_profile)
                    
                    # Get mental health assessment from Groq
                    assessment, _ = groq_agent.analyze_mental_health(user_profile)
                    
                    # Use summarizer for clean display
                    summarized_assessment = summarizer.summarize_health_analysis(
                        assessment, risk_level, user_profile
                    )
                
                # Display risk level indicator
                st.markdown(get_risk_indicator(risk_level), unsafe_allow_html=True)
                
                # Display summarized assessment in bullet format
                st.markdown(f"""
                <div class="ai-analysis">
                    <h4>🎯 Your Wellness Assessment</h4>
                    <div class="bullet-list">
                        {summarized_assessment}
                    </div>
                    <br>
                    <small><em>AI-Powered Analysis • {datetime.now().strftime('%Y-%m-%d %H:%M')}</em></small>
                </div>
                """, unsafe_allow_html=True)
                
                # Save conversation with both original and summarized versions
                db_manager.save_conversation(user_id, {
                    "type": "health_analysis",
                    "assessment": assessment,
                    "summarized_assessment": summarized_assessment,
                    "risk_level": risk_level
                })
                
                # If risk level is concerning, get tasks from Granite
                if risk_level >= 4:  # Medium to high risk
                    st.warning("⚠️ Your assessment indicates some areas that need attention. Let me create a personalized wellness plan for you.")
                    
                    with st.spinner("🎯 Creating personalized wellness tasks..."):
                        # Get tasks from Granite
                        tasks = granite_agent.assign_wellness_tasks(user_profile, assessment, risk_level)
                    
                    if tasks:
                        st.success(f"✅ Created {len(tasks)} personalized wellness tasks for you!")
                        
                        # Save tasks to database
                        for task in tasks:
                            task_id = db_manager.save_task(user_id, task)
                            if task_id:
                                st.markdown(f"""
                                <div class="task-card">
                                    <h5>🎯 {task['title']}</h5>
                                    <p><strong>Type:</strong> {task['task_type'].replace('_', ' ').title()}</p>
                                    <p><strong>Description:</strong> {task['description']}</p>
                                    <p><strong>Duration:</strong> {task['duration_days']} days</p>
                                    <p><strong>Difficulty:</strong> {task['difficulty'].title()}</p>
                                    <p><strong>Instructions:</strong> {task['instructions']}</p>
                                    <p><strong>Completion Criteria:</strong> {task['completion_criteria']}</p>
                                    <p><strong>Reward:</strong> {reward_system.calculate_task_reward(task['task_type'], task['difficulty'])} coins</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        st.info("📋 Your new tasks have been added to the 'My Tasks' tab. Complete them to earn coins!")
                    else:
                        st.error("❌ Unable to create tasks at this time. Please try again later.")
                else:
                    st.success("🎉 Great news! Your mental health profile looks good. Here are some tips to maintain your wellness:")
                    
                    # Get wellness tips from Groq and summarize them
                    with st.spinner("💡 Getting personalized tips..."):
                        tips = groq_agent.get_health_tips(user_profile)
                        summarized_tips = summarizer.summarize_wellness_tips(tips, user_profile)
                    
                    # Display summarized tips
                    st.markdown(f"""
                    <div class="wellness-tips">
                        <h4>💡 Personalized Wellness Tips</h4>
                        <div class="bullet-list">
                            {summarized_tips}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Save tips conversation
                    db_manager.save_conversation(user_id, {
                        "type": "wellness_tips",
                        "tips": tips,
                        "summarized_tips": summarized_tips
                    })
        
        with col2:
            st.subheader("📊 Quick Health Insights")
            
            # Calculate health scores based on user profile
            health_factors = calculate_health_scores(user_profile)
            
            fig = go.Figure(go.Bar(
                x=list(health_factors.values()),
                y=list(health_factors.keys()),
                orientation='h',
                marker_color=['green' if v >= 7 else 'orange' if v >= 5 else 'red' for v in health_factors.values()],
                text=[f"{v}/10" for v in health_factors.values()],
                textposition='inside'
            ))
            fig.update_layout(
                title="Health Factors Score (1-10)",
                xaxis_title="Score",
                height=350,
                showlegend=False,
                xaxis=dict(range=[0, 10]),
                font=dict(color="black")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show calculated risk level
            current_risk = calculate_risk_level(user_profile)
            st.markdown(get_risk_indicator(current_risk), unsafe_allow_html=True)
            
            # Quick wellness tips button
            st.subheader("💡 Daily Wellness Tips")
            if st.button("Get Fresh Tips", use_container_width=True):
                with st.spinner("💡 Generating fresh tips..."):
                    # Get tips from Groq and summarize
                    tips = groq_agent.get_health_tips(user_profile)
                    summarized_tips = summarizer.summarize_wellness_tips(tips, user_profile)
                
                st.markdown(f"""
                <div class="wellness-tips">
                    <div class="bullet-list">
                        {summarized_tips}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.header("📋 My Wellness Tasks")
        
        # Get user tasks
        pending_tasks = db_manager.get_user_tasks(user_id, "pending")
        completed_tasks = db_manager.get_user_tasks(user_id, "completed")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⏳ Pending Tasks")
            
            if pending_tasks:
                for task in pending_tasks:
                    st.markdown(f"""
                    <div class="pending-task">
                        <h5>🎯 {task['title']}</h5>
                        <p><strong>Type:</strong> {task['task_type'].replace('_', ' ').title()}</p>
                        <p><strong>Description:</strong> {task['description']}</p>
                        <p><strong>Instructions:</strong> {task['instructions']}</p>
                        <p><strong>Reward:</strong> {reward_system.calculate_task_reward(task['task_type'], task.get('difficulty', 'medium'))} coins</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Task completion form
                    with st.expander(f"✅ Complete: {task['title']}"):
                        st.write(f"**Completion Criteria:** {task['completion_criteria']}")
                        
                        with st.form(f"complete_task_{task['task_id']}"):
                            st.write("Please provide details about your task completion:")
                            
                            completion_notes = st.text_area(
                                "How did you complete this task?",
                                placeholder="Describe what you did, how it went, any challenges..."
                            )
                            
                            quality_rating = st.slider(
                                "Rate the quality of your completion (1-5)",
                                min_value=1, max_value=5, value=3
                            )
                            
                            exceeded_expectations = st.checkbox(
                                "I went above and beyond the basic requirements"
                            )
                            
                            if st.form_submit_button("🎉 Mark as Completed"):
                                completion_data = {
                                    "notes": completion_notes,
                                    "quality_rating": quality_rating,
                                    "exceeded_expectations": exceeded_expectations
                                }
                                
                                coins_earned = reward_system.award_task_completion(
                                    user_id, task['task_id'], completion_data
                                )
                                
                                if coins_earned > 0:
                                    st.markdown(f"""
                                    <div class="reward-notification">
                                        🎉 Congratulations! You earned {coins_earned} coins!
                                    </div>
                                    """, unsafe_allow_html=True)
                                    st.success(f"✅ Task completed successfully! You earned {coins_earned} coins!")
                                    st.balloons()
                                    # Rerun to update the display
                                    st.rerun()
                                else:
                                    st.error("❌ There was an issue completing the task. Please try again.")
            else:
                st.info("🎯 No pending tasks right now. Get an AI health analysis to receive personalized wellness tasks!")
        
        with col2:
            st.subheader("✅ Completed Tasks")
            
            if completed_tasks:
                for task in completed_tasks[-5:]:  # Show last 5 completed tasks
                    completed_date = task.get('completed_at', datetime.now()).strftime('%Y-%m-%d')
                    coins_earned = reward_system.calculate_task_reward(
                        task['task_type'], 
                        task.get('difficulty', 'medium'),
                        task.get('completion_data', {})
                    )
                    
                    st.markdown(f"""
                    <div class="completed-task">
                        <h5>✅ {task['title']}</h5>
                        <p><strong>Completed:</strong> {completed_date}</p>
                        <p><strong>Coins Earned:</strong> {coins_earned}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("🏆 Completed tasks will appear here as you finish them!")
            
            # Progress summary
            if completed_tasks or pending_tasks:
                total_tasks = len(completed_tasks) + len(pending_tasks)
                completion_rate = len(completed_tasks) / total_tasks * 100 if total_tasks > 0 else 0
                
                fig_progress = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=completion_rate,
                    title={'text': "Task Completion Rate"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkgreen"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "lightgreen"}
                        ]
                    }
                ))
                fig_progress.update_layout(height=250, font=dict(color="black"))
                st.plotly_chart(fig_progress, use_container_width=True)
    
    # TAB 3 - UPDATED WITH GRANITE CHAT AGENT
    with tab3:
        st.header("💬 Ask Your AI Health Coach")
        
        # Add indicator that we're using Granite Chat
        st.markdown(f"""
        <div class="granite-chat-indicator">
            🧠 <strong>Powered by IBM Granite Chat AI</strong> - Advanced conversational health coaching with memory
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("*Ask any health-related question and get personalized advice based on your profile with conversation memory*")
        
        # Initialize chat history for this user
        if f"granite_chat_history_{user_id}" not in st.session_state:
            st.session_state[f"granite_chat_history_{user_id}"] = [
                {"role": "assistant", "content": "Hello! I'm your personal AI health coach powered by IBM Granite Chat. I know your profile and I maintain conversation memory to provide better, contextual responses. How can I assist you today?"}
            ]
        
        # Display chat messages with enhanced health-specific styling
        for message in st.session_state[f"granite_chat_history_{user_id}"]:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    content = message["content"]
                    # Check if this message contains medical advice
                    if "consult" in content.lower() and "doctor" in content.lower():
                        # Highlight medical disclaimers
                        st.markdown(f'''
                        <div class="chat-assistant-message" style="color: white !important;">
                            <div style="background: rgba(255,193,7,0.2); padding: 10px; border-radius: 5px; border-left: 3px solid #ffc107;">
                                ⚠️ <strong>Medical Advice:</strong><br>
                                {content}
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-assistant-message" style="color: white !important;">{content}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-user-message" style="color: white !important;">{message["content"]}</div>', unsafe_allow_html=True)
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about health and wellness..."):
            # Add user message
            st.session_state[f"granite_chat_history_{user_id}"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(f'<div class="chat-user-message" style="color: white !important;">{prompt}</div>', unsafe_allow_html=True)
            
            # Generate AI response using Granite Chat agent
            with st.chat_message("assistant"):
                with st.spinner("🧠 Granite Chat AI is thinking..."):
                    try:
                        # Use Granite Chat agent for chat response with conversation memory
                        raw_response = granite_chat_agent.get_chat_response(prompt, user_profile, context=None)
                        
                        # Use health-specific cleaning function
                        response, metadata = clean_health_ai_response(raw_response)

                        if response:
                            # Add medical disclaimer styling if present
                            if metadata.get('has_disclaimer', False):
                                response_html = f'''
                                <div class="chat-assistant-message" style="color: white !important;">
                                    <div style="background: rgba(255,193,7,0.2); padding: 10px; border-radius: 5px; margin: 5px 0; border-left: 3px solid #ffc107;">
                                        ⚠️ <strong>Medical Advice:</strong><br>
                                        {response}
                                    </div>
                                </div>
                                '''
                            else:
                                response_html = f'<div class="chat-assistant-message" style="color: white !important;">{response}</div>'
                            
                            st.markdown(response_html, unsafe_allow_html=True)
                            st.session_state[f"granite_chat_history_{user_id}"].append({"role": "assistant", "content": response})
                            
                            # Save with enhanced metadata
                            db_manager.save_conversation(user_id, {
                                "type": "granite_chat_interaction",
                                "user_question": prompt,
                                "ai_response": response,
                                "agent_used": "granite_chat",
                                "metadata": metadata,
                                "has_medical_disclaimer": metadata.get('has_disclaimer', False),
                                "conversation_length": len(granite_chat_agent.conversation_history)
                            })
                        else:
                            error_msg = "I'm having trouble processing your question right now. Could you please rephrase or try again?"
                            st.markdown(f'<div class="chat-message" style="color: white !important;">{error_msg}</div>', unsafe_allow_html=True)
                            st.session_state[f"granite_chat_history_{user_id}"].append({"role": "assistant", "content": error_msg})
                    except Exception as e:
                        error_msg = f"I encountered an error while processing your question. Please try again or rephrase your question."
                        st.markdown(f'<div class="chat-message" style="color: white !important;">{error_msg}</div>', unsafe_allow_html=True)
                        st.session_state[f"granite_chat_history_{user_id}"].append({"role": "assistant", "content": error_msg})
        
        # Quick question buttons - now using Granite Chat
        st.subheader("🚀 Quick Questions")
        col1, col2, col3 = st.columns(3)
        
        quick_questions = [
            "How can I reduce my stress levels?",
            "What's the best sleep routine for me?",
            "How much exercise should I be doing?",
            "How can I improve my work-life balance?",
            "What foods should I eat for better mood?",
            "How can I manage my social media usage?"
        ]
        
        for i, question in enumerate(quick_questions[:6]):
            col = [col1, col2, col3][i % 3]
            with col:
                if st.button(question, key=f"granite_quick_q_{i}"):
                    # Add question to chat and trigger response
                    st.session_state[f"granite_chat_history_{user_id}"].append({"role": "user", "content": question})
                    
                    # Generate AI response using Granite Chat
                    with st.spinner("🧠 Granite Chat AI is thinking..."):
                        try:
                            # Use Granite Chat agent for chat response
                            raw_response = granite_chat_agent.get_chat_response(question, user_profile, context=None)
                            
                            # Use health-specific cleaning function
                            response, metadata = clean_health_ai_response(raw_response)

                            if response:
                                st.session_state[f"granite_chat_history_{user_id}"].append({"role": "assistant", "content": response})
                                
                                # Save with enhanced metadata
                                db_manager.save_conversation(user_id, {
                                    "type": "granite_chat_interaction",
                                    "user_question": question,
                                    "ai_response": response,
                                    "agent_used": "granite_chat",
                                    "metadata": metadata,
                                    "has_medical_disclaimer": metadata.get('has_disclaimer', False),
                                    "conversation_length": len(granite_chat_agent.conversation_history)
                                })
                            else:
                                error_msg = "I'm having trouble processing your question right now. Please try again."
                                st.session_state[f"granite_chat_history_{user_id}"].append({"role": "assistant", "content": error_msg})
                        except Exception as e:
                            error_msg = f"I encountered an error while processing your question. Please try again."
                            st.session_state[f"granite_chat_history_{user_id}"].append({"role": "assistant", "content": error_msg})
                    
                    # Rerun to show the new messages
                    st.rerun()
        
        # Add Granite Chat-specific features
        st.markdown("---")
        st.subheader("🧠 Granite Chat AI Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💡 Get Wellness Advice", use_container_width=True, key="granite_wellness_advice"):
                with st.spinner("🧠 Granite Chat AI generating wellness advice..."):
                    advice = granite_chat_agent.get_wellness_advice("general wellness based on my profile", user_profile)
                    cleaned_advice, _ = clean_health_ai_response(advice)
                    if cleaned_advice:
                        st.session_state[f"granite_chat_history_{user_id}"].append({"role": "assistant", "content": cleaned_advice})
                        st.rerun()
        
        with col2:
            if st.button("❓ Ask Health Question", use_container_width=True, key="granite_health_question"):
                question = "What should I focus on most for better health based on my profile?"
                with st.spinner("🧠 Granite Chat AI answering..."):
                    answer = granite_chat_agent.answer_question(question, user_profile)
                    cleaned_answer, _ = clean_health_ai_response(answer)
                    if cleaned_answer:
                        st.session_state[f"granite_chat_history_{user_id}"].append({"role": "user", "content": question})
                        st.session_state[f"granite_chat_history_{user_id}"].append({"role": "assistant", "content": cleaned_answer})
                        st.rerun()
        
        with col3:
            if st.button("🤝 Get Support", use_container_width=True, key="granite_support"):
                concern = "feeling overwhelmed with my wellness goals"
                with st.spinner("🧠 Granite Chat AI providing support..."):
                    support = granite_chat_agent.provide_support(concern, user_profile)
                    cleaned_support, _ = clean_health_ai_response(support)
                    if cleaned_support:
                        st.session_state[f"granite_chat_history_{user_id}"].append({"role": "user", "content": f"I'm {concern}"})
                        st.session_state[f"granite_chat_history_{user_id}"].append({"role": "assistant", "content": cleaned_support})
                        st.rerun()
        
        # Granite Chat conversation controls
        st.markdown("---")
        st.subheader("🎛️ Chat Controls")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear Chat History", use_container_width=True, key="granite_clear_chat"):
                st.session_state[f"granite_chat_history_{user_id}"] = [
                    {"role": "assistant", "content": "Hello! I'm your personal AI health coach powered by IBM Granite Chat. How can I assist you today?"}
                ]
                granite_chat_agent.clear_conversation_history()
                st.success("Chat history cleared!")
                st.rerun()
        
        with col2:
            if st.button("📊 Conversation Summary", use_container_width=True, key="granite_conv_summary"):
                summary = granite_chat_agent.get_conversation_summary()
                st.info(summary)
        
        with col3:
            # Chat personality selector
            personality_type = st.selectbox("🎭 Chat Personality", 
                                          ["supportive", "professional", "casual", "direct"],
                                          key="granite_personality_select",
                                          help="Choose how the AI should respond to you")
            if st.button("Set Personality", use_container_width=True, key="granite_set_personality"):
                granite_chat_agent.set_chat_personality(personality_type)
                st.success(f"Chat personality set to: {personality_type}")
        
        # Display conversation statistics
        st.markdown("---")
        st.subheader("📈 Chat Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        total_messages = len(st.session_state.get(f"granite_chat_history_{user_id}", []))
        user_messages = len([msg for msg in st.session_state.get(f"granite_chat_history_{user_id}", []) if msg["role"] == "user"])
        assistant_messages = len([msg for msg in st.session_state.get(f"granite_chat_history_{user_id}", []) if msg["role"] == "assistant"])
        
        with col1:
            st.metric("Total Messages", total_messages)
        with col2:
            st.metric("Your Questions", user_messages)
        with col3:
            st.metric("AI Responses", assistant_messages)
        
        # Show Granite Chat agent internal conversation history length
        granite_internal_history = len(granite_chat_agent.conversation_history)
        st.info(f"🧠 Granite Chat AI is maintaining {granite_internal_history} conversation turns in memory for better context.")
    
    with tab4:
        st.header("📊 My Wellness Progress")
        
        # Get user's progress data
        all_tasks = db_manager.get_user_tasks(user_id)
        completed_tasks = [task for task in all_tasks if task.get('status') == 'completed']
        conversations = list(db_manager.db[config.CONVERSATIONS_COLLECTION].find(
            {"user_id": user_id}
        ).sort("timestamp", -1).limit(10))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Achievement Summary")
            
            # Use summarizer for progress summary
            progress_summary = summarizer.create_progress_summary(completed_tasks, user_profile)
            
            # Display progress summary
            st.markdown(f"""
            <div class="progress-summary">
                <h4>📈 Your Progress Highlights</h4>
                <div class="bullet-list">
                    {progress_summary}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Calculate achievements
            reward_summary = reward_system.get_reward_summary(user_id)
            
            st.metric("Total Coins Earned", reward_summary['total_earned'])
            st.metric("Current Coin Balance", reward_summary['total_coins'])
            st.metric("Tasks Completed", reward_summary['completed_tasks'])
            st.metric("Pending Tasks", reward_summary['pending_tasks'])
            
            # Current risk level display
            current_risk = calculate_risk_level(user_profile)
            st.markdown(f"""
            <div class="risk-indicator risk-{'low' if current_risk <= 3 else 'medium' if current_risk <= 6 else 'high'}">
                <h4>Current Risk Level: {current_risk}/10</h4>
                <p>Based on your health profile analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Task completion by type
            if all_tasks:
                completed_by_type = {}
                for task in all_tasks:
                    if task.get('status') == 'completed':
                        task_type = task['task_type'].replace('_', ' ').title()
                        completed_by_type[task_type] = completed_by_type.get(task_type, 0) + 1
                
                if completed_by_type:
                    fig_tasks = px.pie(
                        values=list(completed_by_type.values()),
                        names=list(completed_by_type.keys()),
                        title="Completed Tasks by Type"
                    )
                    fig_tasks.update_layout(font=dict(color="black"))
                    st.plotly_chart(fig_tasks, use_container_width=True)
        
        with col2:
            st.subheader("📈 Progress Timeline")
            
            if all_tasks:
                # Create timeline of task completions
                task_timeline = []
                for task in all_tasks:
                    if task.get('status') == 'completed' and task.get('completed_at'):
                        task_timeline.append({
                            'date': task['completed_at'].strftime('%Y-%m-%d'),
                            'task': task['title'],
                            'coins': reward_system.calculate_task_reward(
                                task['task_type'], 
                                task.get('difficulty', 'medium')
                            )
                        })
                
                if task_timeline:
                    timeline_df = pd.DataFrame(task_timeline)
                    
                    # Group by date and sum coins
                    daily_coins = timeline_df.groupby('date')['coins'].sum().reset_index()
                    
                    fig_timeline = px.line(
                        daily_coins,
                        x='date',
                        y='coins',
                        title="Daily Coins Earned",
                        markers=True
                    )
                    fig_timeline.update_layout(font=dict(color="black"))
                    st.plotly_chart(fig_timeline, use_container_width=True)
                else:
                    st.info("Complete some tasks to see your progress timeline!")
            else:
                st.info("Your progress will appear here as you complete tasks and interact with the AI coach!")
            
            # Health scores visualization
            st.subheader("📊 Health Factor Trends")
            health_scores = calculate_health_scores(user_profile)
            
            fig_health = go.Figure(data=go.Scatterpolar(
                r=list(health_scores.values()),
                theta=list(health_scores.keys()),
                fill='toself',
                name='Health Scores'
            ))
            fig_health.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 10]
                    )),
                showlegend=False,
                title="Health Factors Radar Chart",
                font=dict(color="black")
            )
            st.plotly_chart(fig_health, use_container_width=True)
        
        # Recent activity with summarized content
        st.subheader("📝 Recent Activity")
        
        if conversations:
            for conv in conversations[:5]:
                timestamp = conv['timestamp'].strftime('%Y-%m-%d %H:%M')
                conv_type = conv.get('type', 'unknown')
                agent_used = conv.get('agent_used', 'unknown')
                
                with st.expander(f"{conv_type.replace('_', ' ').title()} ({agent_used.title()}) - {timestamp}"):
                    if conv_type == 'health_analysis':
                        st.write(f"**Risk Level:** {conv.get('risk_level', 'N/A')}/10")
                        
                        # Use summarized assessment if available, otherwise original
                        assessment_display = conv.get('summarized_assessment', conv.get('assessment', 'N/A'))
                        
                        st.markdown(f"""
                        <div style="color: black;">
                            <strong>Assessment:</strong>
                            <div class="bullet-list">
                                {assessment_display}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    elif conv_type == 'wellness_tips':
                        # Use summarized tips if available
                        tips_display = conv.get('summarized_tips', conv.get('tips', 'N/A'))
                        
                        st.markdown(f"""
                        <div style="color: black;">
                            <strong>Tips:</strong>
                            <div class="bullet-list">
                                {tips_display}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    elif conv_type in ['chat_interaction', 'granite_chat_interaction']:
                        st.write(f"**Question:** {conv.get('user_question', 'N/A')}")
                        
                        # Use summarized response if available
                        response_display = conv.get('summarized_response', conv.get('ai_response', 'N/A'))
                        
                        st.markdown(f"""
                        <div style="color: black;">
                            <strong>Response:</strong>
                            <div class="bullet-list">
                                {response_display}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show conversation metadata if available
                        if conv.get('conversation_length'):
                            st.write(f"**Conversation Context:** {conv['conversation_length']} turns in memory")
                        if conv.get('has_medical_disclaimer'):
                            st.write("⚠️ **Contains Medical Disclaimer**")
        else:
            st.info("Your recent interactions with the AI coach will appear here!")

def main():
    """Main application function"""
    config, db_manager, groq_agent, granite_agent, granite_chat_agent, reward_system, summarizer = initialize_services()
    
    # Check if user profile exists in session
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = None
    
    # If no user profile, show profile collection form
    if st.session_state.user_profile is None:
        st.markdown('<h1 class="main-title">🌟 Welcome to Your Dynamic Wellness Platform</h1>', unsafe_allow_html=True)
        st.markdown("*Get personalized AI health coaching, dynamic task assignments, and earn rewards for your wellness journey*")
        
        # Show features overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🤖 AI Health Coach
            - Advanced risk calculation
            - Groq AI analyzes your profile
            - Summarized, bullet-point insights
            - Smart risk assessment
            """)
        
        with col2:
            st.markdown("""
            ### 🎯 Dynamic Tasks
            - Granite AI assigns wellness tasks
            - Based on calculated risk levels
            - Earn coins for completion
            - Personalized difficulty levels
            """)
        
        with col3:
            st.markdown("""
            ### 💬 Interactive Chat
            - **IBM Granite Chat AI**
            - Conversation memory & context
            - Get personalized health advice
            - Multiple personality modes
            """)
        
        st.markdown("---")
        
        # Show enhanced features
        st.info("""
        🔍 **Enhanced Features with Granite Chat AI**: 
        - **IBM Granite Conversational AI**: Advanced chat capabilities with conversation memory
        - **Context-Aware Responses**: Remembers previous conversations for better continuity
        - **Enhanced Risk Assessment**: Multi-factor risk calculation based on sleep, stress, work-life balance, and lifestyle
        - **Smart Response Processing**: Clean, readable responses with medical disclaimer detection
        - **Contextual Advice**: Personalized recommendations based on your specific profile and conversation history
        - **Multiple Personality Modes**: Supportive, professional, casual, or direct conversation styles
        - **Advanced Chat Features**: Wellness advice, Q&A, support functions with conversation memory
        - **Progress Tracking**: Comprehensive conversation history and achievement summaries
        """)
        
        # Collect user profile
        user_profile = collect_user_profile()
        
        if user_profile:
            # Save to database
            if db_manager.save_user_profile(user_profile):
                st.session_state.user_profile = user_profile
                
                # Show initial risk calculation
                risk_level = calculate_risk_level(user_profile)
                st.markdown(get_risk_indicator(risk_level), unsafe_allow_html=True)
                
                st.success("✅ Profile saved successfully! Redirecting to your dashboard...")
                st.rerun()
            else:
                st.error("❌ Error saving profile. Please try again.")
    
    else:
        # Display main dashboard with granite chat agent
        display_user_dashboard(
            st.session_state.user_profile,
            db_manager,
            groq_agent,
            granite_agent,
            granite_chat_agent,  # Added granite_chat_agent parameter
            reward_system,
            summarizer
        )
        
        # Add a reset button in sidebar
        with st.sidebar:
            st.header("⚙️ Settings")
            
            user_profile = st.session_state.user_profile
            reward_summary = reward_system.get_reward_summary(user_profile['user_id'])
            risk_level = calculate_risk_level(user_profile)
            
            st.metric("💰 Total Coins", reward_summary['total_coins'])
            st.metric("🏆 Tasks Completed", reward_summary['completed_tasks'])
            st.metric("⏳ Pending Tasks", reward_summary['pending_tasks'])
            st.metric("⚠️ Risk Level", f"{risk_level}/10")
            
            st.markdown("---")
            
            # Health scores breakdown
            st.subheader("📊 Health Scores")
            health_scores = calculate_health_scores(user_profile)
            for factor, score in health_scores.items():
                color = "🟢" if score >= 7 else "🟡" if score >= 5 else "🔴"
                st.write(f"{color} {factor}: {score}/10")
            
            st.markdown("---")
            
            if st.button("🔄 New User Profile", type="secondary"):
                st.session_state.user_profile = None
                st.rerun()
            
            st.markdown("---")
            st.markdown("""
            ### 🎯 Enhanced Features:
            1. **IBM Granite Chat AI**: Advanced conversational health coaching with memory
            2. **Context-Aware Conversations**: Remembers your chat history for better responses
            3. **Smart Risk Analysis**: Multi-factor health assessment with personalized insights
            4. **Dynamic Task Assignment**: AI creates tasks based on your specific risk profile
            5. **Interactive Chat Features**: Wellness advice, Q&A, and support with conversation continuity
            6. **Progress Tracking**: View comprehensive achievements and progress highlights
            7. **Reward System**: Earn coins for completing wellness activities
            
            ### 🆕 Latest Updates:
            - **Granite Chat Integration**: Advanced IBM Granite Chat AI with conversation memory
            - **Enhanced Context Awareness**: AI remembers your previous conversations
            - **Improved Response Quality**: Better cleaning and processing of AI responses
            - **Personality Modes**: Choose between supportive, professional, casual, or direct styles
            - **Conversation Statistics**: Track your chat interactions and AI memory usage
            - **Medical Disclaimer Detection**: Automatic detection and highlighting of medical advice
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🌟 <strong>Enhanced Dynamic Wellness Platform with IBM Granite Chat AI</strong></p>
        <p><small>IBM Granite Chat AI with Memory • Advanced Risk Assessment • Personalized Task Assignment • Smart Health Coaching</small></p>
        <p><small>⚠️ This tool provides wellness guidance. Consult healthcare professionals for medical advice.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()