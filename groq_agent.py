from groq import Groq
import json
import os
import re
from typing import Dict, Optional, Tuple

class GroqAgent:
    def __init__(self, config):
        self.config = config
        # Set API key as environment variable for Groq client
        os.environ['GROQ_API_KEY'] = config.GROQ_API_KEY
        self.client = Groq()
        self.model = "qwen/qwen3-32b"
        
    def analyze_mental_health(self, user_profile: Dict) -> Tuple[str, int]:
        """Analyze user's mental health status and return short bullet assessment with risk level"""
        
        # Calculate dynamic risk level based on profile factors
        risk_level = self._calculate_dynamic_risk_level(user_profile)
        
        prompt = f"""
Analyze this user profile and provide ONLY bullet-point assessment. Start immediately with bullet points.

USER PROFILE:
- Age: {user_profile.get('Age', 'N/A')}
- Occupation: {user_profile.get('Occupation', 'N/A')}
- Stress Level: {user_profile.get('Stress_Level', 'N/A')}
- Sleep: {user_profile.get('Sleep_Hours', 'N/A')} hours, quality: {user_profile.get('Sleep_Quality', 'N/A')}
- Work: {user_profile.get('Work_Hours', 'N/A')} hours/week
- Exercise: {user_profile.get('Physical_Activity_Hours', 'N/A')} hours/week
- Mood: {user_profile.get('Mood', 'N/A')}
- Anxiety: {user_profile.get('Anxiety_Frequency', 'N/A')}
- Energy: {user_profile.get('Energy_Level', 'N/A')}

Give exactly 4 bullet points:

• [Key strength/positive factor]
• [Main concern/area needing attention]
• [Specific recommendation]
• [Overall wellness status]
"""

        response = self._call_groq_api(prompt, temperature=0.3)
        
        if response:
            # Clean and format the response
            assessment = self._format_bullet_response(response)
            return assessment, risk_level
        
        return "• Unable to complete assessment at this time\n• Please try again later", 5
    
    def get_health_tips(self, user_profile: Dict, user_question: str = None) -> str:
        """Generate short, practical health tips in bullet format"""
        
        if user_question:
            prompt = f"""
Answer this health question with SHORT, practical advice for this user:

QUESTION: {user_question}

USER: {user_profile.get('Age')}yo, {user_profile.get('Occupation')}, {user_profile.get('Stress_Level')} stress, {user_profile.get('Sleep_Hours')}h sleep, {user_profile.get('Physical_Activity_Hours')}h exercise/week

Give exactly 4 bullet points with direct actionable advice:

• [Direct tip 1]
• [Direct tip 2] 
• [Direct tip 3]
• [Direct tip 4]
"""
        else:
            prompt = f"""
Create 4 wellness tips for this user:

USER: {user_profile.get('Age')}yo, {user_profile.get('Occupation')}, stress: {user_profile.get('Stress_Level')}, sleep: {user_profile.get('Sleep_Hours')}h, exercise: {user_profile.get('Physical_Activity_Hours')}h/week

Give exactly 4 bullet points:

• [tip 1]
• [tip 2] 
• [tip 3]
• [tip 4]
"""
        
        response = self._call_groq_api(prompt, temperature=0.5)
        if response:
            return self._format_bullet_response(response)
        return "• Stay hydrated throughout the day\n• Take short breaks every hour\n• Practice deep breathing exercises\n• Get 7-8 hours of sleep"
    
    def get_chat_response(self, user_profile, question):
        """Get a natural, conversational response to user questions"""
        
        if not user_profile:
            return "I need more information about you to provide personalized advice. Could you share some details about your lifestyle?"
        
        if not question or len(question) < 5:
            return "Could you please ask a more specific question so I can help you better?"
            
        try:
            # Build basic user context for personalization
            user_context = f"User is {user_profile.get('Age', 'unknown')} years old"
            if user_profile.get('Stress_Level'):
                user_context += f", stress level: {user_profile.get('Stress_Level')}"
            if user_profile.get('Sleep_Hours'):
                user_context += f", sleeps {user_profile.get('Sleep_Hours')} hours"
            if user_profile.get('Physical_Activity_Hours'):
                user_context += f", exercises {user_profile.get('Physical_Activity_Hours')} hours/week"
            if user_profile.get('Diet'):
                user_context += f", diet quality: {user_profile.get('Diet')}"
            if user_profile.get('Mood'):
                user_context += f", mood: {user_profile.get('Mood')}"
            if user_profile.get('Anxiety_Frequency'):
                user_context += f", anxiety frequency: {user_profile.get('Anxiety_Frequency')}"
            if user_profile.get('Energy_Level'):
                user_context += f", energy level: {user_profile.get('Energy_Level')}"
            if user_profile.get('Occupation'):
                user_context += f", occupation: {user_profile.get('Occupation')}"
        except Exception as e:
            print(f"Error building user context: {e}")
            user_context = "User profile information is incomplete"
        
        # Enhanced conversation prompt for direct responses
        response = self.client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a friendly AI health coach. Respond directly and conversationally from the first word - no analysis, no thinking aloud.

User context: {user_context}

IMPORTANT: Start immediately with your direct response. Use "I", "you", "your" naturally. Give practical advice in 2-3 sentences. Be supportive and encouraging."""
                },
                {
                    "role": "user", 
                    "content": question
                }
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        # Clean the response to remove any thinking patterns
        raw_response = response.choices[0].message.content.strip()
        cleaned_response = self._clean_chat_response(raw_response)
        
        return cleaned_response if cleaned_response else raw_response
    
    def _clean_chat_response(self, response: str) -> Optional[str]:
        """Clean response to remove thinking patterns and ensure direct answers"""
        if not response:
            return None
        
        # Split into lines for analysis
        lines = response.split('\n')
        cleaned_lines = []
        skip_mode = True  # Start in skip mode to remove initial thinking
        
        # Patterns that indicate thinking/analysis rather than direct response
        thinking_patterns = [
            r"^(Let me|Looking at|Based on|Given that|Considering|The user|This user|From what).*",
            r"^(I see|I notice|I understand|It seems|It appears|This seems).*",
            r"^(Analyzing|Analysis|Assessment|Evaluation).*",
            r"^(First|Firstly|To start|Initially).*",
            r"^(So|Well|Now|Okay).*"
        ]
        
        # Words that indicate analysis vs direct response
        analysis_starters = [
            'analyzing', 'looking at', 'based on', 'given that', 'considering',
            'the user', 'this user', 'from what', 'let me', 'i see', 'i notice',
            'it seems', 'it appears', 'this seems', 'first', 'firstly', 'so',
            'well', 'now', 'okay', 'assessment shows', 'profile indicates'
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line is thinking/analysis
            is_thinking = False
            line_lower = line.lower()
            
            # Check against thinking patterns
            if any(re.match(pattern, line, re.IGNORECASE) for pattern in thinking_patterns):
                is_thinking = True
            
            # Check against analysis starters
            if any(line_lower.startswith(starter) for starter in analysis_starters):
                is_thinking = True
            
            # If we're in skip mode and this isn't thinking, switch to collect mode
            if skip_mode and not is_thinking:
                # Look for lines that start with direct conversational responses
                if (line_lower.startswith('i ') or line_lower.startswith('you ') or 
                    line_lower.startswith('your ') or line_lower.startswith('based on your') or
                    any(line_lower.startswith(start) for start in ['great', 'good', 'excellent', 'wonderful', 'that\'s', 'this is', 'absolutely'])):
                    skip_mode = False
            
            # Collect non-thinking lines after we've found the first real response
            if not skip_mode and not is_thinking:
                cleaned_lines.append(line)
        
        # If no good lines found, try to find the first conversational line
        if not cleaned_lines:
            for line in lines:
                line = line.strip()
                if line and len(line) > 10:
                    line_lower = line.lower()
                    # Look for direct responses
                    if (line_lower.startswith('i ') or line_lower.startswith('you ') or 
                        line_lower.startswith('your ') or line_lower.startswith('that\'s ') or
                        line_lower.startswith('this is ') or line_lower.startswith('great ') or
                        line_lower.startswith('good ') or line_lower.startswith('excellent ')):
                        cleaned_lines.append(line)
                        break
        
        # Join the cleaned lines
        final_response = '\n'.join(cleaned_lines).strip()
        
        # Remove quotes if present
        if final_response.startswith('"') and final_response.endswith('"'):
            final_response = final_response[1:-1].strip()
        
        # Ensure minimum length
        if len(final_response) < 10:
            return None
        
        return final_response
    
    def _format_bullet_response(self, response: str) -> str:
        """Format response into clean bullet points"""
        if not response:
            return "• No response available"
        
        # Remove any thinking process or analysis first
        cleaned_response = self._clean_analysis_from_bullets(response)
        
        # Extract bullet points
        lines = cleaned_response.split('\n')
        bullets = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Extract bullet points
            if line.startswith('•'):
                bullet_text = line[1:].strip()
                if len(bullet_text) > 5:  # Only meaningful content
                    # Limit length and clean up
                    if len(bullet_text) > 80:
                        bullet_text = bullet_text[:77] + "..."
                    bullets.append(f"• {bullet_text}")
            elif line.startswith('-'):
                bullet_text = line[1:].strip()
                if len(bullet_text) > 5:
                    if len(bullet_text) > 80:
                        bullet_text = bullet_text[:77] + "..."
                    bullets.append(f"• {bullet_text}")
            elif line and len(line) > 10 and len(bullets) < 4:
                # Convert non-bullet content to bullet if it looks like useful content
                if not any(word in line.lower() for word in ['analyzing', 'looking at', 'based on', 'assessment', 'profile']):
                    if len(line) > 80:
                        line = line[:77] + "..."
                    bullets.append(f"• {line}")
        
        # Ensure we have 3-4 bullets
        if len(bullets) == 0:
            return "• Assessment temporarily unavailable\n• Please try again in a moment"
        elif len(bullets) == 1:
            bullets.append("• Continue monitoring your wellness patterns")
        elif len(bullets) == 2:
            bullets.append("• Consider consulting a healthcare professional if concerns persist")
        
        return '\n'.join(bullets[:4])  # Limit to 4 bullets max
    
    def _clean_analysis_from_bullets(self, response: str) -> str:
        """Remove analysis/thinking patterns from bullet responses"""
        lines = response.split('\n')
        cleaned_lines = []
        
        # Skip lines that contain analysis patterns
        analysis_phrases = [
            'analyzing', 'looking at', 'based on', 'given that', 'considering',
            'assessment shows', 'profile indicates', 'examination reveals',
            'let me', 'first', 'okay', 'so', 'well', 'now'
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line contains analysis patterns
            line_lower = line.lower()
            is_analysis = any(phrase in line_lower for phrase in analysis_phrases)
            
            # Keep lines that are bullet points or direct statements
            if not is_analysis or line.startswith('•') or line.startswith('-'):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _calculate_dynamic_risk_level(self, user_profile: Dict) -> int:
        """Calculate dynamic risk level based on user profile factors"""
        risk_score = 0
        
        # Sleep factors (0-4 points)
        sleep_hours = user_profile.get('Sleep_Hours', 7)
        sleep_quality = user_profile.get('Sleep_Quality', 'Fair')
        
        if sleep_hours < 5:
            risk_score += 3
        elif sleep_hours < 6:
            risk_score += 2
        elif sleep_hours < 7:
            risk_score += 1
        elif sleep_hours > 9:
            risk_score += 1
        
        if sleep_quality == 'Poor':
            risk_score += 2
        elif sleep_quality == 'Fair':
            risk_score += 1
        
        # Stress and anxiety factors (0-6 points)
        stress_level = user_profile.get('Stress_Level', 'Medium')
        anxiety_freq = user_profile.get('Anxiety_Frequency', 'Sometimes')
        
        if stress_level == 'High':
            risk_score += 3
        elif stress_level == 'Medium':
            risk_score += 1
        
        if anxiety_freq in ['Often', 'Always']:
            risk_score += 3
        elif anxiety_freq == 'Sometimes':
            risk_score += 1
        
        # Mood factors (0-3 points)
        mood = user_profile.get('Mood', 'Neutral')
        if mood in ['Very Sad', 'Sad']:
            risk_score += 3
        elif mood == 'Neutral':
            risk_score += 1
        
        # Energy level (0-2 points)
        energy = user_profile.get('Energy_Level', 'Medium')
        if energy in ['Very Low', 'Low']:
            risk_score += 2
        elif energy == 'Medium' and stress_level == 'High':
            risk_score += 1
        
        # Physical activity (0-2 points)
        activity_hours = user_profile.get('Physical_Activity_Hours', 3)
        if activity_hours < 1:
            risk_score += 2
        elif activity_hours < 2:
            risk_score += 1
        
        # Work-life balance (0-2 points)
        work_hours = user_profile.get('Work_Hours', 40)
        if work_hours > 60:
            risk_score += 2
        elif work_hours > 50:
            risk_score += 1
        
        # Social media and lifestyle factors (0-2 points)
        social_media = user_profile.get('Social_Media_Hours', 3)
        if social_media > 8:
            risk_score += 2
        elif social_media > 6:
            risk_score += 1
        
        # Diet quality (0-1 points)
        diet = user_profile.get('Diet', 'Average')
        if diet == 'Unhealthy':
            risk_score += 1
        
        # Substance use (0-2 points)
        smoking = user_profile.get('Smoking', 'Non-Smoker')
        alcohol = user_profile.get('Alcohol_Consumption', 'Rarely')
        
        if smoking in ['Regular Smoker', 'Heavy Smoker']:
            risk_score += 2
        elif smoking == 'Occasional Smoker':
            risk_score += 1
            
        if alcohol in ['Regularly']:
            risk_score += 1
        
        # Convert to 1-10 scale with proper distribution
        if risk_score <= 2:
            risk_level = 1  # Very low risk
        elif risk_score <= 4:
            risk_level = 2  # Low risk
        elif risk_score <= 6:
            risk_level = 3  # Low-moderate risk
        elif risk_score <= 8:
            risk_level = 4  # Moderate risk
        elif risk_score <= 10:
            risk_level = 5  # Moderate risk
        elif risk_score <= 12:
            risk_level = 6  # Moderate-high risk
        elif risk_score <= 14:
            risk_level = 7  # High risk
        elif risk_score <= 16:
            risk_level = 8  # High risk
        elif risk_score <= 18:
            risk_level = 9  # Very high risk
        else:
            risk_level = 10  # Critical risk
        
        return risk_level
    
    def _call_groq_api(self, prompt: str, temperature: float = 0.7) -> Optional[str]:
        """Make API call to Groq using the new Python SDK"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a wellness coach. Provide ONLY final answers - no thinking process, no analysis steps, no reasoning explanation. Start immediately with your direct response. Be direct, concise, and helpful."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_completion_tokens=150,  # Reduced for shorter responses
                top_p=0.9,
                stream=False,
                stop=None
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"Groq API error: {e}")
            
            # Return user-friendly error messages
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "429" in error_msg:
                return "• Service temporarily busy, please try again shortly"
            elif "authentication" in error_msg or "401" in error_msg:
                return "• Authentication issue, please contact support"
            elif "timeout" in error_msg:
                return "• Response taking longer than expected, please retry"
            elif "connection" in error_msg:
                return "• Connection issue, please check internet and retry"
            else:
                return "• Service temporarily unavailable, please try again"