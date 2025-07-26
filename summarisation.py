from groq import Groq
import os
import re
from typing import Dict, Optional, List
from datetime import datetime

class ResponseSummarizer:
    def __init__(self, config):
        self.config = config
        # Set API key as environment variable for Groq client
        os.environ['GROQ_API_KEY'] = config.GROQ_API_KEY
        self.client = Groq()
        self.model = "qwen/qwen3-32b"
        
    def summarize_health_analysis(self, assessment_text: str, risk_level: int, user_profile: Dict) -> str:
        """Summarize health analysis into 3-4 concise bullet points for UI display"""
        
        if not assessment_text or assessment_text.strip() == "":
            return self._get_fallback_assessment(risk_level, user_profile)
        
        prompt = f"""
You are a wellness coach. Summarize this health assessment into EXACTLY 3-4 bullet points for a user dashboard.

ORIGINAL ASSESSMENT:
{assessment_text}

RISK LEVEL: {risk_level}/10
USER AGE: {user_profile.get('Age', 'N/A')}
STRESS: {user_profile.get('Stress_Level', 'N/A')}
SLEEP: {user_profile.get('Sleep_Hours', 'N/A')} hours

REQUIREMENTS:
- EXACTLY 3-4 bullet points starting with •
- Each bullet 10-15 words maximum
- Focus on most important findings
- Use clear, friendly language
- No medical jargon
- Be encouraging but honest

FORMAT:
• [Key strength or positive finding]
• [Main concern that needs attention]
• [Specific actionable recommendation]
• [Overall wellness status or next step]

Return ONLY the bullet points, nothing else.
"""

        response = self._call_groq_api(prompt, temperature=0.3)
        
        if response:
            formatted_response = self._format_bullet_points(response)
            if self._validate_bullet_format(formatted_response):
                return formatted_response
        
        # Fallback if API fails
        return self._get_fallback_assessment(risk_level, user_profile)
    
    def summarize_wellness_tips(self, tips_text: str, user_profile: Dict, context: str = None) -> str:
        """Summarize wellness tips into 4 actionable bullet points for UI display"""
        
        if not tips_text or tips_text.strip() == "":
            return self._get_fallback_tips(user_profile)
        
        context_info = f"QUESTION CONTEXT: {context}\n" if context else ""
        
        prompt = f"""
You are a wellness coach. Convert these wellness tips into EXACTLY 4 actionable bullet points.

ORIGINAL TIPS:
{tips_text}

{context_info}USER PROFILE:
- Age: {user_profile.get('Age', 'N/A')}
- Occupation: {user_profile.get('Occupation', 'N/A')}
- Stress Level: {user_profile.get('Stress_Level', 'N/A')}
- Sleep: {user_profile.get('Sleep_Hours', 'N/A')} hours
- Exercise: {user_profile.get('Physical_Activity_Hours', 'N/A')} hours/week

REQUIREMENTS:
- EXACTLY 4 bullet points starting with •
- Each bullet 12-18 words maximum
- Start each with an action verb (Try, Practice, Maintain, Reduce, etc.)
- Make them specific and achievable
- Tailor to their profile
- Be practical and encouraging

FORMAT:
• [Action verb] [specific recommendation tailored to their profile]
• [Action verb] [specific recommendation tailored to their profile]
• [Action verb] [specific recommendation tailored to their profile]
• [Action verb] [specific recommendation tailored to their profile]

Return ONLY the bullet points, nothing else.
"""

        response = self._call_groq_api(prompt, temperature=0.4)
        
        if response:
            formatted_response = self._format_bullet_points(response)
            if self._validate_bullet_format(formatted_response, expected_count=4):
                return formatted_response
        
        # Fallback if API fails
        return self._get_fallback_tips(user_profile)
    
    def summarize_chat_response(self, chat_response: str, user_question: str, user_profile: Dict) -> str:
        """Summarize chat response into 2-3 concise bullet points for UI display"""
        
        if not chat_response or chat_response.strip() == "":
            return self._get_fallback_chat_response(user_question)
        
        prompt = f"""
You are a wellness coach. Summarize this response to the user's question into 2-3 concise bullet points.

USER QUESTION: {user_question}

ORIGINAL RESPONSE:
{chat_response}

USER CONTEXT:
- {user_profile.get('Age', 'N/A')} years old, {user_profile.get('Occupation', 'N/A')}
- Stress: {user_profile.get('Stress_Level', 'N/A')}, Sleep: {user_profile.get('Sleep_Hours', 'N/A')}h

REQUIREMENTS:
- EXACTLY 2-3 bullet points starting with •
- Each bullet 15-20 words maximum
- Direct answers to their specific question
- Actionable and personalized
- Friendly and supportive tone
- No repetition between bullets

FORMAT:
• [Direct answer/recommendation specific to their question]
• [Additional helpful tip or consideration]
• [Optional: Follow-up suggestion or next step]

Return ONLY the bullet points, nothing else.
"""

        response = self._call_groq_api(prompt, temperature=0.3)
        
        if response:
            formatted_response = self._format_bullet_points(response)
            if self._validate_bullet_format(formatted_response, expected_count=3):
                return formatted_response
        
        # Fallback if API fails
        return self._get_fallback_chat_response(user_question)
    
    def extract_key_insights(self, long_text: str, max_insights: int = 3) -> List[str]:
        """Extract key insights from long text for dashboard display"""
        
        if not long_text or len(long_text) < 50:
            return ["No significant insights available"]
        
        prompt = f"""
Extract the {max_insights} most important insights from this text for a wellness dashboard.

TEXT:
{long_text}

REQUIREMENTS:
- Extract {max_insights} key insights
- Each insight should be 8-12 words
- Focus on actionable or important information
- No bullets or formatting
- Return as simple list separated by |

EXAMPLE OUTPUT:
Sleep quality needs improvement|Stress levels are manageable|Exercise routine shows good progress

Return ONLY the insights separated by |, nothing else.
"""

        response = self._call_groq_api(prompt, temperature=0.2)
        
        if response:
            insights = [insight.strip() for insight in response.split('|') if insight.strip()]
            return insights[:max_insights] if insights else ["Key insights extracted from analysis"]
        
        return ["Analysis completed successfully"]
    
    def create_progress_summary(self, completed_tasks: List[Dict], user_profile: Dict) -> str:
        """Create a summary of user's progress for dashboard display"""
        
        if not completed_tasks:
            return self._get_fallback_progress_summary()
        
        task_types = [task.get('task_type', '').replace('_', ' ') for task in completed_tasks]
        recent_completions = len([t for t in completed_tasks if self._is_recent_task(t)])
        
        prompt = f"""
Create a 3-bullet progress summary for a wellness dashboard.

COMPLETED TASKS: {len(completed_tasks)} total
RECENT COMPLETIONS: {recent_completions} in last 7 days
TASK TYPES: {', '.join(set(task_types))}

USER: {user_profile.get('Age')}yo, {user_profile.get('Occupation')}, {user_profile.get('Stress_Level')} stress

REQUIREMENTS:
- EXACTLY 3 bullet points starting with •
- Each bullet 10-15 words maximum
- Highlight achievements and progress
- Be encouraging and motivational
- Focus on positive patterns

FORMAT:
• [Achievement or milestone reached]
• [Pattern or consistency highlighted]
• [Encouragement or next step suggestion]

Return ONLY the bullet points, nothing else.
"""

        response = self._call_groq_api(prompt, temperature=0.4)
        
        if response:
            formatted_response = self._format_bullet_points(response)
            if self._validate_bullet_format(formatted_response):
                return formatted_response
        
        return self._get_fallback_progress_summary()
    
    def _format_bullet_points(self, text: str) -> str:
        """Format text into clean bullet points"""
        if not text:
            return ""
        
        lines = text.strip().split('\n')
        bullets = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and non-bullet content
            if not line:
                continue
            
            # Clean up the line
            if line.startswith('•'):
                bullet_text = line[1:].strip()
            elif line.startswith('-'):
                bullet_text = line[1:].strip()
            elif line.startswith('*'):
                bullet_text = line[1:].strip()
            else:
                # If it doesn't start with bullet, check if it's meaningful content
                if len(line) > 10 and not any(skip_word in line.lower() for skip_word in 
                                             ['here are', 'summary:', 'bullet points', 'analysis']):
                    bullet_text = line
                else:
                    continue
            
            # Validate bullet content
            if bullet_text and len(bullet_text) > 5:
                # Ensure it doesn't end with incomplete sentence
                if not bullet_text.endswith(('.', '!', '?')):
                    bullet_text = bullet_text.rstrip(',') + '.'
                
                bullets.append(f"• {bullet_text}")
        
        return '\n'.join(bullets[:4])  # Limit to 4 bullets maximum
    
    def _validate_bullet_format(self, text: str, expected_count: int = 4) -> bool:
        """Validate that the formatted text meets requirements"""
        if not text:
            return False
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Check we have the right number of bullets
        bullet_lines = [line for line in lines if line.startswith('•')]
        
        if len(bullet_lines) < 2 or len(bullet_lines) > expected_count:
            return False
        
        # Check each bullet meets length requirements
        for bullet in bullet_lines:
            content = bullet[1:].strip()
            word_count = len(content.split())
            if word_count < 3 or word_count > 25:  # Reasonable word count range
                return False
        
        return True
    
    def _is_recent_task(self, task: Dict) -> bool:
        """Check if task was completed recently (within 7 days)"""
        if not task.get('completed_at'):
            return False
        
        try:
            completed_date = task['completed_at']
            if isinstance(completed_date, str):
                completed_date = datetime.fromisoformat(completed_date)
            
            days_ago = (datetime.now() - completed_date).days
            return days_ago <= 7
        except:
            return False
    
    def _call_groq_api(self, prompt: str, temperature: float = 0.3) -> Optional[str]:
        """Make API call to Groq using the same configuration as GroqAgent"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a wellness coach specializing in creating concise, actionable summaries. Follow instructions exactly and return only the requested format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_completion_tokens=200,  # Keep responses short
                top_p=0.9,
                stream=False,
                stop=None
            )
            
            return completion.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Groq API error in summarizer: {e}")
            return None
    
    def _get_fallback_assessment(self, risk_level: int, user_profile: Dict) -> str:
        """Generate fallback assessment when API fails"""
        
        stress_level = user_profile.get('Stress_Level', 'Medium')
        sleep_hours = user_profile.get('Sleep_Hours', 7)
        
        if risk_level <= 3:
            return """• Your wellness indicators show positive patterns overall
• Sleep and stress levels appear to be well managed
• Continue maintaining your current healthy routines
• Consider minor optimizations for enhanced wellbeing"""
        elif risk_level <= 6:
            return f"""• Some wellness areas need attention but manageable overall
• {'Sleep duration could be improved' if sleep_hours < 7 else 'Sleep patterns look reasonable'}
• {'Stress management techniques would be beneficial' if stress_level == 'High' else 'Stress levels are within normal range'}
• Focus on consistent daily wellness routines"""
        else:
            return f"""• Several wellness factors require immediate attention
• {'Poor sleep quality is impacting overall health' if sleep_hours < 6 else 'Sleep schedule needs optimization'}
• {'High stress levels need professional stress management' if stress_level == 'High' else 'Stress reduction should be a priority'}
• Consider consulting healthcare professionals for support"""
    
    def _get_fallback_tips(self, user_profile: Dict) -> str:
        """Generate fallback tips when API fails"""
        
        age = user_profile.get('Age', 30)
        stress = user_profile.get('Stress_Level', 'Medium')
        sleep = user_profile.get('Sleep_Hours', 7)
        exercise = user_profile.get('Physical_Activity_Hours', 3)
        
        tips = []
        
        if sleep < 7:
            tips.append("• Establish a consistent bedtime routine for better sleep quality")
        else:
            tips.append("• Maintain your current sleep schedule and optimize sleep environment")
        
        if stress == 'High':
            tips.append("• Practice daily stress reduction techniques like deep breathing")
        else:
            tips.append("• Continue managing stress with regular relaxation activities")
        
        if exercise < 3:
            tips.append("• Gradually increase physical activity to 150 minutes weekly")
        else:
            tips.append("• Maintain your exercise routine and try new activities")
        
        if age > 40:
            tips.append("• Focus on bone health with weight-bearing exercises")
        else:
            tips.append("• Build healthy habits now for long-term wellness")
        
        return '\n'.join(tips[:4])
    
    def _get_fallback_chat_response(self, user_question: str) -> str:
        """Generate fallback chat response when API fails"""
        
        question_lower = user_question.lower()
        
        if any(word in question_lower for word in ['sleep', 'tired', 'rest']):
            return """• Maintain 7-9 hours of sleep nightly with consistent bedtime
• Create a relaxing pre-sleep routine without screens
• Consider consulting a doctor if sleep issues persist"""
        
        elif any(word in question_lower for word in ['stress', 'anxiety', 'worry']):
            return """• Practice deep breathing exercises for 5 minutes daily
• Try progressive muscle relaxation or meditation apps
• Connect with friends or family for emotional support"""
        
        elif any(word in question_lower for word in ['exercise', 'fitness', 'workout']):
            return """• Start with 30 minutes of moderate activity 3 times weekly
• Choose activities you enjoy like walking, swimming, or dancing
• Gradually increase intensity and duration as you build endurance"""
        
        elif any(word in question_lower for word in ['diet', 'food', 'nutrition']):
            return """• Include colorful vegetables and fruits in every meal
• Stay hydrated with 8 glasses of water daily
• Limit processed foods and practice portion control"""
        
        else:
            return """• Focus on maintaining consistent daily wellness routines
• Balance work, rest, exercise, and social connections
• Listen to your body and adjust habits as needed"""
    
    def _get_fallback_progress_summary(self) -> str:
        """Generate fallback progress summary when API fails"""
        return """• You're making steady progress on your wellness journey
• Consistency in completing tasks shows strong commitment
• Keep building on these positive momentum patterns"""

    def format_for_display(self, text: str, display_type: str = "bullet") -> str:
        """Format any text for optimal UI display"""
        
        if not text:
            return "No content available"
        
        if display_type == "bullet":
            # Ensure proper bullet formatting
            if "•" not in text:
                # Convert text to bullets
                sentences = [s.strip() for s in text.split('.') if s.strip() and len(s) > 10]
                bullets = [f"• {sentence}." for sentence in sentences[:4]]
                return '\n'.join(bullets)
            else:
                return self._format_bullet_points(text)
        
        elif display_type == "paragraph":
            # Format as clean paragraph
            clean_text = re.sub(r'\s+', ' ', text.strip())
            if len(clean_text) > 200:
                # Truncate at sentence boundary
                sentences = clean_text.split('.')
                truncated = '. '.join(sentences[:2]) + '.'
                return truncated if len(truncated) < 200 else clean_text[:197] + '...'
            return clean_text
        
        elif display_type == "short":
            # Very brief summary
            clean_text = re.sub(r'\s+', ' ', text.strip())
            return clean_text[:100] + ('...' if len(clean_text) > 100 else '')
        
        return text

# Utility function to create summarizer instance
def create_summarizer(config):
    """Create and return a ResponseSummarizer instance"""
    return ResponseSummarizer(config)   