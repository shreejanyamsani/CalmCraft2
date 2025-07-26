import requests
import json
import time
from typing import Dict, List, Optional, Tuple

class GraniteAgent:
    def __init__(self, config):
        self.config = config
        self.base_url = config.OLLAMA_BASE_URL
        self.model = config.GRANITE_MODEL
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        
    def assign_wellness_tasks(self, user_profile: Dict, mental_health_assessment: str, risk_level: int) -> List[Dict]:
        """Assign wellness tasks based on mental health assessment using Granite AI"""
        
        # First attempt: Comprehensive AI generation
        tasks = self._generate_ai_tasks(user_profile, mental_health_assessment, risk_level)
        
        if tasks:
            return tasks
            
        # Second attempt: Simplified AI generation with retry
        tasks = self._generate_simplified_tasks(user_profile, risk_level)
        
        if tasks:
            return tasks
            
        # Final fallback: Basic AI generation with minimal context
        tasks = self._generate_basic_tasks(risk_level)
        
        if tasks:
            return tasks
            
        # Emergency fallback: Only if AI is completely unavailable
        print("WARNING: Granite AI completely unavailable, using emergency preset tasks")
        return self._get_emergency_preset_tasks(risk_level)
    
    def _generate_ai_tasks(self, user_profile: Dict, mental_health_assessment: str, risk_level: int) -> Optional[List[Dict]]:
        """Primary method: Generate comprehensive personalized tasks using Granite AI"""
        
        prompt = self._build_comprehensive_prompt(user_profile, mental_health_assessment, risk_level)
        
        for attempt in range(self.max_retries):
            try:
                response = self._call_granite_api(prompt, temperature=0.5)
                
                if response:
                    tasks = self._extract_and_validate_tasks(response, risk_level)
                    if tasks and len(tasks) >= 3:  # Ensure we have enough tasks
                        print(f"✓ Generated {len(tasks)} comprehensive AI tasks")
                        return tasks
                        
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for comprehensive generation: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        return None
    
    def _generate_simplified_tasks(self, user_profile: Dict, risk_level: int) -> Optional[List[Dict]]:
        """Fallback method: Generate simplified tasks with reduced context"""
        
        prompt = self._build_simplified_prompt(user_profile, risk_level)
        
        for attempt in range(self.max_retries):
            try:
                response = self._call_granite_api(prompt, temperature=0.4)
                
                if response:
                    tasks = self._extract_and_validate_tasks(response, risk_level)
                    if tasks and len(tasks) >= 2:
                        print(f"✓ Generated {len(tasks)} simplified AI tasks")
                        return tasks
                        
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for simplified generation: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        return None
    
    def _generate_basic_tasks(self, risk_level: int) -> Optional[List[Dict]]:
        """Secondary fallback: Generate basic tasks with minimal context"""
        
        prompt = self._build_basic_prompt(risk_level)
        
        for attempt in range(self.max_retries):
            try:
                response = self._call_granite_api(prompt, temperature=0.3)
                
                if response:
                    tasks = self._extract_and_validate_tasks(response, risk_level, min_tasks=1)
                    if tasks:
                        print(f"✓ Generated {len(tasks)} basic AI tasks")
                        return tasks
                        
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for basic generation: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        return None
    
    def _build_comprehensive_prompt(self, user_profile: Dict, mental_health_assessment: str, risk_level: int) -> str:
        """Build detailed prompt for comprehensive task generation"""
        
        return f"""
You are a professional wellness coach. Create personalized wellness tasks based on user data.

GUIDELINES:
- Use professional, clear language
- Focus on evidence-based wellness practices
- Ensure all recommendations are safe and appropriate
- Use inclusive, respectful language

USER CONTEXT:
Risk Level: {risk_level}/10 (10 = highest risk)
Assessment: {mental_health_assessment}

USER PROFILE:
- Stress Level: {user_profile.get('Stress_Level', 'Not specified')}
- Sleep: {user_profile.get('Sleep_Hours', 'Not specified')} hours/night, quality: {user_profile.get('Sleep_Quality', 'Not specified')}
- Work: {user_profile.get('Work_Hours', 'Not specified')} hours/week
- Exercise: {user_profile.get('Physical_Activity_Hours', 'Not specified')} hours/week
- Occupation: {user_profile.get('Occupation', 'Not specified')}
- Age: {user_profile.get('Age', 'Not specified')}
- Mood: {user_profile.get('Mood', 'Not specified')}
- Anxiety: {user_profile.get('Anxiety_Frequency', 'Not specified')}
- Energy: {user_profile.get('Energy_Level', 'Not specified')}

TASK REQUIREMENTS:
{self._get_risk_specific_requirements(risk_level)}

Generate 4-6 personalized wellness tasks. Each task must be:
1. Relevant to their specific situation
2. Practical and achievable
3. Evidence-based for mental health improvement
4. Appropriate for their risk level
5. Include clear, actionable instructions

MANDATORY JSON FORMAT (return ONLY valid JSON):
[
    {{
        "task_type": "select from: meditation, exercise, sleep_schedule, social_connection, journaling, breathing_exercise, nature_walk, healthy_meal, screen_break, gratitude_practice, professional_help, mindfulness, stress_management, routine_building, creative_activity, relaxation_technique",
        "title": "Clear, engaging title",
        "description": "Brief description explaining the benefits",
        "duration_days": appropriate_number,
        "difficulty": "easy/medium/hard",
        "instructions": "Step-by-step instructions tailored to their profile",
        "completion_criteria": "Clear, measurable success criteria",
        "personalization_notes": "Why this task fits their specific situation"
    }}
]
"""
    
    def _build_simplified_prompt(self, user_profile: Dict, risk_level: int) -> str:
        """Build simplified prompt when comprehensive generation fails"""
        
        key_factors = []
        if user_profile.get('Stress_Level'):
            key_factors.append(f"Stress: {user_profile['Stress_Level']}")
        if user_profile.get('Sleep_Hours'):
            key_factors.append(f"Sleep: {user_profile['Sleep_Hours']}h")
        if user_profile.get('Work_Hours'):
            key_factors.append(f"Work: {user_profile['Work_Hours']}h/week")
        
        factors_str = ", ".join(key_factors) if key_factors else "Limited profile data"
        
        return f"""
You are a wellness coach. Create practical wellness tasks.

USER: {factors_str}
Risk Level: {risk_level}/10

{self._get_risk_specific_requirements(risk_level)}

Generate 3-4 practical wellness tasks as JSON array:
[
    {{
        "task_type": "meditation|exercise|sleep_schedule|journaling|breathing_exercise|professional_help|stress_management",
        "title": "Clear task title",
        "description": "Brief helpful description",
        "duration_days": number,
        "difficulty": "easy|medium|hard",
        "instructions": "Step-by-step instructions",
        "completion_criteria": "How to measure success"
    }}
]
"""
    
    def _build_basic_prompt(self, risk_level: int) -> str:
        """Build minimal prompt for basic task generation"""
        
        return f"""
You are a wellness coach. Generate wellness tasks for risk level {risk_level}/10.

{self._get_risk_specific_requirements(risk_level)}

Return 2-3 tasks as JSON:
[
    {{
        "task_type": "breathing_exercise|meditation|professional_help|journaling",
        "title": "Task title",
        "description": "What this helps with",
        "duration_days": 1-7,
        "difficulty": "easy|medium",
        "instructions": "Clear instructions",
        "completion_criteria": "Success measure"
    }}
]
"""
    
    def _get_risk_specific_requirements(self, risk_level: int) -> str:
        """Get specific requirements based on risk level"""
        
        if risk_level >= 8:
            return """
CRITICAL PRIORITY:
- MUST include immediate professional help seeking
- Focus on crisis intervention and safety
- Include emergency resources and contacts
- Tasks should provide immediate coping mechanisms
- Maximum task duration: 1-2 days
"""
        elif risk_level >= 6:
            return """
HIGH PRIORITY:
- Strongly recommend professional consultation within 1 week
- Include daily anxiety/stress management techniques
- Focus on stabilization and routine building
- Provide structured, manageable activities
- Task duration: 2-7 days
"""
        elif risk_level >= 4:
            return """
MODERATE PRIORITY:
- Include both self-care and gradual improvement activities
- Balance mental and physical wellness approaches
- Encourage social connection and support
- Build sustainable, healthy habits
- Task duration: 5-14 days
"""
        else:
            return """
MAINTENANCE/PREVENTION:
- Focus on wellness enhancement and prevention
- Include enjoyable, engaging activities
- Support long-term habit building
- Promote overall life satisfaction
- Task duration: 7-21 days
"""
    
    def _extract_and_validate_tasks(self, response: str, risk_level: int, min_tasks: int = 2) -> Optional[List[Dict]]:
        """Extract and validate tasks from AI response"""
        
        try:
            # Clean the response
            cleaned_response = self._clean_json_string(response)
            if not cleaned_response:
                return None
            
            # Try to parse JSON
            try:
                tasks = json.loads(cleaned_response)
            except json.JSONDecodeError:
                # Try to extract individual task objects
                tasks = self._extract_individual_tasks(response)
                if not tasks:
                    return None
            
            # Validate task structure
            validated_tasks = self._validate_task_structure(tasks, risk_level)
            
            if validated_tasks and len(validated_tasks) >= min_tasks:
                return validated_tasks
            
        except Exception as e:
            print(f"Error extracting tasks: {e}")
        
        return None
    
    def _clean_json_string(self, json_str: str) -> Optional[str]:
        """Clean malformed JSON string"""
        try:
            # Remove common issues
            cleaned = json_str.strip()
            
            # Find JSON array boundaries
            start_idx = cleaned.find('[')
            end_idx = cleaned.rfind(']')
            
            if start_idx == -1 or end_idx == -1:
                return None
            
            cleaned = cleaned[start_idx:end_idx + 1]
            
            # Remove newlines and extra whitespace
            cleaned = cleaned.replace('\n', ' ').replace('\r', '')
            cleaned = ' '.join(cleaned.split())
            
            # Fix common JSON issues
            cleaned = cleaned.replace(',}', '}')  # Remove trailing commas
            cleaned = cleaned.replace(',]', ']')
            
            return cleaned
            
        except Exception:
            return None
    
    def _extract_individual_tasks(self, response: str) -> Optional[List[Dict]]:
        """Extract individual task objects when array parsing fails"""
        tasks = []
        
        # Find individual task objects
        import re
        task_pattern = r'\{[^{}]*"task_type"[^{}]*\}'
        matches = re.findall(task_pattern, response, re.DOTALL)
        
        for match in matches:
            try:
                # Clean the match
                clean_match = match.replace('\n', ' ').replace('\r', '')
                clean_match = ' '.join(clean_match.split())
                clean_match = clean_match.replace(',}', '}')
                
                task = json.loads(clean_match)
                tasks.append(task)
            except:
                continue
                
        return tasks if tasks else None
    
    def _validate_task_structure(self, tasks: List[Dict], risk_level: int) -> Optional[List[Dict]]:
        """Validate and sanitize task structure"""
        
        if not isinstance(tasks, list):
            return None
            
        valid_tasks = []
        required_fields = ['task_type', 'title', 'description', 'duration_days', 'difficulty', 'instructions', 'completion_criteria']
        
        valid_task_types = {
            'meditation', 'exercise', 'sleep_schedule', 'social_connection', 
            'journaling', 'breathing_exercise', 'nature_walk', 'healthy_meal', 
            'screen_break', 'gratitude_practice', 'professional_help', 
            'mindfulness', 'stress_management', 'routine_building', 
            'creative_activity', 'relaxation_technique'
        }
        
        # Ensure high-risk users get professional help
        has_professional_help = False
        
        for task in tasks:
            if not isinstance(task, dict):
                continue
                
            # Check required fields
            if not all(field in task and str(task[field]).strip() for field in required_fields):
                continue
                
            # Validate and clean task
            validated_task = self._clean_and_validate_task(task, valid_task_types)
            if validated_task:
                valid_tasks.append(validated_task)
                
                if validated_task['task_type'] == 'professional_help':
                    has_professional_help = True
        
        # Ensure high-risk users have professional help task
        if risk_level >= 7 and not has_professional_help and len(valid_tasks) > 0:
            professional_task = self._generate_professional_help_task(risk_level)
            valid_tasks.insert(0, professional_task)
        
        return valid_tasks if len(valid_tasks) >= 1 else None
    
    def _clean_and_validate_task(self, task: Dict, valid_task_types: set) -> Optional[Dict]:
        """Clean and validate individual task"""
        
        try:
            # Validate task_type
            if task.get('task_type') not in valid_task_types:
                return None
                
            # Clean and validate duration
            try:
                duration = int(task.get('duration_days', 7))
                duration = max(1, min(30, duration))  # Clamp between 1-30 days
            except (ValueError, TypeError):
                duration = 7
                
            # Validate difficulty
            difficulty = task.get('difficulty', 'medium').lower()
            if difficulty not in ['easy', 'medium', 'hard']:
                difficulty = 'medium'
                
            # Clean text fields
            cleaned_task = {
                'task_type': task['task_type'],
                'title': str(task['title']).strip()[:100],
                'description': str(task['description']).strip()[:300],
                'duration_days': duration,
                'difficulty': difficulty,
                'instructions': str(task['instructions']).strip()[:1000],
                'completion_criteria': str(task['completion_criteria']).strip()[:200]
            }
            
            # Add optional fields if present
            if 'personalization_notes' in task:
                cleaned_task['personalization_notes'] = str(task['personalization_notes']).strip()[:200]
                
            return cleaned_task
            
        except Exception as e:
            print(f"Error cleaning task: {e}")
            return None
    
    def _generate_professional_help_task(self, risk_level: int) -> Dict:
        """Generate professional help task for high-risk users"""
        
        urgency = "immediately" if risk_level >= 8 else "within 1-2 days"
        
        return {
            "task_type": "professional_help",
            "title": "Seek Professional Mental Health Support",
            "description": f"Contact a mental health professional {urgency} for assessment and support",
            "duration_days": 2,
            "difficulty": "medium",
            "instructions": f"Contact your healthcare provider, call a mental health helpline, or visit a mental health clinic {urgency}. If in immediate crisis, call emergency services (911) or go to the nearest emergency room.",
            "completion_criteria": "Make contact with a mental health professional or crisis support service"
        }
    
    def _get_emergency_preset_tasks(self, risk_level: int) -> List[Dict]:
        """ONLY used when Granite AI is completely unavailable - minimal preset tasks"""
        
        print("EMERGENCY: Using preset tasks - Granite AI unavailable")
        
        if risk_level >= 7:
            return [
                {
                    "task_type": "professional_help",
                    "title": "Emergency Professional Help",
                    "description": "Seek immediate professional mental health support",
                    "duration_days": 1,
                    "difficulty": "medium",
                    "instructions": "Contact emergency mental health services, your doctor, or call a crisis helpline immediately.",
                    "completion_criteria": "Make contact with professional help"
                }
            ]
        elif risk_level >= 4:
            return [
                {
                    "task_type": "breathing_exercise",
                    "title": "Daily Breathing Practice",
                    "description": "Use breathing exercises to manage stress and anxiety",
                    "duration_days": 7,
                    "difficulty": "easy",
                    "instructions": "Practice 4-7-8 breathing: Inhale for 4 counts, hold for 7 counts, exhale for 8 counts. Repeat 4 times, twice daily.",
                    "completion_criteria": "Complete breathing exercise twice daily for one week"
                },
                {
                    "task_type": "sleep_schedule",
                    "title": "Improve Sleep Routine",
                    "description": "Establish a consistent sleep schedule for better rest",
                    "duration_days": 14,
                    "difficulty": "medium",
                    "instructions": "Go to bed and wake up at the same time daily. Create a 30-minute wind-down routine before bed.",
                    "completion_criteria": "Maintain consistent sleep schedule for 2 weeks"
                }
            ]
        else:
            return [
                {
                    "task_type": "gratitude_practice",
                    "title": "Daily Gratitude Journal",
                    "description": "Practice gratitude to boost mood and well-being",
                    "duration_days": 14,
                    "difficulty": "easy",
                    "instructions": "Write down 3 things you're grateful for each morning. Be specific and reflect on why you appreciate them.",
                    "completion_criteria": "Complete gratitude entries for 14 consecutive days"
                },
                {
                    "task_type": "nature_walk",
                    "title": "Weekly Nature Walks",
                    "description": "Connect with nature to reduce stress and improve mood",
                    "duration_days": 21,
                    "difficulty": "easy",
                    "instructions": "Take a 20-30 minute walk in a park, garden, or natural area twice per week. Focus on your surroundings.",
                    "completion_criteria": "Complete 6 nature walks over 3 weeks"
                }
            ]
    
    def _call_granite_api(self, prompt: str, temperature: float = 0.3) -> Optional[str]:
        """Enhanced API call with better error handling and logging"""
        
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "num_predict": 2000,  # Increased for better task generation
                "repeat_penalty": 1.1,
                "stop": ["Human:", "Assistant:", "\n\n---"]
            }
        }
        
        try:
            print(f"🤖 Calling Granite API (temp={temperature})...")
            response = requests.post(self.base_url, json=data, timeout=90)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get('response', '').strip()
                
                if ai_response:
                    print(f"✓ Granite API responded ({len(ai_response)} chars)")
                    return ai_response
                else:
                    print("⚠ Granite API returned empty response")
                    return None
            else:
                print(f"❌ Granite API error: {response.status_code}")
                return None
                
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to Granite (Ollama not running?)")
            return None
        except requests.exceptions.Timeout:
            print("❌ Granite API timeout")
            return None
        except Exception as e:
            print(f"❌ Granite API error: {e}")
            return None