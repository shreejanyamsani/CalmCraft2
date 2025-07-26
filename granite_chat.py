import json
import time
import re
from typing import Dict, List, Optional, Tuple, Any
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_ibm import WatsonxLLM

class GraniteChatAgent:
    def __init__(self, config):
        self.config = config
        self.max_retries = 3
        self.retry_delay = 2
        self.conversation_history = []
        self.llm = WatsonxLLM(
            model_id=config.GRANITE_MODEL,
            url=config.WATSON_URL,
            apikey=config.WATSON_API_KEY,
            project_id=config.WATSON_PROJECT_ID,
            params={
                GenParams.DECODING_METHOD: "greedy",
                GenParams.TEMPERATURE: 0.7,
                GenParams.MIN_NEW_TOKENS: 20,
                GenParams.MAX_NEW_TOKENS: 500,
                GenParams.STOP_SEQUENCES: ["Human:", "User:", "\n\nHuman:", "\n\nUser:"],
            },
        )

    def get_chat_response(self, user_message: str, user_profile: Optional[Dict] = None, context: Optional[str] = None) -> str:
        self.conversation_history.append({"role": "user", "content": user_message})
        response = self._generate_direct_response(user_message, user_profile, context)
        if not response:
            response = self._generate_fallback_response(user_message, user_profile)
        if not response:
            response = self._generate_basic_response(user_message)
        if not response:
            response = "I apologize, but I'm having trouble processing your request right now. Could you please rephrase your question or try again?"
        self.conversation_history.append({"role": "assistant", "content": response})
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        return response

    def get_wellness_advice(self, topic: str, user_profile: Optional[Dict] = None) -> str:
        advice_prompt = f"Please provide wellness advice about {topic}"
        if user_profile:
            advice_prompt += f" for someone with these characteristics: {self._format_user_profile(user_profile)}"
        return self.get_chat_response(advice_prompt, user_profile)

    def answer_question(self, question: str, user_profile: Optional[Dict] = None) -> str:
        return self.get_chat_response(question, user_profile)

    def provide_support(self, concern: str, user_profile: Optional[Dict] = None) -> str:
        support_message = f"I'm concerned about {concern} and could use some support and guidance."
        return self.get_chat_response(support_message, user_profile)

    def _generate_direct_response(self, user_message: str, user_profile: Optional[Dict], context: Optional[str]) -> Optional[str]:
        prompt = self._build_chat_prompt(user_message, user_profile, context)
        for attempt in range(self.max_retries):
            try:
                response = self._call_granite_api(prompt, temperature=0.7)
                if response and len(response.strip()) > 10:
                    cleaned_response = self._clean_chat_response(response)
                    if cleaned_response:
                        print(f"✓ Generated direct chat response ({len(cleaned_response)} chars)")
                        return cleaned_response
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for direct response: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        return None

    def _generate_fallback_response(self, user_message: str, user_profile: Optional[Dict]) -> Optional[str]:
        prompt = self._build_simple_chat_prompt(user_message, user_profile)
        for attempt in range(self.max_retries):
            try:
                response = self._call_granite_api(prompt, temperature=0.6)
                if response and len(response.strip()) > 5:
                    cleaned_response = self._clean_chat_response(response)
                    if cleaned_response:
                        print(f"✓ Generated fallback chat response")
                        return cleaned_response
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for fallback response: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        return None

    def _generate_basic_response(self, user_message: str) -> Optional[str]:
        prompt = self._build_basic_chat_prompt(user_message)
        for attempt in range(self.max_retries):
            try:
                response = self._call_granite_api(prompt, temperature=0.5)
                if response and len(response.strip()) > 3:
                    cleaned_response = self._clean_chat_response(response)
                    if cleaned_response:
                        print(f"✓ Generated basic chat response")
                        return cleaned_response
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for basic response: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        return None

    def _build_chat_prompt(self, user_message: str, user_profile: Optional[Dict], context: Optional[str]) -> str:
        conversation_context = self._build_conversation_context()
        user_context = ""
        if user_profile:
            user_context = f"""
USER CONTEXT:
- Stress Level: {user_profile.get('Stress_Level', 'Unknown')}
- Sleep: {user_profile.get('Sleep_Hours', 'Unknown')} hours/night
- Work: {user_profile.get('Work_Hours', 'Unknown')} hours/week
- Mood: {user_profile.get('Mood', 'Unknown')}
- Age: {user_profile.get('Age', 'Unknown')}
- Occupation: {user_profile.get('Occupation', 'Unknown')}
"""
        additional_context = f"\nADDITIONAL CONTEXT:\n{context}\n" if context else ""
        return f"""You are a helpful, empathetic, and knowledgeable AI assistant specializing in wellness and mental health support. Respond directly in first person as if you are speaking to the user face-to-face.

IMPORTANT: Start your response immediately with your direct answer. Do not think aloud, analyze, or explain your reasoning process. Give a direct, conversational response from the first word.

Guidelines:
1. SPEAK DIRECTLY: Use "I", "you", "your" - respond as if in conversation
2. NO ANALYSIS: Don't show your thinking process or reasoning
3. BE IMMEDIATE: Start with your actual response, not explanations
4. STAY SUPPORTIVE: Be empathetic and helpful
5. BE CONCISE: Give practical, actionable advice

{user_context}{additional_context}

CONVERSATION HISTORY:
{conversation_context}

USER MESSAGE: {user_message}

Your direct response (start immediately, no analysis):"""

    def _build_simple_chat_prompt(self, user_message: str, user_profile: Optional[Dict]) -> str:
        user_info = ""
        if user_profile:
            key_info = []
            if user_profile.get('Stress_Level'):
                key_info.append(f"Stress: {user_profile['Stress_Level']}")
            if user_profile.get('Mood'):
                key_info.append(f"Mood: {user_profile['Mood']}")
            if key_info:
                user_info = f"User info: {', '.join(key_info)}\n"
        recent_context = ""
        if len(self.conversation_history) > 0:
            recent_context = f"Previous message: {self.conversation_history[-1]['content'][:100]}...\n"
        return f"""You are a helpful wellness AI assistant. Give a direct, first-person response. Start immediately with your answer - no thinking aloud or analysis.

{user_info}{recent_context}
User: {user_message}

Direct response:"""

    def _build_basic_chat_prompt(self, user_message: str) -> str:
        return f"""You are a helpful AI assistant. Respond directly in first person. Start immediately with your response - no analysis or thinking aloud.

User: {user_message}

Direct response:"""

    def _build_conversation_context(self) -> str:
        if not self.conversation_history:
            return "No previous conversation."
        context_lines = []
        recent_history = self.conversation_history[-6:]
        for entry in recent_history:
            role = "User" if entry["role"] == "user" else "Assistant"
            content = entry["content"][:150] + "..." if len(entry["content"]) > 150 else entry["content"]
            context_lines.append(f"{role}: {content}")
        return "\n".join(context_lines)

    def _format_user_profile(self, user_profile: Dict) -> str:
        profile_parts = []
        for key, value in user_profile.items():
            if value and str(value).strip().lower() not in ['unknown', 'not specified', '']:
                readable_key = key.replace('_', ' ').title()
                profile_parts.append(f"{readable_key}: {value}")
        return ", ".join(profile_parts) if profile_parts else "Limited profile information"

    def _clean_chat_response(self, response: str) -> Optional[str]:
        if not response:
            return None
        
        cleaned = response.strip()
        
        # Remove common system/thinking indicators
        system_indicators = [
            "You are a helpful", "RESPONSE:", "Assistant:", "AI:", "Human:", "User:",
            "Direct response:", "Your direct response:", "Based on", "Looking at",
            "Let me", "I need to", "First,", "The user is", "From what", "Since",
            "Given that", "Considering", "In this case", "It seems", "This appears"
        ]
        
        # Remove thinking patterns - lines that start with analysis
        thinking_patterns = [
            r"^(Okay|So|Well|Now|Let me|I see|Looking at|Based on|From what|Given that|Since|Considering).*",
            r"^The user (is|has|wants|needs|seems).*",
            r"^This (is|seems|appears|looks|sounds).*",
            r"^It (seems|appears|looks|sounds).*",
            r"^They (are|have|want|need|seem).*"
        ]
        
        lines = cleaned.split('\n')
        cleaned_lines = []
        skip_mode = True  # Start in skip mode to remove initial thinking
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line is thinking/analysis
            is_thinking = False
            
            # Check system indicators
            if any(line.startswith(indicator) for indicator in system_indicators):
                is_thinking = True
            
            # Check thinking patterns
            if any(re.match(pattern, line, re.IGNORECASE) for pattern in thinking_patterns):
                is_thinking = True
            
            # Check for analysis keywords at start of line
            analysis_starts = ['okay', 'so', 'well', 'now', 'let me', 'i see', 'looking', 'based on', 
                             'from what', 'given that', 'since', 'considering', 'the user', 'this is',
                             'it seems', 'they are', 'i think', 'i believe', 'i should', 'i need to']
            
            line_lower = line.lower()
            if any(line_lower.startswith(start) for start in analysis_starts):
                is_thinking = True
            
            # If we're in skip mode and this isn't thinking, switch to collect mode
            if skip_mode and not is_thinking:
                skip_mode = False
            
            # Collect non-thinking lines after we've found the first real response
            if not skip_mode and not is_thinking:
                cleaned_lines.append(line)
        
        # If no good lines found, try to salvage something
        if not cleaned_lines:
            # Look for lines that start with "I" or common first-person responses
            for line in lines:
                line = line.strip()
                if line and (line.startswith('I ') or line.startswith('You ') or 
                           line.startswith('Your ') or line.startswith('Based on your')):
                    cleaned_lines.append(line)
                    break
        
        # Join the cleaned lines
        final_response = '\n'.join(cleaned_lines).strip()
        
        # Additional cleanup
        if final_response.startswith('"') and final_response.endswith('"'):
            final_response = final_response[1:-1].strip()
        
        # Ensure minimum length
        if len(final_response) < 10:
            return None
        
        # Limit length
        if len(final_response) > 1000:
            final_response = final_response[:1000] + "..."
        
        return final_response if final_response else None

    def _call_granite_api(self, prompt: str, temperature: float = 0.7) -> Optional[str]:
        try:
            print(f"🤖 Calling Watson Granite API (temp={temperature})...")
            self.llm.params[GenParams.TEMPERATURE] = temperature
            response = self.llm.invoke(prompt)
            if response and response.strip():
                print(f"✓ Watson Granite API responded ({len(response)} chars)")
                return response.strip()
            else:
                print("⚠ Watson Granite API returned empty response")
                return None
        except Exception as e:
            print(f"❌ Watson Granite API error: {e}")
            return None

    def clear_conversation_history(self):
        self.conversation_history = []
        print("✓ Conversation history cleared")

    def get_conversation_summary(self) -> str:
        if not self.conversation_history:
            return "No conversation history available."
        total_messages = len(self.conversation_history)
        user_messages = len([msg for msg in self.conversation_history if msg["role"] == "user"])
        assistant_messages = len([msg for msg in self.conversation_history if msg["role"] == "assistant"])
        recent_topics = []
        for msg in self.conversation_history[-4:]:
            if msg["role"] == "user":
                content = msg["content"][:100]
                recent_topics.append(content)
        summary = f"""Conversation Summary:
- Total messages: {total_messages}
- User messages: {user_messages}
- Assistant messages: {assistant_messages}
- Recent topics discussed: {'; '.join(recent_topics) if recent_topics else 'General conversation'}"""
        return summary.strip()

    def set_chat_personality(self, personality_type: str = "supportive"):
        personality_configs = {
            "supportive": {
                "temperature": 0.7,
                "max_tokens": 500,
                "style": "empathetic and encouraging"
            },
            "professional": {
                "temperature": 0.5,
                "max_tokens": 400,
                "style": "clinical and informative"
            },
            "casual": {
                "temperature": 0.8,
                "max_tokens": 300,
                "style": "friendly and conversational"
            },
            "direct": {
                "temperature": 0.4,
                "max_tokens": 250,
                "style": "concise and straightforward"
            }
        }
        if personality_type in personality_configs:
            config = personality_configs[personality_type]
            self.llm.params[GenParams.TEMPERATURE] = config["temperature"]
            self.llm.params[GenParams.MAX_NEW_TOKENS] = config["max_tokens"]
            print(f"✓ Chat personality set to: {personality_type} ({config['style']})")
        else:
            print(f"⚠ Unknown personality type: {personality_type}. Available: {list(personality_configs.keys())}")

class ChatIntegration:
    def __init__(self, granite_agent: GraniteChatAgent):
        self.agent = granite_agent

    async def handle_chat_message(self, message: str, user_id: str, user_profile: Optional[Dict] = None) -> Dict:
        try:
            response = self.agent.get_chat_response(message, user_profile)
            return {
                "success": True,
                "response": response,
                "user_id": user_id,
                "timestamp": time.time(),
                "response_type": "direct_answer",
                "confidence": "high" if len(response) > 50 else "medium"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I'm having technical difficulties. Please try again.",
                "user_id": user_id,
                "timestamp": time.time(),
                "response_type": "error"
            }

    def get_quick_responses(self, user_message: str) -> List[str]:
        message_lower = user_message.lower()
        if any(word in message_lower for word in ['stressed', 'anxiety', 'worried']):
            return [
                "Tell me more about what's causing your stress",
                "Would you like some immediate stress relief techniques?",
                "How long have you been feeling this way?"
            ]
        elif any(word in message_lower for word in ['sleep', 'tired', 'insomnia']):
            return [
                "What's your current sleep schedule like?",
                "Would you like tips for better sleep hygiene?",
                "How many hours of sleep do you typically get?"
            ]
        elif any(word in message_lower for word in ['sad', 'depressed', 'down']):
            return [
                "I'm here to support you through this",
                "Would you like to talk about what's making you feel this way?",
                "Have you considered speaking with a counselor?"
            ]
        else:
            return [
                "Can you tell me more about that?",
                "How can I best support you with this?",
                "What would be most helpful for you right now?"
            ]