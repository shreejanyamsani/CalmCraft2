import pymongo
from datetime import datetime
import json
import uuid
from typing import Dict, List, Optional

class DatabaseManager:
    def __init__(self, config):
        self.config = config
        self.client = pymongo.MongoClient(config.MONGODB_URI)
        self.db = self.client[config.DATABASE_NAME]
        self._init_collections()
    
    def _init_collections(self):
        """Initialize collections with indexes"""
        self.db[self.config.USERS_COLLECTION].create_index("user_id", unique=True)
        self.db[self.config.CONVERSATIONS_COLLECTION].create_index([("user_id", 1), ("timestamp", -1)])
        self.db[self.config.TASKS_COLLECTION].create_index([("user_id", 1), ("created_at", -1)])
        self.db[self.config.REWARDS_COLLECTION].create_index([("user_id", 1), ("timestamp", -1)])
    
    def save_user_profile(self, user_data: Dict) -> bool:
        """Save or update user profile"""
        try:
            user_data["last_updated"] = datetime.now()
            if "coins" not in user_data:
                user_data["coins"] = 0
            if "total_coins_earned" not in user_data:
                user_data["total_coins_earned"] = 0
                
            result = self.db[self.config.USERS_COLLECTION].update_one(
                {"user_id": user_data["user_id"]},
                {"$set": user_data},
                upsert=True
            )
            return True
        except Exception as e:
            print(f"Error saving user profile: {e}")
            return False
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile"""
        return self.db[self.config.USERS_COLLECTION].find_one({"user_id": user_id})
    
    def save_conversation(self, user_id: str, conversation_data: Dict) -> bool:
        """Save conversation data"""
        try:
            conversation_data.update({
                "user_id": user_id,
                "timestamp": datetime.now(),
                "conversation_id": str(uuid.uuid4())
            })
            self.db[self.config.CONVERSATIONS_COLLECTION].insert_one(conversation_data)
            return True
        except Exception as e:
            print(f"Error saving conversation: {e}")
            return False
    
    def save_task(self, user_id: str, task_data: Dict) -> str:
        """Save assigned task and return task ID"""
        try:
            task_id = str(uuid.uuid4())
            task_data.update({
                "user_id": user_id,
                "task_id": task_id,
                "created_at": datetime.now(),
                "status": "pending",
                "completed_at": None
            })
            self.db[self.config.TASKS_COLLECTION].insert_one(task_data)
            return task_id
        except Exception as e:
            print(f"Error saving task: {e}")
            return None
    
    def get_user_tasks(self, user_id: str, status: str = None) -> List[Dict]:
        """Get user tasks, optionally filtered by status"""
        query = {"user_id": user_id}
        if status:
            query["status"] = status
        
        cursor = self.db[self.config.TASKS_COLLECTION].find(query).sort("created_at", -1)
        return list(cursor)
    
    def complete_task(self, task_id: str, completion_data: Dict = None) -> bool:
        """Mark task as completed"""
        try:
            update_data = {
                "status": "completed",
                "completed_at": datetime.now()
            }
            if completion_data:
                update_data["completion_data"] = completion_data
                
            result = self.db[self.config.TASKS_COLLECTION].update_one(
                {"task_id": task_id},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error completing task: {e}")
            return False
    
    def update_user_coins(self, user_id: str, coins_to_add: int, reward_type: str) -> bool:
        """Update user coins and log reward"""
        try:
            # Update user coins
            user_result = self.db[self.config.USERS_COLLECTION].update_one(
                {"user_id": user_id},
                {
                    "$inc": {
                        "coins": coins_to_add,
                        "total_coins_earned": coins_to_add
                    }
                }
            )
            
            if user_result.modified_count > 0:
                # Log reward
                reward_data = {
                    "user_id": user_id,
                    "reward_type": reward_type,
                    "coins": coins_to_add,
                    "timestamp": datetime.now()
                }
                self.db[self.config.REWARDS_COLLECTION].insert_one(reward_data)
                return True
            return False
        except Exception as e:
            print(f"Error updating coins: {e}")
            return False