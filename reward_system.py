from typing import Dict, List

class RewardSystem:
    def __init__(self, config, db_manager):
        self.config = config
        self.db = db_manager
        self.task_rewards = config.TASK_REWARDS
    
    def calculate_task_reward(self, task_type: str, difficulty: str, completion_data: Dict = None) -> int:
        """Calculate coins for completed task"""
        base_reward = self.task_rewards.get(task_type, 10)
        
        # Difficulty multiplier
        difficulty_multipliers = {
            "easy": 1.0,
            "medium": 1.3,
            "hard": 1.6
        }
        
        multiplier = difficulty_multipliers.get(difficulty, 1.0)
        
        # Quality bonus based on completion data
        quality_bonus = 0
        if completion_data:
            if completion_data.get('quality_rating', 0) >= 4:
                quality_bonus = 5
            if completion_data.get('exceeded_expectations', False):
                quality_bonus += 3
        
        total_reward = int(base_reward * multiplier) + quality_bonus
        return total_reward
    
    def award_task_completion(self, user_id: str, task_id: str, completion_data: Dict = None) -> int:
        """Award coins for completing a task"""
        # Get task details
        task = self.db.db[self.config.TASKS_COLLECTION].find_one({"task_id": task_id})
        
        if not task or task['status'] == 'completed':
            return 0
        
        # Calculate reward
        coins = self.calculate_task_reward(
            task['task_type'], 
            task.get('difficulty', 'medium'), 
            completion_data
        )
        
        # Update task as completed
        self.db.complete_task(task_id, completion_data)
        
        # Award coins
        reward_type = f"task_completion_{task['task_type']}"
        if self.db.update_user_coins(user_id, coins, reward_type):
            return coins
        
        return 0
    
    def get_reward_summary(self, user_id: str) -> Dict:
        """Get user's reward summary"""
        user = self.db.get_user_profile(user_id)
        completed_tasks = self.db.get_user_tasks(user_id, "completed")
        
        return {
            "total_coins": user.get('coins', 0) if user else 0,
            "total_earned": user.get('total_coins_earned', 0) if user else 0,
            "completed_tasks": len(completed_tasks),
            "pending_tasks": len(self.db.get_user_tasks(user_id, "pending"))
        }