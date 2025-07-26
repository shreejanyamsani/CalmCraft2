import json
import time
import random
import threading
from datetime import datetime
from typing import Dict, Optional
import numpy as np

class FitnessDataSimulator:
    def __init__(self, config):
        self.config = config
        self.running = False
        self.thread = None
        
        # Base patterns for realistic simulation
        self.sleep_pattern = {"base": 7.5, "variance": 1.5}
        self.steps_pattern = {"base": 8000, "variance": 3000}
        self.mood_pattern = {"base": 7, "variance": 2}
        self.calories_pattern = {"base": 400, "variance": 200}
        self.water_pattern = {"base": 6, "variance": 2}
        
        # Daily progression
        self.daily_progression = 0
    
    def generate_realistic_data(self) -> Dict:
        """Generate realistic fitness data with daily patterns"""
        hour = datetime.now().hour
        
        # Adjust patterns based on time of day
        steps_multiplier = self._get_activity_multiplier(hour)
        mood_adjustment = self._get_mood_adjustment(hour)
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "sleep_hours": max(4, min(12, 
                np.random.normal(self.sleep_pattern["base"], self.sleep_pattern["variance"])
            )),
            "steps": max(0, int(
                np.random.normal(self.steps_pattern["base"] * steps_multiplier, self.steps_pattern["variance"])
            )),
            "mood_score": max(1, min(10, 
                np.random.normal(self.mood_pattern["base"] + mood_adjustment, self.mood_pattern["variance"])
            )),
            "calories_burned": max(0, int(
                np.random.normal(self.calories_pattern["base"] * steps_multiplier, self.calories_pattern["variance"])
            )),
            "water_intake": max(0, 
                np.random.normal(self.water_pattern["base"], self.water_pattern["variance"])
            ),
            "heart_rate": random.randint(60, 100),
            "active_minutes": random.randint(20, 120)
        }
        
        return data
    
    def _get_activity_multiplier(self, hour: int) -> float:
        """Get activity multiplier based on hour of day"""
        if 6 <= hour <= 9:  # Morning
            return 1.2
        elif 12 <= hour <= 14:  # Lunch
            return 1.1
        elif 17 <= hour <= 19:  # Evening
            return 1.3
        elif 22 <= hour or hour <= 5:  # Night
            return 0.3
        else:
            return 1.0
    
    def _get_mood_adjustment(self, hour: int) -> float:
        """Get mood adjustment based on hour of day"""
        if 7 <= hour <= 11:  # Morning
            return 0.5
        elif 14 <= hour <= 16:  # Afternoon dip
            return -0.3
        elif 18 <= hour <= 21:  # Evening
            return 0.3
        else:
            return 0
    
    def save_data(self, data: Dict):
        """Save data to JSON file"""
        try:
            with open(self.config.FIT_STREAM_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def _simulate_loop(self):
        """Main simulation loop"""
        while self.running:
            try:
                data = self.generate_realistic_data()
                self.save_data(data)
                time.sleep(5)  # Update every 5 seconds
            except Exception as e:
                print(f"Simulation error: {e}")
                time.sleep(5)
    
    def start_simulation(self):
        """Start the data simulation"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._simulate_loop, daemon=True)
            self.thread.start()
            print("✅ Fitness data simulation started")
    
    def stop_simulation(self):
        """Stop the data simulation"""
        self.running = False
        if self.thread:
            self.thread.join()
        print("🛑 Fitness data simulation stopped")
    
    def get_latest_data(self) -> Optional[Dict]:
        """Get the latest fitness data"""
        try:
            with open(self.config.FIT_STREAM_FILE, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except Exception as e:
            print(f"Error reading data: {e}")
            return None
