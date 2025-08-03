"""
Resource management for computational and API resources.
"""
import asyncio
import time
import psutil
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiting for API calls and evolution requests"""
    
    def __init__(self, max_evolutions_per_hour: int = 10, max_api_calls_per_minute: int = 60):
        self.max_evolutions_per_hour = max_evolutions_per_hour
        self.max_api_calls_per_minute = max_api_calls_per_minute
        self.requests = defaultdict(deque)
        self.lock = asyncio.Lock()
    
    async def check_limit(self, key: str) -> bool:
        """Check if request is within rate limit"""
        async with self.lock:
            now = time.time()
            
            # Determine time window based on key type
            if key.startswith("evolution:"):
                window = 3600  # 1 hour in seconds
                max_requests = self.max_evolutions_per_hour
            else:
                window = 60  # 1 minute in seconds
                max_requests = self.max_api_calls_per_minute
            
            # Remove old requests outside the window
            cutoff = now - window
            while self.requests[key] and self.requests[key][0] < cutoff:
                self.requests[key].popleft()
            
            # Check if within limit
            if len(self.requests[key]) >= max_requests:
                return False
            
            # Record this request
            self.requests[key].append(now)
            return True
    
    async def wait_if_needed(self, key: str) -> float:
        """Wait if rate limited and return wait time"""
        if await self.check_limit(key):
            return 0.0
        
        # Calculate wait time
        async with self.lock:
            now = time.time()
            
            if key.startswith("evolution:"):
                window = 3600
            else:
                window = 60
            
            if self.requests[key]:
                oldest = self.requests[key][0]
                wait_time = (oldest + window) - now
                return max(0, wait_time)
        
        return 0.0
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current rate limit usage statistics"""
        stats = {}
        
        for key, timestamps in self.requests.items():
            # Clean old entries
            now = time.time()
            window = 3600 if key.startswith("evolution:") else 60
            cutoff = now - window
            
            active_requests = [t for t in timestamps if t >= cutoff]
            
            stats[key] = {
                "current": len(active_requests),
                "limit": self.max_evolutions_per_hour if key.startswith("evolution:") else self.max_api_calls_per_minute,
                "window": "1 hour" if key.startswith("evolution:") else "1 minute"
            }
        
        return stats


class CostTracker:
    """Track API usage costs"""
    
    def __init__(self, costs_file: str = "api_costs.json"):
        self.costs_file = Path(costs_file)
        self.costs = self._load_costs()
        
        # API pricing (example rates)
        self.pricing = {
            "openai_gpt4": 0.03 / 1000,  # per token
            "openai_gpt3.5": 0.002 / 1000,  # per token
            "openai_embedding": 0.0001 / 1000,  # per token
            "supabase_storage": 0.021,  # per GB per month
            "supabase_api": 0.00001,  # per request
        }
        
        self.lock = asyncio.Lock()
    
    def _load_costs(self) -> Dict[str, Any]:
        """Load costs from file"""
        if self.costs_file.exists():
            try:
                with open(self.costs_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load costs file: {e}")
        
        return {
            "daily": defaultdict(float),
            "monthly": defaultdict(float),
            "total": 0.0
        }
    
    async def _save_costs(self):
        """Save costs to file"""
        try:
            with open(self.costs_file, 'w') as f:
                json.dump(self.costs, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save costs file: {e}")
    
    async def record(self, api_type: str, units: float, unit_type: str = "tokens"):
        """Record API usage cost"""
        async with self.lock:
            # Calculate cost
            if api_type in self.pricing:
                cost = units * self.pricing[api_type]
            else:
                cost = 0.0
                logger.warning(f"Unknown API type for cost tracking: {api_type}")
            
            # Record by date
            today = datetime.now().strftime("%Y-%m-%d")
            month = datetime.now().strftime("%Y-%m")
            
            self.costs["daily"][today] = self.costs["daily"].get(today, 0.0) + cost
            self.costs["monthly"][month] = self.costs["monthly"].get(month, 0.0) + cost
            self.costs["total"] += cost
            
            # Save periodically
            await self._save_costs()
            
            logger.info(f"Recorded {api_type} cost: ${cost:.4f} ({units} {unit_type})")
    
    async def get_daily_cost(self, date: Optional[str] = None) -> float:
        """Get cost for specific day"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        return self.costs["daily"].get(date, 0.0)
    
    async def get_monthly_cost(self, month: Optional[str] = None) -> float:
        """Get cost for specific month"""
        if month is None:
            month = datetime.now().strftime("%Y-%m")
        
        return self.costs["monthly"].get(month, 0.0)
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get comprehensive cost summary"""
        today = datetime.now().strftime("%Y-%m-%d")
        month = datetime.now().strftime("%Y-%m")
        
        # Calculate 7-day average
        last_week_costs = []
        for i in range(7):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            last_week_costs.append(self.costs["daily"].get(date, 0.0))
        
        avg_daily = sum(last_week_costs) / 7 if last_week_costs else 0.0
        
        return {
            "today": self.costs["daily"].get(today, 0.0),
            "this_month": self.costs["monthly"].get(month, 0.0),
            "total": self.costs["total"],
            "average_daily": avg_daily,
            "projected_monthly": avg_daily * 30
        }


class ResourceMonitor:
    """Monitor system resources"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 measurements
        self.alerts = []
        self.thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0
        }
    
    async def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        try:
            # Get CPU usage (non-blocking)
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024 ** 3)
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free_gb = disk.free / (1024 ** 3)
            
            # Get process-specific metrics
            process = psutil.Process()
            process_memory_mb = process.memory_info().rss / (1024 ** 2)
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_available_gb": memory_available_gb,
                "disk_percent": disk_percent,
                "disk_free_gb": disk_free_gb,
                "process_memory_mb": process_memory_mb
            }
            
            # Store in history
            self.metrics_history.append(metrics)
            
            # Check thresholds
            await self._check_thresholds(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get resource usage: {e}")
            return {}
    
    async def _check_thresholds(self, metrics: Dict[str, float]):
        """Check if any thresholds are exceeded"""
        for metric, threshold in self.thresholds.items():
            if metric in metrics and metrics[metric] > threshold:
                alert = {
                    "timestamp": datetime.now().isoformat(),
                    "metric": metric,
                    "value": metrics[metric],
                    "threshold": threshold,
                    "severity": "HIGH" if metrics[metric] > threshold * 1.1 else "MEDIUM"
                }
                self.alerts.append(alert)
                logger.warning(f"Resource alert: {metric} at {metrics[metric]:.1f}% (threshold: {threshold}%)")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of resource metrics"""
        if not self.metrics_history:
            return {"message": "No metrics collected yet"}
        
        # Calculate averages
        cpu_values = [m["cpu_percent"] for m in self.metrics_history if "cpu_percent" in m]
        memory_values = [m["memory_percent"] for m in self.metrics_history if "memory_percent" in m]
        
        return {
            "current": self.metrics_history[-1] if self.metrics_history else {},
            "averages": {
                "cpu_percent": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                "memory_percent": sum(memory_values) / len(memory_values) if memory_values else 0
            },
            "peaks": {
                "cpu_percent": max(cpu_values) if cpu_values else 0,
                "memory_percent": max(memory_values) if memory_values else 0
            },
            "recent_alerts": self.alerts[-10:] if self.alerts else []
        }


class ResourceManager:
    """Manage computational and API resources"""
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.cost_tracker = CostTracker()
        self.resource_monitor = ResourceMonitor()
        self.evolution_quota = {}  # Track per-user quotas
    
    async def check_evolution_quota(self, user_id: str) -> bool:
        """Check if user has evolution quota remaining"""
        key = f"evolution:{user_id}"
        return await self.rate_limiter.check_limit(key)
    
    async def wait_for_quota(self, user_id: str) -> float:
        """Wait for quota to become available"""
        key = f"evolution:{user_id}"
        wait_time = await self.rate_limiter.wait_if_needed(key)
        
        if wait_time > 0:
            logger.info(f"User {user_id} rate limited. Wait time: {wait_time:.1f} seconds")
            await asyncio.sleep(wait_time)
        
        return wait_time
    
    async def track_api_cost(self, api_call: str, tokens: int):
        """Track API usage costs"""
        await self.cost_tracker.record(api_call, tokens)
    
    async def get_resource_usage(self) -> Dict[str, Any]:
        """Get comprehensive resource usage report"""
        current_metrics = await self.resource_monitor.get_current_usage()
        
        return {
            "system_resources": current_metrics,
            "rate_limits": self.rate_limiter.get_usage_stats(),
            "costs": self.cost_tracker.get_cost_summary(),
            "resource_summary": self.resource_monitor.get_metrics_summary()
        }
    
    async def can_execute_evolution(self, user_id: str, estimated_cost: float = 0.0) -> Dict[str, Any]:
        """Check if evolution can be executed based on resources"""
        # Check rate limit
        quota_available = await self.check_evolution_quota(user_id)
        
        # Check system resources
        resources = await self.resource_monitor.get_current_usage()
        cpu_ok = resources.get("cpu_percent", 100) < 80
        memory_ok = resources.get("memory_percent", 100) < 85
        
        # Check costs
        daily_cost = await self.cost_tracker.get_daily_cost()
        daily_limit = 10.0  # $10 daily limit example
        cost_ok = (daily_cost + estimated_cost) < daily_limit
        
        can_execute = quota_available and cpu_ok and memory_ok and cost_ok
        
        return {
            "can_execute": can_execute,
            "quota_available": quota_available,
            "resources_available": cpu_ok and memory_ok,
            "within_cost_limit": cost_ok,
            "details": {
                "cpu_usage": resources.get("cpu_percent", 0),
                "memory_usage": resources.get("memory_percent", 0),
                "daily_cost": daily_cost,
                "estimated_total": daily_cost + estimated_cost
            }
        }