"""
Version control and rollback system for system evolution.
"""
import git
import json
import shutil
import tempfile
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class EvolutionVersionControl:
    """Git-based version control for system evolution"""
    
    def __init__(self, repo_path: str = "./evolution_history"):
        self.repo_path = Path(repo_path)
        self.versions = []
        self._initialize_repo()
    
    def _initialize_repo(self):
        """Initialize git repository for version control"""
        try:
            if self.repo_path.exists():
                self.repo = git.Repo(self.repo_path)
            else:
                self.repo_path.mkdir(parents=True, exist_ok=True)
                self.repo = git.Repo.init(self.repo_path)
                
                # Create initial commit
                readme_path = self.repo_path / "README.md"
                readme_path.write_text("# Evolution History\n\nThis repository tracks system evolution history.")
                self.repo.index.add([str(readme_path)])
                self.repo.index.commit("Initial commit")
            
            # Load existing versions
            self._load_versions()
            
        except Exception as e:
            logger.error(f"Failed to initialize git repository: {e}")
            raise
    
    def _load_versions(self):
        """Load version history from git commits"""
        self.versions = []
        
        try:
            for commit in self.repo.iter_commits():
                # Parse version info from commit message
                if commit.message.startswith("Checkpoint"):
                    version_info = self._parse_commit_message(commit.message)
                    if version_info:
                        self.versions.append({
                            "id": version_info["id"],
                            "timestamp": datetime.fromtimestamp(commit.committed_date),
                            "description": version_info["description"],
                            "commit_hash": commit.hexsha,
                            "author": str(commit.author),
                            "changes": version_info.get("changes", {})
                        })
        except Exception as e:
            logger.error(f"Failed to load versions: {e}")
    
    def _parse_commit_message(self, message: str) -> Optional[Dict]:
        """Parse version info from commit message"""
        try:
            # Expected format: "Checkpoint v1-20231215120000: Description"
            parts = message.split(":", 1)
            if len(parts) == 2:
                checkpoint_part = parts[0].strip()
                description = parts[1].strip()
                
                # Extract version ID
                version_id = checkpoint_part.split()[1] if len(checkpoint_part.split()) > 1 else None
                
                return {
                    "id": version_id,
                    "description": description
                }
        except Exception as e:
            logger.error(f"Failed to parse commit message: {e}")
        
        return None
    
    async def create_checkpoint(self, changes: Dict[str, Any]) -> str:
        """Create version checkpoint before applying changes"""
        try:
            version_id = f"v{len(self.versions)}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Copy current source files to repo
            src_path = Path(__file__).parent.parent
            
            # Files and directories to track
            track_items = [
                "agents",
                "evaluation", 
                "evolution",
                "crawl4ai_mcp.py",
                "utils.py"
            ]
            
            for item in track_items:
                src_item = src_path / item
                dst_item = self.repo_path / item
                
                if src_item.exists():
                    if src_item.is_dir():
                        # Remove existing directory if present
                        if dst_item.exists():
                            shutil.rmtree(dst_item)
                        shutil.copytree(src_item, dst_item)
                    else:
                        dst_item.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_item, dst_item)
            
            # Save changes metadata
            metadata_path = self.repo_path / f"{version_id}_metadata.json"
            metadata_path.write_text(json.dumps({
                "version_id": version_id,
                "timestamp": datetime.now().isoformat(),
                "changes": changes
            }, indent=2))
            
            # Add all files to git
            self.repo.index.add("*")
            
            # Commit changes
            commit_message = f"Checkpoint {version_id}: {changes.get('description', 'System evolution')}"
            self.repo.index.commit(commit_message)
            
            # Update versions list
            self.versions.append({
                "id": version_id,
                "timestamp": datetime.now(),
                "changes": changes,
                "commit_hash": self.repo.head.commit.hexsha,
                "description": changes.get("description", "")
            })
            
            logger.info(f"Created checkpoint: {version_id}")
            return version_id
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            raise
    
    async def rollback(self, version_id: str) -> bool:
        """Rollback to previous version"""
        try:
            # Find the version
            version = next((v for v in self.versions if v["id"] == version_id), None)
            if not version:
                logger.error(f"Version {version_id} not found")
                return False
            
            # Create backup of current state
            backup_id = await self.create_checkpoint({
                "description": f"Backup before rollback to {version_id}",
                "type": "rollback_backup"
            })
            
            # Checkout the version
            self.repo.git.checkout(version["commit_hash"])
            
            # Copy files back to source
            src_path = Path(__file__).parent.parent
            
            track_items = [
                "agents",
                "evaluation",
                "evolution", 
                "crawl4ai_mcp.py",
                "utils.py"
            ]
            
            for item in track_items:
                repo_item = self.repo_path / item
                dst_item = src_path / item
                
                if repo_item.exists():
                    if repo_item.is_dir():
                        if dst_item.exists():
                            shutil.rmtree(dst_item)
                        shutil.copytree(repo_item, dst_item)
                    else:
                        shutil.copy2(repo_item, dst_item)
            
            # Return to main branch
            self.repo.git.checkout("master")
            
            logger.info(f"Successfully rolled back to version: {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback to version {version_id}: {e}")
            return False
    
    def get_version_history(self) -> List[Dict[str, Any]]:
        """Get complete version history"""
        return sorted(self.versions, key=lambda v: v["timestamp"], reverse=True)
    
    def get_version_diff(self, version_id1: str, version_id2: str) -> Optional[str]:
        """Get diff between two versions"""
        try:
            v1 = next((v for v in self.versions if v["id"] == version_id1), None)
            v2 = next((v for v in self.versions if v["id"] == version_id2), None)
            
            if not v1 or not v2:
                return None
            
            diff = self.repo.git.diff(v1["commit_hash"], v2["commit_hash"])
            return diff
            
        except Exception as e:
            logger.error(f"Failed to get diff: {e}")
            return None
    
    async def create_evolution_branch(self, feature_name: str) -> str:
        """Create a new branch for feature evolution"""
        try:
            branch_name = f"evolution/{feature_name.lower().replace(' ', '-')}"
            
            # Create and checkout new branch
            new_branch = self.repo.create_head(branch_name)
            new_branch.checkout()
            
            logger.info(f"Created evolution branch: {branch_name}")
            return branch_name
            
        except Exception as e:
            logger.error(f"Failed to create evolution branch: {e}")
            raise
    
    async def merge_evolution(self, branch_name: str, description: str) -> bool:
        """Merge evolution branch back to main"""
        try:
            # Checkout main branch
            self.repo.heads.master.checkout()
            
            # Merge the evolution branch
            self.repo.git.merge(branch_name, "--no-ff", "-m", f"Merge evolution: {description}")
            
            # Delete the branch
            self.repo.delete_head(branch_name, force=True)
            
            logger.info(f"Successfully merged evolution branch: {branch_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to merge evolution branch: {e}")
            return False