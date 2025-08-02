"""
Dependency validator agent for checking and fixing system dependencies.
"""
from typing import Dict, List, Any, Optional
import subprocess
import os
import re
import toml
import json
from pathlib import Path
from packaging import version
from .base_agent import BaseAgent

class DependencyValidatorAgent(BaseAgent):
    """Validates and fixes dependency issues"""
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check and fix dependencies"""
        self.logger.info("Starting dependency validation")
        
        # Check Python dependencies
        dependencies_valid = await self._check_dependencies()
        fixes_applied = []
        
        if not dependencies_valid["all_valid"]:
            fixes_applied = await self._apply_dependency_fixes(dependencies_valid)
        
        # Check Docker configuration if needed
        docker_valid = await self._check_docker_config()
        
        return {
            "agent": self.name,
            "dependencies_valid": dependencies_valid["all_valid"],
            "dependency_details": dependencies_valid,
            "fixes_applied": fixes_applied,
            "docker_valid": docker_valid
        }
    
    async def _check_dependencies(self) -> Dict[str, Any]:
        """Check all system dependencies"""
        results = {
            "all_valid": True,
            "python_version": None,
            "missing_packages": [],
            "outdated_packages": [],
            "conflicting_packages": []
        }
        
        # Check Python version
        try:
            python_version_output = subprocess.run(
                ["python", "--version"], 
                capture_output=True, 
                text=True
            )
            current_version = python_version_output.stdout.strip().split()[-1]
            results["python_version"] = current_version
            
            # Check if Python version meets requirements (3.12+)
            if version.parse(current_version) < version.parse("3.12.0"):
                results["all_valid"] = False
                results["python_version_issue"] = f"Python {current_version} is below required 3.12+"
        except Exception as e:
            results["all_valid"] = False
            results["python_version_issue"] = str(e)
        
        # Check installed packages
        try:
            # Read pyproject.toml
            pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
            if pyproject_path.exists():
                with open(pyproject_path, 'r') as f:
                    pyproject = toml.load(f)
                    required_deps = pyproject.get("project", {}).get("dependencies", [])
                
                # Get installed packages
                installed_packages = subprocess.run(
                    ["pip", "list", "--format=json"],
                    capture_output=True,
                    text=True
                )
                installed = json.loads(installed_packages.stdout)
                installed_dict = {pkg["name"].lower(): pkg["version"] for pkg in installed}
                
                # Check for missing packages
                for dep in required_deps:
                    # Parse package name from requirement string
                    package_name = re.split(r'[><=!~]', dep)[0].strip().lower()
                    if package_name not in installed_dict:
                        results["missing_packages"].append(dep)
                        results["all_valid"] = False
                
                # Check for specific critical packages
                critical_packages = [
                    "langgraph", "mem0", "crawl4ai", "supabase", 
                    "openai", "sentence-transformers", "mcp"
                ]
                for pkg in critical_packages:
                    if pkg not in installed_dict:
                        if pkg not in [p.split()[0] for p in results["missing_packages"]]:
                            results["missing_packages"].append(pkg)
                            results["all_valid"] = False
        
        except Exception as e:
            self.logger.error(f"Error checking dependencies: {e}")
            results["all_valid"] = False
            results["error"] = str(e)
        
        return results
    
    async def _apply_dependency_fixes(self, validation_results: Dict[str, Any]) -> List[str]:
        """Apply automatic fixes for dependency issues"""
        fixes = []
        
        # Fix missing packages
        if validation_results.get("missing_packages"):
            for package in validation_results["missing_packages"]:
                try:
                    self.logger.info(f"Installing missing package: {package}")
                    result = subprocess.run(
                        ["pip", "install", package],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        fixes.append(f"Installed {package}")
                    else:
                        fixes.append(f"Failed to install {package}: {result.stderr}")
                except Exception as e:
                    fixes.append(f"Error installing {package}: {str(e)}")
        
        # Update requirements files
        await self._update_requirements_files()
        fixes.append("Updated requirements files")
        
        return fixes
    
    async def _check_docker_config(self) -> bool:
        """Check if Docker configuration is valid"""
        dockerfile_path = Path(__file__).parent.parent.parent / "Dockerfile"
        
        if not dockerfile_path.exists():
            return False
        
        try:
            with open(dockerfile_path, 'r') as f:
                dockerfile_content = f.read()
                
            # Check for required components
            required_components = [
                "FROM python:3.12",
                "RUN pip install uv",
                "crawl4ai-setup"
            ]
            
            for component in required_components:
                if component not in dockerfile_content:
                    return False
            
            return True
        except Exception as e:
            self.logger.error(f"Error checking Docker config: {e}")
            return False
    
    async def _update_requirements_files(self):
        """Update requirements.txt and pyproject.toml with current dependencies"""
        try:
            # Generate requirements.txt
            result = subprocess.run(
                ["pip", "freeze"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                req_path = Path(__file__).parent.parent.parent / "requirements.txt"
                with open(req_path, 'w') as f:
                    f.write(result.stdout)
                self.logger.info("Updated requirements.txt")
        except Exception as e:
            self.logger.error(f"Error updating requirements: {e}")