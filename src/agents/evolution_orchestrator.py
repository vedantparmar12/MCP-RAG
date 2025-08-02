"""
Evolution orchestrator for managing the self-improvement process.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
import asyncio
from langgraph.graph import StateGraph, END
from mem0 import Memory
import openai
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_agent import BaseAgent
from .dependency_validator import DependencyValidatorAgent
from .code_debugger import CodeDebuggerAgent
from .integration_tester import IntegrationTesterAgent
from evolution.version_control import EvolutionVersionControl
from security.sandbox import SecuritySandbox
from resource_management.manager import ResourceManager
from testing.automated_tester import AutomatedTestGenerator

class EvolutionOrchestrator:
    """Orchestrates the self-improvement process"""
    
    def __init__(self):
        self.memory = Memory()
        self.agents = {
            "dependency_validator": DependencyValidatorAgent("DependencyValidator", self.memory),
            "code_debugger": CodeDebuggerAgent("CodeDebugger", self.memory),
            "integration_tester": IntegrationTesterAgent("IntegrationTester", self.memory)
        }
        self.workflow = self._build_workflow()
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize new components
        self.version_control = EvolutionVersionControl()
        self.security_sandbox = SecuritySandbox()
        self.resource_manager = ResourceManager()
        self.test_generator = AutomatedTestGenerator()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for agent orchestration"""
        workflow = StateGraph()
        
        # Define the state schema
        workflow.add_node("analyze_request", self.analyze_user_request)
        workflow.add_node("validate_dependencies", self.validate_dependencies)
        workflow.add_node("generate_code", self.generate_enhancement)
        workflow.add_node("debug_code", self.debug_generated_code)
        workflow.add_node("test_integration", self.test_integration)
        workflow.add_node("deploy_changes", self.deploy_changes)
        
        # Define the flow
        workflow.add_edge("analyze_request", "validate_dependencies")
        workflow.add_conditional_edges(
            "validate_dependencies",
            self.check_dependency_status,
            {
                "pass": "generate_code",
                "fail": END
            }
        )
        workflow.add_edge("generate_code", "debug_code")
        workflow.add_conditional_edges(
            "debug_code",
            self.check_debug_status,
            {
                "pass": "test_integration",
                "fail": "generate_code"
            }
        )
        workflow.add_conditional_edges(
            "test_integration",
            self.check_test_status,
            {
                "pass": "deploy_changes",
                "fail": "debug_code"
            }
        )
        
        workflow.set_entry_point("analyze_request")
        return workflow.compile()
    
    async def evolve(self, user_request: str, documentation_urls: Optional[List[str]] = None, user_id: str = "default") -> Dict[str, Any]:
        """Main entry point for system evolution"""
        evolution_id = str(uuid.uuid4())
        
        try:
            # Check resource availability
            can_execute = await self.resource_manager.can_execute_evolution(user_id)
            if not can_execute["can_execute"]:
                return {
                    "success": False,
                    "evolution_id": evolution_id,
                    "error": "Resource limits exceeded",
                    "details": can_execute
                }
            
            # Create version checkpoint
            checkpoint_id = await self.version_control.create_checkpoint({
                "description": f"Before evolution: {user_request[:100]}",
                "type": "pre_evolution",
                "evolution_id": evolution_id
            })
            
            initial_state = {
                "user_request": user_request,
                "documentation_urls": documentation_urls or [],
                "timestamp": datetime.now(),
                "evolution_id": evolution_id,
                "checkpoint_id": checkpoint_id,
                "user_id": user_id,
                "status": "started"
            }
            
            # Store evolution request in memory
            self.memory.add(
                messages=[{"role": "user", "content": user_request}],
                metadata={"type": "evolution_request", "id": evolution_id}
            )
            
            # Crawl documentation if provided
            if documentation_urls:
                await self._crawl_documentation(documentation_urls)
            
            # Execute the workflow
            result = await self.workflow.arun(initial_state)
            
            # Store evolution result
            self.memory.add(
                messages=[{"role": "system", "content": str(result)}],
                metadata={"type": "evolution_result", "id": evolution_id}
            )
            
            return result
            
        except Exception as e:
            # Rollback on failure
            if "checkpoint_id" in locals():
                await self.version_control.rollback(checkpoint_id)
            
            return {
                "success": False,
                "evolution_id": evolution_id,
                "error": str(e),
                "status": "failed"
            }
    
    async def analyze_user_request(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the user's evolution request"""
        request = state["user_request"]
        
        # Use LLM to analyze the request
        response = self.openai_client.chat.completions.create(
            model=os.getenv("MODEL_CHOICE", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are an expert at analyzing feature requests for a RAG MCP server. Extract the key requirements and implementation details."},
                {"role": "user", "content": f"Analyze this feature request and extract key requirements:\n\n{request}"}
            ],
            temperature=0.3
        )
        
        analysis = response.choices[0].message.content
        
        state["request_analysis"] = {
            "raw_analysis": analysis,
            "feature_type": self._classify_feature_type(request),
            "complexity": self._estimate_complexity(request),
            "dependencies": self._identify_dependencies(request)
        }
        
        return state
    
    async def validate_dependencies(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate system dependencies"""
        validator = self.agents["dependency_validator"]
        result = await validator.run(state)
        
        state["dependency_validation"] = result
        state["dependencies_valid"] = result.get("dependencies_valid", False)
        
        return state
    
    async def generate_enhancement(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code for the requested enhancement"""
        analysis = state.get("request_analysis", {})
        
        # Generate code based on the analysis
        prompt = self._build_code_generation_prompt(state)
        
        # Track API cost
        response = self.openai_client.chat.completions.create(
            model=os.getenv("MODEL_CHOICE", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are an expert Python developer specializing in MCP servers and RAG systems. Generate high-quality, production-ready code."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        # Track token usage
        if hasattr(response, 'usage'):
            await self.resource_manager.track_api_cost(
                "openai_gpt4", 
                response.usage.total_tokens
            )
        
        generated_code = response.choices[0].message.content
        
        # Extract code from response
        code_blocks = self._extract_code_blocks(generated_code)
        
        # Validate code security
        validated_blocks = []
        for block in code_blocks:
            validation = await self.security_sandbox.validate_code(block["code"])
            block["validation"] = validation
            if validation["safe"]:
                validated_blocks.append(block)
            else:
                # Log security issues
                print(f"Code block failed security validation: {validation['violations']}")
        
        # Generate tests for the code
        if validated_blocks:
            tests = await self.test_generator.generate_tests(
                validated_blocks[0]["code"],
                analysis.get("feature_type", "auto")
            )
            state["generated_tests"] = tests
        
        state["generated_code"] = validated_blocks
        state["generation_status"] = "completed"
        
        return state
    
    async def debug_generated_code(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Debug the generated code"""
        code_blocks = state.get("generated_code", [])
        
        debug_results = []
        for code_block in code_blocks:
            debugger = self.agents["code_debugger"]
            result = await debugger.run({
                "generated_code": code_block["code"],
                "code_type": code_block.get("language", "python"),
                "auto_fix": True
            })
            debug_results.append(result)
        
        state["debug_results"] = debug_results
        state["debug_passed"] = all(r.get("syntax_valid", False) for r in debug_results)
        
        return state
    
    async def test_integration(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Test the integration"""
        tester = self.agents["integration_tester"]
        
        # Prepare test context
        test_context = {
            "test_code": state.get("generated_code", [{}])[0].get("code", "")
        }
        
        result = await tester.run(test_context)
        
        state["test_results"] = result
        state["tests_passed"] = result.get("all_tests_passed", False)
        
        return state
    
    async def deploy_changes(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy the changes to the system"""
        try:
            # Run generated tests first
            if state.get("generated_tests"):
                test_results = await self.test_generator.run_tests(
                    state["generated_tests"],
                    coverage=True
                )
                
                if not test_results["success"]:
                    state["deployment_status"] = "failed"
                    state["status"] = "failed"
                    state["error"] = "Generated tests failed"
                    return state
            
            # Create deployment checkpoint
            deployment_checkpoint = await self.version_control.create_checkpoint({
                "description": f"Deploying evolution: {state['evolution_id']}",
                "type": "deployment",
                "evolution_id": state["evolution_id"],
                "code_blocks": len(state.get("generated_code", []))
            })
            
            # Deploy code blocks
            deployed_files = []
            for code_block in state.get("generated_code", []):
                # Determine file path and deploy
                # This is simplified - in reality would need proper file path resolution
                file_path = self._determine_file_path(code_block, state)
                if file_path:
                    # Write the file
                    with open(file_path, 'w') as f:
                        f.write(code_block["code"])
                    deployed_files.append(file_path)
            
            # Run regression tests
            regression_results = await self.test_generator.run_regression_suite()
            
            if regression_results["success"]:
                state["deployment_status"] = "completed"
                state["status"] = "success"
                state["deployed_files"] = deployed_files
                state["regression_results"] = regression_results
            else:
                # Rollback if regression tests fail
                await self.version_control.rollback(deployment_checkpoint)
                state["deployment_status"] = "rolled_back"
                state["status"] = "failed"
                state["error"] = "Regression tests failed after deployment"
            
        except Exception as e:
            # Rollback on any error
            if "deployment_checkpoint" in locals():
                await self.version_control.rollback(deployment_checkpoint)
            
            state["deployment_status"] = "failed"
            state["status"] = "failed"
            state["error"] = str(e)
        
        return state
    
    def check_dependency_status(self, state: Dict[str, Any]) -> str:
        """Check if dependencies are valid"""
        return "pass" if state.get("dependencies_valid", False) else "fail"
    
    def check_debug_status(self, state: Dict[str, Any]) -> str:
        """Check if debugging passed"""
        return "pass" if state.get("debug_passed", False) else "fail"
    
    def check_test_status(self, state: Dict[str, Any]) -> str:
        """Check if tests passed"""
        return "pass" if state.get("tests_passed", False) else "fail"
    
    async def _crawl_documentation(self, urls: List[str]):
        """Crawl and index documentation"""
        # This would use the existing crawl tools
        for url in urls:
            # Use smart_crawl_url tool to index documentation
            pass
    
    def _classify_feature_type(self, request: str) -> str:
        """Classify the type of feature being requested"""
        request_lower = request.lower()
        
        if "tool" in request_lower or "mcp" in request_lower:
            return "mcp_tool"
        elif "agent" in request_lower:
            return "agent"
        elif "rag" in request_lower or "search" in request_lower:
            return "rag_enhancement"
        elif "fix" in request_lower or "bug" in request_lower:
            return "bug_fix"
        else:
            return "general_enhancement"
    
    def _estimate_complexity(self, request: str) -> str:
        """Estimate the complexity of the request"""
        # Simple heuristic based on keywords
        complex_keywords = ["orchestrate", "integrate", "refactor", "redesign", "migrate"]
        medium_keywords = ["add", "implement", "create", "update", "modify"]
        
        request_lower = request.lower()
        
        if any(keyword in request_lower for keyword in complex_keywords):
            return "high"
        elif any(keyword in request_lower for keyword in medium_keywords):
            return "medium"
        else:
            return "low"
    
    def _identify_dependencies(self, request: str) -> List[str]:
        """Identify potential dependencies from the request"""
        dependencies = []
        
        # Check for common dependencies
        dep_patterns = {
            "langgraph": ["agent", "orchestrat", "workflow"],
            "mem0": ["memory", "persist", "history"],
            "numpy": ["array", "matrix", "numerical"],
            "pandas": ["dataframe", "csv", "excel"],
            "fastapi": ["api", "endpoint", "rest"],
            "pytest": ["test", "testing", "unit test"]
        }
        
        request_lower = request.lower()
        for dep, keywords in dep_patterns.items():
            if any(keyword in request_lower for keyword in keywords):
                dependencies.append(dep)
        
        return dependencies
    
    def _build_code_generation_prompt(self, state: Dict[str, Any]) -> str:
        """Build a prompt for code generation"""
        analysis = state.get("request_analysis", {})
        
        prompt = f"""
Generate Python code for the following feature request:

Request: {state['user_request']}

Analysis:
- Feature Type: {analysis.get('feature_type', 'unknown')}
- Complexity: {analysis.get('complexity', 'medium')}
- Dependencies: {', '.join(analysis.get('dependencies', []))}

Requirements:
1. Follow the existing MCP server patterns
2. Include proper error handling
3. Add comprehensive docstrings
4. Ensure compatibility with the existing codebase
5. Use async/await patterns where appropriate

Generate the complete implementation including:
- Import statements
- Class/function definitions
- Integration points with existing code
- Example usage

Code:
"""
        return prompt
    
    def _extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """Extract code blocks from the generated text"""
        import re
        
        code_blocks = []
        
        # Find all code blocks
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for language, code in matches:
            code_blocks.append({
                "language": language or "python",
                "code": code.strip()
            })
        
        # If no code blocks found, treat the entire text as code
        if not code_blocks and text.strip():
            code_blocks.append({
                "language": "python",
                "code": text.strip()
            })
        
        return code_blocks
    
    def _determine_file_path(self, code_block: Dict[str, Any], state: Dict[str, Any]) -> Optional[str]:
        """Determine the file path for deploying code"""
        # This is a simplified implementation
        # In production, would need proper analysis of code and imports
        
        feature_type = state.get("request_analysis", {}).get("feature_type", "")
        
        if feature_type == "mcp_tool":
            # Extract tool name from code
            import re
            tool_match = re.search(r'async def (\w+)\(', code_block["code"])
            if tool_match:
                tool_name = tool_match.group(1)
                return f"src/tools/{tool_name}.py"
        elif feature_type == "agent":
            # Extract agent class name
            import re
            class_match = re.search(r'class (\w+Agent)', code_block["code"])
            if class_match:
                agent_name = class_match.group(1)
                return f"src/agents/{agent_name.lower()}.py"
        
        # Default to a generated features directory
        return f"src/generated/{state['evolution_id'][:8]}.py"