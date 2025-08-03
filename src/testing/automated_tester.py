"""
Automated testing framework for evolved features.
"""
import ast
import json
import subprocess
import tempfile
import os
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.manager import LLMManager

logger = logging.getLogger(__name__)

class AutomatedTestGenerator:
    """Generate and execute tests for evolved features"""
    
    def __init__(self):
        self.llm_manager = LLMManager(default_provider=os.getenv("LLM_PROVIDER", "auto"))
        self.test_templates = self._load_test_templates()
        self.coverage_threshold = 80.0  # Minimum coverage percentage
    
    def _load_test_templates(self) -> Dict[str, str]:
        """Load test templates for different patterns"""
        return {
            "mcp_tool": '''
import pytest
import asyncio
from unittest.mock import Mock, patch
from {module_path} import {tool_name}

class Test{tool_name_camel}:
    """Test suite for {tool_name} MCP tool"""
    
    @pytest.fixture
    def mock_context(self):
        """Create mock MCP context"""
        context = Mock()
        context.request_context.lifespan_context.supabase_client = Mock()
        context.request_context.lifespan_context.crawler = Mock()
        return context
    
    @pytest.mark.asyncio
    async def test_{tool_name}_success(self, mock_context):
        """Test successful execution of {tool_name}"""
        # Test implementation
        {test_success_implementation}
    
    @pytest.mark.asyncio
    async def test_{tool_name}_invalid_input(self, mock_context):
        """Test {tool_name} with invalid input"""
        # Test implementation
        {test_invalid_input_implementation}
    
    @pytest.mark.asyncio
    async def test_{tool_name}_error_handling(self, mock_context):
        """Test error handling in {tool_name}"""
        # Test implementation
        {test_error_handling_implementation}
''',
            "agent": '''
import pytest
import asyncio
from unittest.mock import Mock, patch
from {module_path} import {agent_name}

class Test{agent_name}:
    """Test suite for {agent_name} agent"""
    
    @pytest.fixture
    def agent(self):
        """Create agent instance"""
        memory = Mock()
        return {agent_name}("{agent_name}", memory)
    
    @pytest.mark.asyncio
    async def test_execute_success(self, agent):
        """Test successful agent execution"""
        context = {test_context}
        result = await agent.execute(context)
        
        assert result["agent"] == "{agent_name}"
        assert "error" not in result
        {additional_assertions}
    
    @pytest.mark.asyncio
    async def test_validate_preconditions(self, agent):
        """Test precondition validation"""
        context = {test_context}
        valid = await agent.validate_preconditions(context)
        assert valid is True
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, agent):
        """Test agent error recovery"""
        {error_test_implementation}
''',
            "function": '''
import pytest
from {module_path} import {function_name}

def test_{function_name}_basic():
    """Test basic functionality of {function_name}"""
    {basic_test}

def test_{function_name}_edge_cases():
    """Test edge cases for {function_name}"""
    {edge_case_tests}

def test_{function_name}_error_handling():
    """Test error handling in {function_name}"""
    {error_tests}
'''
        }
    
    async def generate_tests(self, feature_code: str, feature_type: str = "auto") -> str:
        """Generate comprehensive tests for new feature"""
        try:
            # Analyze code to understand structure
            analysis = self._analyze_code(feature_code)
            
            if feature_type == "auto":
                feature_type = self._detect_feature_type(analysis)
            
            # Generate test code using LLM
            prompt = self._build_test_generation_prompt(feature_code, analysis, feature_type)
            
            messages = [
                {"role": "system", "content": "You are an expert test engineer. Generate comprehensive pytest tests including edge cases, error handling, and integration tests."},
                {"role": "user", "content": prompt}
            ]
            
            generated_tests = asyncio.run(
                self.llm_manager.chat_completion(
                    messages=messages,
                    temperature=0.2
                )
            )
            
            # Post-process and validate tests
            tests = self._post_process_tests(generated_tests, analysis)
            
            return tests
            
        except Exception as e:
            logger.error(f"Failed to generate tests: {e}")
            # Fallback to template-based generation
            return self._generate_from_template(feature_code, feature_type)
    
    def _analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze code structure"""
        try:
            tree = ast.parse(code)
            
            analysis = {
                "functions": [],
                "classes": [],
                "imports": [],
                "decorators": [],
                "async_functions": []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "is_async": False,
                        "decorators": [self._get_decorator_name(d) for d in node.decorator_list],
                        "docstring": ast.get_docstring(node)
                    }
                    analysis["functions"].append(func_info)
                    
                elif isinstance(node, ast.AsyncFunctionDef):
                    func_info = {
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "is_async": True,
                        "decorators": [self._get_decorator_name(d) for d in node.decorator_list],
                        "docstring": ast.get_docstring(node)
                    }
                    analysis["async_functions"].append(func_info)
                    
                elif isinstance(node, ast.ClassDef):
                    analysis["classes"].append({
                        "name": node.name,
                        "bases": [self._get_name(base) for base in node.bases],
                        "methods": []  # Could be expanded
                    })
                    
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    analysis["imports"].append(self._get_import_info(node))
            
            return analysis
            
        except Exception as e:
            logger.error(f"Code analysis failed: {e}")
            return {}
    
    def _get_decorator_name(self, decorator) -> str:
        """Extract decorator name"""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return f"{self._get_name(decorator.value)}.{decorator.attr}"
        elif isinstance(decorator, ast.Call):
            return self._get_decorator_name(decorator.func)
        return str(decorator)
    
    def _get_name(self, node) -> str:
        """Get name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return str(node)
    
    def _get_import_info(self, node) -> Dict[str, Any]:
        """Extract import information"""
        if isinstance(node, ast.Import):
            return {
                "type": "import",
                "names": [alias.name for alias in node.names]
            }
        elif isinstance(node, ast.ImportFrom):
            return {
                "type": "from",
                "module": node.module,
                "names": [alias.name for alias in node.names]
            }
        return {}
    
    def _detect_feature_type(self, analysis: Dict[str, Any]) -> str:
        """Detect the type of feature from code analysis"""
        # Check for MCP tool decorator
        for func in analysis.get("functions", []) + analysis.get("async_functions", []):
            if "mcp.tool" in func.get("decorators", []):
                return "mcp_tool"
        
        # Check for agent pattern
        for cls in analysis.get("classes", []):
            if any("Agent" in base for base in cls.get("bases", [])):
                return "agent"
        
        return "function"
    
    def _build_test_generation_prompt(self, code: str, analysis: Dict, feature_type: str) -> str:
        """Build prompt for test generation"""
        return f"""
Generate comprehensive pytest tests for the following {feature_type} code:

```python
{code}
```

Code Analysis:
- Functions: {len(analysis.get('functions', []))}
- Async Functions: {len(analysis.get('async_functions', []))}
- Classes: {len(analysis.get('classes', []))}

Requirements:
1. Test all public functions and methods
2. Include tests for edge cases and error conditions
3. Mock external dependencies (database, API calls, etc.)
4. Use pytest fixtures for setup
5. Include integration tests where appropriate
6. Ensure tests are deterministic and isolated
7. Add docstrings to all test functions
8. Use meaningful assertions

Generate complete, runnable pytest code:
"""
    
    def _post_process_tests(self, generated_tests: str, analysis: Dict) -> str:
        """Post-process and validate generated tests"""
        # Extract code from markdown if present
        if "```python" in generated_tests:
            import re
            code_blocks = re.findall(r'```python\n(.*?)```', generated_tests, re.DOTALL)
            generated_tests = '\n\n'.join(code_blocks)
        
        # Ensure proper imports
        if "import pytest" not in generated_tests:
            generated_tests = "import pytest\n" + generated_tests
        
        # Validate syntax
        try:
            ast.parse(generated_tests)
        except SyntaxError as e:
            logger.error(f"Generated tests have syntax errors: {e}")
            # Try to fix common issues
            generated_tests = self._fix_common_syntax_issues(generated_tests)
        
        return generated_tests
    
    def _fix_common_syntax_issues(self, code: str) -> str:
        """Fix common syntax issues in generated code"""
        # Fix indentation issues
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Fix tabs to spaces
            line = line.replace('\t', '    ')
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _generate_from_template(self, code: str, feature_type: str) -> str:
        """Fallback template-based test generation"""
        analysis = self._analyze_code(code)
        
        if feature_type == "mcp_tool" and "mcp_tool" in self.test_templates:
            # Extract tool info
            tool_func = next((f for f in analysis.get("async_functions", []) 
                             if "mcp.tool" in f.get("decorators", [])), None)
            
            if tool_func:
                return self.test_templates["mcp_tool"].format(
                    module_path="src.crawl4ai_mcp",
                    tool_name=tool_func["name"],
                    tool_name_camel=self._to_camel_case(tool_func["name"]),
                    test_success_implementation="# TODO: Implement success test",
                    test_invalid_input_implementation="# TODO: Implement invalid input test",
                    test_error_handling_implementation="# TODO: Implement error handling test"
                )
        
        # Default to basic function template
        return "# TODO: Generate tests for this feature"
    
    def _to_camel_case(self, snake_str: str) -> str:
        """Convert snake_case to CamelCase"""
        components = snake_str.split('_')
        return ''.join(x.title() for x in components)
    
    async def run_tests(self, test_file: str, coverage: bool = True) -> Dict[str, Any]:
        """Execute tests and return results"""
        try:
            # Write test file
            with tempfile.NamedTemporaryFile(mode='w', suffix='_test.py', delete=False) as f:
                f.write(test_file)
                temp_test_file = f.name
            
            # Prepare pytest command
            cmd = ["pytest", temp_test_file, "-v", "--tb=short"]
            
            if coverage:
                cmd.extend(["--cov=src", "--cov-report=json"])
            
            # Run tests
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse results
            test_results = {
                "success": result.returncode == 0,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "tests_run": self._parse_test_count(result.stdout),
                "tests_passed": 0,
                "tests_failed": 0,
                "coverage": None
            }
            
            # Parse test results
            if "passed" in result.stdout:
                import re
                passed_match = re.search(r'(\d+) passed', result.stdout)
                if passed_match:
                    test_results["tests_passed"] = int(passed_match.group(1))
            
            if "failed" in result.stdout:
                import re
                failed_match = re.search(r'(\d+) failed', result.stdout)
                if failed_match:
                    test_results["tests_failed"] = int(failed_match.group(1))
            
            # Parse coverage if available
            if coverage and os.path.exists("coverage.json"):
                test_results["coverage"] = self._parse_coverage_report()
            
            # Clean up
            os.unlink(temp_test_file)
            
            return test_results
            
        except Exception as e:
            logger.error(f"Failed to run tests: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _parse_test_count(self, output: str) -> int:
        """Parse number of tests from pytest output"""
        import re
        match = re.search(r'collected (\d+) items?', output)
        return int(match.group(1)) if match else 0
    
    def _parse_coverage_report(self) -> Dict[str, Any]:
        """Parse coverage JSON report"""
        try:
            with open("coverage.json", 'r') as f:
                coverage_data = json.load(f)
            
            # Extract summary
            total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
            
            # Clean up
            os.unlink("coverage.json")
            
            return {
                "total_percentage": total_coverage,
                "meets_threshold": total_coverage >= self.coverage_threshold,
                "threshold": self.coverage_threshold
            }
        except Exception as e:
            logger.error(f"Failed to parse coverage report: {e}")
            return None
    
    async def run_regression_suite(self) -> Dict[str, Any]:
        """Execute full regression test suite"""
        try:
            # Find all test files
            test_files = list(Path("tests").glob("**/test_*.py"))
            
            if not test_files:
                return {
                    "success": False,
                    "error": "No test files found"
                }
            
            # Run pytest on all tests
            result = subprocess.run(
                ["pytest", "tests/", "--tb=short", "--cov=src", "--cov-report=json"],
                capture_output=True,
                text=True
            )
            
            # Parse results
            regression_results = {
                "success": result.returncode == 0,
                "exit_code": result.returncode,
                "summary": self._extract_test_summary(result.stdout),
                "coverage": self._parse_coverage_report() if os.path.exists("coverage.json") else None,
                "duration": self._extract_duration(result.stdout),
                "timestamp": datetime.now().isoformat()
            }
            
            return regression_results
            
        except Exception as e:
            logger.error(f"Failed to run regression suite: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _extract_test_summary(self, output: str) -> Dict[str, int]:
        """Extract test summary from pytest output"""
        import re
        
        summary = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0
        }
        
        for key in summary:
            match = re.search(rf'(\d+) {key}', output)
            if match:
                summary[key] = int(match.group(1))
        
        return summary
    
    def _extract_duration(self, output: str) -> float:
        """Extract test duration from pytest output"""
        import re
        match = re.search(r'in ([\d.]+)s', output)
        return float(match.group(1)) if match else 0.0