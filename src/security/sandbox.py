"""
Security sandbox for safe code execution.
"""
import re
import ast
import subprocess
import tempfile
import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)

class SecuritySandbox:
    """Secure execution environment for generated code"""
    
    def __init__(self):
        self.security_rules = {
            "max_memory": "512m",
            "max_cpu": "0.5",
            "timeout": 30,  # seconds
            "network_access": False,
            "filesystem_access": "readonly"
        }
        
        # Dangerous patterns to check for
        self.dangerous_patterns = [
            (r'exec\s*\(', "Use of exec() is dangerous"),
            (r'eval\s*\(', "Use of eval() is dangerous"),
            (r'__import__\s*\(', "Dynamic imports can be dangerous"),
            (r'compile\s*\(', "Use of compile() can be dangerous"),
            (r'globals\s*\(', "Access to globals() is restricted"),
            (r'locals\s*\(', "Access to locals() is restricted"),
            (r'vars\s*\(', "Access to vars() is restricted"),
            (r'open\s*\(.*[\'"]w', "Write file access is restricted"),
            (r'os\.system\s*\(', "Use of os.system() is dangerous"),
            (r'subprocess\.(call|run|Popen)', "Direct subprocess usage is restricted"),
            (r'socket\s*\.', "Network access is restricted"),
            (r'urllib\s*\.', "Network access is restricted"),
            (r'requests\s*\.', "Network access is restricted"),
            (r'pickle\.loads?\s*\(', "Pickle can execute arbitrary code"),
            (r'marshal\.loads?\s*\(', "Marshal can be dangerous"),
            (r'__file__', "Access to __file__ is restricted"),
            (r'__builtins__', "Access to __builtins__ is restricted")
        ]
        
        # Allowed safe builtins
        self.safe_builtins = {
            'abs', 'all', 'any', 'bool', 'bytes', 'chr', 'dict',
            'enumerate', 'filter', 'float', 'format', 'frozenset',
            'int', 'len', 'list', 'map', 'max', 'min', 'ord',
            'pow', 'range', 'reversed', 'round', 'set', 'sorted',
            'str', 'sum', 'tuple', 'type', 'zip'
        }
    
    async def validate_code(self, code: str) -> Dict[str, Any]:
        """Security validation for generated code"""
        violations = []
        warnings = []
        
        # Check for dangerous patterns
        for pattern, message in self.dangerous_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                violations.append({
                    "line": line_num,
                    "pattern": pattern,
                    "message": message,
                    "match": match.group()
                })
        
        # AST-based validation
        try:
            tree = ast.parse(code)
            validator = CodeValidator(self.safe_builtins)
            validator.visit(tree)
            violations.extend(validator.violations)
            warnings.extend(validator.warnings)
        except SyntaxError as e:
            violations.append({
                "line": e.lineno,
                "message": f"Syntax error: {e.msg}",
                "type": "syntax_error"
            })
        except Exception as e:
            violations.append({
                "message": f"Failed to parse code: {str(e)}",
                "type": "parse_error"
            })
        
        # Check imports
        import_issues = self._check_imports(code)
        violations.extend(import_issues)
        
        return {
            "safe": len(violations) == 0,
            "violations": violations,
            "warnings": warnings,
            "risk_level": self._assess_risk_level(violations, warnings)
        }
    
    def _check_imports(self, code: str) -> List[Dict[str, Any]]:
        """Check for dangerous imports"""
        violations = []
        
        # Dangerous modules
        dangerous_modules = {
            'os', 'sys', 'subprocess', 'socket', 'urllib',
            'requests', 'pickle', 'marshal', 'importlib',
            'builtins', '__builtin__', 'imp', 'zipimport'
        }
        
        # Parse imports
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split('.')[0] in dangerous_modules:
                            violations.append({
                                "line": node.lineno,
                                "message": f"Import of '{alias.name}' is restricted",
                                "type": "dangerous_import"
                            })
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.split('.')[0] in dangerous_modules:
                        violations.append({
                            "line": node.lineno,
                            "message": f"Import from '{node.module}' is restricted",
                            "type": "dangerous_import"
                        })
        except:
            pass
        
        return violations
    
    def _assess_risk_level(self, violations: List[Dict], warnings: List[Dict]) -> str:
        """Assess overall risk level of code"""
        if not violations and not warnings:
            return "LOW"
        elif not violations and len(warnings) <= 2:
            return "MEDIUM"
        elif len(violations) <= 2 and all(v.get("type") != "dangerous_import" for v in violations):
            return "MEDIUM"
        else:
            return "HIGH"
    
    async def execute_sandboxed(self, code: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Execute code in isolated environment"""
        timeout = timeout or self.security_rules["timeout"]
        
        # First validate the code
        validation = await self.validate_code(code)
        if not validation["safe"]:
            return {
                "success": False,
                "error": "Code failed security validation",
                "violations": validation["violations"]
            }
        
        # Execute in subprocess with restrictions
        try:
            # Create temporary file for the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Wrap code in safety harness
                wrapped_code = self._wrap_code_safely(code)
                f.write(wrapped_code)
                temp_file = f.name
            
            # Execute with resource limits
            result = await self._execute_with_limits(temp_file, timeout)
            
            # Clean up
            os.unlink(temp_file)
            
            return result
            
        except Exception as e:
            logger.error(f"Sandboxed execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "exit_code": -1
            }
    
    def _wrap_code_safely(self, code: str) -> str:
        """Wrap code with safety restrictions"""
        wrapped = f"""
import sys
import resource

# Set resource limits
resource.setrlimit(resource.RLIMIT_CPU, (30, 30))  # CPU time limit
resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))  # Memory limit

# Restricted builtins
__builtins__ = {{
    k: v for k, v in __builtins__.items() 
    if k in {self.safe_builtins}
}}

# Add print for output
__builtins__['print'] = print

# User code
try:
{self._indent_code(code)}
except Exception as e:
    print(f"ERROR: {{type(e).__name__}}: {{str(e)}}", file=sys.stderr)
    sys.exit(1)
"""
        return wrapped
    
    def _indent_code(self, code: str) -> str:
        """Indent code for wrapping"""
        lines = code.split('\n')
        return '\n'.join(f"    {line}" for line in lines)
    
    async def _execute_with_limits(self, file_path: str, timeout: int) -> Dict[str, Any]:
        """Execute file with resource limits"""
        start_time = time.time()
        
        try:
            # Run in subprocess
            process = await asyncio.create_subprocess_exec(
                'python', file_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1024 * 1024  # 1MB output limit
            )
            
            # Wait with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    "success": False,
                    "error": f"Execution timed out after {timeout} seconds",
                    "exit_code": -1,
                    "execution_time": time.time() - start_time
                }
            
            execution_time = time.time() - start_time
            
            return {
                "success": process.returncode == 0,
                "exit_code": process.returncode,
                "stdout": stdout.decode('utf-8', errors='replace'),
                "stderr": stderr.decode('utf-8', errors='replace'),
                "execution_time": execution_time
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "exit_code": -1,
                "execution_time": time.time() - start_time
            }
    
    async def analyze_code_behavior(self, code: str) -> Dict[str, Any]:
        """Analyze code behavior without executing"""
        try:
            tree = ast.parse(code)
            analyzer = CodeBehaviorAnalyzer()
            analyzer.visit(tree)
            
            return {
                "functions": analyzer.functions,
                "classes": analyzer.classes,
                "imports": analyzer.imports,
                "global_vars": analyzer.global_vars,
                "has_side_effects": analyzer.has_side_effects,
                "complexity": analyzer.calculate_complexity()
            }
        except Exception as e:
            return {
                "error": str(e),
                "analyzable": False
            }


class CodeValidator(ast.NodeVisitor):
    """AST visitor for code validation"""
    
    def __init__(self, safe_builtins: set):
        self.safe_builtins = safe_builtins
        self.violations = []
        self.warnings = []
        self.scope_stack = []
    
    def visit_Call(self, node):
        """Check function calls"""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            # Check if calling unsafe builtin
            if func_name not in self.safe_builtins and func_name in {'eval', 'exec', 'compile', '__import__'}:
                self.violations.append({
                    "line": node.lineno,
                    "message": f"Call to unsafe function '{func_name}'",
                    "type": "unsafe_call"
                })
        
        self.generic_visit(node)
    
    def visit_Attribute(self, node):
        """Check attribute access"""
        # Check for dangerous attribute access patterns
        if isinstance(node.value, ast.Name):
            if node.value.id in {'os', 'sys', 'subprocess'}:
                self.violations.append({
                    "line": node.lineno,
                    "message": f"Access to '{node.value.id}.{node.attr}' is restricted",
                    "type": "restricted_access"
                })
        
        self.generic_visit(node)


class CodeBehaviorAnalyzer(ast.NodeVisitor):
    """Analyze code behavior and complexity"""
    
    def __init__(self):
        self.functions = []
        self.classes = []
        self.imports = []
        self.global_vars = []
        self.has_side_effects = False
        self.complexity_score = 0
    
    def visit_FunctionDef(self, node):
        self.functions.append({
            "name": node.name,
            "args": [arg.arg for arg in node.args.args],
            "lineno": node.lineno,
            "docstring": ast.get_docstring(node)
        })
        self.complexity_score += 1
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        self.classes.append({
            "name": node.name,
            "lineno": node.lineno,
            "docstring": ast.get_docstring(node),
            "bases": [self._get_name(base) for base in node.bases]
        })
        self.complexity_score += 2
        self.generic_visit(node)
    
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append({
                "module": alias.name,
                "alias": alias.asname,
                "lineno": node.lineno
            })
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.imports.append({
                "module": f"{node.module}.{alias.name}" if node.module else alias.name,
                "alias": alias.asname,
                "lineno": node.lineno,
                "from_import": True
            })
        self.generic_visit(node)
    
    def visit_Assign(self, node):
        # Check for global variable assignments
        for target in node.targets:
            if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store):
                self.global_vars.append(target.id)
        self.generic_visit(node)
    
    def visit_Call(self, node):
        # Check for side effects (file I/O, network, etc.)
        if isinstance(node.func, ast.Name):
            if node.func.id in {'print', 'open', 'write'}:
                self.has_side_effects = True
        self.generic_visit(node)
    
    def _get_name(self, node):
        """Get name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return str(node)
    
    def calculate_complexity(self) -> int:
        """Calculate cyclomatic complexity score"""
        return self.complexity_score