"""
Code debugger agent for analyzing and fixing generated code.
"""
from typing import Dict, List, Any, Optional
import ast
import subprocess
import tempfile
import os
import re
from pathlib import Path
from .base_agent import BaseAgent

class CodeDebuggerAgent(BaseAgent):
    """Analyzes newly generated code for syntax and logical errors"""
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Debug generated code"""
        self.logger.info("Starting code debugging")
        
        code = context.get("generated_code", "")
        code_type = context.get("code_type", "python")
        
        if not code:
            return {
                "agent": self.name,
                "status": "skipped",
                "reason": "No code provided for debugging"
            }
        
        # Perform various checks
        syntax_check = await self._check_syntax(code, code_type)
        import_check = await self._check_imports(code)
        security_check = await self._check_security(code)
        style_check = await self._check_code_style(code)
        
        # Suggest fixes if needed
        fixes = []
        if not syntax_check["valid"]:
            fixes.extend(await self._suggest_syntax_fixes(code, syntax_check["errors"]))
        
        if not import_check["valid"]:
            fixes.extend(await self._suggest_import_fixes(code, import_check["missing"]))
        
        # Apply automatic fixes if possible
        fixed_code = code
        if fixes and context.get("auto_fix", False):
            fixed_code = await self._apply_fixes(code, fixes)
        
        return {
            "agent": self.name,
            "status": "completed",
            "syntax_valid": syntax_check["valid"],
            "syntax_errors": syntax_check.get("errors", []),
            "imports_valid": import_check["valid"],
            "missing_imports": import_check.get("missing", []),
            "security_issues": security_check.get("issues", []),
            "style_issues": style_check.get("issues", []),
            "suggested_fixes": fixes,
            "fixed_code": fixed_code if fixed_code != code else None
        }
    
    async def _check_syntax(self, code: str, code_type: str) -> Dict[str, Any]:
        """Check code syntax"""
        if code_type != "python":
            return {"valid": True, "message": f"Syntax check not implemented for {code_type}"}
        
        try:
            ast.parse(code)
            return {"valid": True}
        except SyntaxError as e:
            return {
                "valid": False,
                "errors": [{
                    "line": e.lineno,
                    "offset": e.offset,
                    "message": e.msg,
                    "text": e.text
                }]
            }
        except Exception as e:
            return {
                "valid": False,
                "errors": [{"message": str(e)}]
            }
    
    async def _check_imports(self, code: str) -> Dict[str, Any]:
        """Check if all imports are available"""
        import_pattern = r'^(?:from\s+(\S+)\s+)?import\s+(.+)$'
        imports = re.findall(import_pattern, code, re.MULTILINE)
        
        missing_imports = []
        for from_module, import_names in imports:
            module_to_check = from_module if from_module else import_names.split(',')[0].strip()
            
            # Check if module is available
            try:
                __import__(module_to_check)
            except ImportError:
                missing_imports.append(module_to_check)
        
        return {
            "valid": len(missing_imports) == 0,
            "missing": missing_imports
        }
    
    async def _check_security(self, code: str) -> Dict[str, Any]:
        """Check for security issues in code"""
        security_patterns = [
            (r'eval\s*\(', "Use of eval() is dangerous"),
            (r'exec\s*\(', "Use of exec() is dangerous"),
            (r'__import__\s*\(', "Dynamic imports can be dangerous"),
            (r'subprocess\.call\s*\(.*shell\s*=\s*True', "Shell injection vulnerability"),
            (r'os\.system\s*\(', "Use subprocess instead of os.system"),
            (r'pickle\.loads?\s*\(', "Pickle can execute arbitrary code"),
            (r'SECRET|PASSWORD|KEY.*=.*["\'].*["\']', "Hardcoded secrets detected")
        ]
        
        issues = []
        for pattern, message in security_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                issues.append({
                    "line": line_num,
                    "type": "security",
                    "message": message,
                    "match": match.group()
                })
        
        return {"issues": issues}
    
    async def _check_code_style(self, code: str) -> Dict[str, Any]:
        """Check code style using flake8 or similar"""
        issues = []
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Run flake8 if available
            result = subprocess.run(
                ["flake8", "--max-line-length=120", temp_file],
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(':', 3)
                        if len(parts) >= 4:
                            issues.append({
                                "line": int(parts[1]),
                                "column": int(parts[2]),
                                "message": parts[3].strip()
                            })
        except FileNotFoundError:
            # flake8 not installed
            self.logger.warning("flake8 not installed, skipping style check")
        finally:
            os.unlink(temp_file)
        
        return {"issues": issues}
    
    async def _suggest_syntax_fixes(self, code: str, errors: List[Dict]) -> List[Dict]:
        """Suggest fixes for syntax errors"""
        fixes = []
        
        for error in errors:
            if "expected" in error.get("message", "").lower():
                fixes.append({
                    "type": "syntax",
                    "line": error.get("line"),
                    "suggestion": "Check for missing parentheses, brackets, or quotes",
                    "error": error["message"]
                })
            elif "indent" in error.get("message", "").lower():
                fixes.append({
                    "type": "syntax",
                    "line": error.get("line"),
                    "suggestion": "Fix indentation - ensure consistent use of spaces or tabs",
                    "error": error["message"]
                })
        
        return fixes
    
    async def _suggest_import_fixes(self, code: str, missing: List[str]) -> List[Dict]:
        """Suggest fixes for missing imports"""
        fixes = []
        
        for module in missing:
            # Suggest installation command
            install_cmd = f"pip install {module}"
            
            # Check if it might be a typo of a known module
            similar_modules = self._find_similar_modules(module)
            
            fixes.append({
                "type": "import",
                "module": module,
                "suggestion": f"Install with: {install_cmd}",
                "alternatives": similar_modules
            })
        
        return fixes
    
    def _find_similar_modules(self, module: str) -> List[str]:
        """Find similar module names (simple implementation)"""
        common_modules = [
            "numpy", "pandas", "requests", "flask", "django",
            "matplotlib", "scipy", "sklearn", "tensorflow",
            "torch", "fastapi", "sqlalchemy", "pytest"
        ]
        
        similar = []
        for common in common_modules:
            if module.lower() in common or common in module.lower():
                similar.append(common)
        
        return similar
    
    async def _apply_fixes(self, code: str, fixes: List[Dict]) -> str:
        """Apply automatic fixes to code"""
        fixed_code = code
        
        # Apply simple fixes
        for fix in fixes:
            if fix["type"] == "import" and fix.get("alternatives"):
                # Replace with most likely alternative
                old_module = fix["module"]
                new_module = fix["alternatives"][0]
                fixed_code = fixed_code.replace(
                    f"import {old_module}",
                    f"import {new_module}"
                )
                fixed_code = fixed_code.replace(
                    f"from {old_module}",
                    f"from {new_module}"
                )
        
        return fixed_code