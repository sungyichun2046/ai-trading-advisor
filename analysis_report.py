#!/usr/bin/env python3
"""
Codebase Analysis Report Generator
Analyzes current project structure to plan consolidation approach.
"""

import os
import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime
from collections import defaultdict, Counter


class CodebaseAnalyzer:
    """Analyzes codebase structure and dependencies."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.target_dirs = ["src/core", "src/data", "src/airflow_dags"]
        self.results = {
            "analysis_timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "scanned_directories": self.target_dirs,
            "file_analysis": {},
            "dependency_graph": {},
            "duplicate_methods": {},
            "complexity_metrics": {},
            "consolidation_recommendations": {}
        }
    
    def scan_files(self) -> List[Path]:
        """Scan all Python files in target directories."""
        python_files = []
        
        for target_dir in self.target_dirs:
            dir_path = self.project_root / target_dir
            if dir_path.exists():
                for py_file in dir_path.rglob("*.py"):
                    if py_file.name != "__init__.py":
                        python_files.append(py_file)
                        
        print(f"Found {len(python_files)} Python files to analyze")
        return python_files
    
    def parse_file(self, file_path: Path) -> Dict:
        """Parse a Python file and extract structural information."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            analysis = {
                "file_path": str(file_path.relative_to(self.project_root)),
                "lines": len(content.splitlines()),
                "imports": self.extract_imports(tree),
                "classes": self.extract_classes(tree),
                "functions": self.extract_functions(tree),
                "methods": self.extract_methods(tree),
                "duplicate_candidates": self.find_duplicate_candidates(tree)
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return {
                "file_path": str(file_path.relative_to(self.project_root)),
                "error": str(e),
                "lines": 0,
                "imports": [],
                "classes": [],
                "functions": [],
                "methods": [],
                "duplicate_candidates": []
            }
    
    def extract_imports(self, tree: ast.AST) -> List[Dict]:
        """Extract import statements."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        "type": "import",
                        "module": alias.name,
                        "alias": alias.asname
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ""
                for alias in node.names:
                    imports.append({
                        "type": "from_import",
                        "module": module,
                        "name": alias.name,
                        "alias": alias.asname
                    })
        
        return imports
    
    def extract_classes(self, tree: ast.AST) -> List[Dict]:
        """Extract class definitions."""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append({
                            "name": item.name,
                            "line": item.lineno,
                            "args": [arg.arg for arg in item.args.args]
                        })
                
                classes.append({
                    "name": node.name,
                    "line": node.lineno,
                    "methods": methods,
                    "method_count": len(methods),
                    "bases": [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases]
                })
        
        return classes
    
    def extract_functions(self, tree: ast.AST) -> List[Dict]:
        """Extract function definitions (not methods)."""
        functions = []
        
        # Get top-level functions only
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    "name": node.name,
                    "line": node.lineno,
                    "args": [arg.arg for arg in node.args.args]
                })
        
        return functions
    
    def extract_methods(self, tree: ast.AST) -> List[Dict]:
        """Extract all method names for duplicate detection."""
        methods = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if it's inside a class
                parent = getattr(node, 'parent_class', None)
                methods.append({
                    "name": node.name,
                    "line": node.lineno,
                    "in_class": bool(parent)
                })
        
        return methods
    
    def find_duplicate_candidates(self, tree: ast.AST) -> List[str]:
        """Find methods that match duplicate patterns."""
        candidates = []
        patterns = [r'^calculate_.*', r'^collect_.*', r'^analyze_.*', r'^get_.*', r'^process_.*']
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for pattern in patterns:
                    if re.match(pattern, node.name):
                        candidates.append(node.name)
                        break
        
        return candidates
    
    def build_dependency_graph(self, file_analyses: List[Dict]) -> Dict:
        """Build dependency graph between modules."""
        dependencies = defaultdict(set)
        
        for analysis in file_analyses:
            if "error" in analysis:
                continue
                
            file_path = analysis["file_path"]
            module_name = file_path.replace("/", ".").replace(".py", "")
            
            for imp in analysis["imports"]:
                if imp["type"] == "from_import" and imp["module"]:
                    # Check if it's an internal import
                    if any(target in imp["module"] for target in ["src.core", "src.data", "src.airflow_dags"]):
                        dependencies[module_name].add(imp["module"])
                elif imp["type"] == "import":
                    if any(target in imp["module"] for target in ["src.core", "src.data", "src.airflow_dags"]):
                        dependencies[module_name].add(imp["module"])
        
        # Convert sets to lists for JSON serialization
        return {k: list(v) for k, v in dependencies.items()}
    
    def find_duplicate_methods(self, file_analyses: List[Dict]) -> Dict:
        """Find duplicate method names across files."""
        method_locations = defaultdict(list)
        
        for analysis in file_analyses:
            if "error" in analysis:
                continue
                
            file_path = analysis["file_path"]
            
            # Check functions and methods
            all_methods = []
            
            # Extract function names
            for func in analysis["functions"]:
                if isinstance(func, dict) and "name" in func:
                    all_methods.append(func["name"])
                elif isinstance(func, str):
                    all_methods.append(func)
            
            # Extract duplicate candidate names
            all_methods.extend(analysis["duplicate_candidates"])
            
            # Remove duplicates within same file and add to locations
            for method_name in set(all_methods):
                method_locations[method_name].append(file_path)
        
        # Find actual duplicates
        duplicates = {
            method: locations for method, locations in method_locations.items()
            if len(locations) > 1
        }
        
        # Group by pattern
        patterns = {
            "calculate_methods": {k: v for k, v in duplicates.items() if k.startswith("calculate_")},
            "collect_methods": {k: v for k, v in duplicates.items() if k.startswith("collect_")},
            "analyze_methods": {k: v for k, v in duplicates.items() if k.startswith("analyze_")},
            "get_methods": {k: v for k, v in duplicates.items() if k.startswith("get_")},
            "other_duplicates": {k: v for k, v in duplicates.items() 
                               if not any(k.startswith(prefix) for prefix in ["calculate_", "collect_", "analyze_", "get_"])}
        }
        
        return patterns
    
    def calculate_complexity_metrics(self, file_analyses: List[Dict]) -> Dict:
        """Calculate complexity metrics for files and modules."""
        metrics = {
            "by_directory": defaultdict(lambda: {"files": 0, "total_lines": 0, "total_classes": 0, "total_methods": 0}),
            "by_file": {},
            "summary": {"total_files": 0, "total_lines": 0, "total_classes": 0, "total_methods": 0}
        }
        
        for analysis in file_analyses:
            if "error" in analysis:
                continue
                
            file_path = analysis["file_path"]
            directory = "/".join(file_path.split("/")[:-1])
            
            lines = analysis["lines"]
            classes = len(analysis["classes"])
            methods = sum(cls["method_count"] for cls in analysis["classes"]) + len(analysis["functions"])
            
            # By directory
            metrics["by_directory"][directory]["files"] += 1
            metrics["by_directory"][directory]["total_lines"] += lines
            metrics["by_directory"][directory]["total_classes"] += classes
            metrics["by_directory"][directory]["total_methods"] += methods
            
            # By file
            metrics["by_file"][file_path] = {
                "lines": lines,
                "classes": classes,
                "methods": methods,
                "complexity_score": lines + (classes * 10) + (methods * 5)  # Weighted complexity
            }
            
            # Summary
            metrics["summary"]["total_files"] += 1
            metrics["summary"]["total_lines"] += lines
            metrics["summary"]["total_classes"] += classes
            metrics["summary"]["total_methods"] += methods
        
        # Convert defaultdict to regular dict
        metrics["by_directory"] = dict(metrics["by_directory"])
        
        return metrics
    
    def generate_consolidation_recommendations(self, dependency_graph: Dict, duplicates: Dict, complexity: Dict) -> Dict:
        """Generate consolidation recommendations."""
        recommendations = {
            "data_module_consolidation": {
                "target_file": "src/core/data_manager.py",
                "source_files": [],
                "rationale": "Consolidate all data collection and storage functionality",
                "estimated_lines": 500
            },
            "analysis_module_consolidation": {
                "target_file": "src/core/analysis_engine.py", 
                "source_files": [],
                "rationale": "Consolidate all analysis capabilities",
                "estimated_lines": 600
            },
            "trading_module_consolidation": {
                "target_file": "src/core/trading_engine.py",
                "source_files": [],
                "rationale": "Consolidate all trading and risk logic",
                "estimated_lines": 400
            },
            "dag_consolidation": {
                "target_files": [
                    "src/dags/data_collection_dag.py",
                    "src/dags/analysis_dag.py", 
                    "src/dags/trading_dag.py"
                ],
                "source_files": [],
                "rationale": "Simplify from 12+ DAGs to 3 focused DAGs",
                "estimated_lines": 1050
            },
            "duplicate_method_resolution": {},
            "circular_dependencies": [],
            "high_complexity_files": []
        }
        
        # Identify source files for consolidation
        for file_path in complexity["by_file"]:
            if "src/data/" in file_path:
                recommendations["data_module_consolidation"]["source_files"].append(file_path)
            elif "src/core/" in file_path and any(keyword in file_path.lower() for keyword in 
                                                 ["analysis", "technical", "pattern", "trend", "sector", "correlation"]):
                recommendations["analysis_module_consolidation"]["source_files"].append(file_path)
            elif "src/core/" in file_path and any(keyword in file_path.lower() for keyword in 
                                                 ["risk", "recommendation", "strategy", "user", "trading"]):
                recommendations["trading_module_consolidation"]["source_files"].append(file_path)
            elif "src/airflow_dags/" in file_path:
                recommendations["dag_consolidation"]["source_files"].append(file_path)
        
        # Analyze duplicate methods
        total_duplicates = sum(len(methods) for methods in duplicates.values())
        recommendations["duplicate_method_resolution"] = {
            "total_duplicate_methods": total_duplicates,
            "calculate_methods": len(duplicates.get("calculate_methods", {})),
            "collect_methods": len(duplicates.get("collect_methods", {})),
            "analyze_methods": len(duplicates.get("analyze_methods", {})),
            "resolution_strategy": "Keep best implementation, remove duplicates during consolidation"
        }
        
        # Find circular dependencies
        recommendations["circular_dependencies"] = self.find_circular_dependencies(dependency_graph)
        
        # Identify high complexity files
        recommendations["high_complexity_files"] = [
            {"file": file, "score": metrics["complexity_score"]}
            for file, metrics in complexity["by_file"].items()
            if metrics["complexity_score"] > 1000
        ]
        
        # Calculate reduction statistics
        total_current_files = len(complexity["by_file"])
        total_new_files = 6  # 3 core + 3 DAGs
        recommendations["consolidation_benefits"] = {
            "file_reduction": {
                "before": total_current_files,
                "after": total_new_files,
                "reduction_percentage": round((1 - total_new_files / total_current_files) * 100, 1)
            },
            "line_reduction": {
                "before": complexity["summary"]["total_lines"],
                "after": 2550,  # Estimated consolidated lines
                "reduction_percentage": round((1 - 2550 / complexity["summary"]["total_lines"]) * 100, 1)
            }
        }
        
        return recommendations
    
    def find_circular_dependencies(self, dependency_graph: Dict) -> List[List[str]]:
        """Find circular dependencies in the dependency graph."""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return
            
            if node in visited:
                return
                
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in dependency_graph.get(node, []):
                dfs(neighbor, path + [node])
            
            rec_stack.remove(node)
        
        for node in dependency_graph:
            if node not in visited:
                dfs(node, [])
        
        return cycles
    
    def run_analysis(self) -> Dict:
        """Run complete codebase analysis."""
        print("🔍 Starting codebase analysis...")
        
        # Scan files
        python_files = self.scan_files()
        
        # Analyze each file
        print("📊 Analyzing file structure...")
        file_analyses = []
        for file_path in python_files:
            analysis = self.parse_file(file_path)
            file_analyses.append(analysis)
            self.results["file_analysis"][str(file_path.relative_to(self.project_root))] = analysis
        
        # Build dependency graph
        print("🔗 Building dependency graph...")
        self.results["dependency_graph"] = self.build_dependency_graph(file_analyses)
        
        # Find duplicates
        print("🔄 Finding duplicate methods...")
        self.results["duplicate_methods"] = self.find_duplicate_methods(file_analyses)
        
        # Calculate complexity
        print("📏 Calculating complexity metrics...")
        self.results["complexity_metrics"] = self.calculate_complexity_metrics(file_analyses)
        
        # Generate recommendations
        print("💡 Generating consolidation recommendations...")
        self.results["consolidation_recommendations"] = self.generate_consolidation_recommendations(
            self.results["dependency_graph"],
            self.results["duplicate_methods"], 
            self.results["complexity_metrics"]
        )
        
        print("✅ Analysis complete!")
        return self.results
    
    def save_report(self, filename: str = "codebase_analysis_report.json"):
        """Save analysis results to JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"📄 Report saved to {filename}")


def main():
    """Main execution function."""
    print("🚀 AI Trading Advisor - Codebase Analysis Report")
    print("=" * 50)
    
    analyzer = CodebaseAnalyzer()
    results = analyzer.run_analysis()
    analyzer.save_report()
    
    # Print summary
    print("\n📋 ANALYSIS SUMMARY")
    print("=" * 30)
    
    complexity = results["complexity_metrics"]
    recommendations = results["consolidation_recommendations"]
    
    print(f"📁 Files analyzed: {complexity['summary']['total_files']}")
    print(f"📝 Total lines: {complexity['summary']['total_lines']:,}")
    print(f"🏗️  Total classes: {complexity['summary']['total_classes']}")
    print(f"⚙️  Total methods: {complexity['summary']['total_methods']}")
    
    print(f"\n🔄 Duplicate methods found:")
    for pattern, methods in results["duplicate_methods"].items():
        if methods:
            print(f"   {pattern}: {len(methods)} duplicates")
    
    print(f"\n📉 Consolidation benefits:")
    benefits = recommendations["consolidation_benefits"]
    print(f"   File reduction: {benefits['file_reduction']['before']} → {benefits['file_reduction']['after']} "
          f"({benefits['file_reduction']['reduction_percentage']}% reduction)")
    print(f"   Line reduction: {benefits['line_reduction']['before']:,} → {benefits['line_reduction']['after']:,} "
          f"({benefits['line_reduction']['reduction_percentage']}% reduction)")
    
    print(f"\n💡 Consolidation strategy:")
    print(f"   📦 Data module: {len(recommendations['data_module_consolidation']['source_files'])} files → data_manager.py")
    print(f"   🔬 Analysis module: {len(recommendations['analysis_module_consolidation']['source_files'])} files → analysis_engine.py")
    print(f"   💰 Trading module: {len(recommendations['trading_module_consolidation']['source_files'])} files → trading_engine.py")
    print(f"   📅 DAG consolidation: {len(recommendations['dag_consolidation']['source_files'])} files → 3 DAGs")
    
    print(f"\n🎯 Next steps:")
    print(f"   1. Review codebase_analysis_report.json for detailed analysis")
    print(f"   2. Begin consolidation with data module (lowest complexity)")
    print(f"   3. Proceed with analysis module consolidation")
    print(f"   4. Complete with trading module and DAG simplification")


if __name__ == "__main__":
    main()