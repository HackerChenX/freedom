#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive Technical Indicator Audit
Identifies all indicators and their pattern registration status
"""

import os
import re
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Set, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)

class IndicatorAudit:
    def __init__(self, indicators_path: str = "indicators"):
        self.indicators_path = indicators_path
        self.all_indicator_files = []
        self.indicators_with_patterns = {}
        self.indicators_without_patterns = []
        self.failed_imports = []
        
    def scan_indicator_files(self) -> List[str]:
        """Scan all Python files in indicators directory"""
        indicator_files = []
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(self.indicators_path):
            # Skip __pycache__ directories
            dirs[:] = [d for d in dirs if d != '__pycache__']
            
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    file_path = os.path.join(root, file)
                    # Convert to module path
                    module_path = file_path.replace('/', '.').replace('\\', '.').replace('.py', '')
                    indicator_files.append(module_path)
        
        self.all_indicator_files = indicator_files
        return indicator_files
    
    def check_register_patterns_method(self, file_path: str) -> Tuple[bool, List[str]]:
        """Check if a file contains register_patterns method and extract pattern IDs"""
        try:
            with open(file_path.replace('.', '/') + '.py', 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for register_patterns method
            has_register_patterns = 'def register_patterns(' in content
            
            # Extract pattern IDs from register_pattern_to_registry calls
            pattern_ids = []
            pattern_regex = r'pattern_id=["\']([^"\']+)["\']'
            matches = re.findall(pattern_regex, content)
            pattern_ids.extend(matches)
            
            # Also check for registry.register calls
            registry_regex = r'registry\.register\(\s*pattern_id=["\']([^"\']+)["\']'
            registry_matches = re.findall(registry_regex, content)
            pattern_ids.extend(registry_matches)
            
            return has_register_patterns, pattern_ids
            
        except Exception as e:
            logger.debug(f"Error reading file {file_path}: {e}")
            return False, []
    
    def try_import_and_check_class(self, module_path: str) -> Dict:
        """Try to import module and check for indicator classes"""
        try:
            module = importlib.import_module(module_path)
            
            # Find classes that might be indicators
            indicator_classes = []
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    hasattr(obj, 'register_patterns') and
                    obj.__module__ == module_path):
                    indicator_classes.append(name)
            
            return {
                'success': True,
                'classes': indicator_classes,
                'module': module
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'classes': []
            }
    
    def analyze_pattern_registration(self, module_path: str) -> Dict:
        """Analyze pattern registration for a specific module"""
        # Check file content first
        has_method, pattern_ids_from_file = self.check_register_patterns_method(module_path)
        
        # Try to import and instantiate
        import_result = self.try_import_and_check_class(module_path)
        
        result = {
            'module_path': module_path,
            'has_register_patterns_method': has_method,
            'pattern_ids_from_file': pattern_ids_from_file,
            'importable': import_result['success'],
            'indicator_classes': import_result['classes'],
            'import_error': import_result.get('error', None)
        }
        
        # Try to get actual patterns by instantiating
        if import_result['success'] and import_result['classes']:
            for class_name in import_result['classes']:
                try:
                    indicator_class = getattr(import_result['module'], class_name)
                    
                    # Try to instantiate with different parameter combinations
                    instance = None
                    try:
                        instance = indicator_class()
                    except:
                        try:
                            instance = indicator_class(periods=[5, 10, 20])
                        except:
                            try:
                                instance = indicator_class(period=14)
                            except:
                                pass
                    
                    if instance and hasattr(instance, 'register_patterns'):
                        result['instantiable'] = True
                        result['main_class'] = class_name
                        break
                        
                except Exception as e:
                    logger.debug(f"Error instantiating {class_name}: {e}")
        
        return result
    
    def run_comprehensive_audit(self) -> Dict:
        """Run comprehensive audit of all indicators"""
        print("🔍 Starting comprehensive indicator audit...")
        
        # Scan all indicator files
        indicator_files = self.scan_indicator_files()
        print(f"📁 Found {len(indicator_files)} indicator files")
        
        # Analyze each file
        indicators_with_patterns = {}
        indicators_without_patterns = []
        failed_imports = []
        
        for module_path in indicator_files:
            print(f"🔍 Analyzing: {module_path}")
            
            analysis = self.analyze_pattern_registration(module_path)
            
            if analysis['has_register_patterns_method'] or analysis['pattern_ids_from_file']:
                indicators_with_patterns[module_path] = analysis
                print(f"  ✅ Has patterns: {len(analysis['pattern_ids_from_file'])} patterns found")
            else:
                indicators_without_patterns.append(module_path)
                print(f"  ❌ No patterns found")
            
            if not analysis['importable']:
                failed_imports.append({
                    'module': module_path,
                    'error': analysis['import_error']
                })
                print(f"  ⚠️  Import failed: {analysis['import_error']}")
        
        # Store results
        self.indicators_with_patterns = indicators_with_patterns
        self.indicators_without_patterns = indicators_without_patterns
        self.failed_imports = failed_imports
        
        return {
            'total_files': len(indicator_files),
            'with_patterns': len(indicators_with_patterns),
            'without_patterns': len(indicators_without_patterns),
            'failed_imports': len(failed_imports),
            'indicators_with_patterns': indicators_with_patterns,
            'indicators_without_patterns': indicators_without_patterns,
            'failed_imports': failed_imports
        }
    
    def generate_audit_report(self, results: Dict) -> str:
        """Generate detailed audit report"""
        report = []
        report.append("# 📊 Comprehensive Technical Indicator Audit Report")
        report.append("")
        report.append("## 📈 Summary Statistics")
        report.append(f"- **Total indicator files**: {results['total_files']}")
        report.append(f"- **Files with pattern registration**: {results['with_patterns']}")
        report.append(f"- **Files without patterns**: {results['without_patterns']}")
        report.append(f"- **Failed imports**: {results['failed_imports']}")
        report.append("")
        
        # Indicators with patterns
        report.append("## ✅ Indicators WITH Pattern Registration")
        report.append("")
        total_patterns = 0
        for module_path, analysis in results['indicators_with_patterns'].items():
            pattern_count = len(analysis['pattern_ids_from_file'])
            total_patterns += pattern_count
            status = "✅" if analysis['importable'] else "⚠️"
            report.append(f"- {status} **{module_path}**: {pattern_count} patterns")
            if analysis['indicator_classes']:
                report.append(f"  - Classes: {', '.join(analysis['indicator_classes'])}")
        
        report.append(f"\n**Total patterns found**: {total_patterns}")
        report.append("")
        
        # Indicators without patterns
        report.append("## ❌ Indicators WITHOUT Pattern Registration")
        report.append("")
        for module_path in results['indicators_without_patterns']:
            report.append(f"- ❌ **{module_path}**")
        report.append("")
        
        # Failed imports
        if results['failed_imports']:
            report.append("## ⚠️ Import Failures")
            report.append("")
            for failure in results['failed_imports']:
                report.append(f"- ⚠️ **{failure['module']}**: {failure['error']}")
            report.append("")
        
        return "\n".join(report)

if __name__ == "__main__":
    auditor = IndicatorAudit()
    results = auditor.run_comprehensive_audit()
    
    # Generate and save report
    report = auditor.generate_audit_report(results)
    
    with open("comprehensive_indicator_audit_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n📋 Audit complete!")
    print(f"📊 Summary: {results['with_patterns']}/{results['total_files']} files have pattern registration")
    print(f"📄 Report saved to: comprehensive_indicator_audit_report.md")
