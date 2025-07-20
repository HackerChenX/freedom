#!/usr/bin/env python3
"""
生成最终的清理和架构合规报告

汇总所有清理和修复操作的结果，生成完整的项目状态报告
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


class FinalReportGenerator:
    """最终报告生成器"""
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'project_status': {},
            'cleanup_summary': {},
            'architecture_status': {},
            'recommendations': [],
            'next_steps': []
        }
    
    def analyze_project_status(self) -> Dict[str, Any]:
        """分析项目当前状态"""
        print("📊 分析项目当前状态...")
        
        status = {
            'total_files': 0,
            'python_files': 0,
            'test_files': 0,
            'config_files': 0,
            'documentation_files': 0,
            'data_files': 0,
            'directory_structure': {},
            'file_size_distribution': {}
        }
        
        # 统计文件数量和类型
        for file_path in self.root_dir.rglob("*"):
            if file_path.is_file() and not self._should_skip_file(file_path):
                status['total_files'] += 1
                
                suffix = file_path.suffix.lower()
                if suffix == '.py':
                    status['python_files'] += 1
                    if 'test' in file_path.name.lower():
                        status['test_files'] += 1
                elif suffix in ['.yaml', '.yml', '.json', '.ini', '.cfg']:
                    status['config_files'] += 1
                elif suffix in ['.md', '.rst', '.txt']:
                    status['documentation_files'] += 1
                elif suffix in ['.csv', '.json', '.xml']:
                    status['data_files'] += 1
        
        # 分析目录结构
        main_dirs = ['bin', 'api', 'strategy', 'analysis', 'indicators', 'formula', 
                    'db', 'utils', 'config', 'enums', 'tests', 'docs', 'data']
        
        for dir_name in main_dirs:
            dir_path = self.root_dir / dir_name
            if dir_path.exists():
                file_count = len(list(dir_path.rglob("*.py")))
                status['directory_structure'][dir_name] = {
                    'exists': True,
                    'python_files': file_count
                }
            else:
                status['directory_structure'][dir_name] = {'exists': False}
        
        self.report['project_status'] = status
        return status
    
    def load_cleanup_summary(self) -> Dict[str, Any]:
        """加载清理操作总结"""
        print("📋 加载清理操作总结...")
        
        # 查找最新的清理总结文件
        cleanup_files = list(self.root_dir.glob('cleanup_summary_*.json'))
        if cleanup_files:
            latest_cleanup = max(cleanup_files, key=lambda x: x.stat().st_mtime)
            with open(latest_cleanup, 'r', encoding='utf-8') as f:
                cleanup_data = json.load(f)
            
            summary = {
                'cleanup_executed': True,
                'files_deleted': cleanup_data.get('total_files_deleted', 0),
                'space_freed_mb': cleanup_data.get('total_size_freed', 0) / 1024 / 1024,
                'categories_cleaned': list(cleanup_data.get('categories', {}).keys()),
                'cleanup_timestamp': cleanup_data.get('timestamp')
            }
        else:
            summary = {
                'cleanup_executed': False,
                'files_deleted': 0,
                'space_freed_mb': 0,
                'categories_cleaned': [],
                'cleanup_timestamp': None
            }
        
        self.report['cleanup_summary'] = summary
        return summary
    
    def analyze_architecture_compliance(self) -> Dict[str, Any]:
        """分析架构合规性"""
        print("🏗️ 分析架构合规性...")
        
        # 查找架构合规报告
        compliance_file = self.root_dir / 'architecture_compliance_report.json'
        if compliance_file.exists():
            with open(compliance_file, 'r', encoding='utf-8') as f:
                compliance_data = json.load(f)
            
            status = {
                'compliance_checked': True,
                'total_violations': compliance_data.get('summary', {}).get('total_violations', 0),
                'critical_violations': compliance_data.get('summary', {}).get('critical_violations', 0),
                'compliance_score': compliance_data.get('summary', {}).get('compliance_score', 0),
                'cross_layer_violations': len(compliance_data.get('cross_layer_violations', [])),
                'naming_violations': len(compliance_data.get('naming_violations', [])),
                'code_duplication': len(compliance_data.get('code_duplication', [])),
                'check_timestamp': compliance_data.get('timestamp')
            }
        else:
            status = {
                'compliance_checked': False,
                'total_violations': 'unknown',
                'critical_violations': 'unknown',
                'compliance_score': 'unknown',
                'cross_layer_violations': 'unknown',
                'naming_violations': 'unknown',
                'code_duplication': 'unknown',
                'check_timestamp': None
            }
        
        self.report['architecture_status'] = status
        return status
    
    def generate_recommendations(self):
        """生成改进建议"""
        print("💡 生成改进建议...")
        
        recommendations = []
        
        # 基于清理结果的建议
        cleanup = self.report['cleanup_summary']
        if cleanup['cleanup_executed']:
            recommendations.append({
                'category': 'maintenance',
                'priority': 'low',
                'title': '定期清理维护',
                'description': f'已成功清理 {cleanup["files_deleted"]} 个文件，释放 {cleanup["space_freed_mb"]:.1f} MB 空间',
                'action': '建议每月执行一次项目清理，保持代码库整洁'
            })
        else:
            recommendations.append({
                'category': 'maintenance',
                'priority': 'medium',
                'title': '执行项目清理',
                'description': '项目中存在大量备份文件和临时文件',
                'action': '运行清理脚本删除无用文件，释放磁盘空间'
            })
        
        # 基于架构合规性的建议
        arch = self.report['architecture_status']
        if arch['compliance_checked']:
            if arch['critical_violations'] > 0:
                recommendations.append({
                    'category': 'architecture',
                    'priority': 'critical',
                    'title': '修复架构违规',
                    'description': f'发现 {arch["critical_violations"]} 个严重架构违规',
                    'action': '使用依赖注入模式重构跨层依赖，确保分层架构合规'
                })
            
            if arch['compliance_score'] < 80:
                recommendations.append({
                    'category': 'architecture',
                    'priority': 'high',
                    'title': '提升架构合规性',
                    'description': f'当前合规分数: {arch["compliance_score"]}/100',
                    'action': '重构违规代码，提升架构质量和可维护性'
                })
        
        # 基于项目结构的建议
        project = self.report['project_status']
        missing_dirs = [name for name, info in project['directory_structure'].items() 
                       if not info['exists']]
        
        if missing_dirs:
            recommendations.append({
                'category': 'structure',
                'priority': 'medium',
                'title': '完善目录结构',
                'description': f'缺少标准目录: {", ".join(missing_dirs)}',
                'action': '创建缺失的目录，完善项目结构'
            })
        
        # 测试覆盖率建议
        if project['test_files'] < project['python_files'] * 0.3:
            recommendations.append({
                'category': 'testing',
                'priority': 'high',
                'title': '增加测试覆盖率',
                'description': f'测试文件数量 ({project["test_files"]}) 相对于代码文件 ({project["python_files"]}) 偏少',
                'action': '编写更多单元测试和集成测试，提升代码质量'
            })
        
        self.report['recommendations'] = recommendations
    
    def generate_next_steps(self):
        """生成后续步骤"""
        print("📋 生成后续步骤...")
        
        next_steps = []
        
        # 基于当前状态确定优先级
        arch = self.report['architecture_status']
        cleanup = self.report['cleanup_summary']
        
        if arch['compliance_checked'] and arch['critical_violations'] > 0:
            next_steps.append({
                'step': 1,
                'title': '修复严重架构违规',
                'description': '运行架构修复工具，解决跨层依赖问题',
                'command': 'python3 fix_architecture_violations.py',
                'estimated_time': '2-4 小时'
            })
        
        if not cleanup['cleanup_executed']:
            next_steps.append({
                'step': 2,
                'title': '执行项目清理',
                'description': '清理无用文件，释放磁盘空间',
                'command': 'python3 execute_project_cleanup.py',
                'estimated_time': '30 分钟'
            })
        
        next_steps.append({
            'step': 3,
            'title': '建立持续集成检查',
            'description': '设置自动化架构合规检查和代码质量监控',
            'command': '配置 CI/CD 流水线',
            'estimated_time': '1-2 小时'
        })
        
        next_steps.append({
            'step': 4,
            'title': '完善测试覆盖率',
            'description': '为核心模块编写单元测试和集成测试',
            'command': '编写测试用例',
            'estimated_time': '4-8 小时'
        })
        
        next_steps.append({
            'step': 5,
            'title': '文档更新',
            'description': '更新架构文档和使用指南',
            'command': '更新 README 和技术文档',
            'estimated_time': '1-2 小时'
        })
        
        self.report['next_steps'] = next_steps
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """判断是否应该跳过文件"""
        skip_patterns = [
            '__pycache__', '.git', 'venv', 'docker', 'archive',
            '.pytest_cache', '.coverage', 'node_modules'
        ]
        return any(pattern in str(file_path) for pattern in skip_patterns)


def main():
    """主函数"""
    print("🚀 生成最终清理和合规报告")
    print("=" * 50)
    
    generator = FinalReportGenerator('.')
    
    # 1. 分析项目状态
    project_status = generator.analyze_project_status()
    
    # 2. 加载清理总结
    cleanup_summary = generator.load_cleanup_summary()
    
    # 3. 分析架构合规性
    arch_status = generator.analyze_architecture_compliance()
    
    # 4. 生成建议和后续步骤
    generator.generate_recommendations()
    generator.generate_next_steps()
    
    # 5. 输出报告摘要
    print(f"\n📊 项目状态摘要:")
    print(f"  总文件数: {project_status['total_files']}")
    print(f"  Python 文件: {project_status['python_files']}")
    print(f"  测试文件: {project_status['test_files']}")
    print(f"  配置文件: {project_status['config_files']}")
    print(f"  文档文件: {project_status['documentation_files']}")
    
    print(f"\n🧹 清理状态:")
    if cleanup_summary['cleanup_executed']:
        print(f"  ✅ 已执行清理")
        print(f"  删除文件: {cleanup_summary['files_deleted']} 个")
        print(f"  释放空间: {cleanup_summary['space_freed_mb']:.1f} MB")
    else:
        print(f"  ❌ 未执行清理")
    
    print(f"\n🏗️ 架构合规性:")
    if arch_status['compliance_checked']:
        print(f"  合规分数: {arch_status['compliance_score']}/100")
        print(f"  严重违规: {arch_status['critical_violations']} 个")
        print(f"  总违规数: {arch_status['total_violations']} 个")
    else:
        print(f"  ❌ 未检查架构合规性")
    
    print(f"\n💡 改进建议 ({len(generator.report['recommendations'])} 项):")
    for i, rec in enumerate(generator.report['recommendations'][:3], 1):
        priority_icon = "🔴" if rec['priority'] == 'critical' else "🟡" if rec['priority'] == 'high' else "🟢"
        print(f"  {i}. {priority_icon} {rec['title']}")
    
    print(f"\n📋 后续步骤 ({len(generator.report['next_steps'])} 步):")
    for step in generator.report['next_steps'][:3]:
        print(f"  {step['step']}. {step['title']} (预计: {step['estimated_time']})")
    
    # 6. 保存完整报告
    report_file = f'final_cleanup_compliance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(generator.report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 完整报告已保存到: {report_file}")
    
    # 7. 生成 Markdown 报告
    md_report = generate_markdown_report(generator.report)
    md_file = f'FINAL_CLEANUP_COMPLIANCE_REPORT_{datetime.now().strftime("%Y%m%d")}.md'
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_report)
    
    print(f"📄 Markdown 报告已保存到: {md_file}")
    
    return 0


def generate_markdown_report(report: Dict[str, Any]) -> str:
    """生成 Markdown 格式的报告"""
    
    md_content = f"""# 项目清理和架构合规最终报告

**生成时间**: {report['timestamp']}

## 📊 项目状态概览

| 指标 | 数值 |
|------|------|
| 总文件数 | {report['project_status']['total_files']} |
| Python 文件 | {report['project_status']['python_files']} |
| 测试文件 | {report['project_status']['test_files']} |
| 配置文件 | {report['project_status']['config_files']} |
| 文档文件 | {report['project_status']['documentation_files']} |

## 🧹 清理操作结果

"""
    
    cleanup = report['cleanup_summary']
    if cleanup['cleanup_executed']:
        md_content += f"""✅ **清理已完成**

- 删除文件: {cleanup['files_deleted']} 个
- 释放空间: {cleanup['space_freed_mb']:.1f} MB
- 清理类别: {', '.join(cleanup['categories_cleaned'])}
- 执行时间: {cleanup['cleanup_timestamp']}

"""
    else:
        md_content += "❌ **清理未执行**\n\n"
    
    md_content += "## 🏗️ 架构合规性状态\n\n"
    
    arch = report['architecture_status']
    if arch['compliance_checked']:
        md_content += f"""✅ **架构检查已完成**

| 指标 | 数值 |
|------|------|
| 合规分数 | {arch['compliance_score']}/100 |
| 严重违规 | {arch['critical_violations']} 个 |
| 总违规数 | {arch['total_violations']} 个 |
| 跨层依赖违规 | {arch['cross_layer_violations']} 个 |
| 命名规范违规 | {arch['naming_violations']} 个 |
| 代码重复问题 | {arch['code_duplication']} 个 |

"""
    else:
        md_content += "❌ **架构检查未执行**\n\n"
    
    md_content += "## 💡 改进建议\n\n"
    
    for i, rec in enumerate(report['recommendations'], 1):
        priority_icon = "🔴" if rec['priority'] == 'critical' else "🟡" if rec['priority'] == 'high' else "🟢"
        md_content += f"### {i}. {priority_icon} {rec['title']}\n\n"
        md_content += f"**描述**: {rec['description']}\n\n"
        md_content += f"**建议行动**: {rec['action']}\n\n"
    
    md_content += "## 📋 后续步骤\n\n"
    
    for step in report['next_steps']:
        md_content += f"### 步骤 {step['step']}: {step['title']}\n\n"
        md_content += f"**描述**: {step['description']}\n\n"
        md_content += f"**命令**: `{step['command']}`\n\n"
        md_content += f"**预计时间**: {step['estimated_time']}\n\n"
    
    md_content += """## 📈 项目健康度评估

基于当前分析结果，项目健康度评估如下：

"""
    
    # 计算健康度分数
    health_score = 100
    
    if not cleanup['cleanup_executed']:
        health_score -= 10
    
    if arch['compliance_checked']:
        health_score -= (100 - arch['compliance_score']) * 0.5
    else:
        health_score -= 30
    
    if report['project_status']['test_files'] < report['project_status']['python_files'] * 0.3:
        health_score -= 20
    
    health_score = max(0, health_score)
    
    if health_score >= 90:
        health_level = "优秀 ✅"
    elif health_score >= 70:
        health_level = "良好 🟡"
    elif health_score >= 50:
        health_level = "一般 🟠"
    else:
        health_level = "需要改进 🔴"
    
    md_content += f"**项目健康度**: {health_score:.0f}/100 ({health_level})\n\n"
    
    md_content += """## 🎯 总结

本次清理和架构合规检查已完成。请按照后续步骤逐步改进项目质量，确保代码库的长期可维护性。

---

*此报告由项目清理和架构合规检查工具自动生成*
"""
    
    return md_content


if __name__ == "__main__":
    sys.exit(main())
