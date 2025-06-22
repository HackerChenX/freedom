#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
系统清理工具

识别和清理系统中过时的文件和无用的脚本
"""

import os
import re
import json
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Set, Any
from collections import defaultdict

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class SystemCleanupTool:
    """系统清理工具"""
    
    def __init__(self, project_root: str = "."):
        """初始化清理工具"""
        self.project_root = Path(project_root).resolve()
        self.cleanup_report = {
            'scan_date': datetime.now().isoformat(),
            'categories': {},
            'recommendations': [],
            'cleanup_actions': []
        }
        
        # 定义过时文件模式
        self.obsolete_patterns = {
            'backup_files': [
                r'\.backup$', r'\.bak$', r'\.old$', r'\.orig$',
                r'\.broken_backup$', r'_backup\.py$'
            ],
            'temporary_files': [
                r'\.tmp$', r'\.temp$', r'_temp\.py$', r'temp_.*\.py$',
                r'test_.*\.png$', r'\.pyc$'
            ],
            'duplicate_files': [
                r'\.py\.new$', r'_new\.py$', r'_copy\.py$'
            ],
            'log_files': [
                r'\.log$'
            ],
            'result_files': [
                r'validation_results_.*\.json$',
                r'perfect_validation_results_.*\.json$',
                r'comprehensive_.*_results_.*\.json$',
                r'demo_validation_results_.*\.json$',
                r'enhanced_validation_results_.*\.json$',
                r'ultimate_validation_results_.*\.json$',
                r'optimized_.*_results_.*\.json$',
                r'selection_results_.*\.csv$'
            ]
        }
        
        # 定义可以安全删除的目录
        self.safe_cleanup_dirs = {
            '__pycache__',
            '.pytest_cache',
            'logs',
            'cache',
            'tmp',
            'temp',
            'metadata_dropped',
            'preprocessed_configs'
        }
        
        # 定义过时的脚本文件（根据功能重复或已被替代）
        self.obsolete_scripts = {
            # 已被新系统替代的旧脚本
            'analyze_migration_candidates.py',
            'analyze_pattern_polarity.py',
            'batch_fix_polarity.py',
            'batch_standardize_p1_indicators.py',
            'batch_standardize_p2_indicators.py',
            'batch_standardize_remaining.py',
            'check_missing_polarity_indicators.py',
            'check_missing_raw_score.py',
            'complete_remaining_standardization.py',
            'comprehensive_indicator_audit.py',
            'comprehensive_polarity_validation.py',
            'comprehensive_system_optimizer.py',
            'debug_indicators.py',
            'detailed_indicator_check.py',
            'direct_syntax_fixer.py',
            'final_comprehensive_validation.py',
            'final_quality_validator.py',
            'final_review_gate.py',
            'final_standardizer.py',
            'final_system_optimization_report.py',
            'final_system_validator.py',
            'find_missing_polarity.py',
            'fix_amplitude_elasticity.py',
            'fix_missing_raw_score.py',
            'fix_polarity_by_technical_analysis.py',
            'fix_polarity_consistency.py',
            'generate_cleanup_report.py',
            'indicator_pattern_audit.py',
            'indicator_quality_enhancer.py',
            'indicator_quality_optimizer.py',
            'intelligent_batch_standardizer.py',
            'manual_standardizer.py',
            'perfect_indicator_fixer.py',
            'perfect_schema_enhancer.py',
            'perfect_system_optimizer.py',
            'precise_syntax_fixer.py',
            'rapid_standardizer.py',
            'schema_aware_quality_enhancer.py',
            'schema_completion_tool.py',
            'schema_validator_fixer.py',
            'syntax_error_fixer.py',
            'syntax_fixer_and_standardizer.py',
            'system_integration_enhancer.py',
            'system_integration_fixer.py',
            'ultimate_perfect_validator.py',
            'validate_critical_polarity.py',
            'verify_fixes.py'
        }
        
        logger.info("系统清理工具初始化完成")
    
    def scan_obsolete_files(self) -> Dict[str, List[str]]:
        """扫描过时文件"""
        logger.info("开始扫描过时文件...")
        
        obsolete_files = defaultdict(list)
        
        # 扫描所有文件
        for file_path in self.project_root.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(self.project_root)
                file_name = file_path.name
                
                # 检查各种过时文件模式
                for category, patterns in self.obsolete_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, file_name):
                            obsolete_files[category].append(str(relative_path))
                            break
                
                # 检查过时脚本
                if file_name in self.obsolete_scripts:
                    obsolete_files['obsolete_scripts'].append(str(relative_path))
        
        # 更新报告
        self.cleanup_report['categories'] = dict(obsolete_files)
        
        # 统计信息
        total_files = sum(len(files) for files in obsolete_files.values())
        logger.info(f"扫描完成，发现 {total_files} 个过时文件")
        
        for category, files in obsolete_files.items():
            logger.info(f"  {category}: {len(files)} 个文件")
        
        return dict(obsolete_files)
    
    def scan_empty_directories(self) -> List[str]:
        """扫描空目录"""
        logger.info("扫描空目录...")
        
        empty_dirs = []
        
        for dir_path in self.project_root.rglob('*'):
            if dir_path.is_dir():
                try:
                    # 检查目录是否为空（忽略隐藏文件）
                    contents = [f for f in dir_path.iterdir() if not f.name.startswith('.')]
                    if not contents:
                        relative_path = dir_path.relative_to(self.project_root)
                        empty_dirs.append(str(relative_path))
                except PermissionError:
                    continue
        
        logger.info(f"发现 {len(empty_dirs)} 个空目录")
        return empty_dirs
    
    def scan_large_log_files(self, size_limit_mb: int = 10) -> List[Dict[str, Any]]:
        """扫描大型日志文件"""
        logger.info(f"扫描大于 {size_limit_mb}MB 的日志文件...")
        
        large_logs = []
        size_limit_bytes = size_limit_mb * 1024 * 1024
        
        log_patterns = [r'\.log$', r'\.log\.\d+$']
        
        for file_path in self.project_root.rglob('*'):
            if file_path.is_file():
                file_name = file_path.name
                
                # 检查是否为日志文件
                is_log = any(re.search(pattern, file_name) for pattern in log_patterns)
                
                if is_log:
                    try:
                        file_size = file_path.stat().st_size
                        if file_size > size_limit_bytes:
                            relative_path = file_path.relative_to(self.project_root)
                            large_logs.append({
                                'path': str(relative_path),
                                'size_mb': file_size / (1024 * 1024),
                                'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                            })
                    except (OSError, PermissionError):
                        continue
        
        # 按大小排序
        large_logs.sort(key=lambda x: x['size_mb'], reverse=True)
        
        logger.info(f"发现 {len(large_logs)} 个大型日志文件")
        return large_logs
    
    def scan_old_files(self, days_old: int = 30) -> List[Dict[str, Any]]:
        """扫描旧文件"""
        logger.info(f"扫描 {days_old} 天前的文件...")
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        old_files = []
        
        # 只检查特定类型的文件
        check_patterns = [
            r'\.json$',  # 结果文件
            r'\.csv$',   # 数据文件
            r'\.log$',   # 日志文件
            r'\.png$',   # 图片文件
            r'\.backup$', # 备份文件
        ]
        
        for file_path in self.project_root.rglob('*'):
            if file_path.is_file():
                file_name = file_path.name
                
                # 检查文件类型
                should_check = any(re.search(pattern, file_name) for pattern in check_patterns)
                
                if should_check:
                    try:
                        modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if modified_time < cutoff_date:
                            relative_path = file_path.relative_to(self.project_root)
                            old_files.append({
                                'path': str(relative_path),
                                'modified': modified_time.isoformat(),
                                'size_mb': file_path.stat().st_size / (1024 * 1024)
                            })
                    except (OSError, PermissionError):
                        continue
        
        # 按修改时间排序
        old_files.sort(key=lambda x: x['modified'])
        
        logger.info(f"发现 {len(old_files)} 个旧文件")
        return old_files
    
    def generate_cleanup_recommendations(self) -> List[Dict[str, Any]]:
        """生成清理建议"""
        logger.info("生成清理建议...")
        
        recommendations = []
        
        # 扫描各种过时文件
        obsolete_files = self.scan_obsolete_files()
        empty_dirs = self.scan_empty_directories()
        large_logs = self.scan_large_log_files()
        old_files = self.scan_old_files()
        
        # 备份文件建议
        if 'backup_files' in obsolete_files:
            recommendations.append({
                'category': 'backup_files',
                'priority': 'high',
                'action': 'delete',
                'description': f"删除 {len(obsolete_files['backup_files'])} 个备份文件",
                'files': obsolete_files['backup_files'],
                'safe': True
            })
        
        # 临时文件建议
        if 'temporary_files' in obsolete_files:
            recommendations.append({
                'category': 'temporary_files',
                'priority': 'high',
                'action': 'delete',
                'description': f"删除 {len(obsolete_files['temporary_files'])} 个临时文件",
                'files': obsolete_files['temporary_files'],
                'safe': True
            })
        
        # 过时脚本建议
        if 'obsolete_scripts' in obsolete_files:
            recommendations.append({
                'category': 'obsolete_scripts',
                'priority': 'medium',
                'action': 'archive_or_delete',
                'description': f"归档或删除 {len(obsolete_files['obsolete_scripts'])} 个过时脚本",
                'files': obsolete_files['obsolete_scripts'],
                'safe': False,
                'note': "建议先归档，确认无用后再删除"
            })
        
        # 结果文件建议
        if 'result_files' in obsolete_files:
            recommendations.append({
                'category': 'result_files',
                'priority': 'medium',
                'action': 'archive',
                'description': f"归档 {len(obsolete_files['result_files'])} 个结果文件",
                'files': obsolete_files['result_files'],
                'safe': False,
                'note': "建议归档到专门目录"
            })
        
        # 空目录建议
        if empty_dirs:
            recommendations.append({
                'category': 'empty_directories',
                'priority': 'low',
                'action': 'delete',
                'description': f"删除 {len(empty_dirs)} 个空目录",
                'files': empty_dirs,
                'safe': True
            })
        
        # 大型日志文件建议
        if large_logs:
            recommendations.append({
                'category': 'large_logs',
                'priority': 'medium',
                'action': 'archive_or_compress',
                'description': f"处理 {len(large_logs)} 个大型日志文件",
                'files': [log['path'] for log in large_logs],
                'safe': False,
                'note': "建议压缩或归档"
            })
        
        # 旧文件建议
        if old_files:
            recommendations.append({
                'category': 'old_files',
                'priority': 'low',
                'action': 'review',
                'description': f"检查 {len(old_files)} 个旧文件",
                'files': [f['path'] for f in old_files],
                'safe': False,
                'note': "需要人工检查是否还需要"
            })
        
        self.cleanup_report['recommendations'] = recommendations
        
        logger.info(f"生成了 {len(recommendations)} 条清理建议")
        return recommendations
    
    def execute_safe_cleanup(self, dry_run: bool = True) -> Dict[str, Any]:
        """执行安全清理"""
        logger.info(f"执行安全清理 (dry_run={dry_run})...")
        
        cleanup_results = {
            'dry_run': dry_run,
            'actions_taken': [],
            'files_deleted': 0,
            'space_freed_mb': 0,
            'errors': []
        }
        
        recommendations = self.generate_cleanup_recommendations()
        
        for rec in recommendations:
            if rec.get('safe', False) and rec['priority'] == 'high':
                category = rec['category']
                files = rec['files']
                
                logger.info(f"处理 {category}: {len(files)} 个文件")
                
                for file_path in files:
                    full_path = self.project_root / file_path
                    
                    try:
                        if full_path.exists():
                            # 计算文件大小
                            if full_path.is_file():
                                size_mb = full_path.stat().st_size / (1024 * 1024)
                            else:
                                size_mb = 0
                            
                            if not dry_run:
                                if full_path.is_file():
                                    full_path.unlink()
                                elif full_path.is_dir():
                                    shutil.rmtree(full_path)
                                
                                cleanup_results['files_deleted'] += 1
                                cleanup_results['space_freed_mb'] += size_mb
                            
                            cleanup_results['actions_taken'].append({
                                'action': 'delete',
                                'path': file_path,
                                'category': category,
                                'size_mb': size_mb
                            })
                            
                    except Exception as e:
                        error_msg = f"删除 {file_path} 失败: {e}"
                        logger.error(error_msg)
                        cleanup_results['errors'].append(error_msg)
        
        self.cleanup_report['cleanup_actions'] = cleanup_results
        
        if dry_run:
            logger.info("干运行完成，未实际删除文件")
        else:
            logger.info(f"清理完成，删除了 {cleanup_results['files_deleted']} 个文件，"
                       f"释放空间 {cleanup_results['space_freed_mb']:.2f} MB")
        
        return cleanup_results
    
    def generate_cleanup_report(self, output_file: str = None) -> str:
        """生成清理报告"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"system_cleanup_report_{timestamp}.json"
        
        # 确保有完整的扫描数据
        if not self.cleanup_report['categories']:
            self.generate_cleanup_recommendations()
        
        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.cleanup_report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"清理报告已保存: {output_file}")
        return output_file


def main():
    """主函数"""
    print("🧹 系统清理工具")
    print("=" * 50)
    
    try:
        # 创建清理工具
        cleanup_tool = SystemCleanupTool()
        
        # 生成清理建议
        recommendations = cleanup_tool.generate_cleanup_recommendations()
        
        # 显示建议摘要
        print("\n📋 清理建议摘要:")
        print("-" * 30)
        
        total_files = 0
        for rec in recommendations:
            print(f"• {rec['description']} ({rec['priority']} 优先级)")
            total_files += len(rec['files'])
        
        print(f"\n📊 总计: {total_files} 个文件需要处理")
        
        # 询问是否执行安全清理
        print("\n🔍 执行安全清理预览...")
        dry_run_results = cleanup_tool.execute_safe_cleanup(dry_run=True)
        
        print(f"\n预览结果:")
        print(f"• 可安全删除: {len(dry_run_results['actions_taken'])} 个文件")
        print(f"• 可释放空间: {dry_run_results['space_freed_mb']:.2f} MB")
        
        # 生成详细报告
        report_file = cleanup_tool.generate_cleanup_report()
        print(f"\n📄 详细报告已保存: {report_file}")
        
        # 询问是否执行实际清理
        response = input("\n是否执行安全清理? (y/N): ").strip().lower()
        if response == 'y':
            print("\n🗑️ 执行实际清理...")
            actual_results = cleanup_tool.execute_safe_cleanup(dry_run=False)
            
            print(f"✅ 清理完成!")
            print(f"• 删除文件: {actual_results['files_deleted']} 个")
            print(f"• 释放空间: {actual_results['space_freed_mb']:.2f} MB")
            
            if actual_results['errors']:
                print(f"⚠️ 错误: {len(actual_results['errors'])} 个")
        else:
            print("\n取消清理操作")
        
        print("\n💡 建议:")
        print("1. 查看详细报告了解所有发现的文件")
        print("2. 手动检查标记为'不安全'的文件")
        print("3. 考虑归档重要的结果文件")
        print("4. 定期运行此工具保持系统整洁")
        
    except Exception as e:
        print(f"❌ 清理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
