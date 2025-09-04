#!/usr/bin/env python3
"""
项目清理执行脚本

基于清理分析报告，安全地执行项目清理操作
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


class ProjectCleaner:
    """项目清理执行器"""
    
    def __init__(self, root_dir: str, report_file: str):
        self.root_dir = Path(root_dir)
        self.report_file = report_file
        self.backup_dir = self.root_dir / 'archive' / 'cleanup_backup' / datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 加载清理报告
        with open(report_file, 'r', encoding='utf-8') as f:
            self.report = json.load(f)
    
    def create_backup(self) -> bool:
        """创建备份目录"""
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ 创建备份目录: {self.backup_dir}")
            return True
        except Exception as e:
            print(f"❌ 创建备份目录失败: {e}")
            return False
    
    def cleanup_backup_files(self) -> Dict[str, Any]:
        """清理备份文件"""
        print("\n🧹 清理备份文件...")
        
        results = {'deleted': [], 'errors': [], 'size_freed': 0}
        backup_files = self.report['cleanup_analysis'].get('backup_files', [])
        
        for file_path in backup_files:
            full_path = self.root_dir / file_path
            try:
                if full_path.exists() and full_path.is_file():
                    # 检查是否为重要备份文件
                    if self._is_important_backup(file_path):
                        print(f"  ⚠️  保留重要备份: {file_path}")
                        continue
                    
                    size = full_path.stat().st_size
                    full_path.unlink()
                    results['deleted'].append(file_path)
                    results['size_freed'] += size
                    
                    if len(results['deleted']) % 50 == 0:
                        print(f"  已删除 {len(results['deleted'])} 个备份文件...")
                        
            except Exception as e:
                results['errors'].append(f"删除 {file_path} 失败: {str(e)}")
        
        print(f"  ✅ 删除了 {len(results['deleted'])} 个备份文件")
        print(f"  💾 释放空间: {results['size_freed'] / 1024 / 1024:.1f} MB")
        
        return results
    
    def cleanup_temp_files(self) -> Dict[str, Any]:
        """清理临时文件"""
        print("\n🧹 清理临时文件...")
        
        results = {'deleted': [], 'errors': [], 'size_freed': 0}
        
        # 清理缓存文件
        cache_files = self.report['cleanup_analysis'].get('cache_files', [])
        for file_path in cache_files:
            full_path = self.root_dir / file_path
            try:
                if full_path.exists():
                    if full_path.is_file():
                        size = full_path.stat().st_size
                        full_path.unlink()
                        results['size_freed'] += size
                    elif full_path.is_dir():
                        shutil.rmtree(full_path)
                    results['deleted'].append(file_path)
            except Exception as e:
                results['errors'].append(f"删除 {file_path} 失败: {str(e)}")
        
        # 清理临时文件
        temp_files = self.report['cleanup_analysis'].get('temp_files', [])
        for file_path in temp_files:
            full_path = self.root_dir / file_path
            try:
                if full_path.exists() and full_path.is_file():
                    size = full_path.stat().st_size
                    full_path.unlink()
                    results['deleted'].append(file_path)
                    results['size_freed'] += size
            except Exception as e:
                results['errors'].append(f"删除 {file_path} 失败: {str(e)}")
        
        print(f"  ✅ 删除了 {len(results['deleted'])} 个临时/缓存文件")
        print(f"  💾 释放空间: {results['size_freed'] / 1024 / 1024:.1f} MB")
        
        return results
    
    def cleanup_outdated_results(self) -> Dict[str, Any]:
        """清理过时的测试结果文件"""
        print("\n🧹 清理过时的测试结果...")
        
        results = {'deleted': [], 'errors': [], 'size_freed': 0}
        outdated_files = self.report['cleanup_analysis'].get('outdated_test_results', [])
        
        for file_path in outdated_files:
            full_path = self.root_dir / file_path
            try:
                if full_path.exists() and full_path.is_file():
                    size = full_path.stat().st_size
                    full_path.unlink()
                    results['deleted'].append(file_path)
                    results['size_freed'] += size
            except Exception as e:
                results['errors'].append(f"删除 {file_path} 失败: {str(e)}")
        
        print(f"  ✅ 删除了 {len(results['deleted'])} 个过时结果文件")
        print(f"  💾 释放空间: {results['size_freed'] / 1024 / 1024:.1f} MB")
        
        return results
    
    def cleanup_empty_directories(self) -> Dict[str, Any]:
        """清理空目录"""
        print("\n🧹 清理空目录...")
        
        results = {'deleted': [], 'errors': []}
        empty_dirs = self.report['cleanup_analysis'].get('empty_directories', [])
        
        # 按深度排序，先删除深层目录
        empty_dirs.sort(key=lambda x: x.count('/'), reverse=True)
        
        for dir_path in empty_dirs:
            full_path = self.root_dir / dir_path
            try:
                if full_path.exists() and full_path.is_dir():
                    # 再次检查是否为空
                    if not any(full_path.iterdir()):
                        full_path.rmdir()
                        results['deleted'].append(dir_path)
            except Exception as e:
                results['errors'].append(f"删除目录 {dir_path} 失败: {str(e)}")
        
        print(f"  ✅ 删除了 {len(results['deleted'])} 个空目录")
        
        return results
    
    def cleanup_log_files(self) -> Dict[str, Any]:
        """清理日志文件（保留最近的）"""
        print("\n🧹 清理日志文件...")
        
        results = {'deleted': [], 'errors': [], 'size_freed': 0}
        log_files = self.report['cleanup_analysis'].get('log_files', [])
        
        # 按目录分组日志文件
        log_groups = {}
        for file_path in log_files:
            dir_path = str(Path(file_path).parent)
            if dir_path not in log_groups:
                log_groups[dir_path] = []
            log_groups[dir_path].append(file_path)
        
        # 每个目录保留最新的5个日志文件
        for dir_path, files in log_groups.items():
            if len(files) > 5:
                # 按修改时间排序
                files_with_time = []
                for file_path in files:
                    full_path = self.root_dir / file_path
                    try:
                        if full_path.exists():
                            mtime = full_path.stat().st_mtime
                            files_with_time.append((file_path, mtime))
                    except OSError:
                        pass
                
                # 排序并删除旧文件
                files_with_time.sort(key=lambda x: x[1], reverse=True)
                files_to_delete = files_with_time[5:]  # 保留最新的5个
                
                for file_path, _ in files_to_delete:
                    full_path = self.root_dir / file_path
                    try:
                        if full_path.exists():
                            size = full_path.stat().st_size
                            full_path.unlink()
                            results['deleted'].append(file_path)
                            results['size_freed'] += size
                    except Exception as e:
                        results['errors'].append(f"删除日志 {file_path} 失败: {str(e)}")
        
        print(f"  ✅ 删除了 {len(results['deleted'])} 个旧日志文件")
        print(f"  💾 释放空间: {results['size_freed'] / 1024 / 1024:.1f} MB")
        
        return results
    
    def _is_important_backup(self, file_path: str) -> bool:
        """判断是否为重要的备份文件"""
        important_patterns = [
            'final_migration_backup',
            'singleton_fix_backup',
            'cleanup_backup',
            'refactor_backup'
        ]
        return any(pattern in file_path for pattern in important_patterns)
    
    def generate_cleanup_summary(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """生成清理总结"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_files_deleted': 0,
            'total_directories_deleted': 0,
            'total_size_freed': 0,
            'total_errors': 0,
            'categories': {}
        }
        
        for category, result in results.items():
            summary['categories'][category] = {
                'files_deleted': len(result.get('deleted', [])),
                'size_freed': result.get('size_freed', 0),
                'errors': len(result.get('errors', []))
            }
            
            summary['total_files_deleted'] += len(result.get('deleted', []))
            summary['total_size_freed'] += result.get('size_freed', 0)
            summary['total_errors'] += len(result.get('errors', []))
        
        return summary


def main():
    """主函数"""
    print("🚀 项目清理执行工具")
    print("=" * 50)
    
    # 查找最新的清理报告
    report_files = list(Path('.').glob('comprehensive_cleanup_report_*.json'))
    if not report_files:
        print("❌ 未找到清理报告文件，请先运行清理分析工具")
        return 1
    
    latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
    print(f"📄 使用清理报告: {latest_report}")
    
    # 确认执行清理
    print(f"\n⚠️  即将执行项目清理操作")
    print(f"   这将删除备份文件、临时文件、缓存文件等")
    print(f"   重要文件将被保留")
    
    if input("\n确认执行清理？(yes/no): ").lower() != 'yes':
        print("❌ 用户取消操作")
        return 0
    
    # 执行清理
    cleaner = ProjectCleaner('.', str(latest_report))
    
    if not cleaner.create_backup():
        return 1
    
    results = {}
    
    # 1. 清理备份文件
    results['backup_files'] = cleaner.cleanup_backup_files()
    
    # 2. 清理临时文件
    results['temp_files'] = cleaner.cleanup_temp_files()
    
    # 3. 清理过时结果文件
    results['outdated_results'] = cleaner.cleanup_outdated_results()
    
    # 4. 清理空目录
    results['empty_directories'] = cleaner.cleanup_empty_directories()
    
    # 5. 清理日志文件
    results['log_files'] = cleaner.cleanup_log_files()
    
    # 生成清理总结
    summary = cleaner.generate_cleanup_summary(results)
    
    print(f"\n📊 清理总结:")
    print(f"  删除文件总数: {summary['total_files_deleted']}")
    print(f"  删除目录总数: {summary['total_directories_deleted']}")
    print(f"  释放空间总计: {summary['total_size_freed'] / 1024 / 1024:.1f} MB")
    print(f"  错误总数: {summary['total_errors']}")
    
    # 保存清理总结
    summary_file = f'cleanup_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 清理总结已保存到: {summary_file}")
    
    if summary['total_errors'] > 0:
        print(f"\n⚠️  清理过程中出现 {summary['total_errors']} 个错误")
        return 1
    else:
        print(f"\n✅ 清理完成，无错误")
        return 0


if __name__ == "__main__":
    sys.exit(main())
