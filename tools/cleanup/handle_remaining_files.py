#!/usr/bin/env python3
"""
处理剩余未分类文件的工具

将根目录下剩余的文件移动到合适的位置
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime


class RemainingFilesHandler:
    """剩余文件处理器"""
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.backup_dir = self.root_dir / 'archive' / 'remaining_files_backup' / datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 剩余文件的处理规则
        self.file_mappings = {
            # 调试和分析脚本
            'deep_debug_603359.py': 'tools/debug',
            'analyze_centralized_mapping_status.py': 'tools/analysis',
            'system_integration_analyzer.py': 'tools/analysis',
            'performance_optimizer.py': 'tools/optimization',
            
            # 架构和清理工具
            'final_architecture_fix.py': 'tools/fixes',
            'quick_architecture_fix.py': 'tools/fixes',
            'execute_project_cleanup.py': 'tools/cleanup',
            'root_directory_organizer.py': 'tools/cleanup',
            'generate_final_cleanup_report.py': 'tools/cleanup',
            
            # 测试和验证脚本
            'system_integration_test.py': 'tests/integration',
            'run_comprehensive_test.py': 'tests/scripts',
            'run_production_validation.py': 'tests/production',
            'final_review_gate.py': 'tests/validation',
            'quick_compliance_check.py': 'tools/validation',
            
            # 数据处理脚本
            'import_native_data.py': 'tools/data',
            'populate_stock_database.py': 'tools/data',
            
            # 风险检测
            'run_risk_detection.py': 'tools/risk',
            
            # 配置文件
            '.flake8': '.',  # 保留在根目录
            
            # 安装脚本
            'get-pip.py': 'tools/setup',
        }
        
        # 需要删除的文件
        self.files_to_delete = [
            'uuid'  # 这个看起来是临时文件或目录
        ]
    
    def handle_remaining_files(self) -> dict:
        """处理剩余文件"""
        print("🔧 处理剩余未分类文件...")
        
        results = {
            'moved_files': [],
            'deleted_files': [],
            'kept_files': [],
            'errors': []
        }
        
        # 创建备份目录
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 处理文件映射
        for file_name, target_dir in self.file_mappings.items():
            src_path = self.root_dir / file_name
            
            if not src_path.exists():
                continue
            
            try:
                if target_dir == '.':
                    # 保留在根目录
                    results['kept_files'].append(file_name)
                    continue
                
                # 移动到目标目录
                dst_dir = self.root_dir / target_dir
                dst_path = dst_dir / file_name
                
                # 创建目标目录
                dst_dir.mkdir(parents=True, exist_ok=True)
                
                # 备份原文件
                backup_path = self.backup_dir / file_name
                if src_path.is_file():
                    shutil.copy2(src_path, backup_path)
                
                # 移动文件
                shutil.move(str(src_path), str(dst_path))
                
                results['moved_files'].append({
                    'file': file_name,
                    'from': '.',
                    'to': target_dir
                })
                
                print(f"  ✅ 移动: {file_name} -> {target_dir}")
                
            except Exception as e:
                results['errors'].append(f"处理 {file_name} 失败: {str(e)}")
                print(f"  ❌ 错误: {file_name} - {str(e)}")
        
        # 删除不需要的文件
        for file_name in self.files_to_delete:
            src_path = self.root_dir / file_name
            
            if not src_path.exists():
                continue
            
            try:
                # 备份后删除
                if src_path.is_file():
                    backup_path = self.backup_dir / file_name
                    shutil.copy2(src_path, backup_path)
                    src_path.unlink()
                elif src_path.is_dir():
                    backup_path = self.backup_dir / file_name
                    shutil.copytree(src_path, backup_path)
                    shutil.rmtree(src_path)
                
                results['deleted_files'].append(file_name)
                print(f"  🗑️ 删除: {file_name}")
                
            except Exception as e:
                results['errors'].append(f"删除 {file_name} 失败: {str(e)}")
                print(f"  ❌ 错误: {file_name} - {str(e)}")
        
        return results
    
    def create_tool_directories(self) -> list:
        """创建工具目录结构"""
        print("🏗️ 创建工具目录结构...")
        
        tool_dirs = [
            'tools/debug',
            'tools/analysis',
            'tools/optimization',
            'tools/cleanup',
            'tools/data',
            'tools/risk',
            'tools/setup',
            'tests/production'
        ]
        
        created_dirs = []
        for dir_path in tool_dirs:
            full_path = self.root_dir / dir_path
            if not full_path.exists():
                full_path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(dir_path)
                print(f"  📁 创建目录: {dir_path}")
        
        return created_dirs
    
    def check_root_directory_status(self) -> dict:
        """检查根目录清理状态"""
        print("📊 检查根目录清理状态...")
        
        # 获取根目录下的所有文件（不包括子目录）
        root_files = [f for f in self.root_dir.iterdir() if f.is_file()]
        
        # 应该保留在根目录的文件
        expected_files = {
            'README.md',
            'requirements.txt',
            'pyproject.toml',
            'pytest.ini',
            'docker-compose.yml',
            '__init__.py',
            '.gitignore',
            '.flake8'
        }
        
        status = {
            'total_files': len(root_files),
            'expected_files': [],
            'unexpected_files': [],
            'missing_files': []
        }
        
        for file_path in root_files:
            file_name = file_path.name
            if file_name in expected_files:
                status['expected_files'].append(file_name)
            else:
                status['unexpected_files'].append(file_name)
        
        # 检查缺失的重要文件
        for expected_file in expected_files:
            if expected_file not in [f.name for f in root_files]:
                status['missing_files'].append(expected_file)
        
        return status


def main():
    """主函数"""
    print("🚀 处理剩余未分类文件工具")
    print("=" * 50)
    
    handler = RemainingFilesHandler('.')
    
    # 1. 检查当前根目录状态
    current_status = handler.check_root_directory_status()
    print(f"\n📊 当前根目录状态:")
    print(f"  文件总数: {current_status['total_files']}")
    print(f"  预期文件: {len(current_status['expected_files'])}")
    print(f"  意外文件: {len(current_status['unexpected_files'])}")
    
    if current_status['unexpected_files']:
        print(f"\n❓ 意外文件:")
        for file_name in current_status['unexpected_files']:
            print(f"    {file_name}")
    
    if current_status['missing_files']:
        print(f"\n⚠️ 缺失的重要文件:")
        for file_name in current_status['missing_files']:
            print(f"    {file_name}")
    
    # 2. 询问是否处理剩余文件
    if input("\n是否处理剩余文件？(y/N): ").lower() == 'y':
        
        # 3. 创建工具目录
        created_dirs = handler.create_tool_directories()
        
        # 4. 处理剩余文件
        results = handler.handle_remaining_files()
        
        print(f"\n📊 处理结果:")
        print(f"  移动文件: {len(results['moved_files'])}")
        print(f"  删除文件: {len(results['deleted_files'])}")
        print(f"  保留文件: {len(results['kept_files'])}")
        print(f"  错误数量: {len(results['errors'])}")
        
        if results['errors']:
            print(f"\n❌ 错误信息:")
            for error in results['errors']:
                print(f"  {error}")
        
        # 5. 再次检查根目录状态
        final_status = handler.check_root_directory_status()
        print(f"\n📊 最终根目录状态:")
        print(f"  文件总数: {final_status['total_files']}")
        print(f"  预期文件: {len(final_status['expected_files'])}")
        print(f"  意外文件: {len(final_status['unexpected_files'])}")
        
        if final_status['unexpected_files']:
            print(f"\n⚠️ 仍有意外文件:")
            for file_name in final_status['unexpected_files']:
                print(f"    {file_name}")
        else:
            print(f"\n✅ 根目录已完全清理！")
        
        print(f"\n📁 备份位置: {handler.backup_dir}")
        
        # 6. 生成最终报告
        report = f"""# 剩余文件处理报告

**执行时间**: {datetime.now().isoformat()}

## 📊 处理统计

- **移动文件**: {len(results['moved_files'])} 个
- **删除文件**: {len(results['deleted_files'])} 个
- **保留文件**: {len(results['kept_files'])} 个
- **错误数量**: {len(results['errors'])} 个

## 📁 文件移动详情

"""
        
        for move_info in results['moved_files']:
            report += f"- {move_info['file']} -> {move_info['to']}\n"
        
        if results['deleted_files']:
            report += f"\n## 🗑️ 删除的文件\n\n"
            for file_name in results['deleted_files']:
                report += f"- {file_name}\n"
        
        if results['kept_files']:
            report += f"\n## 🗂️ 保留的文件\n\n"
            for file_name in results['kept_files']:
                report += f"- {file_name}\n"
        
        report += f"\n## 📊 最终根目录状态\n\n"
        report += f"- **文件总数**: {final_status['total_files']}\n"
        report += f"- **预期文件**: {len(final_status['expected_files'])}\n"
        report += f"- **意外文件**: {len(final_status['unexpected_files'])}\n"
        
        if final_status['unexpected_files']:
            report += f"\n### ⚠️ 仍需处理的文件\n\n"
            for file_name in final_status['unexpected_files']:
                report += f"- {file_name}\n"
        
        report += f"\n## 📁 备份位置\n\n"
        report += f"原文件已备份到: `{handler.backup_dir}`\n"
        
        report_file = f'remaining_files_handling_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 处理报告已保存到: {report_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
