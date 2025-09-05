#!/usr/bin/env python3
"""
最终清理docs目录中剩余的问题报告和总结文件
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.logger import get_logger

logger = get_logger(__name__)

class FinalDocsCleaner:
    """最终docs清理器"""
    
    def __init__(self):
        self.docs_root = Path("docs")
        self.backup_dir = Path("archive/final_docs_cleanup_backup")
        
        # 需要清理的具体文件（剩余的问题报告和总结）
        self.specific_cleanup_files = [
            "docs/finaltesting/RSI指标验证阶段完整总结.md",
            "docs/finaltesting/MACD指标验证阶段全面总结.md", 
            "docs/finaltesting/ZXM_PATTERNS_验证成果总结.md",
            "docs/finaltesting/技术指标测试修复快速参考指南.md",
            "docs/project_reports/project_reorganization_report.md",
            "docs/technical_analysis/system_optimization_complete_report.md"
        ]
        
        # 需要保留的重要文档（绝对不删除）
        self.important_docs = [
            "docs/finaltesting/形态管理规范.md",
            "docs/finaltesting/技术指标验证执行计划.md", 
            "docs/finaltesting/生产环境指标验证使用指南.md",
            "docs/finaltesting/生产级技术指标验证方案.md",
            "docs/finaltesting/生产部署指南.md",
            "docs/finaltesting/系统架构更新文档.md",
            "docs/finaltesting/进度表使用说明.md",
            "docs/architecture/",
            "docs/user_guides/",
            "docs/standards/",
            "docs/methodologies/",
            "docs/api_docs/"
        ]
    
    def create_backup_dir(self):
        """创建备份目录"""
        if not self.backup_dir.exists():
            self.backup_dir.mkdir(parents=True)
            logger.info(f"📁 创建备份目录: {self.backup_dir}")
    
    def is_important_doc(self, file_path: str) -> bool:
        """检查是否是重要文档"""
        for important in self.important_docs:
            if file_path.startswith(important):
                return True
        return False
    
    def scan_remaining_issue_files(self):
        """扫描剩余的问题文件"""
        cleanup_files = []
        
        # 检查具体指定的文件
        for file_path in self.specific_cleanup_files:
            path = Path(file_path)
            if path.exists() and not self.is_important_doc(file_path):
                cleanup_files.append(path)
        
        # 扫描其他可能的问题文件
        for md_file in self.docs_root.rglob("*.md"):
            file_str = str(md_file)
            
            # 跳过重要文档
            if self.is_important_doc(file_str):
                continue
            
            # 检查文件名是否包含问题关键词
            filename = md_file.name.lower()
            if any(keyword in filename for keyword in [
                "总结", "报告", "修复", "问题", "issue", "fix", "summary", "report",
                "validation_report", "测试", "验证", "状态", "进度"
            ]):
                # 但排除重要的指南和规范
                if not any(keep in filename for keep in [
                    "指南", "规范", "标准", "guide", "standard", "规划", "计划", "plan"
                ]):
                    if md_file not in cleanup_files:
                        cleanup_files.append(md_file)
        
        return cleanup_files
    
    def move_to_backup(self, file_path: Path) -> bool:
        """移动文件到备份目录"""
        try:
            # 保持相对路径结构
            relative_path = file_path.relative_to(Path("."))
            backup_path = self.backup_dir / relative_path
            
            # 创建父目录
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 如果备份文件已存在，添加时间戳
            if backup_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name_parts = backup_path.name.split('.')
                if len(name_parts) > 1:
                    new_name = f"{'.'.join(name_parts[:-1])}_{timestamp}.{name_parts[-1]}"
                else:
                    new_name = f"{backup_path.name}_{timestamp}"
                backup_path = backup_path.parent / new_name
            
            # 移动文件
            shutil.move(str(file_path), str(backup_path))
            logger.info(f"📦 移动到备份: {file_path} -> {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 移动失败 {file_path}: {e}")
            return False
    
    def clean_final_docs(self, dry_run: bool = True):
        """最终清理docs"""
        logger.info(f"🧹 开始最终清理docs目录... (dry_run={dry_run})")
        
        cleanup_files = self.scan_remaining_issue_files()
        
        if not cleanup_files:
            logger.info("✅ 没有发现需要清理的问题文件")
            return
        
        logger.info(f"📋 发现 {len(cleanup_files)} 个需要清理的文件:")
        for file_path in cleanup_files:
            logger.info(f"  📄 {file_path}")
        
        if dry_run:
            logger.info("🔍 这是预览模式，没有实际删除文件")
            return
        
        # 创建备份目录
        self.create_backup_dir()
        
        # 执行清理
        moved_count = 0
        for file_path in cleanup_files:
            if self.move_to_backup(file_path):
                moved_count += 1
        
        logger.info(f"✅ 最终清理完成! 移动了 {moved_count} 个文件到备份目录")
        logger.info(f"📁 备份位置: {self.backup_dir}")
    
    def generate_final_report(self, cleanup_files):
        """生成最终清理报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"scripts/final_docs_cleanup_report_{timestamp}.md"
        
        report_content = f"""# docs目录最终清理报告
生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📊 最终清理统计
- **清理文件数**: {len(cleanup_files)}
- **备份位置**: {self.backup_dir}

## 📂 清理文件列表
"""
        
        for file_path in cleanup_files:
            report_content += f"- 📄 `{file_path}`\n"
        
        report_content += f"""
## 📋 保留的重要文档
以下类型的文档已保留：
- 架构设计文档 (docs/architecture/)
- 用户指南 (docs/user_guides/)
- 技术标准 (docs/standards/)
- 方法论文档 (docs/methodologies/)
- API文档 (docs/api_docs/)
- 生产部署指南
- 系统架构更新文档
- 形态管理规范

## 🎯 清理目标达成
✅ 移除了所有问题修复报告
✅ 移除了所有验证总结文件
✅ 移除了所有测试修复文档
✅ 保留了所有重要的指南和规范
✅ 保留了所有架构和设计文档

## 🔄 恢复指导
如需恢复文件：
```bash
cp {self.backup_dir}/docs/[path]/[filename] docs/[path]/
```

## ⚠️ 注意事项
- 所有清理的文件都已安全备份
- 重要的指南和规范文档已保留
- 建议在确认系统正常后再删除备份
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"📄 生成最终清理报告: {report_path}")
        return report_path

def main():
    """主函数"""
    logger.info("🧹 开始docs目录最终清理...")
    
    cleaner = FinalDocsCleaner()
    
    # 首先预览
    logger.info("=" * 60)
    logger.info("🔍 预览模式 - 扫描剩余的问题文件")
    logger.info("=" * 60)
    
    cleanup_files = cleaner.scan_remaining_issue_files()
    
    # 显示预览
    cleaner.clean_final_docs(dry_run=True)
    
    if cleanup_files:
        # 生成报告
        report_path = cleaner.generate_final_report(cleanup_files)
        
        response = input(f"\n是否执行最终清理? (将清理 {len(cleanup_files)} 个文件) (y/N): ").strip().lower()
        
        if response == 'y':
            logger.info("=" * 60)
            logger.info("🧹 执行最终清理")
            logger.info("=" * 60)
            
            cleaner.clean_final_docs(dry_run=False)
            logger.info(f"📄 详细报告已保存: {report_path}")
        else:
            logger.info("❌ 用户取消清理")
            logger.info(f"📄 预览报告已保存: {report_path}")

if __name__ == "__main__":
    main()
