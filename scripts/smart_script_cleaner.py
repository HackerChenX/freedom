#!/usr/bin/env python3
"""
智能脚本清理工具
专门清理可能造成混淆的过时脚本，保持系统整洁
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Set
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.logger import get_logger

logger = get_logger(__name__)

class SmartScriptCleaner:
    """智能脚本清理器"""
    
    def __init__(self):
        self.scripts_dir = Path("scripts")
        self.backup_dir = Path("scripts/deprecated_backup")
        
        # 需要保留的核心脚本
        self.core_scripts = {
            # 核心测试脚本
            "unified_indicator_quality_monitor.py",
            "comprehensive_indicator_quality_monitor.py",
            
            # 有效的验证脚本（从测试报告中提取）
            "validate_enhanced_macd_trend.py",
            "validate_enhanced_boll_indicators.py", 
            "validate_enhanced_kdj.py",
            "validate_enhanced_wr.py",
            "validate_rsi_derivatives.py",
            "validate_vol.py",
            "validate_volume_score.py",
            "validate_trend_indicators.py",
            "validate_score_indicators.py",
            "validate_fibonacci_tools.py",
            "validate_market_env.py",
            "validate_sentiment_analysis.py",
            "validate_trend_classification.py",
            "validate_trend_strength.py",
            "validate_time_cycle_analysis.py",
            "validate_multi_period_resonance.py",
            "validate_intraday_volatility.py",
            "validate_adx_indicator.py",
            
            # 核心工具脚本
            "register_indicators.py",
            "run_strategy.py",
            "test_clickhouse_connection.py",
            "manage_db_config.py",
            "cleanup_deprecated_scripts.py",
            "smart_script_cleaner.py",
            
            # ClickHouse相关
            "start_clickhouse.sh",
            "stop_clickhouse.sh",
            "clickhouse_aliases.sh",
        }
        
        # 需要保留的目录
        self.core_directories = {
            "analysis",
            "backtest", 
            "cleanup",
            "fixes",
            "optimization",
            "risk",
            "utils",
            "validation",
            "__pycache__",
        }
        
        # 明确要清理的过时脚本模式
        self.deprecated_patterns = [
            # 调试和诊断脚本
            "debug_",
            "diagnose_",
            
            # 修复脚本（已完成修复）
            "fix_",
            "batch_fix_",
            "smart_fix_",
            "quick_fix_",
            
            # 分析脚本（已完成分析）
            "analyze_",
            "check_",
            "find_",
            "verify_",
            "count_",
            
            # 临时脚本
            "final_",
            "complete_",
            "enhanced_batch_",
            "factory_",
            "lightweight_",
            "optimized_",
            "realistic_",
            "simple_",
            "systematic_",
            
            # 特定日期的脚本
            "_20250",
            
            # 重复功能脚本
            "batch_",
            "manual_",
            "remaining_",
            "update_",
            "view_",
            
            # 测试脚本（除了核心测试）
            "test_",
        ]
        
        # 特殊处理的脚本
        self.special_scripts = {
            "comprehensive_indicator_quality_monitor.py": "深度质量检查工具，保留但标记为特殊用途",
        }
    
    def create_backup_dir(self):
        """创建备份目录"""
        if not self.backup_dir.exists():
            self.backup_dir.mkdir(parents=True)
            logger.info(f"📁 创建备份目录: {self.backup_dir}")
    
    def is_deprecated_script(self, script_path: Path) -> bool:
        """判断脚本是否过时"""
        script_name = script_path.name
        
        # 保留核心脚本
        if script_name in self.core_scripts:
            return False
        
        # 保留核心目录
        if script_path.is_dir() and script_name in self.core_directories:
            return False
        
        # 检查过时模式
        for pattern in self.deprecated_patterns:
            if pattern in script_name:
                return True
        
        return False
    
    def scan_deprecated_scripts(self) -> List[Path]:
        """扫描过时脚本"""
        deprecated_scripts = []
        
        for item in self.scripts_dir.iterdir():
            if item.name.startswith('.'):
                continue
                
            if self.is_deprecated_script(item):
                deprecated_scripts.append(item)
        
        return deprecated_scripts
    
    def move_to_backup(self, script_path: Path) -> bool:
        """移动脚本到备份目录"""
        try:
            backup_path = self.backup_dir / script_path.name
            
            # 如果备份文件已存在，添加序号
            counter = 1
            while backup_path.exists():
                name_parts = script_path.name.split('.')
                if len(name_parts) > 1:
                    new_name = f"{'.'.join(name_parts[:-1])}_{counter}.{name_parts[-1]}"
                else:
                    new_name = f"{script_path.name}_{counter}"
                backup_path = self.backup_dir / new_name
                counter += 1
            
            if script_path.is_dir():
                shutil.copytree(str(script_path), str(backup_path))
                shutil.rmtree(str(script_path))
            else:
                shutil.move(str(script_path), str(backup_path))
            
            logger.info(f"📦 移动到备份: {script_path.name} -> {backup_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 移动失败 {script_path.name}: {e}")
            return False
    
    def clean_deprecated_scripts(self, dry_run: bool = True) -> Dict[str, int]:
        """清理过时脚本"""
        logger.info(f"🔍 扫描过时脚本... (dry_run={dry_run})")
        
        deprecated_scripts = self.scan_deprecated_scripts()
        
        stats = {
            'total_found': len(deprecated_scripts),
            'files_moved': 0,
            'dirs_moved': 0,
            'errors': 0
        }
        
        if not deprecated_scripts:
            logger.info("✅ 没有发现过时脚本")
            return stats
        
        logger.info(f"📋 发现 {len(deprecated_scripts)} 个过时脚本:")
        for script in deprecated_scripts:
            script_type = "📁" if script.is_dir() else "📄"
            logger.info(f"  {script_type} {script.name}")
        
        if dry_run:
            logger.info("🔍 这是预览模式，没有实际删除文件")
            return stats
        
        # 创建备份目录
        self.create_backup_dir()
        
        # 移动过时脚本
        for script in deprecated_scripts:
            if self.move_to_backup(script):
                if script.is_dir():
                    stats['dirs_moved'] += 1
                else:
                    stats['files_moved'] += 1
            else:
                stats['errors'] += 1
        
        return stats
    
    def generate_cleanup_report(self, stats: Dict[str, int]):
        """生成清理报告"""
        report = f"""# 智能脚本清理报告

## 📊 清理统计

- **发现过时脚本**: {stats['total_found']} 个
- **移动文件**: {stats['files_moved']} 个
- **移动目录**: {stats['dirs_moved']} 个
- **错误**: {stats['errors']} 个

## 🎯 清理策略

### 保留的核心脚本
- `unified_indicator_quality_monitor.py` - 主要质量监控工具 ✅
- `comprehensive_indicator_quality_monitor.py` - 深度质量检查工具 ⚠️
- 所有有效的验证脚本 (validate_*.py) ✅
- 核心工具和配置脚本 ✅

### 清理的过时脚本类型
- 调试脚本 (debug_*, diagnose_*)
- 修复脚本 (fix_*, batch_fix_*)
- 分析脚本 (analyze_*, check_*)
- 临时脚本 (final_*, complete_*)
- 重复功能脚本

## 📁 备份位置
所有移动的脚本都保存在 `scripts/deprecated_backup/` 目录中，可以随时恢复。

## ✅ 清理效果
清理后的scripts目录更加整洁，避免了脚本混淆，提高了开发效率。

## 🔧 使用建议
- 使用 `unified_indicator_quality_monitor.py` 进行日常质量监控
- 使用 `comprehensive_indicator_quality_monitor.py` 进行深度质量检查
- 避免混淆两种不同的测试方法
"""
        
        report_path = Path("scripts/cleanup_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report.strip())
        
        logger.info(f"📄 清理报告已保存: {report_path}")

def main():
    """主函数"""
    logger.info("🧹 开始智能清理过时脚本...")
    
    cleaner = SmartScriptCleaner()
    
    # 首先预览
    logger.info("=" * 60)
    logger.info("🔍 预览模式 - 扫描过时脚本")
    logger.info("=" * 60)
    
    stats = cleaner.clean_deprecated_scripts(dry_run=True)
    
    if stats['total_found'] > 0:
        print(f"\n发现 {stats['total_found']} 个过时脚本")
        response = input("是否执行清理? (y/N): ").strip().lower()
        
        if response == 'y':
            logger.info("=" * 60)
            logger.info("🧹 执行清理")
            logger.info("=" * 60)
            
            final_stats = cleaner.clean_deprecated_scripts(dry_run=False)
            cleaner.generate_cleanup_report(final_stats)
            
            logger.info("✅ 清理完成!")
            logger.info(f"📦 移动了 {final_stats['files_moved']} 个文件和 {final_stats['dirs_moved']} 个目录")
        else:
            logger.info("❌ 用户取消清理")
    else:
        logger.info("✅ 没有需要清理的脚本")

if __name__ == "__main__":
    main()
