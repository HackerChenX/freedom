#!/usr/bin/env python3
"""
最终清理剩余过时脚本
专门清理那些仍然存在的过时脚本
"""

import os
import shutil
from pathlib import Path
from typing import List, Set
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.logger import get_logger

logger = get_logger(__name__)

class FinalScriptCleaner:
    """最终脚本清理器"""
    
    def __init__(self):
        self.scripts_dir = Path("scripts")
        self.backup_dir = Path("scripts/final_cleanup_backup")
        
        # 绝对需要保留的核心脚本
        self.essential_scripts = {
            # 核心测试脚本
            "unified_indicator_quality_monitor.py",
            
            # 核心工具脚本
            "register_indicators.py",
            "run_strategy.py",
            "test_clickhouse_connection.py",
            "manage_db_config.py",
            
            # ClickHouse相关
            "start_clickhouse.sh",
            "stop_clickhouse.sh",
            "clickhouse_aliases.sh",
            "start_clickhouse_compose.sh",
            "start_clickhouse_docker.sh",
            
            # 清理脚本
            "cleanup_deprecated_scripts.py",
            "smart_script_cleaner.py",
            "final_cleanup_remaining_scripts.py",
            
            # 回测相关
            "zxm_kdj_strategy_demo.py",
            
            # 数据相关
            "akshare_to_clickhouse.py",
        }
        
        # 需要保留的目录
        self.essential_directories = {
            "analysis",
            "backtest", 
            "cleanup",
            "fixes",
            "optimization",
            "risk",
            "utils",
            "validation",
            "debug",
            "sql",
            "__pycache__",
            "deprecated_backup",
            "final_cleanup_backup",
        }
        
        # 需要清理的过时脚本（明确指定）
        self.scripts_to_remove = {
            # 添加方法的脚本（已完成）
            "add_get_pattern_info_methods.py",
            "add_minimum_periods_to_indicators.py", 
            "add_zxm_get_pattern_info.py",
            
            # 架构检查脚本（已完成）
            "architecture_checker.py",
            "architecture_compliance_check.py",
            
            # 自动化风险检测（已完成）
            "automated_risk_detection.py",
            
            # 基准测试脚本（已完成）
            "benchmark_unified_indicator_engine.py",
            
            # 缓存优化分析（已完成）
            "cache_optimization_analyzer.py",
            
            # 代码重复检查（已完成）
            "code_duplication_checker.py",
            
            # 比较分析引擎（已完成）
            "compare_analysis_engines.py",
            
            # 综合指标分析（已完成）
            "comprehensive_indicator_analysis.py",
            "comprehensive_indicator_tester.py",
            "comprehensive_system_check.py",
            "comprehensive_system_test.py",
            "comprehensive_unified_engine_test.py",
            
            # 持续质量保证（已完成）
            "continuous_quality_assurance.py",
            
            # 正确验证分析（已完成）
            "correct_validation_analysis.py",
            
            # 数据库优化（已完成）
            "database_optimization.py",
            
            # 增强模式指标（已完成）
            "enhance_pattern_indicators.py",
            
            # 执行策略（已完成）
            "execute_zxm_absorb_volume_strategy.py",
            
            # 扩展股票池测试（已完成）
            "expanded_stock_pool_tester.py",
            
            # 生成闭环摘要（已完成）
            "generate_closed_loop_summary.py",
            
            # 历史信号验证（已完成）
            "historical_signal_validator.py",
            
            # 识别缺失指标（已完成）
            "identify_missing_indicators.py",
            
            # 实现模式指标增强（已完成）
            "implement_pattern_indicators_enhancement.py",
            
            # 指标相关脚本（已完成）
            "indicator_closed_loop_validator.py",
            "indicator_generator.py",
            "indicator_logic_validator.py",
            "indicator_multi_pattern_validator.py",
            "indicator_optimization_plan.py",
            "indicator_validation_demo.py",
            "indicator_validation_framework.py",
            
            # MACD指标修复（已完成）
            "macd_indicator_repairer.py",
            
            # 迁移脚本（已完成）
            "migrate_patterns.py",
            "migrate_strategy_configs.py",
            
            # 命名约定检查（已完成）
            "naming_convention_checker.py",
            
            # 优化策略（已完成）
            "optimize_strategy.py",
            
            # 模式相关脚本（已完成）
            "pattern_name_standardizer.py",
            "pattern_polarity_classifier.py",
            
            # 性能基准（已完成）
            "performance_benchmark.py",
            
            # 极性验证（已完成）
            "polarity_validation.py",
            
            # 精确指标检查（已完成）
            "precise_indicator_check.py",
            
            # 生产相关脚本（已完成）
            "production_database_test.py",
            "production_indicator_validator.py",
            "production_quality_monitor.py",
            "production_strategy_validator.py",
            
            # 项目文件重组（已完成）
            "project_file_reorganization.py",
            
            # 快速验证脚本（已完成）
            "quick_indicator_verification.py",
            "quick_quality_check.py",
            "quick_validate_high_priority.py",
            
            # 真实数据反向验证（已完成）
            "real_data_reverse_validation.py",
            
            # 重新生成报告（已完成）
            "regenerate_buypoint_report.py",
            "rerun_buypoint_analysis.py",
            
            # 反向验证（已完成）
            "reverse_validation.py",
            
            # 运行测试脚本（已完成）
            "run_all_tests.py",
            "run_full_quality_test.py",
            "run_tests.py",
            
            # 智能指标匹配（已完成）
            "smart_indicator_matching.py",
            
            # 独立配置测试（已完成）
            "standalone_config_test.py",
            
            # 启动统一引擎测试（已完成）
            "start_unified_engine_test.py",
            
            # 系统清理工具（已完成）
            "system_cleanup_tool.py",
            "system_integration_test.py",
            
            # ClickHouse连接摘要（已完成）
            "clickhouse_connection_summary.py",
            
            # 清理报告（已生成）
            "cleanup_report.md",
        }
        
        # 需要清理的验证脚本（大部分已完成验证）
        self.validation_scripts_to_remove = {
            # 深度优化验证脚本（已完成）
            "validate_adx_deep_optimization_99.py",
            "validate_adx_indicator_strict_99.py",
            "validate_mfi_five_stage_99.py",
            "validate_obv_five_stage_99.py",
            "validate_roc_deep_optimization_99.py",
            "validate_vix_deep_optimization_99.py",
            "validate_vix_indicator_strict.py",
            "validate_mtm_indicator_strict.py",
            "validate_synergy_indicator_strict.py",
            "validate_unified_ma_indicator_strict.py",
            
            # 基础验证脚本（已完成）
            "validate_base_zxm_indicator.py",
            "validate_chip_distribution.py",
            "validate_buypoint_strategy.py",
            "validate_buy_point_indicators.py",
            
            # ZXM指标当前状态（已完成）
            "validate_zxm_indicators_current_status.py",
            "validate_zxm_patterns_simple.py",
            
            # 技术标准验证（已完成）
            "validate_technical_standards.py",
        }
    
    def create_backup_dir(self):
        """创建备份目录"""
        if not self.backup_dir.exists():
            self.backup_dir.mkdir(parents=True)
            logger.info(f"📁 创建备份目录: {self.backup_dir}")
    
    def should_remove_script(self, script_path: Path) -> bool:
        """判断脚本是否应该被移除"""
        script_name = script_path.name
        
        # 保留核心脚本
        if script_name in self.essential_scripts:
            return False
        
        # 保留核心目录
        if script_path.is_dir() and script_name in self.essential_directories:
            return False
        
        # 移除明确指定的脚本
        if script_name in self.scripts_to_remove:
            return True
        
        # 移除明确指定的验证脚本
        if script_name in self.validation_scripts_to_remove:
            return True
        
        return False
    
    def scan_scripts_to_remove(self) -> List[Path]:
        """扫描需要移除的脚本"""
        scripts_to_remove = []
        
        for item in self.scripts_dir.iterdir():
            if item.name.startswith('.'):
                continue
                
            if self.should_remove_script(item):
                scripts_to_remove.append(item)
        
        return scripts_to_remove
    
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
            
            logger.info(f"📦 移动到备份: {script_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 移动失败 {script_path.name}: {e}")
            return False
    
    def clean_scripts(self, dry_run: bool = True):
        """清理脚本"""
        logger.info(f"🔍 扫描需要清理的脚本... (dry_run={dry_run})")
        
        scripts_to_remove = self.scan_scripts_to_remove()
        
        if not scripts_to_remove:
            logger.info("✅ 没有发现需要清理的脚本")
            return
        
        logger.info(f"📋 发现 {len(scripts_to_remove)} 个需要清理的脚本:")
        for script in scripts_to_remove:
            script_type = "📁" if script.is_dir() else "📄"
            logger.info(f"  {script_type} {script.name}")
        
        if dry_run:
            logger.info("🔍 这是预览模式，没有实际删除文件")
            return
        
        # 创建备份目录
        self.create_backup_dir()
        
        # 移动脚本
        moved_count = 0
        for script in scripts_to_remove:
            if self.move_to_backup(script):
                moved_count += 1
        
        logger.info(f"✅ 清理完成! 移动了 {moved_count} 个脚本")

def main():
    """主函数"""
    logger.info("🧹 开始最终清理剩余过时脚本...")
    
    cleaner = FinalScriptCleaner()
    
    # 首先预览
    logger.info("=" * 60)
    logger.info("🔍 预览模式 - 扫描需要清理的脚本")
    logger.info("=" * 60)
    
    cleaner.clean_scripts(dry_run=True)
    
    response = input("\n是否执行清理? (y/N): ").strip().lower()
    
    if response == 'y':
        logger.info("=" * 60)
        logger.info("🧹 执行清理")
        logger.info("=" * 60)
        
        cleaner.clean_scripts(dry_run=False)
    else:
        logger.info("❌ 用户取消清理")

if __name__ == "__main__":
    main()
