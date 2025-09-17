#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略配置迁移工具

用于将现有的策略配置文件迁移到统一格式
遵循六层架构规范，仅依赖L2基础设施层
"""

import os
import json
import yaml
import shutil
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import sys

from utils.logger import getLogger
from utils.decorators import performance_monitor, time_it
from utils.strategy_validator import UnifiedStrategyConfigValidator
from config.unified_config_manager import get_config
from db.sql_manager import SQLManager, QueryType

logger = getLogger(__name__)

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)


class StrategyConfigMigrator:
    """
    策略配置迁移器
    
    负责将现有的各种格式的策略配置迁移到统一格式
    """
    
    def __init__(self):
        """初始化迁移器"""
        self.config = get_config()
        self.validator = UnifiedStrategyConfigValidator()
        
        # 配置路径
        self.legacy_config_dir = os.path.join(root_dir, 'config', 'strategies')
        self.unified_config_dir = os.path.join(root_dir, 'config', 'strategies', 'standardized')
        self.backup_dir = os.path.join(root_dir, 'config', 'strategies', 'legacy')
        
        # 确保目录存在
        os.makedirs(self.unified_config_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        
        logger.info("策略配置迁移器初始化完成")
    
    @performance_monitor(threshold=5.0)
    def migrate_all_strategies(self, backup_original: bool = True) -> Dict[str, Any]:
        """
        迁移所有策略配置文件
        
        Args:
            backup_original: 是否备份原始文件
            
        Returns:
            Dict[str, Any]: 迁移结果统计
        """
        migration_result = {
            'total_files': 0,
            'migrated_successfully': 0,
            'migration_failed': 0,
            'skipped_files': 0,
            'failed_files': [],
            'migrated_files': [],
            'start_time': datetime.now().isoformat(),
            'end_time': None
        }
        
        try:
            # 获取所有策略配置文件
            strategy_files = self._get_strategy_files()
            migration_result['total_files'] = len(strategy_files)
            
            logger.info(f"开始迁移 {len(strategy_files)} 个策略配置文件")
            
            for file_path in strategy_files:
                try:
                    # 检查是否已经是统一格式
                    if self._is_unified_format(file_path):
                        migration_result['skipped_files'] += 1
                        logger.debug(f"跳过已统一格式文件: {file_path}")
                        continue
                    
                    # 执行迁移
                    migrate_result = self._migrate_single_file(file_path, backup_original)
                    
                    if migrate_result['success']:
                        migration_result['migrated_successfully'] += 1
                        migration_result['migrated_files'].append({
                            'original_file': file_path,
                            'unified_file': migrate_result['unified_file'],
                            'backup_file': migrate_result.get('backup_file')
                        })
                        logger.info(f"成功迁移策略配置: {os.path.basename(file_path)}")
                    else:
                        migration_result['migration_failed'] += 1
                        migration_result['failed_files'].append({
                            'file': file_path,
                            'error': migrate_result['error']
                        })
                        logger.error(f"迁移失败: {file_path}, 错误: {migrate_result['error']}")
                
                except Exception as e:
                    migration_result['migration_failed'] += 1
                    migration_result['failed_files'].append({
                        'file': file_path,
                        'error': str(e)
                    })
                    logger.error(f"迁移文件异常: {file_path}, 错误: {e}")
            
            migration_result['end_time'] = datetime.now().isoformat()
            
            # 生成迁移报告
            self._generate_migration_report(migration_result)
            
            logger.info(f"策略配置迁移完成，成功: {migration_result['migrated_successfully']}, "
                       f"失败: {migration_result['migration_failed']}, "
                       f"跳过: {migration_result['skipped_files']}")
            
            return migration_result
            
        except Exception as e:
            logger.error(f"策略配置迁移过程异常: {e}")
            migration_result['end_time'] = datetime.now().isoformat()
            return migration_result
    
    def _get_strategy_files(self) -> List[str]:
        """获取所有策略配置文件"""
        strategy_files = []
        
        if not os.path.exists(self.legacy_config_dir):
            logger.warning(f"策略配置目录不存在: {self.legacy_config_dir}")
            return strategy_files
        
        for root, dirs, files in os.walk(self.legacy_config_dir):
            # 跳过已统一格式的目录
            if 'standardized' in root or 'legacy' in root:
                continue
            
            for file in files:
                if file.endswith(('.yaml', '.yml', '.json')):
                    file_path = os.path.join(root, file)
                    strategy_files.append(file_path)
        
        return strategy_files
    
    def _is_unified_format(self, file_path: str) -> bool:
        """检查文件是否已经是统一格式"""
        try:
            config = self._load_config_file(file_path)
            if not config:
                return False
            
            # 检查是否包含统一格式的必要字段
            required_sections = ['strategy', 'technical_indicators', 'time_criteria', 'validation']
            for section in required_sections:
                if section not in config:
                    return False
            
            # 检查策略信息是否符合统一格式
            strategy_info = config.get('strategy', {})
            required_fields = ['id', 'name', 'description', 'version']
            for field in required_fields:
                if field not in strategy_info:
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"检查统一格式异常: {file_path}, 错误: {e}")
            return False
    
    def _migrate_single_file(self, file_path: str, backup_original: bool) -> Dict[str, Any]:
        """迁移单个策略配置文件"""
        result = {
            'success': False,
            'unified_file': None,
            'backup_file': None,
            'error': None
        }
        
        try:
            # 加载原始配置
            original_config = self._load_config_file(file_path)
            if not original_config:
                result['error'] = "无法加载原始配置文件"
                return result
            
            # 转换为统一格式
            unified_config = self._convert_to_unified_format(original_config, file_path)
            
            # 验证统一格式配置
            validation_result = self.validator.validate_strategy_config(unified_config)
            if not validation_result['is_valid']:
                result['error'] = f"统一格式验证失败: {validation_result['errors']}"
                return result
            
            # 生成统一格式文件名
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            unified_file_name = f"{base_name}_unified.json"
            unified_file_path = os.path.join(self.unified_config_dir, unified_file_name)
            
            # 保存统一格式配置
            with open(unified_file_path, 'w', encoding='utf-8') as f:
                json.dump(unified_config, f, ensure_ascii=False, indent=2)
            
            result['unified_file'] = unified_file_path
            
            # 备份原始文件
            if backup_original:
                backup_file_name = f"{base_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
                backup_file_path = os.path.join(self.backup_dir, backup_file_name)
                shutil.copy2(file_path, backup_file_path)
                result['backup_file'] = backup_file_path
            
            result['success'] = True
            return result
            
        except Exception as e:
            result['error'] = str(e)
            return result
    
    def _load_config_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """加载配置文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    return json.load(f)
                else:  # yaml/yml
                    return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {file_path}, 错误: {e}")
            return None
    
    def _convert_to_unified_format(self, original_config: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """将原始配置转换为统一格式"""
        # 生成基础策略信息
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        strategy_id = base_name.upper().replace('-', '_').replace(' ', '_')
        
        unified_config = {
            "strategy": {
                "id": strategy_id,
                "name": original_config.get('name', base_name.replace('_', ' ').title()),
                "description": original_config.get('description', f"从 {base_name} 迁移的策略"),
                "version": "1.0.0",
                "author": "system_migration",
                "category": self._infer_strategy_category(original_config),
                "risk_level": "medium",
                "created_date": datetime.now().strftime('%Y-%m-%d'),
                "updated_date": datetime.now().strftime('%Y-%m-%d')
            },
            "technical_indicators": self._convert_technical_indicators(original_config),
            "time_criteria": self._convert_time_criteria(original_config),
            "filters": self._convert_filters(original_config),
            "selection_parameters": self._convert_selection_parameters(original_config),
            "validation": {
                "enable_closed_loop": True,
                "validation_method": "entry_point_analysis",
                "validation_threshold": 0.8,
                "sample_size": 10
            },
            "performance_config": {
                "enable_cache": True,
                "parallel_processing": True,
                "max_workers": 4,
                "timeout_seconds": 300
            }
        }
        
        return unified_config
    
    def _infer_strategy_category(self, config: Dict[str, Any]) -> str:
        """推断策略类别"""
        # 简单的策略类别推断逻辑
        name = config.get('name', '').lower()
        description = config.get('description', '').lower()
        
        if any(keyword in name + description for keyword in ['macd', 'kdj', 'rsi', 'momentum']):
            return 'momentum'
        elif any(keyword in name + description for keyword in ['volume', 'vol', '成交量']):
            return 'volume'
        elif any(keyword in name + description for keyword in ['trend', '趋势']):
            return 'trend'
        elif any(keyword in name + description for keyword in ['reversal', '反转']):
            return 'reversal'
        else:
            return 'composite'
    
    def _convert_technical_indicators(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """转换技术指标配置"""
        # 这里需要根据具体的原始配置格式进行转换
        # 简化处理，提供默认配置
        return {
            "primary_indicators": [
                {
                    "indicator_id": "MACD",
                    "parameters": {
                        "fast_period": 12,
                        "slow_period": 26,
                        "signal_period": 9
                    },
                    "conditions": [
                        {
                            "field": "macd_cross_signal",
                            "operator": "cross_up",
                            "value": 0,
                            "lookback_days": 1
                        }
                    ]
                }
            ],
            "combination_logic": "AND"
        }
    
    def _convert_time_criteria(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """转换时间条件配置"""
        return {
            "time_frames": [
                {
                    "level": "daily",
                    "priority": 1,
                    "required": True
                }
            ],
            "date_range": {
                "target_date": datetime.now().strftime('%Y-%m-%d')
            },
            "market_timing": {
                "trading_session": "full_day",
                "exclude_holidays": True
            }
        }
    
    def _convert_filters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """转换过滤条件配置"""
        return {
            "market_filters": {
                "markets": ["主板", "创业板", "科创板"],
                "exclude_st": True
            },
            "financial_filters": {
                "market_cap": {
                    "min": 10,
                    "max": 5000
                },
                "max": 200.0
                }
            }
        }
    
    def _convert_selection_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """转换选股参数配置"""
        return {
            "max_selections": 50,
            "min_score": 0.6,
            "ranking_method": "score"
        }
    
    def _generate_migration_report(self, migration_result: Dict[str, Any]) -> None:
        """生成迁移报告"""
        report_file = os.path.join(self.unified_config_dir, 'migration_report.json')
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(migration_result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"迁移报告已生成: {report_file}")
            
        except Exception as e:
            logger.error(f"生成迁移报告失败: {e}")


# 便捷函数
def migrate_strategies(backup_original: bool = True) -> Dict[str, Any]:
    """
    迁移所有策略配置的便捷函数
    
    Args:
        backup_original: 是否备份原始文件
        
    Returns:
        Dict[str, Any]: 迁移结果
    """
    migrator = StrategyConfigMigrator()
    return migrator.migrate_all_strategies(backup_original)


if __name__ == "__main__":
    # 命令行执行迁移
    result = migrate_strategies()
    print(f"迁移完成，成功: {result['migrated_successfully']}, 失败: {result['migration_failed']}")
