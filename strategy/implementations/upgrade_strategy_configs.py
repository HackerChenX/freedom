from utils.container import container
from strategy.unified_base_strategy import UnifiedBaseStrategy
#!/usr/bin/env python3
"""
策略配置文件升级工具

将旧格式的策略配置文件转换为新的标准化格式
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yaml
import json
from pathlib import Path
from typing import Dict, List, Any
from utils.parameter_standardizer import ParameterStandardizer
from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


class StrategyConfigUpgrader:
    """策略配置升级器"""
    
    def __init__(self):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        self.standardizer = ParameterStandardizer()
        self.upgrade_stats = {
            'total_files': 0,
            'successful_upgrades': 0,
            'failed_upgrades': 0,
            'conditions_upgraded': 0,
            'conditions_failed': 0
        }
    
    def upgrade_strategy_file(self, file_path: str) -> bool:
        """升级单个策略配置文件"""
        try:
            print(f"\n升级策略文件: {file_path}")
            
            # 读取原始配置
            with open(file_path, 'r', encoding='utf-8') as f:
                original_config = yaml.safe_load(f)
            
            if not original_config or 'strategy' not in original_config:
                print(f"  ⚠️ 文件格式不正确，跳过")
                return False
            
            # 升级配置
            upgraded_config = self.upgrade_strategy_config(original_config)
            
            if upgraded_config:
                # 备份原文件
                backup_path = f"{file_path}.backup"
                with open(backup_path, 'w', encoding='utf-8') as f:
                    yaml.dump(original_config, f, default_flow_style=False, allow_unicode=True)
                print(f"  ✓ 原文件已备份到: {backup_path}")
                
                # 写入升级后的配置
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(upgraded_config, f, default_flow_style=False, allow_unicode=True)
                
                print(f"  ✓ 策略配置升级成功")
                self.upgrade_stats['successful_upgrades'] += 1
                return True
            else:
                print(f"  ✗ 策略配置升级失败")
                self.upgrade_stats['failed_upgrades'] += 1
                return False
                
        except Exception as e:
            print(f"  ✗ 升级文件时出错: {e}")
            self.upgrade_stats['failed_upgrades'] += 1
            return False
    
    def upgrade_strategy_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """升级策略配置"""
        try:
            strategy = config['strategy']
            
            # 升级条件列表
            if 'conditions' in strategy and strategy['conditions']:
                upgraded_conditions = []
                
                for condition in strategy['conditions']:
                    if isinstance(condition, dict) and 'indicator_id' in condition:
                        # 这是一个指标条件，需要升级
                        upgraded_condition = self.upgrade_condition(condition)
                        if upgraded_condition:
                            upgraded_conditions.append(upgraded_condition)
                            self.upgrade_stats['conditions_upgraded'] += 1
                        else:
                            # 保留原条件
                            upgraded_conditions.append(condition)
                            self.upgrade_stats['conditions_failed'] += 1
                    else:
                        # 逻辑操作符或其他，直接保留
                        upgraded_conditions.append(condition)
                
                strategy['conditions'] = upgraded_conditions
            
            return config
            
        except Exception as e:
            logger.error(f"升级策略配置时出错: {e}")
            return None
    
    def upgrade_condition(self, condition: Dict[str, Any]) -> Dict[str, Any]:
        """升级单个条件"""
        try:
            # 转换为标准化格式
            standardized_condition = {
                'type': 'indicator',
                'indicator': condition.get('indicator_id', ''),
                'signal_type': condition.get('signal_type', 'BUY'),
                'parameter': '',  # 让标准化器自动填充
                'operator': '=',
                'value': 1
            }
            
            # 添加参数
            if 'parameters' in condition:
                standardized_condition['parameters'] = condition['parameters']
            
            # 使用标准化器处理
            upgraded_condition = self.standardizer.standardize_indicator_condition(standardized_condition)
            
            # 保留原有的period信息
            if 'period' in condition:
                upgraded_condition['period'] = condition['period']
            
            return upgraded_condition
            
        except Exception as e:
            logger.warning(f"升级条件时出错: {e}")
            return None
    
    def upgrade_all_strategies(self, strategies_dir: str = "config/strategies") -> Dict[str, Any]:
        """升级所有策略配置文件"""
        print(f"🔧 开始升级策略配置文件")
        print(f"策略目录: {strategies_dir}")
        print("=" * 60)
        
        strategies_path = Path(strategies_dir)
        if not strategies_path.exists():
            print(f"❌ 策略目录不存在: {strategies_dir}")
            return self.upgrade_stats
        
        # 查找所有YAML文件
        yaml_files = list(strategies_path.glob("*.yaml")) + list(strategies_path.glob("*.yml"))
        
        self.upgrade_stats['total_files'] = len(yaml_files)
        print(f"发现 {len(yaml_files)} 个策略配置文件")
        
        # 逐个升级
        for yaml_file in yaml_files:
            self.upgrade_strategy_file(str(yaml_file))
        
        return self.upgrade_stats


def create_sample_upgraded_strategy():
    """创建一个示例升级后的策略配置"""
    sample_strategy = {
        'strategy': {
            'id': 'UPGRADED_SAMPLE_STRATEGY',
            'name': '升级后示例策略',
            'description': '展示新标准化格式的示例策略',
            'version': '2.0',
            'author': 'system',
            'create_time': '2025-06-21 23:30:00',
            'update_time': '2025-06-21 23:30:00',
            
            'conditions': [
                # KDJ金叉条件
                {
                    'type': 'indicator',
                    'indicator_id': 'KDJ',
                    'signal_type': 'BUY',
                    'parameter': 'kdj_golden_cross',
                    'operator': '=',
                    'value': 1,
                    'period': 'DAILY',
                    'parameters': {
                        'n': 9,
                        'm1': 3,
                        'm2': 3
                    }
                },
                
                {'logic': 'AND'},
                
                # MACD金叉条件
                {
                    'type': 'indicator',
                    'indicator_id': 'MACD',
                    'signal_type': 'BUY',
                    'parameter': 'macd_golden_cross',
                    'operator': '=',
                    'value': 1,
                    'period': 'DAILY',
                    'parameters': {
                        'fast_period': 12,
                        'slow_period': 26,
                        'signal_period': 9
                    }
                },
                
                {'logic': 'OR'},
                
                # RSI超卖条件
                {
                    'type': 'indicator',
                    'indicator_id': 'RSI',
                    'signal_type': 'BUY',
                    'parameter': 'rsi_oversold',
                    'operator': '=',
                    'value': 1,
                    'period': 'DAILY',
                    'parameters': {
                        'period': 14,
                        'overbought': 70.0,
                        'oversold': 30.0
                    }
                }
            ],
            
            'filters': {
                'market': [],
                'market_cap': {
                    'min': 0,
                    'max': 10000
                },
                'price': {
                    'min': 0,
                    'max': 500
                }
            },
            
            'sort': [
                {
                    'field': 'signal_strength',
                    'direction': 'DESC'
                },
                {
                    'field': 'market_cap',
                    'direction': 'ASC'
                }
            ]
        }
    }
    
    # 保存示例策略
    sample_path = "config/strategies/upgraded_sample_strategy.yaml"
    with open(sample_path, 'w', encoding='utf-8') as f:
        yaml.dump(sample_strategy, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✓ 创建示例升级策略: {sample_path}")
    return sample_path


def test_upgraded_strategy(strategy_path: str):
    """测试升级后的策略配置"""
    try:
        print(f"\n测试升级后的策略: {strategy_path}")
        
        # 读取策略配置
        with open(strategy_path, 'r', encoding='utf-8') as f:
            strategy_config = yaml.safe_load(f)
        
        # 使用标准化器验证
        standardizer = ParameterStandardizer()
        validated_strategy = standardizer.validate_and_standardize(strategy_config)
        
        if validated_strategy:
            conditions = validated_strategy['strategy']['conditions']
            indicator_conditions = [c for c in conditions if isinstance(c, dict) and 'indicator_id' in c]
            
            print(f"  ✓ 策略验证成功")
            print(f"  条件总数: {len(conditions)}")
            print(f"  指标条件: {len(indicator_conditions)}")
            
            # 显示指标条件详情
            for i, condition in enumerate(indicator_conditions, 1):
                indicator = condition.get('indicator_id')
                parameter = condition.get('parameter')
                print(f"    {i}. {indicator} -> {parameter}")
            
            return True
        else:
            print(f"  ✗ 策略验证失败")
            return False
            
    except Exception as e:
        print(f"  ✗ 测试策略时出错: {e}")
        return False


def mainUpgradestrategyconfigs():
    """主函数"""
    print("🔧 策略配置文件升级工具")
    print("=" * 60)
    
    try:
        # 创建升级器
        upgrader = StrategyConfigUpgrader()
        
        # 创建示例升级策略
        sample_path = create_sample_upgraded_strategy()
        
        # 测试示例策略
        test_upgraded_strategy(sample_path)
        
        # 升级所有策略文件
        stats = upgrader.upgrade_all_strategies()
        
        # 显示升级统计
        print("\n" + "=" * 60)
        print("📊 策略配置升级统计")
        print("=" * 60)
        print(f"总文件数: {stats['total_files']}")
        print(f"升级成功: {stats['successful_upgrades']}")
        print(f"升级失败: {stats['failed_upgrades']}")
        print(f"条件升级成功: {stats['conditions_upgraded']}")
        print(f"条件升级失败: {stats['conditions_failed']}")
        
        success_rate = (stats['successful_upgrades'] / stats['total_files']) * 100 if stats['total_files'] > 0 else 0
        print(f"成功率: {success_rate:.1f}%")
        
        if success_rate >= 80:
            grade = "✅ 优秀"
        elif success_rate >= 60:
            grade = "✅ 良好"
        else:
            grade = "⚠️ 需要改进"
        
        print(f"评级: {grade}")
        
        print("\n" + "=" * 60)
        print("✅ 策略配置文件升级完成")
        
        return stats
        
    except Exception as e:
        print(f"\n❌ 升级过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    mainUpgradestrategyconfigs()
