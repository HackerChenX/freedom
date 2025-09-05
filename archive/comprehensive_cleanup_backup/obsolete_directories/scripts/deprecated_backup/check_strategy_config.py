#!/usr/bin/env python3
"""
检查策略配置生成

验证指标验证框架生成的策略配置是否符合策略执行器的要求
"""

import os
import sys
import json

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from analysis.engines.indicator_validation_framework import (
    IndicatorValidationFramework, 
    IndicatorValidationConfig, 
    ValidationMode
)
from utils.logger import get_logger

logger = get_logger(__name__)


def check_strategy_config():
    """检查策略配置生成"""
    print("🔍 检查策略配置生成")
    print("=" * 50)
    
    try:
        # 创建验证框架
        config = IndicatorValidationConfig(
            mode=ValidationMode.QUICK,
            debug_mode=True
        )
        
        framework = IndicatorValidationFramework(config)
        print(f"✅ 验证框架创建成功")
        
        # 测试几个核心指标的策略生成
        test_indicators = ['MA', 'MACD', 'RSI', 'KDJ', 'BOLL', 'EMA']
        
        for indicator in test_indicators:
            print(f"\n📊 测试指标: {indicator}")
            
            try:
                # 生成策略配置
                strategy_config = framework._generate_indicator_strategy(indicator)
                
                # 检查必要字段
                required_fields = ['strategy_id', 'name', 'conditions']
                missing_fields = []
                
                for field in required_fields:
                    if field not in strategy_config:
                        missing_fields.append(field)
                
                if missing_fields:
                    print(f"❌ 缺少必要字段: {missing_fields}")
                else:
                    print(f"✅ 必要字段完整")
                
                # 检查条件
                conditions = strategy_config.get('conditions', [])
                if not conditions:
                    print(f"❌ 缺少条件")
                else:
                    print(f"✅ 条件数量: {len(conditions)}")
                    
                    # 检查条件格式
                    for i, condition in enumerate(conditions):
                        print(f"   条件 {i+1}: {condition}")
                        
                        # 检查条件必要字段
                        if condition.get('type') == 'indicator':
                            if 'indicator_id' not in condition:
                                print(f"   ❌ 指标条件缺少 indicator_id")
                            if 'period' not in condition:
                                print(f"   ❌ 指标条件缺少 period")
                        elif condition.get('type') == 'basic':
                            if 'field' not in condition:
                                print(f"   ❌ 基础条件缺少 field")
                            if 'operator' not in condition:
                                print(f"   ❌ 基础条件缺少 operator")
                            if 'value' not in condition:
                                print(f"   ❌ 基础条件缺少 value")
                
                # 打印完整配置（调试用）
                if config.debug_mode:
                    print(f"   完整配置: {json.dumps(strategy_config, indent=2, ensure_ascii=False)}")
                
            except Exception as e:
                print(f"❌ 生成策略配置失败: {e}")
                logger.error(f"生成指标 {indicator} 策略配置失败", exc_info=True)
        
        print(f"\n✅ 策略配置检查完成")
        
    except Exception as e:
        print(f"❌ 检查过程出错: {e}")
        logger.error("策略配置检查失败", exc_info=True)


if __name__ == "__main__":
    check_strategy_config() 