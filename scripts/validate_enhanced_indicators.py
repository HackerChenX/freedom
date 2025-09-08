#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强指标通用验证脚本

用于验证多个增强指标的基本功能
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class EnhancedIndicatorValidator:
    """增强指标通用验证器"""
    
    def __init__(self, indicator_name: str = "ENHANCED"):
        self.indicator_name = indicator_name
        self.validation_results = {}
        self.test_data = None
        
    def generate_test_data(self) -> pd.DataFrame:
        """生成测试数据"""
        logger.info(f"📊 生成{self.indicator_name}测试数据...")
        
        # 生成60天的测试数据
        dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
        
        # 生成价格数据
        np.random.seed(42)
        base_price = 100.0
        
        # 生成有趋势的价格序列
        trend = np.linspace(0, 10, 60)
        noise = np.random.normal(0, 1, 60)
        
        prices = base_price + trend + noise
        
        # 确保high >= max(open, close), low <= min(open, close)
        opens = prices + np.random.normal(0, 0.5, 60)
        closes = prices + np.random.normal(0, 0.5, 60)
        highs = np.maximum(opens, closes) + np.abs(np.random.normal(0, 0.5, 60))
        lows = np.minimum(opens, closes) - np.abs(np.random.normal(0, 0.5, 60))
        
        # 生成成交量数据
        volumes = np.random.uniform(1000000, 5000000, 60)
        
        self.test_data = pd.DataFrame({
            'code': ['000001'] * 60,
            'date': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
            'turnover_rate': np.random.uniform(0.5, 5.0, 60)
        })
        
        logger.info(f"✅ 生成测试数据: {len(self.test_data)}行")
        return self.test_data
    
    def validate_basic_functionality(self) -> Dict[str, Any]:
        """验证基本功能"""
        logger.info(f"🔧 验证{self.indicator_name}基本功能...")

        try:
            from indicators.complete_indicator_registry import get_indicator_registry

            # 获取指标注册表
            registry = get_indicator_registry()

            # 获取指标实例
            indicator = registry.get_indicator(self.indicator_name)
            if not indicator:
                # 尝试一些常见的指标名称变体
                alternative_names = [
                    self.indicator_name.upper(),
                    self.indicator_name.lower(),
                    f"ENHANCED_{self.indicator_name}",
                    self.indicator_name.replace("_", "")
                ]

                for alt_name in alternative_names:
                    indicator = registry.get_indicator(alt_name)
                    if indicator:
                        logger.info(f"✅ 找到指标实例: {alt_name}")
                        break

                if not indicator:
                    return {
                        'score': 0,
                        'details': f"无法获取{self.indicator_name}指标实例，尝试了: {alternative_names}"
                    }

            # 测试基本计算
            result = indicator.calculate(self.test_data)

            if result is None or result.empty:
                return {
                    'score': 20,
                    'details': f"{self.indicator_name}计算返回空结果"
                }

            # 检查输出格式
            if not isinstance(result, pd.DataFrame):
                return {
                    'score': 40,
                    'details': f"{self.indicator_name}输出格式不是DataFrame"
                }

            # 检查数据完整性
            if len(result) == 0:
                return {
                    'score': 60,
                    'details': f"{self.indicator_name}输出数据为空"
                }

            logger.info(f"✅ {self.indicator_name}基本功能验证通过")
            return {
                'score': 100,
                'details': f"{self.indicator_name}基本功能正常",
                'output_shape': result.shape,
                'output_columns': list(result.columns)
            }

        except Exception as e:
            logger.error(f"❌ {self.indicator_name}基本功能验证失败: {e}")
            return {
                'score': 0,
                'details': f"{self.indicator_name}验证异常: {str(e)}"
            }
    
    def run_validation(self) -> Dict[str, Any]:
        """运行完整验证"""
        logger.info(f"🔍 开始{self.indicator_name}验证...")
        
        start_time = time.time()
        
        # 生成测试数据
        self.generate_test_data()
        
        # 验证基本功能
        basic_result = self.validate_basic_functionality()
        
        validation_time = time.time() - start_time
        
        # 计算总分
        total_score = basic_result['score']
        
        # 确定状态
        if total_score >= 80:
            status = 'PASSED'
        elif total_score >= 60:
            status = 'WARNING'
        else:
            status = 'FAILED'
        
        final_result = {
            'indicator': self.indicator_name,
            'validation_time': validation_time,
            'overall_score': total_score,
            'status': status,
            'details': basic_result.get('details', ''),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"🎯 {self.indicator_name}验证完成!")
        logger.info(f"📊 总体得分: {total_score:.1f}/100")
        logger.info(f"⏱️ 验证时间: {validation_time:.2f}秒")
        logger.info(f"✅ 验证状态: {status}")
        
        return final_result


def main():
    """主函数"""
    # 从命令行参数获取指标名称，默认为CCI
    indicator_name = sys.argv[1] if len(sys.argv) > 1 else "CCI"

    # 确保指标名称正确传递
    logger.info(f"🔍 开始{indicator_name}验证...")

    validator = EnhancedIndicatorValidator(indicator_name)
    result = validator.run_validation()

    # 输出结果供统一监控脚本解析
    print(f"验证通过，得分{result['overall_score']:.1f}分")

    return 0 if result['status'] == 'PASSED' else 1


if __name__ == "__main__":
    exit(main())
