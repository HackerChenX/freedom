#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
优化版指标验证器
解决重复查询股票池和无限循环的问题
"""

import os
import sys
import logging
import argparse
import json
from datetime import datetime
from typing import Dict, List, Any

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from analysis.engines.indicator_validation_framework import (
    IndicatorValidationFramework, 
    IndicatorValidationConfig,
    ValidationMode
)

logger = get_logger(__name__)

class OptimizedIndicatorValidator:
    """优化版指标验证器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化验证器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path or "config/indicator_closed_loop_config.json"
        self.config = self._load_config()
        self.framework = None
        
        # 缓存控制
        self._stock_pool_cache = None
        self._cache_date = None
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"✅ 加载配置文件: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ 加载配置文件失败: {e}")
            # 返回默认配置
            return {
                "validation": {
                    "date": "2024-12-28",
                    "stock_pool_size": 4378,
                    "max_selection_ratio": 0.1,
                    "min_selection_count": 1,
                    "stop_on_success": True,
                    "stop_on_error": True,
                    "parallel_workers": 1,
                    "timeout_seconds": 300,
                    "debug_mode": True
                }
            }
    
    def _create_validation_config(self) -> IndicatorValidationConfig:
        """创建验证配置"""
        validation_config = self.config.get("validation", {})
        
        config = IndicatorValidationConfig()
        config.mode = ValidationMode.PRIORITY
        config.validation_date = validation_config.get("date", "2024-12-28")
        config.stock_pool_size = validation_config.get("stock_pool_size", 4378)
        config.max_selection_ratio = validation_config.get("max_selection_ratio", 0.1)
        config.min_selection_count = validation_config.get("min_selection_count", 1)
        config.timeout_seconds = validation_config.get("timeout_seconds", 300)
        config.parallel_workers = validation_config.get("parallel_workers", 1)
        config.stop_on_success = validation_config.get("stop_on_success", True)
        config.stop_on_error = validation_config.get("stop_on_error", True)
        config.debug_mode = validation_config.get("debug_mode", True)
        config.save_details = True
        
        return config
    
    def validate_single_indicator(self, indicator_name: str) -> Dict[str, Any]:
        """
        验证单个指标（优化版本）
        
        Args:
            indicator_name: 指标名称
            
        Returns:
            验证结果
        """
        logger.info(f"🚀 开始验证指标: {indicator_name}")
        
        try:
            # 创建验证配置
            config = self._create_validation_config()
            
            # 创建验证框架
            self.framework = IndicatorValidationFramework(config)
            
            # 执行验证
            start_time = datetime.now()
            result = self.framework.validate_single_indicator(indicator_name)
            end_time = datetime.now()
            
            # 计算执行时间
            execution_time = (end_time - start_time).total_seconds()
            result['execution_time'] = execution_time
            
            # 打印结果摘要
            self._print_result_summary(indicator_name, result, execution_time)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 验证指标 {indicator_name} 失败: {e}")
            return {
                'indicator_name': indicator_name,
                'status': 'error',
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _print_result_summary(self, indicator_name: str, result: Dict[str, Any], execution_time: float):
        """打印单个结果摘要"""
        status = result.get('status', 'unknown')
        selected_count = result.get('selected_count', 0)
        
        print("\n" + "="*60)
        print(f"📊 指标验证结果: {indicator_name}")
        print("="*60)
        print(f"状态: {status}")
        print(f"选股数量: {selected_count}")
        print(f"执行时间: {execution_time:.2f}秒")
        
        if status == 'success' and result.get('selected_stocks'):
            stocks = result['selected_stocks'][:5]  # 只显示前5只
            print(f"选中股票示例: {stocks}")
        
        if result.get('error_message'):
            print(f"错误信息: {result['error_message']}")
        
        print("="*60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="优化版指标验证器")
    parser.add_argument("--indicator", type=str, help="验证单个指标")
    parser.add_argument("--config", type=str, help="配置文件路径")
    
    args = parser.parse_args()
    
    try:
        # 创建验证器
        validator = OptimizedIndicatorValidator(args.config)
        
        if args.indicator:
            # 验证单个指标
            result = validator.validate_single_indicator(args.indicator)
        else:
            # 默认验证MA指标
            logger.info("未指定指标，使用默认MA指标")
            result = validator.validate_single_indicator('MA')
        
        logger.info("🎉 验证完成")
        
    except KeyboardInterrupt:
        logger.warning("🛑 用户中断验证")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 验证过程出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
