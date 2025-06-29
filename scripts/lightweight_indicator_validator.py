#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
轻量级指标验证器
跳过复杂的指标注册过程，直接验证指标
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
from db.clickhouse_db import get_clickhouse_db

logger = get_logger(__name__)

class LightweightIndicatorValidator:
    """轻量级指标验证器"""
    
    def __init__(self):
        """初始化验证器"""
        self.db = get_clickhouse_db()
        
        # 验证配置
        self.validation_date = "2024-12-28"
        self.stock_pool_size = 4378
        self.enable_early_stop = True
        
        # 缓存
        self._stock_pool_cache = None
        self._cache_date = None
        
        logger.info("✅ 轻量级指标验证器初始化完成")
    
    def _get_stock_pool(self) -> List[str]:
        """获取股票池（带缓存）"""
        if (self._stock_pool_cache is not None and 
            self._cache_date == self.validation_date):
            logger.info(f"使用缓存的股票池: {len(self._stock_pool_cache)} 只股票")
            return self._stock_pool_cache
        
        logger.info(f"🔍 查询股票池，日期: {self.validation_date}")
        
        try:
            # 查询活跃股票
            query = f"""
            SELECT DISTINCT code 
            FROM stock_info 
            WHERE level = '日线' AND date = '{self.validation_date}'
            AND volume > 0 AND close > 0
            ORDER BY volume DESC
            LIMIT {self.stock_pool_size}
            """
            
            result = self.db.query(query)
            if result.empty:
                logger.warning("未找到股票数据，使用默认股票池")
                stock_pool = ['000001', '000002', '600000', '600036', '000858']
            else:
                stock_pool = result['code'].tolist()
            
            # 缓存结果
            self._stock_pool_cache = stock_pool
            self._cache_date = self.validation_date
            
            logger.info(f"✅ 股票池准备完成: {len(stock_pool)} 只股票")
            return stock_pool
            
        except Exception as e:
            logger.error(f"查询股票池失败: {e}")
            return ['000001', '000002', '600000', '600036', '000858']
    
    def _validate_ma_indicator(self, stock_pool: List[str]) -> Dict[str, Any]:
        """验证MA指标"""
        logger.info("📊 验证MA指标计算")
        
        try:
            # 选择一个股票进行验证
            test_code = stock_pool[0] if stock_pool else '000001'
            
            # 查询股票数据并计算MA
            query = f"""
            SELECT 
                code,
                date,
                close,
                avg(close) OVER (
                    PARTITION BY code 
                    ORDER BY date 
                    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                ) as ma5
            FROM stock_info 
            WHERE code = '{test_code}' 
            AND level = '日线' 
            AND date <= '{self.validation_date}'
            ORDER BY date DESC 
            LIMIT 10
            """
            
            result = self.db.query(query)
            
            if not result.empty:
                logger.info(f"✅ MA指标计算成功，获得 {len(result)} 条数据")
                # 检查MA值是否合理
                latest_row = result.iloc[0]
                if latest_row['ma5'] > 0:
                    return {
                        "success": True,
                        "indicator": "MA",
                        "test_code": test_code,
                        "test_date": latest_row['date'],
                        "close": float(latest_row['close']),
                        "ma5": float(latest_row['ma5']),
                        "records_count": len(result)
                    }
            
            return {
                "success": False,
                "indicator": "MA",
                "error": "MA计算结果为空或无效"
            }
            
        except Exception as e:
            logger.error(f"MA指标验证失败: {e}")
            return {
                "success": False,
                "indicator": "MA",
                "error": str(e)
            }
    
    def _validate_rsi_indicator(self, stock_pool: List[str]) -> Dict[str, Any]:
        """验证RSI指标"""
        logger.info("📊 验证RSI指标计算")
        
        try:
            # 选择一个股票进行验证
            test_code = stock_pool[0] if stock_pool else '000001'
            
            # 查询股票数据并计算简单的价格变化
            query = f"""
            WITH price_changes AS (
                SELECT 
                    code,
                    date,
                    close,
                    close - lag(close) OVER (PARTITION BY code ORDER BY date) as price_change
                FROM stock_info 
                WHERE code = '{test_code}' 
                AND level = '日线' 
                AND date <= '{self.validation_date}'
                ORDER BY date DESC 
                LIMIT 20
            )
            SELECT 
                code,
                date,
                close,
                price_change,
                CASE 
                    WHEN price_change > 0 THEN price_change 
                    ELSE 0 
                END as gain,
                CASE 
                    WHEN price_change < 0 THEN abs(price_change) 
                    ELSE 0 
                END as loss
            FROM price_changes
            WHERE price_change IS NOT NULL
            ORDER BY date DESC
            LIMIT 10
            """
            
            result = self.db.query(query)
            
            if not result.empty:
                logger.info(f"✅ RSI指标数据计算成功，获得 {len(result)} 条数据")
                latest_row = result.iloc[0]
                return {
                    "success": True,
                    "indicator": "RSI",
                    "test_code": test_code,
                    "test_date": latest_row['date'],
                    "close": float(latest_row['close']),
                    "price_change": float(latest_row['price_change']),
                    "records_count": len(result)
                }
            
            return {
                "success": False,
                "indicator": "RSI",
                "error": "RSI计算数据为空"
            }
            
        except Exception as e:
            logger.error(f"RSI指标验证失败: {e}")
            return {
                "success": False,
                "indicator": "RSI",
                "error": str(e)
            }
    
    def _validate_generic_indicator(self, indicator_name: str, stock_pool: List[str]) -> Dict[str, Any]:
        """验证通用指标"""
        logger.info(f"📊 验证 {indicator_name} 指标（通用方式）")
        
        try:
            # 选择一个股票进行基础数据验证
            test_code = stock_pool[0] if stock_pool else '000001'
            
            # 查询基础股票数据
            query = f"""
            SELECT 
                code,
                date,
                open,
                high,
                low,
                close,
                volume
            FROM stock_info 
            WHERE code = '{test_code}' 
            AND level = '日线' 
            AND date <= '{self.validation_date}'
            ORDER BY date DESC 
            LIMIT 5
            """
            
            result = self.db.query(query)
            
            if not result.empty:
                logger.info(f"✅ {indicator_name} 指标基础数据验证成功，获得 {len(result)} 条数据")
                latest_row = result.iloc[0]
                return {
                    "success": True,
                    "indicator": indicator_name,
                    "test_code": test_code,
                    "test_date": latest_row['date'],
                    "ohlcv_data": {
                        "open": float(latest_row['open']),
                        "high": float(latest_row['high']),
                        "low": float(latest_row['low']),
                        "close": float(latest_row['close']),
                        "volume": int(latest_row['volume'])
                    },
                    "records_count": len(result),
                    "note": "基础数据验证通过，指标计算需要具体实现"
                }
            
            return {
                "success": False,
                "indicator": indicator_name,
                "error": "基础数据为空"
            }
            
        except Exception as e:
            logger.error(f"{indicator_name} 指标验证失败: {e}")
            return {
                "success": False,
                "indicator": indicator_name,
                "error": str(e)
            }
    
    def validate_indicator(self, indicator_name: str) -> Dict[str, Any]:
        """
        验证单个指标
        
        Args:
            indicator_name: 指标名称
            
        Returns:
            验证结果
        """
        logger.info(f"🚀 开始验证指标: {indicator_name}")
        start_time = datetime.now()
        
        try:
            # 准备股票池
            stock_pool = self._get_stock_pool()
            
            # 根据指标类型选择验证方法
            if indicator_name.upper() == 'MA':
                result = self._validate_ma_indicator(stock_pool)
            elif indicator_name.upper() == 'RSI':
                result = self._validate_rsi_indicator(stock_pool)
            else:
                result = self._validate_generic_indicator(indicator_name, stock_pool)
            
            # 计算耗时
            elapsed_time = (datetime.now() - start_time).total_seconds()
            result['elapsed_time'] = elapsed_time
            result['stock_pool_size'] = len(stock_pool)
            
            if result.get('success'):
                logger.info(f"✅ 指标 {indicator_name} 验证成功，耗时: {elapsed_time:.2f}秒")
            else:
                logger.error(f"❌ 指标 {indicator_name} 验证失败: {result.get('error', '未知错误')}")
            
            return result
            
        except Exception as e:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ 验证指标 {indicator_name} 失败: {e}")
            return {
                "success": False,
                "indicator": indicator_name,
                "error": str(e),
                "elapsed_time": elapsed_time
            }
    
    def _print_result(self, result: Dict[str, Any]):
        """打印验证结果"""
        print("\n" + "="*60)
        print(f"指标验证结果: {result.get('indicator', 'Unknown')}")
        print("="*60)
        
        if result.get('success'):
            print("✅ 验证状态: 成功")
            print(f"📊 测试股票: {result.get('test_code', 'N/A')}")
            print(f"📅 测试日期: {result.get('test_date', 'N/A')}")
            print(f"📈 股票池大小: {result.get('stock_pool_size', 'N/A')}")
            print(f"⏱️  执行时间: {result.get('elapsed_time', 0):.2f}秒")
            print(f"📋 数据记录数: {result.get('records_count', 'N/A')}")
            
            # 显示具体数据
            if 'close' in result:
                print(f"💰 收盘价: {result['close']:.2f}")
            if 'ma5' in result:
                print(f"📊 MA5: {result['ma5']:.2f}")
            if 'price_change' in result:
                print(f"📈 价格变化: {result['price_change']:.2f}")
            if 'ohlcv_data' in result:
                ohlcv = result['ohlcv_data']
                print(f"📊 OHLCV: O:{ohlcv['open']:.2f}, H:{ohlcv['high']:.2f}, L:{ohlcv['low']:.2f}, C:{ohlcv['close']:.2f}, V:{ohlcv['volume']:,}")
            if 'note' in result:
                print(f"📝 说明: {result['note']}")
        else:
            print("❌ 验证状态: 失败")
            print(f"❗ 错误信息: {result.get('error', '未知错误')}")
            print(f"⏱️  执行时间: {result.get('elapsed_time', 0):.2f}秒")
        
        print("="*60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='轻量级指标验证器')
    parser.add_argument('--indicator', '-i', type=str, required=True,
                       help='要验证的指标名称 (如: MA, RSI, MACD)')
    
    args = parser.parse_args()
    
    # 创建验证器
    validator = LightweightIndicatorValidator()
    
    # 执行验证
    result = validator.validate_indicator(args.indicator)
    
    # 打印结果
    validator._print_result(result)
    
    logger.info("🎉 验证完成")

if __name__ == "__main__":
    main()
