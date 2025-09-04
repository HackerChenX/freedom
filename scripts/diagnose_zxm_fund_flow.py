#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
诊断ZXM_FUND_FLOW指标的实际状态
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_logger

logger = get_logger(__name__)


def create_test_data_with_fund_flow(length: int = 100) -> pd.DataFrame:
    """创建包含资金流向特征的测试数据"""
    np.random.seed(42)
    
    # 生成模拟股价数据
    dates = pd.date_range('2024-01-01', periods=length, freq='D')
    
    # 生成价格序列
    initial_price = 100.0
    returns = np.random.normal(0.001, 0.02, length)
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # 生成OHLC数据，特意插入资金流向特征
    close_prices = np.array(prices[1:])
    data_length = len(close_prices)
    
    high_prices = []
    low_prices = []
    open_prices = []
    volumes = []
    
    for i in range(data_length):
        close = close_prices[i]
        
        # 在多个位置插入资金流向特征
        if 20 <= i <= 30:  # 资金流入阶段
            # 价格上涨，成交量放大
            open_price = close - 0.5
            close = close_prices[i-1] + 0.8  # 持续上涨
            high_price = close + 0.3
            low_price = open_price - 0.1
            volume = np.random.lognormal(12, 0.3)  # 放量
            close_prices[i] = close
        elif 60 <= i <= 70:  # 资金流出阶段
            # 价格下跌，成交量放大
            open_price = close + 0.5
            close = close_prices[i-1] - 0.8  # 持续下跌
            high_price = open_price + 0.1
            low_price = close - 0.3
            volume = np.random.lognormal(12, 0.3)  # 放量
            close_prices[i] = close
        else:
            # 正常交易
            open_price = close + np.random.normal(0, 0.3)
            high_price = max(open_price, close) + abs(np.random.normal(0, 0.2))
            low_price = min(open_price, close) - abs(np.random.normal(0, 0.2))
            volume = np.random.lognormal(10, 0.3)  # 正常成交量
        
        open_prices.append(open_price)
        high_prices.append(high_price)
        low_prices.append(low_price)
        volumes.append(volume)
    
    data = pd.DataFrame({
        'date': dates[:data_length],
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    return data


def diagnose_zxm_fund_flow():
    """诊断ZXM_FUND_FLOW指标的实际状态"""
    logger.info("🔍 诊断ZXM_FUND_FLOW指标的实际状态...")
    
    try:
        # 创建测试数据
        test_data = create_test_data_with_fund_flow(100)
        logger.info(f"✅ 测试数据创建成功: {len(test_data)} 行")
        
        # 1. 检查指标注册情况
        logger.info("📦 检查指标注册情况...")
        try:
            from indicators.complete_indicator_registry import get_indicator_registry
            registry = get_indicator_registry()
            
            # 尝试获取ZXM_FUND_FLOW指标
            try:
                fund_flow_indicator = registry.create_indicator('ZXM_FUND_FLOW')
                logger.info(f"✅ ZXM_FUND_FLOW指标已注册: {type(fund_flow_indicator)}")
                
                # 测试指标计算
                result = fund_flow_indicator.calculate(test_data)
                logger.info(f"✅ 注册的ZXM_FUND_FLOW指标计算成功，形状: {result.shape}")
                logger.info(f"  - 列名: {list(result.columns)}")
                
                # 检查资金流向相关列
                fund_flow_columns = [col for col in result.columns if any(keyword in col.lower() for keyword in ['fund', 'flow', '资金', '流向', 'money'])]
                logger.info(f"  - 资金流向相关列: {fund_flow_columns}")
                
                if fund_flow_columns:
                    for col in fund_flow_columns:
                        if col in result.columns:
                            values = result[col]
                            if pd.api.types.is_numeric_dtype(values):
                                logger.info(f"    - {col}: 数值型，范围 [{values.min():.3f}, {values.max():.3f}]")
                            else:
                                logger.info(f"    - {col}: 类型 {values.dtype}")
                
            except Exception as e:
                logger.error(f"❌ 获取注册的ZXM_FUND_FLOW指标失败: {e}")
            
        except Exception as e:
            logger.error(f"❌ 检查指标注册失败: {e}")
        
        # 2. 检查ZXM指标实现文件
        logger.info("📦 检查ZXM指标实现文件...")
        try:
            zxm_files = [
                'indicators/zxm/fund_flow.py',
                'indicators/zxm/zxm_fund_flow.py',
                'indicators/zxm/capital_flow.py'
            ]
            
            for file_path in zxm_files:
                full_path = Path(root_dir) / file_path
                if full_path.exists():
                    logger.info(f"✅ 找到实现文件: {file_path}")
                    
                    # 尝试导入
                    try:
                        if 'fund_flow.py' in file_path:
                            from indicators.zxm.fund_flow import ZXMFundFlow
                            indicator = ZXMFundFlow()
                            result = indicator.calculate(test_data)
                            logger.info(f"  - 直接导入成功，计算结果形状: {result.shape}")
                        elif 'zxm_fund_flow.py' in file_path:
                            from indicators.zxm.zxm_fund_flow import ZXMFundFlow
                            indicator = ZXMFundFlow()
                            result = indicator.calculate(test_data)
                            logger.info(f"  - 直接导入成功，计算结果形状: {result.shape}")
                    except Exception as e:
                        logger.warning(f"  ⚠️ 直接导入失败: {e}")
                else:
                    logger.warning(f"⚠️ 实现文件不存在: {file_path}")
            
        except Exception as e:
            logger.error(f"❌ 检查实现文件失败: {e}")
        
        # 3. 手动实现资金流向分析
        logger.info("🧪 手动实现资金流向分析...")
        try:
            # 计算资金流向指标
            close_prices = test_data['close']
            high_prices = test_data['high']
            low_prices = test_data['low']
            volumes = test_data['volume']
            
            # 典型价格
            typical_price = (high_prices + low_prices + close_prices) / 3
            
            # 资金流量倍数
            money_flow_multiplier = pd.Series(index=test_data.index, dtype=float)
            for i in range(1, len(test_data)):
                if typical_price.iloc[i] > typical_price.iloc[i-1]:
                    money_flow_multiplier.iloc[i] = 1  # 资金流入
                elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                    money_flow_multiplier.iloc[i] = -1  # 资金流出
                else:
                    money_flow_multiplier.iloc[i] = 0  # 平衡
            
            # 资金流量
            money_flow = typical_price * volumes * money_flow_multiplier
            
            # 累积资金流量
            cumulative_money_flow = money_flow.cumsum()
            
            # 资金流向强度（20日移动平均）
            fund_flow_strength = money_flow.rolling(window=20).mean()
            
            # 资金流向趋势
            fund_flow_trend = pd.Series(index=test_data.index, dtype=str)
            fund_flow_trend[fund_flow_strength > 0] = '流入'
            fund_flow_trend[fund_flow_strength < 0] = '流出'
            fund_flow_trend[fund_flow_strength == 0] = '平衡'
            
            logger.info(f"  - 资金流量计算完成，范围: [{money_flow.min():.2f}, {money_flow.max():.2f}]")
            logger.info(f"  - 累积资金流量范围: [{cumulative_money_flow.min():.2f}, {cumulative_money_flow.max():.2f}]")
            logger.info(f"  - 资金流向趋势统计: {fund_flow_trend.value_counts().to_dict()}")
            
            # 检测资金流向信号
            flow_in_signals = (fund_flow_strength > fund_flow_strength.quantile(0.7)).sum()
            flow_out_signals = (fund_flow_strength < fund_flow_strength.quantile(0.3)).sum()
            
            logger.info(f"  - 资金流入信号: {flow_in_signals} 个")
            logger.info(f"  - 资金流出信号: {flow_out_signals} 个")
            
            if flow_in_signals > 0 or flow_out_signals > 0:
                logger.info("✅ 手动资金流向分析检测到有效信号")
            else:
                logger.warning("⚠️ 手动资金流向分析未检测到明显信号")
            
        except Exception as e:
            logger.error(f"❌ 手动资金流向分析失败: {e}")
        
        # 4. 检查BaseIndicator抽象方法
        logger.info("🧪 检查BaseIndicator抽象方法...")
        try:
            if 'fund_flow_indicator' in locals():
                required_methods = [
                    '_calculate_baseindicator',
                    'calculate_raw_score_Indicator_Base_Indicator',
                    'get_patterns_Indicator_Base_Indicator',
                    'calculate_confidence_Indicator_Base_Indicator',
                    'set_parameters_Indicator_Base_Indicator'
                ]
                
                missing_methods = [method for method in required_methods if not hasattr(fund_flow_indicator, method)]
                
                if not missing_methods:
                    logger.info("  ✅ BaseIndicator抽象方法完整")
                else:
                    logger.warning(f"  ⚠️ 缺少方法: {missing_methods}")
            else:
                logger.warning("  ⚠️ 无法检查BaseIndicator方法，指标未成功创建")
                
        except Exception as e:
            logger.error(f"❌ BaseIndicator方法检查失败: {e}")
        
        logger.info("🎯 ZXM_FUND_FLOW指标诊断完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 诊断过程中发生异常: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return False


def main():
    """主函数"""
    success = diagnose_zxm_fund_flow()
    
    if success:
        logger.info("✅ ZXM_FUND_FLOW指标诊断完成")
    else:
        logger.error("❌ ZXM_FUND_FLOW指标诊断失败")
    
    return success


if __name__ == "__main__":
    main()
