#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
诊断ZXM_INSTITUTION_BEHAVIOR指标的实际状态
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


def create_test_data_with_institution_behavior(length: int = 100) -> pd.DataFrame:
    """创建包含机构行为特征的测试数据"""
    np.random.seed(42)
    
    # 生成模拟股价数据
    dates = pd.date_range('2024-01-01', periods=length, freq='D')
    
    # 生成价格序列
    initial_price = 100.0
    returns = np.random.normal(0.001, 0.02, length)
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # 生成OHLC数据，特意插入机构行为特征
    close_prices = np.array(prices[1:])
    data_length = len(close_prices)
    
    high_prices = []
    low_prices = []
    open_prices = []
    volumes = []
    
    for i in range(data_length):
        close = close_prices[i]
        
        # 在多个位置插入机构行为特征
        if 20 <= i <= 30:  # 机构建仓阶段
            # 价格缓慢上涨，成交量逐步放大，机构悄悄建仓
            open_price = close - 0.3
            close = close_prices[i-1] + 0.4  # 缓慢上涨
            high_price = close + 0.2
            low_price = open_price - 0.1
            volume = np.random.lognormal(11.2 + (i-20)*0.1, 0.2)  # 逐步放量
            close_prices[i] = close
        elif 60 <= i <= 70:  # 机构派发阶段
            # 价格高位震荡，成交量大幅放大，机构开始派发
            open_price = close + np.random.uniform(-0.8, 0.8)
            close = close_prices[59] + np.random.uniform(-1.0, 1.0)  # 高位震荡
            high_price = max(open_price, close) + abs(np.random.normal(0, 0.4))
            low_price = min(open_price, close) - abs(np.random.normal(0, 0.4))
            volume = np.random.lognormal(12.5, 0.2)  # 大幅放量
            close_prices[i] = close
        elif 80 <= i <= 85:  # 机构砸盘阶段
            # 价格快速下跌，成交量放大，机构集中抛售
            open_price = close + 0.5
            close = close_prices[i-1] - 1.2  # 快速下跌
            high_price = open_price + 0.2
            low_price = close - 0.3
            volume = np.random.lognormal(12.8, 0.15)  # 超大成交量
            close_prices[i] = close
        else:
            # 正常交易
            open_price = close + np.random.normal(0, 0.3)
            high_price = max(open_price, close) + abs(np.random.normal(0, 0.2))
            low_price = min(open_price, close) - abs(np.random.normal(0, 0.2))
            volume = np.random.lognormal(10.8, 0.4)
        
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


def diagnose_zxm_institution_behavior():
    """诊断ZXM_INSTITUTION_BEHAVIOR指标的实际状态"""
    logger.info("🔍 诊断ZXM_INSTITUTION_BEHAVIOR指标的实际状态...")
    
    try:
        # 创建测试数据
        test_data = create_test_data_with_institution_behavior(100)
        logger.info(f"✅ 测试数据创建成功: {len(test_data)} 行")
        
        # 1. 检查指标注册情况
        logger.info("📦 检查指标注册情况...")
        try:
            from indicators.complete_indicator_registry import get_indicator_registry
            registry = get_indicator_registry()
            
            # 尝试获取ZXM_INSTITUTION_BEHAVIOR指标
            try:
                institution_behavior_indicator = registry.create_indicator('ZXM_INSTITUTION_BEHAVIOR')
                logger.info(f"✅ ZXM_INSTITUTION_BEHAVIOR指标已注册: {type(institution_behavior_indicator)}")
                
                # 测试指标计算
                result = institution_behavior_indicator.calculate(test_data)
                logger.info(f"✅ 注册的ZXM_INSTITUTION_BEHAVIOR指标计算成功，形状: {result.shape}")
                logger.info(f"  - 列名: {list(result.columns)}")
                
                # 检查机构行为相关列
                institution_columns = [col for col in result.columns if any(keyword in col.lower() for keyword in ['institution', 'behavior', '机构', '行为', 'smart', 'money'])]
                logger.info(f"  - 机构行为相关列: {institution_columns}")
                
                if institution_columns:
                    for col in institution_columns:
                        if col in result.columns:
                            values = result[col]
                            if pd.api.types.is_numeric_dtype(values):
                                logger.info(f"    - {col}: 数值型，范围 [{values.min():.3f}, {values.max():.3f}]")
                            else:
                                logger.info(f"    - {col}: 类型 {values.dtype}")
                
            except Exception as e:
                logger.error(f"❌ 获取注册的ZXM_INSTITUTION_BEHAVIOR指标失败: {e}")
            
        except Exception as e:
            logger.error(f"❌ 检查指标注册失败: {e}")
        
        # 2. 检查ZXM指标实现文件
        logger.info("📦 检查ZXM指标实现文件...")
        try:
            zxm_files = [
                'indicators/zxm/institution_behavior.py',
                'indicators/zxm/zxm_institution_behavior.py',
                'indicators/zxm/smart_money.py'
            ]
            
            for file_path in zxm_files:
                full_path = Path(root_dir) / file_path
                if full_path.exists():
                    logger.info(f"✅ 找到实现文件: {file_path}")
                    
                    # 尝试导入
                    try:
                        if 'institution_behavior.py' in file_path:
                            from indicators.zxm.institution_behavior import ZXMInstitutionBehavior
                            indicator = ZXMInstitutionBehavior()
                            result = indicator.calculate(test_data)
                            logger.info(f"  - 直接导入成功，计算结果形状: {result.shape}")
                        elif 'zxm_institution_behavior.py' in file_path:
                            from indicators.zxm.zxm_institution_behavior import ZXMInstitutionBehavior
                            indicator = ZXMInstitutionBehavior()
                            result = indicator.calculate(test_data)
                            logger.info(f"  - 直接导入成功，计算结果形状: {result.shape}")
                    except Exception as e:
                        logger.warning(f"  ⚠️ 直接导入失败: {e}")
                else:
                    logger.warning(f"⚠️ 实现文件不存在: {file_path}")
            
        except Exception as e:
            logger.error(f"❌ 检查实现文件失败: {e}")
        
        # 3. 手动实现机构行为分析
        logger.info("🧪 手动实现机构行为分析...")
        try:
            # 计算机构行为指标
            close_prices = test_data['close']
            high_prices = test_data['high']
            low_prices = test_data['low']
            volumes = test_data['volume']
            
            # 机构资金流向分析
            typical_price = (high_prices + low_prices + close_prices) / 3
            
            # 大单净流入（模拟机构行为）
            big_order_flow = pd.Series(index=test_data.index, dtype=float)
            for i in range(1, len(test_data)):
                price_change = typical_price.iloc[i] - typical_price.iloc[i-1]
                volume_ratio = volumes.iloc[i] / volumes.iloc[i-5:i].mean() if i >= 5 else 1
                
                # 大单流入判断：价格上涨且成交量放大
                if price_change > 0 and volume_ratio > 1.5:
                    big_order_flow.iloc[i] = volumes.iloc[i] * price_change
                elif price_change < 0 and volume_ratio > 1.5:
                    big_order_flow.iloc[i] = volumes.iloc[i] * price_change
                else:
                    big_order_flow.iloc[i] = 0
            
            # 机构持仓变化
            institution_position = big_order_flow.cumsum()
            
            # 机构行为强度
            institution_strength = pd.Series(index=test_data.index, dtype=float)
            for i in range(10, len(test_data)):
                recent_flow = big_order_flow.iloc[i-10:i+1]
                strength = abs(recent_flow).sum() / volumes.iloc[i-10:i+1].sum()
                institution_strength.iloc[i] = min(100, strength * 10000)  # 归一化
            
            # 机构行为信号
            institution_signals = pd.Series(0, index=test_data.index)
            for i in range(5, len(test_data)):
                # 机构建仓信号
                if (big_order_flow.iloc[i-5:i+1] > 0).sum() >= 4:
                    institution_signals.iloc[i] = 1  # 机构建仓
                # 机构减仓信号
                elif (big_order_flow.iloc[i-5:i+1] < 0).sum() >= 4:
                    institution_signals.iloc[i] = -1  # 机构减仓
            
            # 机构活跃度
            institution_activity = pd.Series(index=test_data.index, dtype=float)
            for i in range(5, len(test_data)):
                activity = abs(big_order_flow.iloc[i-5:i+1]).mean()
                institution_activity.iloc[i] = activity
            
            logger.info(f"  - 大单净流入计算完成，范围: [{big_order_flow.min():.2f}, {big_order_flow.max():.2f}]")
            logger.info(f"  - 机构持仓变化范围: [{institution_position.min():.2f}, {institution_position.max():.2f}]")
            logger.info(f"  - 机构行为强度范围: [{institution_strength.min():.3f}, {institution_strength.max():.3f}]")
            logger.info(f"  - 机构行为信号统计: 建仓 {(institution_signals == 1).sum()} 个, 减仓 {(institution_signals == -1).sum()} 个")
            
            if (institution_signals != 0).sum() > 0:
                logger.info("✅ 手动机构行为分析检测到有效信号")
            else:
                logger.warning("⚠️ 手动机构行为分析未检测到明显信号")
            
        except Exception as e:
            logger.error(f"❌ 手动机构行为分析失败: {e}")
        
        # 4. 检查BaseIndicator抽象方法
        logger.info("🧪 检查BaseIndicator抽象方法...")
        try:
            if 'institution_behavior_indicator' in locals():
                required_methods = [
                    '_calculate_baseindicator',
                    'calculate_raw_score_Indicator_Base_Indicator',
                    'get_patterns_Indicator_Base_Indicator',
                    'calculate_confidence_Indicator_Base_Indicator',
                    'set_parameters_Indicator_Base_Indicator'
                ]
                
                missing_methods = [method for method in required_methods if not hasattr(institution_behavior_indicator, method)]
                
                if not missing_methods:
                    logger.info("  ✅ BaseIndicator抽象方法完整")
                else:
                    logger.warning(f"  ⚠️ 缺少方法: {missing_methods}")
            else:
                logger.warning("  ⚠️ 无法检查BaseIndicator方法，指标未成功创建")
                
        except Exception as e:
            logger.error(f"❌ BaseIndicator方法检查失败: {e}")
        
        logger.info("🎯 ZXM_INSTITUTION_BEHAVIOR指标诊断完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 诊断过程中发生异常: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return False


def main():
    """主函数"""
    success = diagnose_zxm_institution_behavior()
    
    if success:
        logger.info("✅ ZXM_INSTITUTION_BEHAVIOR指标诊断完成")
    else:
        logger.error("❌ ZXM_INSTITUTION_BEHAVIOR指标诊断失败")
    
    return success


if __name__ == "__main__":
    main()
