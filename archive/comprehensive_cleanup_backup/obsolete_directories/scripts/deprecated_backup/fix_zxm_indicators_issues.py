#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ZXM指标问题修复脚本
修复4个有问题的ZXM指标：ZXM_MARKET_SENTIMENT、ZXM_LIQUIDITY_ANALYSIS、ZXM_VOLATILITY_FORECAST、ZXM_CORRELATION_MATRIX
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class ZXMIndicatorIssuesFixer:
    """ZXM指标问题修复器"""
    
    def __init__(self):
        self.target_score = 95.0  # 目标分数：95分以上
        self.test_data = None
        self.problem_indicators = [
            'ZXM_MARKET_SENTIMENT',      # 返回None问题
            'ZXM_LIQUIDITY_ANALYSIS',    # 未注册问题
            'ZXM_VOLATILITY_FORECAST',   # 未注册问题
            'ZXM_CORRELATION_MATRIX'     # 未注册问题
        ]
        
    def generate_premium_test_data(self) -> pd.DataFrame:
        """生成高质量测试数据"""
        logger.info("📊 生成ZXM指标测试数据...")
        
        # 生成100天的测试数据
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        np.random.seed(42)
        base_price = 100.0
        base_volume = 2000000
        
        # 生成具有市场特征的数据
        market_trend = np.concatenate([
            np.linspace(0, 15, 25),    # 上升趋势
            np.linspace(15, 20, 20),   # 加速上升
            np.linspace(20, 12, 25),   # 高位震荡
            np.linspace(12, 3, 20),    # 下降趋势
            np.linspace(3, 10, 10)     # 底部反弹
        ])
        
        volatility = np.sin(np.linspace(0, 6*np.pi, 100)) * 2 + 3
        volume_pattern = np.cos(np.linspace(0, 4*np.pi, 100)) * 1000000 + base_volume
        
        prices = [base_price]
        volumes = [base_volume]
        
        for i in range(1, 100):
            trend = (market_trend[i] - market_trend[i-1]) * 0.3
            vol = volatility[i] * 0.12
            noise = np.random.normal(0, 0.5)
            
            price_change = trend + vol + noise
            new_price = max(prices[-1] + price_change, 1.0)
            
            volume_factor = (volatility[i] / 4) + 0.8
            new_volume = int(volume_pattern[i] * volume_factor * np.random.uniform(0.8, 1.2))
            
            prices.append(new_price)
            volumes.append(max(new_volume, 100000))
        
        # 生成OHLC数据
        data = []
        for i, (price, volume) in enumerate(zip(prices, volumes)):
            daily_vol = volatility[i] * 0.01
            high = price + np.random.uniform(0, daily_vol * price)
            low = price - np.random.uniform(0, daily_vol * price)
            
            open_price = prices[i-1] if i > 0 else price
            close = price
            
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            data.append({
                'date': dates[i],
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        logger.info(f"✅ 生成ZXM测试数据: {len(df)}行")
        return df
    
    def analyze_zxm_indicator_issues(self, indicator_name: str) -> Dict[str, Any]:
        """分析ZXM指标的具体问题"""
        logger.info(f"🔍 分析ZXM指标问题: {indicator_name}")
        
        try:
            # 导入指标注册表
            from indicators.complete_indicator_registry import get_indicator_registry
            
            # 获取指标注册表实例
            registry = get_indicator_registry()
            
            # 尝试创建指标实例
            try:
                indicator = registry.create_indicator(indicator_name)
                if indicator is None:
                    return {
                        'indicator': indicator_name,
                        'issue_type': 'creation_failed',
                        'description': '指标创建失败，可能未注册',
                        'severity': 'high'
                    }
                
                # 测试计算功能
                try:
                    result = indicator.calculate(self.test_data)
                    if result is None:
                        return {
                            'indicator': indicator_name,
                            'issue_type': 'returns_none',
                            'description': '指标计算返回None',
                            'severity': 'high'
                        }
                    elif isinstance(result, dict) and len(result) == 0:
                        return {
                            'indicator': indicator_name,
                            'issue_type': 'empty_result',
                            'description': '指标计算返回空字典',
                            'severity': 'medium'
                        }
                    else:
                        return {
                            'indicator': indicator_name,
                            'issue_type': 'working',
                            'description': '指标工作正常',
                            'severity': 'none'
                        }
                        
                except Exception as e:
                    return {
                        'indicator': indicator_name,
                        'issue_type': 'calculation_error',
                        'description': f'指标计算异常: {e}',
                        'severity': 'high'
                    }
                    
            except Exception as e:
                return {
                    'indicator': indicator_name,
                    'issue_type': 'not_registered',
                    'description': f'指标未注册或注册失败: {e}',
                    'severity': 'high'
                }
                
        except Exception as e:
            logger.error(f"❌ 分析{indicator_name}失败: {e}")
            return {
                'indicator': indicator_name,
                'issue_type': 'analysis_error',
                'description': f'分析过程异常: {e}',
                'severity': 'critical'
            }
    
    def fix_market_sentiment_indicator(self) -> Dict[str, Any]:
        """修复ZXM_MARKET_SENTIMENT指标的返回None问题"""
        logger.info("🔧 修复ZXM_MARKET_SENTIMENT指标...")
        
        try:
            # 分析问题
            analysis = self.analyze_zxm_indicator_issues('ZXM_MARKET_SENTIMENT')
            logger.info(f"ZXM_MARKET_SENTIMENT问题分析: {analysis}")
            
            if analysis['issue_type'] == 'returns_none':
                # 这是一个返回None的问题，需要检查指标实现
                logger.info("🔍 检查ZXM_MARKET_SENTIMENT指标实现...")
                
                # 检查指标文件是否存在
                indicator_file = "indicators/zxm/zxm_market_sentiment.py"
                if os.path.exists(indicator_file):
                    logger.info(f"✅ 找到指标文件: {indicator_file}")
                    return {
                        'status': 'needs_implementation_fix',
                        'issue': 'returns_none',
                        'file_path': indicator_file,
                        'description': '指标文件存在但calculate方法返回None'
                    }
                else:
                    logger.warning(f"⚠️ 指标文件不存在: {indicator_file}")
                    return {
                        'status': 'needs_file_creation',
                        'issue': 'file_missing',
                        'file_path': indicator_file,
                        'description': '指标文件不存在，需要创建'
                    }
            else:
                return {
                    'status': 'analysis_complete',
                    'issue': analysis['issue_type'],
                    'description': analysis['description']
                }
                
        except Exception as e:
            logger.error(f"❌ ZXM_MARKET_SENTIMENT修复失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def fix_unregistered_indicators(self) -> Dict[str, Any]:
        """修复未注册的ZXM指标"""
        logger.info("🔧 修复未注册的ZXM指标...")
        
        unregistered_indicators = [
            'ZXM_LIQUIDITY_ANALYSIS',
            'ZXM_VOLATILITY_FORECAST', 
            'ZXM_CORRELATION_MATRIX'
        ]
        
        results = {}
        
        for indicator_name in unregistered_indicators:
            logger.info(f"🔍 检查{indicator_name}...")
            
            analysis = self.analyze_zxm_indicator_issues(indicator_name)
            
            if analysis['issue_type'] == 'not_registered':
                # 检查指标文件是否存在
                file_name = indicator_name.lower() + ".py"
                indicator_file = f"indicators/zxm/{file_name}"
                
                if os.path.exists(indicator_file):
                    results[indicator_name] = {
                        'status': 'needs_registration',
                        'issue': 'not_registered',
                        'file_path': indicator_file,
                        'description': '指标文件存在但未注册'
                    }
                else:
                    results[indicator_name] = {
                        'status': 'needs_creation',
                        'issue': 'file_missing',
                        'file_path': indicator_file,
                        'description': '指标文件不存在，需要创建'
                    }
            else:
                results[indicator_name] = {
                    'status': 'analysis_complete',
                    'issue': analysis['issue_type'],
                    'description': analysis['description']
                }
        
        return results
    
    def run_zxm_issues_analysis(self) -> Dict[str, Any]:
        """运行ZXM指标问题分析"""
        logger.info("🚀 开始ZXM指标问题分析...")
        
        start_time = time.time()
        
        # 生成测试数据
        self.test_data = self.generate_premium_test_data()
        
        # 分析结果
        results = {}
        
        # 分析ZXM_MARKET_SENTIMENT
        results['ZXM_MARKET_SENTIMENT'] = self.fix_market_sentiment_indicator()
        
        # 分析未注册的指标
        unregistered_results = self.fix_unregistered_indicators()
        results.update(unregistered_results)
        
        analysis_time = time.time() - start_time
        
        summary = {
            'analysis_type': 'ZXM_INDICATORS_ISSUES_ANALYSIS',
            'indicators_analyzed': len(self.problem_indicators),
            'analysis_time': analysis_time,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"🎯 ZXM指标问题分析完成!")
        logger.info(f"⏱️ 分析时间: {analysis_time:.2f}秒")
        
        return summary


def main():
    """主函数"""
    logger.info("🔍 ZXM指标问题分析开始...")
    
    fixer = ZXMIndicatorIssuesFixer()
    result = fixer.run_zxm_issues_analysis()
    
    # 保存分析报告
    report_file = f"docs/finaltesting/indicators/zxm_indicators_issues_analysis.md"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    # 生成分析报告
    report_content = f"""# ZXM指标问题分析报告

## 分析概览
- **分析类型**: ZXM指标问题分析
- **分析时间**: {result['timestamp']}
- **目标**: 分析4个有问题的ZXM指标的具体问题类型
- **分析指标数**: {result['indicators_analyzed']}个

## 分析结果详情

### ZXM_MARKET_SENTIMENT指标分析
- **问题类型**: {result['results']['ZXM_MARKET_SENTIMENT']['issue']}
- **修复状态**: {result['results']['ZXM_MARKET_SENTIMENT']['status']}
- **问题描述**: {result['results']['ZXM_MARKET_SENTIMENT']['description']}

### ZXM_LIQUIDITY_ANALYSIS指标分析
- **问题类型**: {result['results']['ZXM_LIQUIDITY_ANALYSIS']['issue']}
- **修复状态**: {result['results']['ZXM_LIQUIDITY_ANALYSIS']['status']}
- **问题描述**: {result['results']['ZXM_LIQUIDITY_ANALYSIS']['description']}

### ZXM_VOLATILITY_FORECAST指标分析
- **问题类型**: {result['results']['ZXM_VOLATILITY_FORECAST']['issue']}
- **修复状态**: {result['results']['ZXM_VOLATILITY_FORECAST']['status']}
- **问题描述**: {result['results']['ZXM_VOLATILITY_FORECAST']['description']}

### ZXM_CORRELATION_MATRIX指标分析
- **问题类型**: {result['results']['ZXM_CORRELATION_MATRIX']['issue']}
- **修复状态**: {result['results']['ZXM_CORRELATION_MATRIX']['status']}
- **问题描述**: {result['results']['ZXM_CORRELATION_MATRIX']['description']}

## 修复建议

基于分析结果，修复方向：
1. **返回None问题**: 修复calculate方法的实现逻辑
2. **未注册问题**: 在指标注册表中添加指标注册
3. **文件缺失问题**: 创建缺失的指标实现文件
4. **实现完善**: 确保所有方法正确实现

## 下一步行动
1. 实施具体的代码修复
2. 重新运行95分标准验证
3. 确认所有ZXM指标达到95分以上标准

---
*分析时间: {result['analysis_time']:.2f}秒*
*分析工具: ZXM指标问题分析系统*
"""
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"📄 问题分析报告已保存: {report_file}")
    
    return result


if __name__ == "__main__":
    main()
