#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
综合指标评分系统演示

展示如何使用统一的指标评分系统进行股票技术分析
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from indicators.macd import MACD
from indicators.kdj import KDJ
from indicators.rsi import RSI
from indicators.boll import BOLL
from indicators.obv import OBV
from indicators.wr import WR
from indicators.cci import CCI
from indicators.atr import ATR
from indicators.dmi import DMI
from utils.logger import get_logger

logger = get_logger(__name__)


class ComprehensiveIndicatorScoring:
    """
    综合指标评分系统
    """
    
    def __init__(self):
        """初始化评分系统"""
        self.indicators = {
            'MACD': MACD(),
            'KDJ': KDJ(),
            'RSI': RSI(),
            'BOLL': BOLL(),
            'OBV': OBV(),
            'WR': WR(),
            'CCI': CCI(),
            'ATR': ATR(),
            'DMI': DMI(),
        }
        
        # 指标权重配置
        self.weights = {
            'MACD': 0.15,    # 趋势指标
            'KDJ': 0.12,     # 震荡指标
            'RSI': 0.12,     # 震荡指标
            'BOLL': 0.10,    # 波动率指标
            'OBV': 0.15,     # 成交量指标
            'WR': 0.08,      # 震荡指标
            'CCI': 0.08,     # 震荡指标
            'ATR': 0.10,     # 波动率指标
            'DMI': 0.10,     # 趋势指标
        }
    
    def calculate_comprehensive_score(self, data: pd.DataFrame) -> Dict:
        """
        计算综合评分
        
        Args:
            data: 股票数据
            
        Returns:
            Dict: 综合评分结果
        """
        results = {}
        individual_scores = {}
        patterns_summary = []
        
        print(f"\n{'='*60}")
        print("开始计算各指标评分...")
        print(f"{'='*60}")
        
        # 计算各指标评分
        for name, indicator in self.indicators.items():
            try:
                print(f"\n计算 {name} 指标评分...")
                score_result = indicator.calculate_score(data)
                
                if score_result and 'final_score' in score_result:
                    final_score = score_result['final_score'].iloc[-1]
                    raw_score = score_result['raw_score'].iloc[-1]
                    patterns = score_result['patterns']
                    confidence = score_result['confidence']
                    
                    individual_scores[name] = {
                        'final_score': final_score,
                        'raw_score': raw_score,
                        'patterns': patterns,
                        'confidence': confidence,
                        'weight': self.weights[name]
                    }
                    
                    print(f"  ✓ {name}: {final_score:.1f}分 (置信度: {confidence:.1f}%)")
                    if patterns:
                        print(f"    形态: {', '.join(patterns[:3])}{'...' if len(patterns) > 3 else ''}")
                    
                    patterns_summary.extend([f"{name}-{p}" for p in patterns])
                    
                else:
                    print(f"  ✗ {name}: 评分计算失败")
                    individual_scores[name] = {
                        'final_score': 50.0,
                        'raw_score': 50.0,
                        'patterns': [],
                        'confidence': 50.0,
                        'weight': self.weights[name]
                    }
                    
            except Exception as e:
                print(f"  ✗ {name}: 计算错误 - {str(e)}")
                individual_scores[name] = {
                    'final_score': 50.0,
                    'raw_score': 50.0,
                    'patterns': [],
                    'confidence': 50.0,
                    'weight': self.weights[name]
                }
        
        # 计算加权综合评分
        weighted_score = 0
        total_weight = 0
        confidence_sum = 0
        
        for name, score_data in individual_scores.items():
            weight = score_data['weight']
            score = score_data['final_score']
            confidence = score_data['confidence']
            
            weighted_score += score * weight * (confidence / 100)
            total_weight += weight * (confidence / 100)
            confidence_sum += confidence * weight
        
        if total_weight > 0:
            comprehensive_score = weighted_score / total_weight
            avg_confidence = confidence_sum / sum(self.weights.values())
        else:
            comprehensive_score = 50.0
            avg_confidence = 50.0
        
        # 分类评分结果
        signal_type = self._classify_signal(comprehensive_score)
        
        results = {
            'comprehensive_score': comprehensive_score,
            'signal_type': signal_type,
            'average_confidence': avg_confidence,
            'individual_scores': individual_scores,
            'patterns_summary': patterns_summary,
            'recommendation': self._generate_recommendation(comprehensive_score, avg_confidence)
        }
        
        return results
    
    def _classify_signal(self, score: float) -> str:
        """
        分类信号类型
        
        Args:
            score: 综合评分
            
        Returns:
            str: 信号类型
        """
        if score >= 80:
            return "强烈买入"
        elif score >= 65:
            return "买入"
        elif score >= 55:
            return "弱买入"
        elif score >= 45:
            return "中性"
        elif score >= 35:
            return "弱卖出"
        elif score >= 20:
            return "卖出"
        else:
            return "强烈卖出"
    
    def _generate_recommendation(self, score: float, confidence: float) -> str:
        """
        生成投资建议
        
        Args:
            score: 综合评分
            confidence: 平均置信度
            
        Returns:
            str: 投资建议
        """
        signal_type = self._classify_signal(score)
        
        if confidence >= 80:
            confidence_desc = "高置信度"
        elif confidence >= 60:
            confidence_desc = "中等置信度"
        else:
            confidence_desc = "低置信度"
        
        recommendations = {
            "强烈买入": f"建议积极买入，{confidence_desc}信号",
            "买入": f"建议买入，{confidence_desc}信号",
            "弱买入": f"可考虑小仓位买入，{confidence_desc}信号",
            "中性": f"建议观望，{confidence_desc}信号",
            "弱卖出": f"可考虑减仓，{confidence_desc}信号",
            "卖出": f"建议卖出，{confidence_desc}信号",
            "强烈卖出": f"建议立即卖出，{confidence_desc}信号"
        }
        
        return recommendations.get(signal_type, "建议谨慎操作")
    
    def print_detailed_report(self, results: Dict):
        """
        打印详细报告
        
        Args:
            results: 评分结果
        """
        print(f"\n{'='*60}")
        print("综合技术分析报告")
        print(f"{'='*60}")
        
        # 综合评分
        score = results['comprehensive_score']
        signal_type = results['signal_type']
        confidence = results['average_confidence']
        
        print(f"\n📊 综合评分: {score:.1f}分")
        print(f"📈 信号类型: {signal_type}")
        print(f"🎯 平均置信度: {confidence:.1f}%")
        print(f"💡 投资建议: {results['recommendation']}")
        
        # 各指标详细评分
        print(f"\n{'='*40}")
        print("各指标详细评分")
        print(f"{'='*40}")
        
        individual_scores = results['individual_scores']
        
        # 按评分排序
        sorted_indicators = sorted(
            individual_scores.items(),
            key=lambda x: x[1]['final_score'],
            reverse=True
        )
        
        for name, score_data in sorted_indicators:
            final_score = score_data['final_score']
            confidence = score_data['confidence']
            weight = score_data['weight']
            patterns = score_data['patterns']
            
            print(f"\n{name:>6}: {final_score:>5.1f}分 "
                  f"(权重: {weight:.2f}, 置信度: {confidence:.1f}%)")
            
            if patterns:
                print(f"       形态: {', '.join(patterns[:2])}{'...' if len(patterns) > 2 else ''}")
        
        # 形态汇总
        if results['patterns_summary']:
            print(f"\n{'='*40}")
            print("识别的技术形态汇总")
            print(f"{'='*40}")
            
            # 统计形态出现频率
            pattern_count = {}
            for pattern in results['patterns_summary']:
                indicator, pattern_name = pattern.split('-', 1)
                if pattern_name in pattern_count:
                    pattern_count[pattern_name] += 1
                else:
                    pattern_count[pattern_name] = 1
            
            # 按频率排序
            sorted_patterns = sorted(
                pattern_count.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for pattern, count in sorted_patterns[:10]:  # 显示前10个
                print(f"  • {pattern} (出现{count}次)")
    
    def analyze_market_environment(self, data: pd.DataFrame) -> Dict:
        """
        分析市场环境
        
        Args:
            data: 股票数据
            
        Returns:
            Dict: 市场环境分析结果
        """
        # 使用多个指标的市场环境检测结果
        environments = []
        
        for name, indicator in self.indicators.items():
            try:
                score_result = indicator.calculate_score(data)
                if score_result and 'market_environment' in score_result:
                    env = score_result['market_environment']
                    environments.append(env.value)
            except:
                continue
        
        # 统计最常见的市场环境
        if environments:
            env_count = {}
            for env in environments:
                env_count[env] = env_count.get(env, 0) + 1
            
            dominant_env = max(env_count.items(), key=lambda x: x[1])
            
            return {
                'dominant_environment': dominant_env[0],
                'confidence': dominant_env[1] / len(environments),
                'all_environments': env_count
            }
        
        return {
            'dominant_environment': '震荡市',
            'confidence': 0.5,
            'all_environments': {'震荡市': 1}
        }


def generate_sample_data(symbol: str = "TEST", periods: int = 100) -> pd.DataFrame:
    """
    生成示例股票数据
    
    Args:
        symbol: 股票代码
        periods: 数据周期数
        
    Returns:
        pd.DataFrame: 示例数据
    """
    # 生成日期索引
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
    
    # 设置随机种子以获得可重现的结果
    np.random.seed(42)
    
    # 生成基础价格趋势
    base_price = 100
    trend = np.cumsum(np.random.normal(0.001, 0.02, periods))  # 轻微上升趋势
    noise = np.random.normal(0, 0.01, periods)  # 随机噪声
    
    # 生成收盘价
    close_prices = base_price * (1 + trend + noise)
    close_prices = np.maximum(close_prices, 10)  # 确保价格为正
    
    # 生成其他价格数据
    high_prices = close_prices * (1 + np.random.uniform(0, 0.03, periods))
    low_prices = close_prices * (1 - np.random.uniform(0, 0.03, periods))
    open_prices = close_prices * (1 + np.random.normal(0, 0.01, periods))
    
    # 确保价格关系合理
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
    
    # 生成成交量数据
    base_volume = 1000000
    volume_trend = np.random.uniform(0.5, 2.0, periods)
    volumes = base_volume * volume_trend
    
    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)
    
    return data


def main():
    """
    主演示函数
    """
    print("🚀 综合指标评分系统演示")
    print(f"演示时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 生成示例数据
    print("\n📈 生成示例股票数据...")
    stock_data = generate_sample_data("DEMO", 100)
    print(f"数据范围: {stock_data.index[0].strftime('%Y-%m-%d')} 至 {stock_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"数据形状: {stock_data.shape}")
    print(f"价格范围: {stock_data['close'].min():.2f} - {stock_data['close'].max():.2f}")
    
    # 初始化评分系统
    print("\n🔧 初始化综合评分系统...")
    scoring_system = ComprehensiveIndicatorScoring()
    print(f"已加载 {len(scoring_system.indicators)} 个技术指标")
    
    # 分析市场环境
    print("\n🌍 分析市场环境...")
    market_env = scoring_system.analyze_market_environment(stock_data)
    print(f"主导市场环境: {market_env['dominant_environment']} (置信度: {market_env['confidence']:.1%})")
    
    # 计算综合评分
    print("\n⚡ 开始综合技术分析...")
    results = scoring_system.calculate_comprehensive_score(stock_data)
    
    # 打印详细报告
    scoring_system.print_detailed_report(results)
    
    # 风险提示
    print(f"\n{'='*60}")
    print("⚠️  风险提示")
    print(f"{'='*60}")
    print("1. 本分析仅基于技术指标，不构成投资建议")
    print("2. 投资有风险，入市需谨慎")
    print("3. 建议结合基本面分析和市场环境综合判断")
    print("4. 请根据个人风险承受能力制定投资策略")
    
    print(f"\n✅ 演示完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main() 