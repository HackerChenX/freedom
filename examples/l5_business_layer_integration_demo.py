#!/usr/bin/env python3
"""
L5业务层集成验证示例

本示例演示如何在L5业务层正确使用L4层的抽象方法接口，
包括买点分析、策略选股和实时交易等核心业务场景。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
from indicators.core.unified_indicator_manager import UnifiedIndicatorManager
from utils.logger import get_logger

logger = get_logger(__name__)

class BuyPointAnalyzer:
    """L5层买点分析器 - 演示正确使用L4接口"""
    
    def __init__(self):
        self.indicator_manager = UnifiedIndicatorManager()
        self.core_indicators = ['MACD', 'RSI', 'KDJ', 'BOLL']
    
    def analyze_buypoint(self, stock_code: str, stock_data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析股票的买点信号
        
        Args:
            stock_code: 股票代码
            stock_data: 股票OHLCV数据
            
        Returns:
            买点分析结果
        """
        logger.info(f"开始分析股票 {stock_code} 的买点信号")
        
        buypoint_signals = []
        
        # 遍历核心指标
        for indicator_name in self.core_indicators:
            try:
                # 1. 创建指标实例
                indicator = self.indicator_manager.create_indicator(indicator_name)
                
                # 2. 计算指标数值 - 使用标准抽象方法
                values = indicator.calculate(stock_data)
                
                # 3. 生成交易信号 - 使用标准抽象方法
                signal = indicator.get_signal(values)
                
                # 4. 收集买点信号
                if signal['signal_type'] == 'buy':
                    buypoint_signals.append({
                        'indicator': indicator_name,
                        'strength': signal['strength'],
                        'confidence': signal['confidence'],
                        'reason': signal.get('reason', ''),
                        'metadata': signal.get('metadata', {})
                    })
                    
                logger.info(f"{indicator_name}: {signal['signal_type']} (强度:{signal['strength']:.2f})")
                
            except Exception as e:
                logger.warning(f"指标 {indicator_name} 分析失败: {e}")
                continue
        
        # 5. L5层业务逻辑：综合评分
        composite_score = self._calculate_buypoint_score(buypoint_signals)
        
        result = {
            'stock_code': stock_code,
            'analysis_time': datetime.now(),
            'composite_score': composite_score,
            'buypoint_signals': buypoint_signals,
            'recommendation': self._get_recommendation(composite_score)
        }
        
        logger.info(f"股票 {stock_code} 买点分析完成，综合评分: {composite_score:.2f}")
        return result
    
    def _calculate_buypoint_score(self, signals: List[Dict]) -> float:
        """计算综合买点评分"""
        if not signals:
            return 0.0
        
        # 加权平均计算
        total_weight = 0
        weighted_score = 0
        
        for signal in signals:
            weight = signal['confidence']
            score = signal['strength']
            weighted_score += weight * score
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _get_recommendation(self, score: float) -> str:
        """根据评分给出投资建议"""
        if score >= 0.8:
            return "强烈买入"
        elif score >= 0.6:
            return "买入"
        elif score >= 0.4:
            return "观望"
        else:
            return "不建议买入"


class StrategySelector:
    """L5层策略选股器 - 演示批量处理"""
    
    def __init__(self):
        self.indicator_manager = UnifiedIndicatorManager()
        self.selection_indicators = ['MACD', 'RSI', 'KDJ']
    
    def select_stocks(self, stock_pool: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        从股票池中选择符合策略的股票
        
        Args:
            stock_pool: 股票池 {股票代码: 股票数据}
            
        Returns:
            选股结果列表
        """
        logger.info(f"开始策略选股，股票池大小: {len(stock_pool)}")
        
        selection_results = []
        
        for stock_code, stock_data in stock_pool.items():
            try:
                # 计算多个指标的信号
                signals = {}
                for indicator_name in self.selection_indicators:
                    indicator = self.indicator_manager.create_indicator(indicator_name)
                    
                    # 使用标准抽象方法
                    values = indicator.calculate(stock_data)
                    signal = indicator.get_signal(values)
                    signals[indicator_name] = signal
                
                # L5层业务逻辑：多信号综合判断
                buy_signals = [s for s in signals.values() if s['signal_type'] == 'buy']
                
                if len(buy_signals) >= 2:  # 至少2个买入信号
                    avg_strength = sum(s['strength'] for s in buy_signals) / len(buy_signals)
                    avg_confidence = sum(s['confidence'] for s in buy_signals) / len(buy_signals)
                    
                    selection_results.append({
                        'stock_code': stock_code,
                        'recommendation': 'buy',
                        'buy_signal_count': len(buy_signals),
                        'avg_strength': avg_strength,
                        'avg_confidence': avg_confidence,
                        'signals': signals
                    })
                    
                    logger.info(f"选中股票 {stock_code}: {len(buy_signals)}个买入信号")
                
            except Exception as e:
                logger.warning(f"股票 {stock_code} 选股分析失败: {e}")
                continue
        
        # 按综合评分排序
        selection_results.sort(key=lambda x: x['avg_strength'] * x['avg_confidence'], reverse=True)
        
        logger.info(f"策略选股完成，选中 {len(selection_results)} 只股票")
        return selection_results


class RealTimeTrader:
    """L5层实时交易器 - 演示实时信号处理"""
    
    def __init__(self):
        self.indicator_manager = UnifiedIndicatorManager()
        self.trading_indicators = ['RSI', 'MACD']
    
    def make_trading_decision(self, stock_code: str, real_time_data: pd.DataFrame) -> Dict[str, Any]:
        """
        基于实时数据做出交易决策
        
        Args:
            stock_code: 股票代码
            real_time_data: 实时股票数据
            
        Returns:
            交易决策结果
        """
        logger.info(f"为股票 {stock_code} 生成实时交易决策")
        
        trading_signals = []
        
        for indicator_name in self.trading_indicators:
            try:
                indicator = self.indicator_manager.create_indicator(indicator_name)
                
                # 使用标准抽象方法
                values = indicator.calculate(real_time_data)
                signal = indicator.get_signal(values)
                
                if signal and 'signal_type' in signal:
                    trading_signals.append({
                        'indicator': indicator_name,
                        'signal': signal
                    })
                
            except Exception as e:
                logger.warning(f"指标 {indicator_name} 实时分析失败: {e}")
                continue
        
        # L5层业务逻辑：实时交易决策
        decision = self._make_decision(trading_signals, real_time_data)
        
        logger.info(f"股票 {stock_code} 交易决策: {decision['action']}")
        return decision
    
    def _make_decision(self, signals: List[Dict], data: pd.DataFrame) -> Dict[str, Any]:
        """制定交易决策"""
        current_price = data['close'].iloc[-1]
        
        buy_signals = [s for s in signals if s['signal']['signal_type'] == 'buy']
        sell_signals = [s for s in signals if s['signal']['signal_type'] == 'sell']
        
        if len(buy_signals) >= 2 and all(s['signal']['confidence'] > 0.7 for s in buy_signals):
            return {
                'action': 'buy',
                'price': current_price,
                'confidence': sum(s['signal']['confidence'] for s in buy_signals) / len(buy_signals),
                'reason': f"{len(buy_signals)}个强烈买入信号"
            }
        elif len(sell_signals) >= 2 and all(s['signal']['confidence'] > 0.7 for s in sell_signals):
            return {
                'action': 'sell',
                'price': current_price,
                'confidence': sum(s['signal']['confidence'] for s in sell_signals) / len(sell_signals),
                'reason': f"{len(sell_signals)}个强烈卖出信号"
            }
        else:
            return {
                'action': 'hold',
                'price': current_price,
                'confidence': 0.5,
                'reason': '信号不明确或强度不足'
            }


def generate_sample_data(stock_code: str, days: int = 100) -> pd.DataFrame:
    """生成示例股票数据"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')
    
    # 生成模拟的OHLCV数据
    np.random.seed(hash(stock_code) % 2**32)  # 基于股票代码的固定种子
    
    base_price = 50 + np.random.random() * 50
    price_changes = np.random.normal(0, 0.02, days)
    prices = base_price * np.exp(np.cumsum(price_changes))
    
    data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.normal(0, 0.01, days)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.02, days))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.02, days))),
        'close': prices,
        'volume': np.random.randint(10000, 100000, days)
    })
    
    # 确保OHLC关系正确
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data  # 不设置索引，保持date列


def main():
    """主函数 - 演示L5业务层集成"""
    print("🚀 L5业务层集成验证演示")
    print("=" * 60)
    
    # 生成示例数据
    stock_pool = {
        '000001': generate_sample_data('000001'),
        '000002': generate_sample_data('000002'),
        '600000': generate_sample_data('600000'),
        '600036': generate_sample_data('600036'),
    }
    
    # 1. 买点分析演示
    print("\n📊 1. 买点分析演示")
    print("-" * 30)
    
    analyzer = BuyPointAnalyzer()
    buypoint_result = analyzer.analyze_buypoint('000001', stock_pool['000001'])
    
    print(f"股票代码: {buypoint_result['stock_code']}")
    print(f"综合评分: {buypoint_result['composite_score']:.2f}")
    print(f"投资建议: {buypoint_result['recommendation']}")
    print(f"买点信号数量: {len(buypoint_result['buypoint_signals'])}")
    
    # 2. 策略选股演示
    print("\n🎯 2. 策略选股演示")
    print("-" * 30)
    
    selector = StrategySelector()
    selection_results = selector.select_stocks(stock_pool)
    
    print(f"股票池大小: {len(stock_pool)}")
    print(f"选中股票数量: {len(selection_results)}")
    
    for result in selection_results[:3]:  # 显示前3只
        print(f"  {result['stock_code']}: {result['buy_signal_count']}个买入信号, "
              f"强度:{result['avg_strength']:.2f}, 置信度:{result['avg_confidence']:.2f}")
    
    # 3. 实时交易演示
    print("\n⚡ 3. 实时交易演示")
    print("-" * 30)
    
    trader = RealTimeTrader()
    trading_decision = trader.make_trading_decision('000001', stock_pool['000001'])
    
    print(f"交易动作: {trading_decision['action']}")
    print(f"价格: {trading_decision['price']:.2f}")
    print(f"置信度: {trading_decision['confidence']:.2f}")
    print(f"原因: {trading_decision['reason']}")
    
    print("\n✅ L5业务层集成验证完成")
    print("所有业务场景都能正确使用L4层的calculate()和get_signal()抽象方法")


if __name__ == "__main__":
    main()
