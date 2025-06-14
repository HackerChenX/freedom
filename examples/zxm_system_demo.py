"""
ZXM体系实际应用示例
基于ZXM体系3.0版权威文档的完整交易系统演示
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from indicators.zxm_absorb import ZXMAbsorb
from indicators.zxm_washplate import ZXMWashPlate


def generate_sample_data(periods=150, start_price=100):
    """生成示例股票数据"""
    np.random.seed(42)  # 确保结果可重现

    dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

    # 生成价格序列
    returns = np.random.normal(0.001, 0.02, periods)  # 日收益率
    prices = [start_price]

    for i in range(1, periods):
        # 添加趋势性
        trend = 0.0005 if i < periods * 0.7 else -0.0002
        price = prices[-1] * (1 + returns[i] + trend)
        prices.append(max(price, start_price * 0.5))  # 防止价格过低

    # 生成OHLC数据
    data = []
    for i, price in enumerate(prices):
        volatility = 0.015
        high = price * (1 + np.random.uniform(0, volatility))
        low = price * (1 - np.random.uniform(0, volatility))
        open_price = prices[i-1] if i > 0 else price
        close = price

        # 确保OHLC逻辑正确
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # 生成成交量
        volume = np.random.randint(1000000, 5000000)

        data.append({
            'datetime': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'code': 'DEMO001',
            'name': '演示股票',
            'level': 1,
            'industry': '科技',
            'seq': i,
            'turnover': volume * close,
            'turnover_rate': 0.05,
            'price_change': close - open_price,
            'price_range': high - low
        })

    return pd.DataFrame(data)


class ZXMTradingSystem:
    """ZXM交易体系实现"""
    
    def __init__(self):
        self.zxm_absorb = ZXMAbsorb()
        self.zxm_washplate = ZXMWashPlate()
    
    def analyze_stock(self, data: pd.DataFrame) -> dict:
        """
        分析股票的ZXM信号
        
        Args:
            data: 股票OHLCV数据
            
        Returns:
            dict: 分析结果
        """
        # 计算ZXM吸筹指标
        absorb_result = self.zxm_absorb.calculate(data)
        absorb_score = self.zxm_absorb.calculate_raw_score(data)
        absorb_patterns = self.zxm_absorb.identify_patterns(data)
        
        # 计算ZXM洗盘指标
        washplate_result = self.zxm_washplate.calculate(data)
        washplate_score = self.zxm_washplate.calculate_raw_score(data)
        washplate_patterns = self.zxm_washplate.identify_patterns(data)
        
        # 获取最新信号
        latest_absorb = absorb_result.iloc[-1] if not absorb_result.empty else None
        latest_washplate = washplate_result.iloc[-1] if not washplate_result.empty else None
        
        # 验证买点四要素
        buy_point_validation = None
        if len(data) >= 120:
            buy_point_validation = self.zxm_absorb.validate_buy_point_four_elements(data, len(data)-1)
        
        return {
            'absorb_signal': bool(latest_absorb['BUY']) if latest_absorb is not None else False,
            'absorb_strength': int(latest_absorb['XG']) if latest_absorb is not None else 0,
            'absorb_score': float(absorb_score.iloc[-1]) if not absorb_score.empty else 50.0,
            'absorb_patterns': absorb_patterns,
            'washplate_score': float(washplate_score.iloc[-1]) if not washplate_score.empty else 50.0,
            'washplate_patterns': washplate_patterns,
            'buy_point_validation': buy_point_validation,
            'v11_value': float(latest_absorb['EMA_V11_3']) if latest_absorb is not None else None,
            'v12_value': float(latest_absorb['V12']) if latest_absorb is not None else None,
            'recommendation': self._generate_recommendation(
                absorb_score.iloc[-1] if not absorb_score.empty else 50.0,
                washplate_score.iloc[-1] if not washplate_score.empty else 50.0,
                bool(latest_absorb['BUY']) if latest_absorb is not None else False,
                buy_point_validation
            )
        }
    
    def _generate_recommendation(self, absorb_score: float, washplate_score: float, 
                               absorb_signal: bool, buy_point_validation: dict) -> dict:
        """
        生成投资建议
        
        Args:
            absorb_score: 吸筹评分
            washplate_score: 洗盘评分
            absorb_signal: 吸筹信号
            buy_point_validation: 买点四要素验证
            
        Returns:
            dict: 投资建议
        """
        # 综合评分
        total_score = (absorb_score + washplate_score) / 2
        
        # 买点四要素评分
        four_elements_score = 0
        if buy_point_validation:
            four_elements_score = sum(buy_point_validation.values()) * 25  # 每个要素25分
        
        # 最终评分
        final_score = (total_score * 0.7 + four_elements_score * 0.3)
        
        # 生成建议
        if final_score >= 80 and absorb_signal:
            action = "强烈买入"
            position_size = 0.6
            risk_level = "低"
        elif final_score >= 70 and absorb_signal:
            action = "买入"
            position_size = 0.4
            risk_level = "中低"
        elif final_score >= 60:
            action = "关注"
            position_size = 0.2
            risk_level = "中"
        elif final_score >= 50:
            action = "观望"
            position_size = 0.0
            risk_level = "中"
        else:
            action = "回避"
            position_size = 0.0
            risk_level = "高"
        
        return {
            'action': action,
            'final_score': final_score,
            'position_size': position_size,
            'risk_level': risk_level,
            'absorb_score': absorb_score,
            'washplate_score': washplate_score,
            'four_elements_score': four_elements_score
        }


def demo_zxm_system():
    """演示ZXM体系的使用"""
    print("=" * 60)
    print("ZXM体系3.0版交易系统演示")
    print("基于权威文档的核心吸筹公式和买点四要素")
    print("=" * 60)
    
    # 创建ZXM交易系统
    zxm_system = ZXMTradingSystem()
    
    # 生成模拟股票数据
    print("\n1. 生成模拟股票数据...")
    
    # 场景1：上升趋势中的吸筹信号
    print("\n场景1：上升趋势中的吸筹机会")
    trend_data = generate_sample_data(periods=150, start_price=100)

    analysis1 = zxm_system.analyze_stock(trend_data)
    print_analysis_result("上升趋势股票", analysis1)

    # 场景2：洗盘后的机会
    print("\n场景2：洗盘后的投资机会")
    washplate_data = generate_sample_data(periods=150, start_price=110)

    analysis2 = zxm_system.analyze_stock(washplate_data)
    print_analysis_result("洗盘后股票", analysis2)

    # 场景3：下降趋势中的风险
    print("\n场景3：下降趋势中的风险识别")
    decline_data = generate_sample_data(periods=150, start_price=90)

    analysis3 = zxm_system.analyze_stock(decline_data)
    print_analysis_result("下降趋势股票", analysis3)
    
    print("\n" + "=" * 60)
    print("ZXM体系核心要点总结：")
    print("1. 供需格局是股价波动的第一性原理")
    print("2. 买点四要素：趋势不破、缩量、回踩支撑、BS信号")
    print("3. V11指标低位(≤13)且V12上升(>13)是强吸筹信号")
    print("4. XG≥3表示近6日内满足吸筹条件次数达标")
    print("5. 洗盘形态完成后往往是较好的买入时机")
    print("6. 月均收益5%即可实现长期财富增长")
    print("=" * 60)


def print_analysis_result(stock_name: str, analysis: dict):
    """打印分析结果"""
    print(f"\n📊 {stock_name} ZXM分析结果:")
    print("-" * 40)
    
    # 核心信号
    print(f"🎯 吸筹信号: {'✅ 是' if analysis['absorb_signal'] else '❌ 否'}")
    print(f"💪 吸筹强度: {analysis['absorb_strength']}/6")
    print(f"📈 吸筹评分: {analysis['absorb_score']:.1f}/100")
    print(f"🔄 洗盘评分: {analysis['washplate_score']:.1f}/100")
    
    # V11/V12指标
    if analysis['v11_value'] is not None:
        print(f"📊 V11指标: {analysis['v11_value']:.2f} ({'低位' if analysis['v11_value'] <= 13 else '高位'})")
    if analysis['v12_value'] is not None:
        print(f"📈 V12动量: {analysis['v12_value']:.2f} ({'上升' if analysis['v12_value'] > 0 else '下降'})")
    
    # 买点四要素
    if analysis['buy_point_validation']:
        print(f"\n🎯 买点四要素验证:")
        validation = analysis['buy_point_validation']
        print(f"   趋势不破: {'✅' if validation['trend_intact'] else '❌'}")
        print(f"   缩量确认: {'✅' if validation['volume_shrink'] else '❌'}")
        print(f"   回踩支撑: {'✅' if validation['pullback_support'] else '❌'}")
        print(f"   BS信号: {'✅' if validation['bs_signal'] else '❌'}")
    
    # 形态识别
    if analysis['absorb_patterns']:
        print(f"\n🔍 吸筹形态: {', '.join(analysis['absorb_patterns'])}")
    if analysis['washplate_patterns']:
        print(f"🌊 洗盘形态: {', '.join(analysis['washplate_patterns'])}")
    
    # 投资建议
    rec = analysis['recommendation']
    print(f"\n💡 投资建议:")
    print(f"   操作建议: {rec['action']}")
    print(f"   综合评分: {rec['final_score']:.1f}/100")
    print(f"   建议仓位: {rec['position_size']*100:.0f}%")
    print(f"   风险等级: {rec['risk_level']}")


if __name__ == "__main__":
    demo_zxm_system()
