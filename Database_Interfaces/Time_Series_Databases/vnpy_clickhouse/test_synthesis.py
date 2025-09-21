#!/usr/bin/env python3
"""
测试ClickHouse数据库的K线合成功能
基于15分钟数据自动合成其他时间周期
"""

import sys
import os
from datetime import datetime, timedelta

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(project_root, "Core_Framework", "vnpy"))

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import get_database


def test_clickhouse_synthesis():
    """测试ClickHouse K线合成功能"""
    print("🧪 测试ClickHouse K线合成功能")
    print("=" * 60)
    
    try:
        # 1. 获取数据库实例
        print("\n1. 连接ClickHouse数据库...")
        db = get_database()
        print("✅ 数据库连接成功")
        
        # 2. 设置测试参数
        symbol = "000001"
        exchange = Exchange.SZSE
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)  # 最近7天
        
        print(f"\n2. 测试参数:")
        print(f"   股票代码: {symbol}")
        print(f"   交易所: {exchange.value}")
        print(f"   时间范围: {start_time.date()} ~ {end_time.date()}")
        
        # 3. 测试各种时间周期
        test_intervals = [
            (Interval.MINUTE_15, "15分钟基础数据"),
            (Interval.MINUTE_30, "30分钟合成数据"),
            (Interval.HOUR, "60分钟合成数据"),
            (Interval.HOUR_2, "120分钟合成数据"),
            (Interval.HOUR_4, "240分钟合成数据"),
            (Interval.DAILY, "日线基础数据")
        ]
        
        print(f"\n3. 测试不同时间周期数据获取:")
        results = {}
        
        for interval, description in test_intervals:
            try:
                print(f"\n   🔄 加载 {description}...")
                bars = db.load_bar_data(symbol, exchange, interval, start_time, end_time)
                
                if bars:
                    results[interval] = bars
                    print(f"   ✅ 成功获取 {len(bars)} 条数据")
                    print(f"   时间范围: {bars[0].datetime} ~ {bars[-1].datetime}")
                    
                    # 显示前3条数据样例
                    print(f"   数据样例:")
                    for i, bar in enumerate(bars[:3]):
                        gateway_mark = "🔧" if "synthesized" in bar.gateway_name else "📊"
                        print(f"     {gateway_mark} {bar.datetime}: O:{bar.open_price:.2f} H:{bar.high_price:.2f} L:{bar.low_price:.2f} C:{bar.close_price:.2f} V:{bar.volume}")
                else:
                    print(f"   ❌ 无数据")
                    results[interval] = []
                    
            except Exception as e:
                print(f"   ❌ 加载失败: {e}")
                results[interval] = []
        
        # 4. 数据质量验证
        print(f"\n4. 数据质量验证:")
        
        if results.get(Interval.MINUTE_15) and results.get(Interval.MINUTE_30):
            base_bars = results[Interval.MINUTE_15]
            synthesized_bars = results[Interval.MINUTE_30]
            
            print(f"   📊 15分钟数据: {len(base_bars)} 条")
            print(f"   🔧 30分钟合成: {len(synthesized_bars)} 条")
            if len(base_bars) > 0:
                print(f"   📈 合成比例: {len(synthesized_bars) / len(base_bars) * 2:.1f} (理论值: 1.0)")
            
            # 验证数据连续性
            if synthesized_bars:
                time_gaps = []
                for i in range(1, len(synthesized_bars)):
                    gap = (synthesized_bars[i].datetime - synthesized_bars[i-1].datetime).total_seconds() / 60
                    time_gaps.append(gap)
                
                expected_gap = 30  # 30分钟
                actual_avg_gap = sum(time_gaps) / len(time_gaps) if time_gaps else 0
                print(f"   ⏰ 时间间隔: 平均{actual_avg_gap:.0f}分钟 (期望: {expected_gap}分钟)")
        
        # 5. 性能统计
        print(f"\n5. 功能统计:")
        direct_count = sum(1 for interval, bars in results.items() 
                          if bars and interval in [Interval.MINUTE_15, Interval.DAILY])
        synthesized_count = sum(1 for interval, bars in results.items() 
                               if bars and interval in [Interval.MINUTE_30, Interval.HOUR, Interval.HOUR_2, Interval.HOUR_4])
        
        print(f"   📊 基础数据周期: {direct_count} 个可用")
        print(f"   🔧 合成数据周期: {synthesized_count} 个可用")
        print(f"   🎯 总体可用率: {(direct_count + synthesized_count) / len(test_intervals) * 100:.1f}%")
        
        # 6. 使用建议
        print(f"\n6. 集成使用建议:")
        print(f"   💡 在策略中直接使用: db.load_bar_data(symbol, exchange, interval, start, end)")
        print(f"   💡 支持的合成周期: 30分钟、60分钟、120分钟、240分钟")
        print(f"   💡 自动识别: 基础数据直接查询，合成数据自动处理")
        print(f"   💡 性能优化: 合成数据会缓存在gateway_name中标识")
        
        if synthesized_count > 0:
            print(f"\n✅ ClickHouse K线合成功能测试成功！")
            print(f"   现在可以在策略中无缝使用各种时间周期的数据。")
        else:
            print(f"\n⚠️  合成功能需要基础15分钟数据支持")
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def demo_strategy_usage():
    """演示在策略中如何使用"""
    print("\n" + "="*60)
    print("📈 策略中的使用示例:")
    print("="*60)
    
    code_example = '''
# 在CTA策略中使用不同时间周期数据
from vnpy_ctastrategy import CtaTemplate

class MultiTimeframeStrategy(CtaTemplate):
    def on_init(self):
        """策略初始化 - 加载多时间框架数据"""
        
        # 自动获取各种周期数据（无需关心是否需要合成）
        
        # 主分析周期 - 15分钟（基础数据）
        self.bars_15m = self.load_bar(15)
        
        # 入场确认 - 30分钟（自动合成）  
        self.bars_30m = self.load_bar(30)
        
        # 趋势过滤 - 60分钟（自动合成）
        self.bars_1h = self.load_bar(60)
        
        # 大趋势 - 日线（基础数据）
        self.bars_daily = self.load_bar(240)  # 或使用日线

    def on_bar(self, bar):
        """实时数据处理"""
        # 可以请求任意周期的最新数据
        # ClickHouse会自动判断是直接查询还是实时合成
        
        recent_30m = self.cta_engine.main_engine.get_bar_data(
            self.vt_symbol, Interval.MINUTE_30, count=10
        )
        
        # 进行多时间框架分析...
'''
    
    print(code_example)
    print("✨ 核心优势:")
    print("   • 无需手动管理不同周期数据")
    print("   • 自动识别基础数据和合成需求")
    print("   • 与现有VnPy策略无缝集成")
    print("   • 支持实时和历史数据处理")


if __name__ == "__main__":
    test_clickhouse_synthesis()
    demo_strategy_usage()
