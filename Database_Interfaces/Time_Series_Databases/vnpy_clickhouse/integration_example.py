#!/usr/bin/env python3
"""
ClickHouse多时间周期数据集成使用示例
展示如何在策略中使用扩展的时间周期
"""

import sys
import os
from datetime import datetime, timedelta

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(project_root, "Core_Framework", "vnpy"))

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import get_database


def demo_multi_timeframe_usage():
    """演示多时间周期数据使用"""
    print("📈 ClickHouse多时间周期数据集成演示")
    print("=" * 60)
    
    try:
        # 1. 获取数据库实例
        db = get_database()
        print("✅ 数据库连接成功")
        
        # 2. 演示不同时间周期的自动处理
        symbol = "000001"
        exchange = Exchange.SZSE
        end_time = datetime.now()
        start_time = end_time - timedelta(days=5)
        
        print(f"\n🔧 演示时间周期自动处理:")
        print(f"   标的: {symbol} ({exchange.value})")
        print(f"   时间: {start_time.date()} ~ {end_time.date()}")
        
        # 3. 展示新扩展的Interval常量
        print(f"\n📊 扩展的时间周期常量:")
        intervals = [
            (Interval.MINUTE, "1分钟"),
            (Interval.MINUTE_5, "5分钟"),
            (Interval.MINUTE_15, "15分钟 (基础数据)"),
            (Interval.MINUTE_30, "30分钟 (自动合成)"),
            (Interval.HOUR, "1小时 (自动合成)"),
            (Interval.HOUR_2, "2小时 (自动合成)"),
            (Interval.HOUR_4, "4小时 (自动合成)"),
            (Interval.DAILY, "日线"),
            (Interval.WEEKLY, "周线"),
            (Interval.MONTHLY, "月线")
        ]
        
        for interval, description in intervals:
            print(f"   • {interval.name:12} = '{interval.value:4}' - {description}")
        
        # 4. 演示统一接口调用
        print(f"\n🎯 统一接口调用演示:")
        print(f"   # 无论基础数据还是合成数据，都用同一个接口:")
        print(f"   bars = db.load_bar_data(symbol, exchange, interval, start, end)")
        print(f"")
        
        for interval, description in intervals[2:7]:  # 选择几个重要的周期
            print(f"   # {description}")
            print(f"   bars_{interval.value} = db.load_bar_data('{symbol}', Exchange.{exchange.name}, Interval.{interval.name}, start, end)")
            print(f"   # -> 自动识别: {'直接查询数据库' if '基础' in description else '基于15分钟合成'}")
            print()
        
        # 5. 在CTA策略中的集成示例
        print(f"\n🚀 CTA策略集成示例:")
        strategy_code = '''
class MultiTimeframeStrategy(CtaTemplate):
    """多时间周期策略示例"""
    
    def on_init(self):
        """策略初始化"""
        self.write_log("多时间周期策略初始化")
        
        # 加载不同周期的历史数据
        # VnPy会自动调用load_bar_data，无需关心合成逻辑
        self.load_bar(15)    # 15分钟 - 主分析周期
        self.load_bar(30)    # 30分钟 - 入场确认 (自动合成)
        self.load_bar(60)    # 1小时 - 趋势过滤 (自动合成)
        
    def on_bar(self, bar):
        """K线数据更新"""
        if bar.interval == Interval.MINUTE_15:
            # 主周期分析
            self.analyze_main_timeframe(bar)
            
        elif bar.interval == Interval.MINUTE_30:
            # 确认信号
            self.confirm_signals(bar)
            
        elif bar.interval == Interval.HOUR:
            # 趋势判断
            self.analyze_trend(bar)
    
    def analyze_main_timeframe(self, bar):
        """主要周期分析 (15分钟)"""
        # 获取最近的30分钟数据用于确认
        recent_30m = self.cta_engine.main_engine.database.load_bar_data(
            self.vt_symbol.split('.')[0],
            Exchange.SZSE,
            Interval.MINUTE_30,  # 自动合成
            bar.datetime - timedelta(hours=2),
            bar.datetime
        )
        
        if recent_30m:
            self.write_log(f"获取到 {len(recent_30m)} 条30分钟合成数据")
'''
        
        print(strategy_code)
        
        # 6. 核心优势总结
        print(f"\n✨ 核心优势:")
        advantages = [
            "🔄 自动数据合成: 基于15分钟数据自动生成30分钟、60分钟、120分钟、240分钟数据",
            "🎯 统一接口: 所有时间周期使用相同的load_bar_data接口",
            "⚡ 透明处理: 策略无需关心数据来源（直接查询 vs 实时合成）",
            "📊 扩展Interval: 新增MINUTE_5, MINUTE_15, MINUTE_30, HOUR_2, HOUR_4, MONTHLY",
            "🚀 无缝集成: 与现有VnPy策略框架完全兼容",
            "🔧 智能缓存: 合成数据标记为'clickhouse_synthesized'便于识别",
            "⏰ 精确对齐: 自动处理时间边界对齐和数据完整性"
        ]
        
        for advantage in advantages:
            print(f"   {advantage}")
        
        print(f"\n🎉 集成完成!")
        print(f"   现在可以在任何VnPy策略中使用扩展的时间周期！")
        print(f"   数据库接口会自动判断是直接查询还是实时合成。")
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_multi_timeframe_usage()
