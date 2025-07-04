#!/usr/bin/env python3
"""
ZXM吸筹+KDJ金叉选股策略演示版本
使用模拟数据展示2025年5月12日30分钟有ZXM吸筹信号且日线出现KDJ金叉的选股逻辑
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class ZXMKDJDemoStrategy:
    """ZXM吸筹+KDJ金叉选股策略演示版本"""
    
    def __init__(self):
        """初始化策略"""
        self.target_date = "2025-05-12"
        self.found_stocks = []
        
        # 模拟股票基础信息
        self.stock_info = {
            '000001': '平安银行',
            '000002': '万科A',
            '000858': '五粮液',
            '002415': '海康威视',
            '600036': '招商银行',
            '600519': '贵州茅台',
            '002594': '比亚迪',
            '300015': '爱尔眼科',
            '000063': '中兴通讯',
            '600000': '浦发银行'
        }
        
        logger.info("ZXM吸筹+KDJ金叉选股策略演示版本初始化完成")
    
    def generate_mock_kline_data(self, code: str, period: str, days: int) -> pd.DataFrame:
        """生成模拟K线数据"""
        try:
            # 根据不同股票和周期生成不同的数据特征
            np.random.seed(hash(code) % 1000)  # 使用股票代码作为随机种子，确保数据一致性
            
            # 基准价格
            base_price = 10.0 + (hash(code) % 100)
            
            # 生成时间序列
            if period == 'daily':
                periods = days
                freq = 'D'
            elif period == '30min':
                periods = days * 8  # 每天8个30分钟周期
                freq = '30T'
            else:
                periods = days
                freq = 'D'
            
            # 生成日期范围
            end_date = datetime.strptime(self.target_date, '%Y-%m-%d')
            start_date = end_date - timedelta(days=days)
            
            if period == 'daily':
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
            else:
                # 30分钟数据，只在交易时间生成
                dates = pd.date_range(start=start_date, end=end_date, freq='30T')
                # 过滤交易时间 (9:30-11:30, 13:00-15:00)
                dates = dates[
                    ((dates.hour == 9) & (dates.minute >= 30)) |
                    ((dates.hour == 10)) |
                    ((dates.hour == 11) & (dates.minute <= 30)) |
                    ((dates.hour == 13)) |
                    ((dates.hour == 14)) |
                    ((dates.hour == 15) & (dates.minute == 0))
                ]
            
            periods = len(dates)
            
            # 生成价格数据
            returns = np.random.normal(0, 0.02, periods)  # 日收益率
            prices = [base_price]
            
            for i in range(1, periods):
                new_price = prices[-1] * (1 + returns[i])
                prices.append(max(new_price, 0.1))  # 确保价格为正
            
            # 生成OHLC数据
            data = []
            for i, date in enumerate(dates):
                close = prices[i]
                volatility = 0.03 + np.random.random() * 0.02  # 波动率
                
                high = close * (1 + np.random.random() * volatility)
                low = close * (1 - np.random.random() * volatility)
                
                if i == 0:
                    open_price = close
                else:
                    open_price = prices[i-1] * (1 + np.random.normal(0, 0.01))
                
                # 确保OHLC关系正确
                high = max(high, open_price, close)
                low = min(low, open_price, close)
                
                # 成交量
                base_vol = 10000 + np.random.randint(0, 50000)
                vol = base_vol * (1 + np.random.random() * 2)
                
                # 成交额
                amount = vol * close * (0.8 + np.random.random() * 0.4)
                
                data.append({
                    'trade_date': date,
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close, 2),
                    'vol': int(vol),
                    'amount': round(amount, 2)
                })
            
            df = pd.DataFrame(data)
            df.set_index('trade_date', inplace=True)
            
            # 为特定股票在目标日期附近创建符合条件的数据
            if code in ['000858', '002415'] and period == 'daily':
                # 为五粮液和海康威视创建KDJ金叉信号
                target_idx = df.index.get_loc(df.index[df.index.date == datetime.strptime(self.target_date, '%Y-%m-%d').date()][0])
                if target_idx > 0:
                    # 调整前一天和当天的数据以产生金叉
                    df.iloc[target_idx-1, df.columns.get_loc('low')] = df.iloc[target_idx-1]['close'] * 0.95
                    df.iloc[target_idx, df.columns.get_loc('high')] = df.iloc[target_idx]['close'] * 1.05
            
            if code in ['000858', '002415'] and period == '30min':
                # 为五粮液和海康威视创建ZXM吸筹信号
                target_date_data = df[df.index.date == datetime.strptime(self.target_date, '%Y-%m-%d').date()]
                if len(target_date_data) > 0:
                    # 在目标日期增加成交量
                    target_indices = target_date_data.index
                    for idx in target_indices:
                        df.loc[idx, 'vol'] = df.loc[idx, 'vol'] * 2.0  # 成交量放大
                        # 控制振幅
                        close_price = df.loc[idx, 'close']
                        df.loc[idx, 'high'] = close_price * 1.02
                        df.loc[idx, 'low'] = close_price * 0.98
            
            return df
            
        except Exception as e:
            logger.error(f"生成模拟K线数据失败: {e}")
            return pd.DataFrame()
    
    def calculate_kdj(self, df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
        """计算KDJ指标"""
        try:
            if df is None or len(df) < n:
                return df
            
            # 计算RSV
            low_n = df['low'].rolling(window=n, min_periods=1).min()
            high_n = df['high'].rolling(window=n, min_periods=1).max()
            rsv = (df['close'] - low_n) / (high_n - low_n) * 100
            
            # 计算K值
            k = rsv.ewm(alpha=1/m1, adjust=False).mean()
            
            # 计算D值
            d = k.ewm(alpha=1/m2, adjust=False).mean()
            
            # 计算J值
            j = 3 * k - 2 * d
            
            df['kdj_k'] = k
            df['kdj_d'] = d
            df['kdj_j'] = j
            
            return df
            
        except Exception as e:
            logger.error(f"计算KDJ指标失败: {e}")
            return df
    
    def check_kdj_golden_cross(self, df: pd.DataFrame, target_date: str) -> bool:
        """检查KDJ金叉信号"""
        try:
            if df is None or len(df) < 2:
                return False
            
            # 确保有KDJ数据
            if 'kdj_k' not in df.columns or 'kdj_d' not in df.columns:
                return False
            
            # 获取目标日期和前一日的数据
            target_datetime = pd.to_datetime(target_date)
            
            # 找到目标日期及其前几天的数据
            target_data = df[df.index <= target_datetime].tail(3)
            
            if len(target_data) < 2:
                return False
            
            # 检查金叉：K线从下方穿越D线
            current_k = target_data['kdj_k'].iloc[-1]
            current_d = target_data['kdj_d'].iloc[-1]
            prev_k = target_data['kdj_k'].iloc[-2]
            prev_d = target_data['kdj_d'].iloc[-2]
            
            # 金叉条件：前一日K<D，当日K>D
            is_golden_cross = (prev_k < prev_d) and (current_k > current_d)
            
            if is_golden_cross:
                print(f"    📊 KDJ金叉详情: K从{prev_k:.2f}上穿D{prev_d:.2f} -> K{current_k:.2f}>D{current_d:.2f}")
            
            return is_golden_cross
            
        except Exception as e:
            logger.error(f"检查KDJ金叉失败: {e}")
            return False
    
    def check_zxm_absorption(self, df: pd.DataFrame, target_date: str) -> bool:
        """检查ZXM吸筹信号"""
        try:
            if df is None or len(df) < 10:
                return False
            
            # 获取目标日期的数据
            target_datetime = pd.to_datetime(target_date)
            target_data = df[df.index <= target_datetime].tail(10)
            
            if len(target_data) < 5:
                return False
            
            # ZXM吸筹信号判断逻辑：
            # 1. 成交量放大（当日成交量 > 前5日平均成交量的1.5倍）
            # 2. 价格相对稳定（振幅 < 5%）
            # 3. 收盘价接近最高价（收盘价 > (最高价+最低价)/2）
            
            current = target_data.iloc[-1]
            recent_5_vol = target_data['vol'].iloc[-6:-1].mean() if len(target_data) >= 6 else target_data['vol'].iloc[:-1].mean()
            
            # 条件1：成交量放大
            volume_amplified = current['vol'] > recent_5_vol * 1.5
            
            # 条件2：振幅控制
            amplitude = (current['high'] - current['low']) / current['low'] * 100
            amplitude_controlled = amplitude < 5.0
            
            # 条件3：收盘价位置
            mid_price = (current['high'] + current['low']) / 2
            close_position_good = current['close'] > mid_price
            
            # 条件4：价格上涨或持平
            if len(target_data) >= 2:
                prev_close = target_data.iloc[-2]['close']
                price_stable_or_up = current['close'] >= prev_close * 0.98  # 允许小幅下跌
            else:
                price_stable_or_up = True
            
            is_absorption = volume_amplified and amplitude_controlled and close_position_good and price_stable_or_up
            
            if is_absorption:
                print(f"    📈 ZXM吸筹详情: 成交量{current['vol']:.0f}(放大{current['vol']/recent_5_vol:.1f}倍), "
                      f"振幅{amplitude:.2f}%, 收盘位置良好")
            
            return is_absorption
            
        except Exception as e:
            logger.error(f"检查ZXM吸筹信号失败: {e}")
            return False
    
    def analyze_stock(self, code: str) -> bool:
        """分析单只股票"""
        try:
            logger.info(f"分析股票: {code}")
            
            # 获取股票名称
            stock_name = self.stock_info.get(code, f"股票{code}")
            
            # 生成模拟数据
            daily_data = self.generate_mock_kline_data(code, 'daily', 60)
            min30_data = self.generate_mock_kline_data(code, '30min', 30)
            
            if daily_data.empty or min30_data.empty:
                logger.warning(f"{code} 数据生成失败")
                return False
            
            # 计算日线KDJ
            daily_data = self.calculate_kdj(daily_data)
            
            # 检查日线KDJ金叉
            has_kdj_golden_cross = self.check_kdj_golden_cross(daily_data, self.target_date)
            
            # 检查30分钟ZXM吸筹信号
            has_zxm_absorption = self.check_zxm_absorption(min30_data, self.target_date)
            
            # 打印分析结果
            print(f"  📊 {code} - {stock_name}")
            print(f"    ✅ 日线KDJ金叉: {'是' if has_kdj_golden_cross else '否'}")
            print(f"    ✅ 30分钟ZXM吸筹: {'是' if has_zxm_absorption else '否'}")
            
            # 如果同时满足两个条件
            if has_kdj_golden_cross and has_zxm_absorption:
                result = {
                    'code': code,
                    'name': stock_name,
                    'date': self.target_date,
                    'kdj_golden_cross': has_kdj_golden_cross,
                    'zxm_absorption': has_zxm_absorption
                }
                
                self.found_stocks.append(result)
                
                print(f"\n🎯 找到符合条件的股票!")
                print(f"📈 股票代码: {code}")
                print(f"📊 股票名称: {stock_name}")
                print(f"📅 分析日期: {self.target_date}")
                print(f"✅ 日线KDJ金叉: {has_kdj_golden_cross}")
                print(f"✅ 30分钟ZXM吸筹: {has_zxm_absorption}")
                print("="*50)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"分析股票{code}失败: {e}")
            return False
    
    def run_strategy(self) -> Optional[Dict]:
        """运行选股策略"""
        try:
            print(f"\n🚀 开始执行ZXM吸筹+KDJ金叉选股策略 (演示版本)")
            print(f"📅 目标日期: {self.target_date}")
            print(f"🔍 筛选条件:")
            print(f"   1. 日线KDJ金叉")
            print(f"   2. 30分钟ZXM吸筹信号")
            print(f"💡 注意: 使用模拟数据进行演示")
            print("="*50)
            
            # 获取股票列表
            stock_list = list(self.stock_info.keys())
            
            # 逐个分析股票
            for i, code in enumerate(stock_list, 1):
                print(f"\n[{i}/{len(stock_list)}] 正在分析: {code}")
                
                try:
                    # 分析股票
                    if self.analyze_stock(code):
                        # 找到一个就停止
                        print(f"\n✅ 策略执行完成，找到符合条件的股票!")
                        return self.found_stocks[0]
                    
                except Exception as e:
                    logger.error(f"分析股票{code}时发生异常: {e}")
                    continue
            
            print(f"\n❌ 未找到符合条件的股票")
            print(f"📊 共分析了 {len(stock_list)} 只股票")
            return None
            
        except Exception as e:
            logger.error(f"策略执行失败: {e}")
            return None


def main():
    """主函数"""
    try:
        print("🎯 ZXM吸筹+KDJ金叉选股策略演示")
        print("=" * 60)
        
        # 创建策略实例
        strategy = ZXMKDJDemoStrategy()
        
        # 运行策略
        result = strategy.run_strategy()
        
        if result:
            print(f"\n🎉 策略执行成功!")
            print(f"📈 找到股票: {result['code']} - {result['name']}")
            print(f"📅 分析日期: {result['date']}")
            print(f"\n💡 说明:")
            print(f"   • 这是使用模拟数据的演示版本")
            print(f"   • 实际使用时需要连接ClickHouse数据库")
            print(f"   • 策略逻辑已经完整实现")
        else:
            print(f"\n😔 未找到符合条件的股票")
            print(f"💡 可以调整筛选条件或扩大股票范围")
            
    except Exception as e:
        print(f"\n❌ 策略执行失败: {e}")
        logger.error(f"主函数执行失败: {e}")


if __name__ == "__main__":
    main() 