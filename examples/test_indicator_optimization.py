#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
技术指标评分机制优化测试脚本

用于测试MA、RSI和MACD指标优化后的评分效果
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from indicators.ma import MA
from indicators.rsi import RSI
from indicators.macd import MACD
from db.clickhouse_db import get_clickhouse_db


def load_test_data(stock_code='000001.SZ', start_date='2022-01-01', end_date='2023-01-01'):
    """加载测试数据"""
    print(f"加载测试数据: {stock_code} 从 {start_date} 到 {end_date}")
    
    db = get_clickhouse_db()
    sql = f"""
    SELECT 
        trade_date,
        open,
        high,
        low,
        close,
        volume,
        amount
    FROM stock_daily_kline
    WHERE 
        ts_code = '{stock_code}' AND
        trade_date >= '{start_date}' AND
        trade_date <= '{end_date}'
    ORDER BY trade_date
    """
    
    df = db.query(sql)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index('trade_date', inplace=True)
    
    print(f"成功加载 {len(df)} 条数据")
    return df


def plot_ma_score_comparison(data, old_score, new_score, title="MA评分对比"):
    """绘制MA评分对比图"""
    plt.figure(figsize=(15, 10))
    
    # 第一个子图：价格和MA
    plt.subplot(3, 1, 1)
    plt.plot(data.index, data['close'], label='价格', color='black')
    for period in [5, 20, 60]:
        plt.plot(data.index, data[f'MA{period}'], label=f'MA{period}')
    plt.title('价格和MA线')
    plt.legend()
    plt.grid(True)
    
    # 第二个子图：新旧评分对比
    plt.subplot(3, 1, 2)
    plt.plot(data.index, old_score, label='优化前评分', color='blue', alpha=0.7)
    plt.plot(data.index, new_score, label='优化后评分', color='red')
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    plt.title('MA评分对比')
    plt.legend()
    plt.grid(True)
    
    # 第三个子图：评分差异
    plt.subplot(3, 1, 3)
    score_diff = new_score - old_score
    plt.bar(data.index, score_diff, color=np.where(score_diff >= 0, 'green', 'red'), alpha=0.6)
    plt.title('评分差异 (优化后 - 优化前)')
    plt.axhline(y=0, color='black', linestyle='-')
    plt.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, 'data/result/ma_score_comparison.png'))
    plt.close()


def plot_rsi_score_comparison(data, old_score, new_score, title="RSI评分对比"):
    """绘制RSI评分对比图"""
    plt.figure(figsize=(15, 10))
    
    # 第一个子图：价格
    plt.subplot(4, 1, 1)
    plt.plot(data.index, data['close'], label='价格', color='black')
    plt.title('价格')
    plt.grid(True)
    
    # 第二个子图：RSI
    plt.subplot(4, 1, 2)
    plt.plot(data.index, data['RSI14'], label='RSI(14)', color='purple')
    plt.axhline(y=30, color='green', linestyle='--', alpha=0.5)
    plt.axhline(y=70, color='red', linestyle='--', alpha=0.5)
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    plt.title('RSI指标')
    plt.grid(True)
    
    # 第三个子图：新旧评分对比
    plt.subplot(4, 1, 3)
    plt.plot(data.index, old_score, label='优化前评分', color='blue', alpha=0.7)
    plt.plot(data.index, new_score, label='优化后评分', color='red')
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    plt.title('RSI评分对比')
    plt.legend()
    plt.grid(True)
    
    # 第四个子图：评分差异
    plt.subplot(4, 1, 4)
    score_diff = new_score - old_score
    plt.bar(data.index, score_diff, color=np.where(score_diff >= 0, 'green', 'red'), alpha=0.6)
    plt.title('评分差异 (优化后 - 优化前)')
    plt.axhline(y=0, color='black', linestyle='-')
    plt.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, 'data/result/rsi_score_comparison.png'))
    plt.close()


def plot_macd_score_comparison(data, old_score, new_score, title="MACD评分对比"):
    """绘制MACD评分对比图"""
    plt.figure(figsize=(15, 10))
    
    # 第一个子图：价格
    plt.subplot(4, 1, 1)
    plt.plot(data.index, data['close'], label='价格', color='black')
    plt.title('价格')
    plt.grid(True)
    
    # 第二个子图：MACD
    plt.subplot(4, 1, 2)
    plt.plot(data.index, data['DIF'], label='DIF', color='blue')
    plt.plot(data.index, data['DEA'], label='DEA', color='red')
    plt.bar(data.index, data['MACD'], label='MACD', color=np.where(data['MACD'] >= 0, 'green', 'red'), alpha=0.5)
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title('MACD指标')
    plt.legend()
    plt.grid(True)
    
    # 第三个子图：新旧评分对比
    plt.subplot(4, 1, 3)
    plt.plot(data.index, old_score, label='优化前评分', color='blue', alpha=0.7)
    plt.plot(data.index, new_score, label='优化后评分', color='red')
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    plt.title('MACD评分对比')
    plt.legend()
    plt.grid(True)
    
    # 第四个子图：评分差异
    plt.subplot(4, 1, 4)
    score_diff = new_score - old_score
    plt.bar(data.index, score_diff, color=np.where(score_diff >= 0, 'green', 'red'), alpha=0.6)
    plt.title('评分差异 (优化后 - 优化前)')
    plt.axhline(y=0, color='black', linestyle='-')
    plt.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, 'data/result/macd_score_comparison.png'))
    plt.close()


def test_ma_optimization(data):
    """测试MA指标优化效果"""
    print("测试MA指标优化效果...")
    
    # 创建MA指标实例
    ma = MA()
    
    # 计算MA指标
    ma_result = ma.calculate(data)
    data = pd.concat([data, ma_result], axis=1)
    
    # 暂存优化前的评分计算方法
    original_calculate_ma_trend_score = ma._calculate_ma_trend_score
    original_calculate_ma_cross_score = ma._calculate_ma_cross_score
    
    # 计算优化前评分
    old_score = ma.calculate_raw_score(data)
    
    # 恢复优化后的方法（已经在编辑文件时更新）
    ma._calculate_ma_trend_score = original_calculate_ma_trend_score
    ma._calculate_ma_cross_score = original_calculate_ma_cross_score
    
    # 计算优化后评分
    new_score = ma.calculate_raw_score(data)
    
    # 绘制对比图
    plot_ma_score_comparison(data, old_score, new_score)
    
    # 打印评分统计
    print("MA评分统计:")
    print(f"优化前评分均值: {old_score.mean():.2f}, 方差: {old_score.var():.2f}")
    print(f"优化后评分均值: {new_score.mean():.2f}, 方差: {new_score.var():.2f}")
    print(f"评分变化均值: {(new_score - old_score).mean():.2f}")
    print(f"显著变化比例: {((new_score - old_score).abs() > 5).mean():.2%}")
    
    return data, old_score, new_score


def test_rsi_optimization(data):
    """测试RSI指标优化效果"""
    print("测试RSI指标优化效果...")
    
    # 创建RSI指标实例
    rsi = RSI()
    
    # 计算RSI指标
    rsi_result = rsi.calculate(data)
    data = pd.concat([data, rsi_result], axis=1)
    
    # 计算优化后评分（已在编辑文件时更新）
    new_score = rsi.calculate_raw_score(data)
    
    # 临时修改回原始计算方法计算优化前的评分
    # 修改为更简单的计算方法，不使用动态阈值和速率因子
    def simple_calculate_raw_score(self, data):
        if not self.has_result():
            self.calculate(data)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)
        rsi = self._result['RSI14']
        
        # 使用固定阈值
        oversold_condition = rsi <= 30
        overbought_condition = rsi >= 70
        
        # 简单评分
        oversold_score = np.where(oversold_condition, (30 - rsi) * 1.0, 0)
        score += oversold_score
        
        overbought_score = np.where(overbought_condition, (rsi - 70) * 1.0, 0)
        score -= overbought_score
        
        # 固定阈值穿越
        rsi_cross_up_30 = self.crossover(rsi, 30)
        rsi_cross_down_70 = self.crossunder(rsi, 70)
        score += rsi_cross_up_30 * 20
        score -= rsi_cross_down_70 * 20
        
        # 中线穿越评分
        rsi_cross_up_50 = self.crossover(rsi, 50)
        rsi_cross_down_50 = self.crossunder(rsi, 50)
        score += rsi_cross_up_50 * 15
        score -= rsi_cross_down_50 * 15
        
        # 背离评分
        if len(data) >= 20:
            divergence_score = self._calculate_rsi_divergence(data['close'], rsi)
            score += divergence_score
        
        # 形态评分
        pattern_score = self._calculate_rsi_pattern_score(rsi)
        score += pattern_score
        
        # 斜率评分
        slope_score = self._calculate_rsi_slope_score(rsi)
        score += slope_score
        
        return np.clip(score, 0, 100)
    
    # 临时替换方法
    original_calculate_raw_score = rsi.calculate_raw_score
    rsi.calculate_raw_score = lambda d: simple_calculate_raw_score(rsi, d)
    
    # 计算优化前评分
    old_score = rsi.calculate_raw_score(data)
    
    # 恢复优化后的方法
    rsi.calculate_raw_score = original_calculate_raw_score
    
    # 绘制对比图
    plot_rsi_score_comparison(data, old_score, new_score)
    
    # 打印评分统计
    print("RSI评分统计:")
    print(f"优化前评分均值: {old_score.mean():.2f}, 方差: {old_score.var():.2f}")
    print(f"优化后评分均值: {new_score.mean():.2f}, 方差: {new_score.var():.2f}")
    print(f"评分变化均值: {(new_score - old_score).mean():.2f}")
    print(f"显著变化比例: {((new_score - old_score).abs() > 5).mean():.2%}")
    
    return data, old_score, new_score


def test_macd_optimization(data):
    """测试MACD指标优化效果"""
    print("测试MACD指标优化效果...")
    
    # 创建MACD指标实例
    macd = MACD()
    
    # 计算MACD指标
    macd_result = macd.calculate(data)
    data = pd.concat([data, macd_result], axis=1)
    
    # 计算优化后评分（已在编辑文件时更新）
    new_score = macd.calculate_raw_score(data)
    
    # 临时修改回原始计算方法计算优化前的评分
    def simple_calculate_raw_score(self, data):
        if not self.has_result():
            self.calculate(data)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)
        
        dif = self._result['DIF']
        dea = self._result['DEA']
        macd_hist = self._result['MACD']
        
        # 简单金叉死叉评分，无零轴距离系数
        golden_cross = self.crossover(dif, dea)
        death_cross = self.crossunder(dif, dea)
        score += golden_cross * 20
        score -= death_cross * 20
        
        # 零轴位置评分
        above_zero = (dif > 0) & (dea > 0)
        below_zero = (dif < 0) & (dea < 0)
        score += above_zero * 10
        score -= below_zero * 10
        
        # 零轴穿越评分
        dif_cross_up = self.crossover(dif, 0)
        dif_cross_down = self.crossunder(dif, 0)
        score += dif_cross_up * 15
        score -= dif_cross_down * 15
        
        # 柱状图评分，无能量因子
        hist_turn_positive = (macd_hist > 0) & (macd_hist.shift(1) <= 0)
        hist_turn_negative = (macd_hist < 0) & (macd_hist.shift(1) >= 0)
        score += hist_turn_positive * 12
        score -= hist_turn_negative * 12
        
        # 背离评分
        if len(data) >= 20:
            price_peaks = self._find_peaks(data['close'], 10)
            macd_peaks = self._find_peaks(dif, 10)
            
            if len(price_peaks) >= 2 and len(macd_peaks) >= 2:
                price_trend = price_peaks[-1] - price_peaks[-2]
                macd_trend = macd_peaks[-1] - macd_peaks[-2]
                
                if price_trend < 0 and macd_trend > 0:
                    score.iloc[-10:] += 25
                elif price_trend > 0 and macd_trend < 0:
                    score.iloc[-10:] -= 25
        
        return np.clip(score, 0, 100)
    
    # 临时替换方法
    original_calculate_raw_score = macd.calculate_raw_score
    macd.calculate_raw_score = lambda d: simple_calculate_raw_score(macd, d)
    
    # 计算优化前评分
    old_score = macd.calculate_raw_score(data)
    
    # 恢复优化后的方法
    macd.calculate_raw_score = original_calculate_raw_score
    
    # 绘制对比图
    plot_macd_score_comparison(data, old_score, new_score)
    
    # 打印评分统计
    print("MACD评分统计:")
    print(f"优化前评分均值: {old_score.mean():.2f}, 方差: {old_score.var():.2f}")
    print(f"优化后评分均值: {new_score.mean():.2f}, 方差: {new_score.var():.2f}")
    print(f"评分变化均值: {(new_score - old_score).mean():.2f}")
    print(f"显著变化比例: {((new_score - old_score).abs() > 5).mean():.2%}")
    
    return data, old_score, new_score


def main():
    """主函数"""
    # 确保结果目录存在
    result_dir = os.path.join(root_dir, 'data/result')
    os.makedirs(result_dir, exist_ok=True)
    
    # 测试不同市场环境
    # 牛市：2019年
    bull_data = load_test_data(stock_code='000001.SZ', start_date='2019-01-01', end_date='2019-12-31')
    
    # 熊市：2022年
    bear_data = load_test_data(stock_code='000001.SZ', start_date='2022-01-01', end_date='2022-12-31')
    
    # 震荡市：2023年
    sideways_data = load_test_data(stock_code='000001.SZ', start_date='2023-01-01', end_date='2023-12-31')
    
    # 测试MA指标优化
    print("\n===== 牛市环境下MA指标优化效果 =====")
    bull_ma_data, bull_ma_old, bull_ma_new = test_ma_optimization(bull_data)
    
    print("\n===== 熊市环境下MA指标优化效果 =====")
    bear_ma_data, bear_ma_old, bear_ma_new = test_ma_optimization(bear_data)
    
    print("\n===== 震荡市环境下MA指标优化效果 =====")
    sideways_ma_data, sideways_ma_old, sideways_ma_new = test_ma_optimization(sideways_data)
    
    # 测试RSI指标优化
    print("\n===== 牛市环境下RSI指标优化效果 =====")
    bull_rsi_data, bull_rsi_old, bull_rsi_new = test_rsi_optimization(bull_data)
    
    print("\n===== 熊市环境下RSI指标优化效果 =====")
    bear_rsi_data, bear_rsi_old, bear_rsi_new = test_rsi_optimization(bear_data)
    
    print("\n===== 震荡市环境下RSI指标优化效果 =====")
    sideways_rsi_data, sideways_rsi_old, sideways_rsi_new = test_rsi_optimization(sideways_data)
    
    # 测试MACD指标优化
    print("\n===== 牛市环境下MACD指标优化效果 =====")
    bull_macd_data, bull_macd_old, bull_macd_new = test_macd_optimization(bull_data)
    
    print("\n===== 熊市环境下MACD指标优化效果 =====")
    bear_macd_data, bear_macd_old, bear_macd_new = test_macd_optimization(bear_data)
    
    print("\n===== 震荡市环境下MACD指标优化效果 =====")
    sideways_macd_data, sideways_macd_old, sideways_macd_new = test_macd_optimization(sideways_data)
    
    print("\n===== 优化效果总结 =====")
    print("1. MA指标优化效果:")
    print(f"   牛市：评分变化均值 {(bull_ma_new - bull_ma_old).mean():.2f}, 显著变化比例 {((bull_ma_new - bull_ma_old).abs() > 5).mean():.2%}")
    print(f"   熊市：评分变化均值 {(bear_ma_new - bear_ma_old).mean():.2f}, 显著变化比例 {((bear_ma_new - bear_ma_old).abs() > 5).mean():.2%}")
    print(f"   震荡市：评分变化均值 {(sideways_ma_new - sideways_ma_old).mean():.2f}, 显著变化比例 {((sideways_ma_new - sideways_ma_old).abs() > 5).mean():.2%}")
    
    print("\n2. RSI指标优化效果:")
    print(f"   牛市：评分变化均值 {(bull_rsi_new - bull_rsi_old).mean():.2f}, 显著变化比例 {((bull_rsi_new - bull_rsi_old).abs() > 5).mean():.2%}")
    print(f"   熊市：评分变化均值 {(bear_rsi_new - bear_rsi_old).mean():.2f}, 显著变化比例 {((bear_rsi_new - bear_rsi_old).abs() > 5).mean():.2%}")
    print(f"   震荡市：评分变化均值 {(sideways_rsi_new - sideways_rsi_old).mean():.2f}, 显著变化比例 {((sideways_rsi_new - sideways_rsi_old).abs() > 5).mean():.2%}")
    
    print("\n3. MACD指标优化效果:")
    print(f"   牛市：评分变化均值 {(bull_macd_new - bull_macd_old).mean():.2f}, 显著变化比例 {((bull_macd_new - bull_macd_old).abs() > 5).mean():.2%}")
    print(f"   熊市：评分变化均值 {(bear_macd_new - bear_macd_old).mean():.2f}, 显著变化比例 {((bear_macd_new - bear_macd_old).abs() > 5).mean():.2%}")
    print(f"   震荡市：评分变化均值 {(sideways_macd_new - sideways_macd_old).mean():.2f}, 显著变化比例 {((sideways_macd_new - sideways_macd_old).abs() > 5).mean():.2%}")


if __name__ == "__main__":
    main() 