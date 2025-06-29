#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量指标验证器
只查询一次股票数据，然后用同一份数据验证多个指标
"""

import os
import sys
import logging
import argparse
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from db.clickhouse_db import get_clickhouse_db

logger = get_logger(__name__)

class BatchIndicatorValidator:
    """批量指标验证器 - 一次查询，多次验证"""
    
    def __init__(self):
        """初始化验证器"""
        self.db = get_clickhouse_db()
        
        # 验证配置
        self.validation_date = "2024-12-28"
        self.stock_pool_size = 100  # 扩大到100个股票
        self.test_stock_count = 20   # 增加测试用的股票数量到20个
        
        # 数据缓存
        self._stock_pool_cache = None
        self._stock_data_cache = None
        self._cache_date = None
        
        logger.info("✅ 批量指标验证器初始化完成")
    
    def _get_stock_pool(self) -> List[str]:
        """获取股票池（带缓存）"""
        if (self._stock_pool_cache is not None and 
            self._cache_date == self.validation_date):
            logger.info(f"📋 使用缓存的股票池: {len(self._stock_pool_cache)} 只股票")
            return self._stock_pool_cache
        
        logger.info(f"🔍 查询股票池，日期: {self.validation_date}")
        
        try:
            # 查询活跃股票
            query = f"""
            SELECT DISTINCT code 
            FROM stock_info 
            WHERE level = '日线' AND date = '{self.validation_date}'
            AND volume > 0 AND close > 0
            ORDER BY volume DESC
            LIMIT {self.stock_pool_size}
            """
            
            result = self.db.query(query)
            if result.empty:
                logger.warning("⚠️ 未找到股票数据，使用默认股票池")
                stock_pool = ['000001', '000002', '600000', '600036', '000858']
            else:
                stock_pool = result['code'].tolist()
            
            # 缓存结果
            self._stock_pool_cache = stock_pool
            self._cache_date = self.validation_date
            
            logger.info(f"✅ 股票池准备完成: {len(stock_pool)} 只股票")
            return stock_pool
            
        except Exception as e:
            logger.error(f"❌ 查询股票池失败: {e}")
            return ['000001', '000002', '600000', '600036', '000858']
    
    def _load_stock_data(self, stock_codes: List[str]) -> pd.DataFrame:
        """加载股票数据（一次性查询所有需要的数据）"""
        if (self._stock_data_cache is not None and 
            self._cache_date == self.validation_date):
            logger.info(f"📊 使用缓存的股票数据: {len(self._stock_data_cache)} 条记录")
            return self._stock_data_cache
        
        logger.info(f"📥 加载股票数据，股票数量: {len(stock_codes)}")
        
        try:
            # 构建股票代码列表
            codes_str = "', '".join(stock_codes[:self.test_stock_count])
            
            # 一次性查询所有需要的股票数据（包含足够的历史数据用于指标计算）
            query = f"""
            SELECT 
                code,
                date,
                open,
                high,
                low,
                close,
                volume
            FROM stock_info 
            WHERE code IN ('{codes_str}')
            AND level = '日线' 
            AND date <= '{self.validation_date}'
            ORDER BY code, date DESC
            """
            
            result = self.db.query(query)
            
            if result.empty:
                logger.warning("⚠️ 未找到股票数据，创建模拟数据")
                # 创建模拟数据用于测试
                dates = pd.date_range(end=self.validation_date, periods=30, freq='D')
                mock_data = []
                for code in stock_codes[:self.test_stock_count]:
                    for i, date in enumerate(dates):
                        price = 10.0 + i * 0.1  # 模拟价格
                        mock_data.append({
                            'code': code,
                            'date': date.strftime('%Y-%m-%d'),
                            'open': price,
                            'high': price * 1.02,
                            'low': price * 0.98,
                            'close': price + 0.05,
                            'volume': 1000000 + i * 10000
                        })
                result = pd.DataFrame(mock_data)
            
            # 缓存数据
            self._stock_data_cache = result
            self._cache_date = self.validation_date
            
            logger.info(f"✅ 股票数据加载完成: {len(result)} 条记录，覆盖 {result['code'].nunique()} 只股票")
            return result
            
        except Exception as e:
            logger.error(f"❌ 加载股票数据失败: {e}")
            return pd.DataFrame()
    
    def _validate_ma_indicator(self, stock_data: pd.DataFrame, period: int = 5) -> Dict[str, Any]:
        """验证MA指标"""
        logger.info(f"📊 验证MA({period})指标")
        
        try:
            if stock_data.empty:
                return {"success": False, "error": "股票数据为空"}
            
            results = []
            
            for code in stock_data['code'].unique()[:5]:  # 测试前5只股票
                code_data = stock_data[stock_data['code'] == code].copy()
                code_data = code_data.sort_values('date')
                
                if len(code_data) >= period:
                    # 计算MA
                    code_data['ma'] = code_data['close'].rolling(window=period).mean()
                    
                    # 获取最新数据
                    latest = code_data.iloc[-1]
                    current_price = latest['close']
                    current_ma = latest['ma']
                    current_date = latest['date']
                    
                    # 判断信号
                    if current_price > current_ma:
                        signal = 'BUY'
                    else:
                        signal = 'HOLD'
                    
                    results.append({
                        'code': code,
                        'signal': signal,
                        'date': current_date.strftime('%Y-%m-%d') if hasattr(current_date, 'strftime') else str(current_date),
                        'close_price': float(current_price),
                        'ma_value': float(current_ma),
                        'price_vs_ma': f"{((current_price - current_ma) / current_ma * 100):+.2f}%"
                    })
            
            if results:
                buy_signals = sum(1 for r in results if r['signal'] == 'BUY')
                avg_ma = sum(r['ma_value'] for r in results) / len(results)
                
                return {
                    "success": True,
                    "results": results,
                    "summary": {
                        "tested_stocks": len(results),
                        "buy_signals": buy_signals,
                        "avg_ma": avg_ma
                    }
                }
            else:
                return {"success": False, "error": "无法计算MA指标"}
                
        except Exception as e:
            logger.error(f"❌ MA指标验证失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _validate_rsi_indicator(self, stock_data: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
        """验证RSI指标"""
        logger.info(f"📊 验证RSI({period})指标")
        
        try:
            if stock_data.empty:
                return {"success": False, "error": "股票数据为空"}
            
            results = []
            
            for code in stock_data['code'].unique()[:5]:  # 测试前5只股票
                code_data = stock_data[stock_data['code'] == code].copy()
                code_data = code_data.sort_values('date')
                
                if len(code_data) >= period + 1:
                    # 计算RSI
                    delta = code_data['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                    rs = gain / loss
                    code_data['rsi'] = 100 - (100 / (1 + rs))
                    
                    # 获取最新数据
                    latest = code_data.iloc[-1]
                    current_rsi = latest['rsi']
                    current_price = latest['close']
                    current_date = latest['date']
                    
                    # 判断信号
                    if current_rsi < 30:
                        signal = 'OVERSOLD'
                    elif current_rsi > 70:
                        signal = 'OVERBOUGHT'
                    else:
                        signal = 'NORMAL'
                    
                    results.append({
                        'code': code,
                        'signal': signal,
                        'date': current_date.strftime('%Y-%m-%d') if hasattr(current_date, 'strftime') else str(current_date),
                        'close_price': float(current_price),
                        'rsi_value': float(current_rsi),
                        'rsi_level': f"RSI({current_rsi:.1f})"
                    })
            
            if results:
                oversold_count = sum(1 for r in results if r['signal'] == 'OVERSOLD')
                overbought_count = sum(1 for r in results if r['signal'] == 'OVERBOUGHT')
                normal_count = sum(1 for r in results if r['signal'] == 'NORMAL')
                avg_rsi = sum(r['rsi_value'] for r in results) / len(results)
                
                return {
                    "success": True,
                    "results": results,
                    "summary": {
                        "tested_stocks": len(results),
                        "oversold_count": oversold_count,
                        "overbought_count": overbought_count,
                        "normal_count": normal_count,
                        "avg_rsi": avg_rsi
                    }
                }
            else:
                return {"success": False, "error": "无法计算RSI指标"}
                
        except Exception as e:
            logger.error(f"❌ RSI指标验证失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _validate_macd_indicator(self, stock_data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, Any]:
        """验证MACD指标"""
        logger.info(f"📊 验证MACD({fast},{slow},{signal})指标")
        
        try:
            if stock_data.empty:
                return {"success": False, "error": "股票数据为空"}
            
            results = []
            
            for code in stock_data['code'].unique()[:5]:  # 测试前5只股票
                code_data = stock_data[stock_data['code'] == code].copy()
                code_data = code_data.sort_values('date')
                
                if len(code_data) >= slow + signal:
                    # 计算EMA
                    code_data['ema_fast'] = code_data['close'].ewm(span=fast).mean()
                    code_data['ema_slow'] = code_data['close'].ewm(span=slow).mean()
                    
                    # 计算MACD
                    code_data['macd'] = code_data['ema_fast'] - code_data['ema_slow']
                    code_data['signal_line'] = code_data['macd'].ewm(span=signal).mean()
                    code_data['histogram'] = code_data['macd'] - code_data['signal_line']
                    
                    # 获取最新数据
                    latest = code_data.iloc[-1]
                    current_macd = latest['macd']
                    current_signal = latest['signal_line']
                    current_histogram = latest['histogram']
                    current_price = latest['close']
                    current_date = latest['date']
                    
                    # 判断信号
                    if current_macd > current_signal and current_histogram > 0:
                        signal_type = 'BUY'
                    elif current_macd < current_signal and current_histogram < 0:
                        signal_type = 'SELL'
                    else:
                        signal_type = 'HOLD'
                    
                    results.append({
                        'code': code,
                        'signal': signal_type,
                        'date': current_date.strftime('%Y-%m-%d') if hasattr(current_date, 'strftime') else str(current_date),
                        'close_price': float(current_price),
                        'macd_value': float(current_macd),
                        'signal_line': float(current_signal),
                        'histogram': float(current_histogram),
                        'macd_status': f"MACD({current_macd:.4f})"
                    })
            
            if results:
                buy_signals = sum(1 for r in results if r['signal'] == 'BUY')
                sell_signals = sum(1 for r in results if r['signal'] == 'SELL')
                avg_macd = sum(r['macd_value'] for r in results) / len(results)
                
                return {
                    "success": True,
                    "results": results,
                    "summary": {
                        "tested_stocks": len(results),
                        "buy_signals": buy_signals,
                        "sell_signals": sell_signals,
                        "avg_macd": avg_macd
                    }
                }
            else:
                return {"success": False, "error": "无法计算MACD指标"}
                
        except Exception as e:
            logger.error(f"❌ MACD指标验证失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _validate_bollinger_bands(self, stock_data: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> Dict[str, Any]:
        """验证布林带指标"""
        logger.info(f"📊 验证布林带({period},{std_dev})指标")
        
        try:
            if stock_data.empty:
                return {"success": False, "error": "股票数据为空"}
            
            results = []
            
            for code in stock_data['code'].unique()[:5]:  # 测试前5只股票
                code_data = stock_data[stock_data['code'] == code].copy()
                code_data = code_data.sort_values('date')
                
                if len(code_data) >= period:
                    # 计算布林带
                    code_data['middle_band'] = code_data['close'].rolling(window=period).mean()
                    code_data['std'] = code_data['close'].rolling(window=period).std()
                    code_data['upper_band'] = code_data['middle_band'] + (code_data['std'] * std_dev)
                    code_data['lower_band'] = code_data['middle_band'] - (code_data['std'] * std_dev)
                    
                    # 获取最新数据
                    latest = code_data.iloc[-1]
                    current_price = latest['close']
                    upper_band = latest['upper_band']
                    lower_band = latest['lower_band']
                    middle_band = latest['middle_band']
                    current_date = latest['date']
                    
                    # 判断信号
                    if current_price >= upper_band:
                        signal = 'OVERBOUGHT'
                    elif current_price <= lower_band:
                        signal = 'OVERSOLD'
                    else:
                        signal = 'NORMAL'
                    
                    # 计算价格在布林带中的位置百分比
                    band_width = upper_band - lower_band
                    price_position = ((current_price - lower_band) / band_width * 100) if band_width > 0 else 50
                    
                    results.append({
                        'code': code,
                        'signal': signal,
                        'date': current_date.strftime('%Y-%m-%d') if hasattr(current_date, 'strftime') else str(current_date),
                        'close_price': float(current_price),
                        'upper_band': float(upper_band),
                        'middle_band': float(middle_band),
                        'lower_band': float(lower_band),
                        'band_width': float(band_width),
                        'price_position': f"{price_position:.1f}%"
                    })
            
            if results:
                oversold_count = sum(1 for r in results if r['signal'] == 'OVERSOLD')
                overbought_count = sum(1 for r in results if r['signal'] == 'OVERBOUGHT')
                normal_count = sum(1 for r in results if r['signal'] == 'NORMAL')
                
                return {
                    "success": True,
                    "results": results,
                    "summary": {
                        "tested_stocks": len(results),
                        "oversold_count": oversold_count,
                        "overbought_count": overbought_count,
                        "normal_count": normal_count
                    }
                }
            else:
                return {"success": False, "error": "无法计算布林带指标"}
                
        except Exception as e:
            logger.error(f"❌ 布林带指标验证失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _validate_kdj_indicator(self, stock_data: pd.DataFrame, period: int = 9, k_period: int = 3, d_period: int = 3) -> Dict[str, Any]:
        """验证KDJ指标"""
        logger.info(f"📊 验证KDJ({period},{k_period},{d_period})指标")
        
        try:
            if stock_data.empty:
                return {"success": False, "error": "股票数据为空"}
            
            results = []
            
            for code in stock_data['code'].unique()[:5]:  # 测试前5只股票
                code_data = stock_data[stock_data['code'] == code].copy()
                code_data = code_data.sort_values('date')
                
                if len(code_data) >= period + k_period + d_period:
                    # 计算RSV
                    code_data['lowest_low'] = code_data['low'].rolling(window=period).min()
                    code_data['highest_high'] = code_data['high'].rolling(window=period).max()
                    code_data['rsv'] = ((code_data['close'] - code_data['lowest_low']) / 
                                       (code_data['highest_high'] - code_data['lowest_low'] + 1e-10)) * 100
                    
                    # 计算K值
                    code_data['k'] = code_data['rsv'].ewm(alpha=1/k_period).mean()
                    
                    # 计算D值
                    code_data['d'] = code_data['k'].ewm(alpha=1/d_period).mean()
                    
                    # 计算J值
                    code_data['j'] = 3 * code_data['k'] - 2 * code_data['d']
                    
                    # 获取最新数据
                    latest = code_data.iloc[-1]
                    if pd.notna(latest['k']) and pd.notna(latest['d']):
                        k_value = float(latest['k'])
                        d_value = float(latest['d'])
                        j_value = float(latest['j'])
                        
                        # 判断信号
                        if k_value < 20 and d_value < 20:
                            signal = 'OVERSOLD'
                        elif k_value > 80 and d_value > 80:
                            signal = 'OVERBOUGHT'
                        elif k_value > d_value:
                            signal = 'BUY'
                        else:
                            signal = 'SELL'
                        
                        results.append({
                            'code': code,
                            'signal': signal,
                            'date': latest['date'].strftime('%Y-%m-%d') if hasattr(latest['date'], 'strftime') else str(latest['date']),
                            'close_price': float(latest['close']),
                            'k_value': float(k_value),
                            'd_value': float(d_value),
                            'j_value': float(j_value),
                            'kdj_status': f"K={k_value:.1f}, D={d_value:.1f}, J={j_value:.1f}"
                        })
            
            if results:
                return {
                    "success": True,
                    "indicator": f"KDJ({period},{k_period},{d_period})",
                    "results": results,
                    "summary": {
                        "tested_stocks": len(results),
                        "buy_signals": sum(1 for r in results if r['signal'] == 'BUY'),
                        "oversold_count": sum(1 for r in results if r['signal'] == 'OVERSOLD'),
                        "overbought_count": sum(1 for r in results if r['signal'] == 'OVERBOUGHT'),
                        "avg_k": sum(r['k_value'] for r in results) / len(results),
                        "avg_d": sum(r['d_value'] for r in results) / len(results)
                    }
                }
            else:
                return {"success": False, "error": "无法计算KDJ指标"}
                
        except Exception as e:
            logger.error(f"❌ KDJ指标验证失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _validate_cci_indicator(self, stock_data: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
        """验证CCI指标"""
        logger.info(f"📊 验证CCI({period})指标")
        
        try:
            if stock_data.empty:
                return {"success": False, "error": "股票数据为空"}
            
            results = []
            
            for code in stock_data['code'].unique()[:5]:  # 测试前5只股票
                code_data = stock_data[stock_data['code'] == code].copy()
                code_data = code_data.sort_values('date')
                
                if len(code_data) >= period:
                    # 计算典型价格
                    code_data['tp'] = (code_data['high'] + code_data['low'] + code_data['close']) / 3
                    
                    # 计算移动平均
                    code_data['ma_tp'] = code_data['tp'].rolling(window=period).mean()
                    
                    # 计算平均绝对偏差
                    code_data['mad'] = code_data['tp'].rolling(window=period).apply(
                        lambda x: abs(x - x.mean()).mean()
                    )
                    
                    # 计算CCI
                    code_data['cci'] = (code_data['tp'] - code_data['ma_tp']) / (0.015 * code_data['mad'])
                    
                    # 获取最新数据
                    latest = code_data.iloc[-1]
                    if pd.notna(latest['cci']):
                        cci_value = float(latest['cci'])
                        
                        if cci_value > 100:
                            signal = 'OVERBOUGHT'
                        elif cci_value < -100:
                            signal = 'OVERSOLD'
                        else:
                            signal = 'NORMAL'
                        
                        results.append({
                            'code': code,
                            'date': latest['date'],
                            'close': float(latest['close']),
                            'cci': cci_value,
                            'signal': signal
                        })
            
            if results:
                return {
                    "success": True,
                    "indicator": f"CCI({period})",
                    "results": results,
                    "summary": {
                        "tested_stocks": len(results),
                        "oversold_count": sum(1 for r in results if r['signal'] == 'OVERSOLD'),
                        "overbought_count": sum(1 for r in results if r['signal'] == 'OVERBOUGHT'),
                        "normal_count": sum(1 for r in results if r['signal'] == 'NORMAL'),
                        "avg_cci": sum(r['cci'] for r in results) / len(results)
                    }
                }
            else:
                return {"success": False, "error": "无法计算CCI指标"}
                
        except Exception as e:
            logger.error(f"❌ CCI指标验证失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _validate_wr_indicator(self, stock_data: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
        """验证威廉指标WR"""
        logger.info(f"📊 验证WR({period})指标")
        
        try:
            if stock_data.empty:
                return {"success": False, "error": "股票数据为空"}
            
            results = []
            
            for code in stock_data['code'].unique()[:5]:  # 测试前5只股票
                code_data = stock_data[stock_data['code'] == code].copy()
                code_data = code_data.sort_values('date')
                
                if len(code_data) >= period:
                    # 计算最高价和最低价
                    code_data['highest_high'] = code_data['high'].rolling(window=period).max()
                    code_data['lowest_low'] = code_data['low'].rolling(window=period).min()
                    
                    # 计算WR
                    code_data['wr'] = ((code_data['highest_high'] - code_data['close']) / 
                                      (code_data['highest_high'] - code_data['lowest_low'] + 1e-10)) * -100
                    
                    # 获取最新数据
                    latest = code_data.iloc[-1]
                    if pd.notna(latest['wr']):
                        wr_value = float(latest['wr'])
                        
                        if wr_value > -20:
                            signal = 'OVERBOUGHT'
                        elif wr_value < -80:
                            signal = 'OVERSOLD'
                        else:
                            signal = 'NORMAL'
                        
                        results.append({
                            'code': code,
                            'date': latest['date'],
                            'close': float(latest['close']),
                            'wr': wr_value,
                            'signal': signal
                        })
            
            if results:
                return {
                    "success": True,
                    "indicator": f"WR({period})",
                    "results": results,
                    "summary": {
                        "tested_stocks": len(results),
                        "oversold_count": sum(1 for r in results if r['signal'] == 'OVERSOLD'),
                        "overbought_count": sum(1 for r in results if r['signal'] == 'OVERBOUGHT'),
                        "normal_count": sum(1 for r in results if r['signal'] == 'NORMAL'),
                        "avg_wr": sum(r['wr'] for r in results) / len(results)
                    }
                }
            else:
                return {"success": False, "error": "无法计算WR指标"}
                
        except Exception as e:
            logger.error(f"❌ WR指标验证失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _validate_stoch_indicator(self, stock_data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, Any]:
        """验证随机指标STOCH"""
        logger.info(f"📊 验证STOCH({k_period},{d_period})指标")
        
        try:
            if stock_data.empty:
                return {"success": False, "error": "股票数据为空"}
            
            results = []
            
            for code in stock_data['code'].unique()[:5]:  # 测试前5只股票
                code_data = stock_data[stock_data['code'] == code].copy()
                code_data = code_data.sort_values('date')
                
                if len(code_data) >= k_period + d_period:
                    # 计算%K
                    code_data['lowest_low'] = code_data['low'].rolling(window=k_period).min()
                    code_data['highest_high'] = code_data['high'].rolling(window=k_period).max()
                    code_data['stoch_k'] = ((code_data['close'] - code_data['lowest_low']) / 
                                           (code_data['highest_high'] - code_data['lowest_low'] + 1e-10)) * 100
                    
                    # 计算%D
                    code_data['stoch_d'] = code_data['stoch_k'].rolling(window=d_period).mean()
                    
                    # 获取最新数据
                    latest = code_data.iloc[-1]
                    if pd.notna(latest['stoch_k']) and pd.notna(latest['stoch_d']):
                        k_value = float(latest['stoch_k'])
                        d_value = float(latest['stoch_d'])
                        
                        if k_value > 80 and d_value > 80:
                            signal = 'OVERBOUGHT'
                        elif k_value < 20 and d_value < 20:
                            signal = 'OVERSOLD'
                        elif k_value > d_value:
                            signal = 'BUY'
                        else:
                            signal = 'SELL'
                        
                        results.append({
                            'code': code,
                            'date': latest['date'],
                            'close': float(latest['close']),
                            'stoch_k': k_value,
                            'stoch_d': d_value,
                            'signal': signal
                        })
            
            if results:
                return {
                    "success": True,
                    "indicator": f"STOCH({k_period},{d_period})",
                    "results": results,
                    "summary": {
                        "tested_stocks": len(results),
                        "buy_signals": sum(1 for r in results if r['signal'] == 'BUY'),
                        "oversold_count": sum(1 for r in results if r['signal'] == 'OVERSOLD'),
                        "overbought_count": sum(1 for r in results if r['signal'] == 'OVERBOUGHT'),
                        "avg_k": sum(r['stoch_k'] for r in results) / len(results),
                        "avg_d": sum(r['stoch_d'] for r in results) / len(results)
                    }
                }
            else:
                return {"success": False, "error": "无法计算STOCH指标"}
                
        except Exception as e:
            logger.error(f"❌ STOCH指标验证失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _validate_roc_indicator(self, stock_data: pd.DataFrame, period: int = 12) -> Dict[str, Any]:
        """验证ROC变动率指标"""
        logger.info(f"📊 验证ROC({period})指标")
        
        try:
            if stock_data.empty:
                return {"success": False, "error": "股票数据为空"}
            
            results = []
            
            for code in stock_data['code'].unique()[:5]:  # 测试前5只股票
                code_data = stock_data[stock_data['code'] == code].copy()
                code_data = code_data.sort_values('date')
                
                if len(code_data) >= period + 1:
                    # 计算ROC
                    code_data['roc'] = ((code_data['close'] - code_data['close'].shift(period)) / 
                                       code_data['close'].shift(period)) * 100
                    
                    # 获取最新数据
                    latest = code_data.iloc[-1]
                    if pd.notna(latest['roc']):
                        roc_value = float(latest['roc'])
                        
                        if roc_value > 10:
                            signal = 'STRONG_BUY'
                        elif roc_value > 5:
                            signal = 'BUY'
                        elif roc_value < -10:
                            signal = 'STRONG_SELL'
                        elif roc_value < -5:
                            signal = 'SELL'
                        else:
                            signal = 'HOLD'
                        
                        results.append({
                            'code': code,
                            'date': latest['date'],
                            'close': float(latest['close']),
                            'roc': roc_value,
                            'signal': signal
                        })
            
            if results:
                return {
                    "success": True,
                    "indicator": f"ROC({period})",
                    "results": results,
                    "summary": {
                        "tested_stocks": len(results),
                        "strong_buy_count": sum(1 for r in results if r['signal'] == 'STRONG_BUY'),
                        "buy_count": sum(1 for r in results if r['signal'] == 'BUY'),
                        "hold_count": sum(1 for r in results if r['signal'] == 'HOLD'),
                        "sell_count": sum(1 for r in results if r['signal'] == 'SELL'),
                        "strong_sell_count": sum(1 for r in results if r['signal'] == 'STRONG_SELL'),
                        "avg_roc": sum(r['roc'] for r in results) / len(results)
                    }
                }
            else:
                return {"success": False, "error": "无法计算ROC指标"}
                
        except Exception as e:
            logger.error(f"❌ ROC指标验证失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _validate_atr_indicator(self, stock_data: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
        """验证ATR真实波动范围指标"""
        logger.info(f"📊 验证ATR({period})指标")
        
        try:
            if stock_data.empty:
                return {"success": False, "error": "股票数据为空"}
            
            results = []
            
            for code in stock_data['code'].unique()[:5]:  # 测试前5只股票
                code_data = stock_data[stock_data['code'] == code].copy()
                code_data = code_data.sort_values('date')
                
                if len(code_data) >= period + 1:
                    # 计算真实范围TR
                    code_data['prev_close'] = code_data['close'].shift(1)
                    code_data['tr1'] = code_data['high'] - code_data['low']
                    code_data['tr2'] = abs(code_data['high'] - code_data['prev_close'])
                    code_data['tr3'] = abs(code_data['low'] - code_data['prev_close'])
                    code_data['tr'] = code_data[['tr1', 'tr2', 'tr3']].max(axis=1)
                    
                    # 计算ATR
                    code_data['atr'] = code_data['tr'].rolling(window=period).mean()
                    
                    # 获取最新数据
                    latest = code_data.iloc[-1]
                    if pd.notna(latest['atr']):
                        atr_value = float(latest['atr'])
                        close_price = float(latest['close'])
                        atr_percent = (atr_value / close_price) * 100
                        
                        if atr_percent > 5:
                            signal = 'HIGH_VOLATILITY'
                        elif atr_percent > 3:
                            signal = 'MEDIUM_VOLATILITY'
                        else:
                            signal = 'LOW_VOLATILITY'
                        
                        results.append({
                            'code': code,
                            'date': latest['date'],
                            'close': close_price,
                            'atr': atr_value,
                            'atr_percent': atr_percent,
                            'signal': signal
                        })
            
            if results:
                return {
                    "success": True,
                    "indicator": f"ATR({period})",
                    "results": results,
                    "summary": {
                        "tested_stocks": len(results),
                        "high_vol_count": sum(1 for r in results if r['signal'] == 'HIGH_VOLATILITY'),
                        "medium_vol_count": sum(1 for r in results if r['signal'] == 'MEDIUM_VOLATILITY'),
                        "low_vol_count": sum(1 for r in results if r['signal'] == 'LOW_VOLATILITY'),
                        "avg_atr": sum(r['atr'] for r in results) / len(results),
                        "avg_atr_percent": sum(r['atr_percent'] for r in results) / len(results)
                    }
                }
            else:
                return {"success": False, "error": "无法计算ATR指标"}
                
        except Exception as e:
            logger.error(f"❌ ATR指标验证失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _validate_obv_indicator(self, stock_data: pd.DataFrame) -> Dict[str, Any]:
        """验证OBV能量潮指标"""
        logger.info("📊 验证OBV指标")
        
        try:
            if stock_data.empty:
                return {"success": False, "error": "股票数据为空"}
            
            results = []
            
            for code in stock_data['code'].unique()[:5]:  # 测试前5只股票
                code_data = stock_data[stock_data['code'] == code].copy()
                code_data = code_data.sort_values('date')
                
                if len(code_data) >= 2:
                    # 计算OBV
                    code_data['price_change'] = code_data['close'].diff()
                    code_data['volume_direction'] = 0
                    code_data.loc[code_data['price_change'] > 0, 'volume_direction'] = 1
                    code_data.loc[code_data['price_change'] < 0, 'volume_direction'] = -1
                    code_data['obv_change'] = code_data['volume'] * code_data['volume_direction']
                    code_data['obv'] = code_data['obv_change'].cumsum()
                    
                    # 获取最新数据
                    latest = code_data.iloc[-1]
                    prev = code_data.iloc[-2] if len(code_data) >= 2 else latest
                    
                    if pd.notna(latest['obv']) and pd.notna(prev['obv']):
                        obv_value = float(latest['obv'])
                        obv_change = obv_value - float(prev['obv'])
                        
                        if obv_change > 0 and latest['price_change'] > 0:
                            signal = 'BULLISH_CONFIRMATION'
                        elif obv_change < 0 and latest['price_change'] < 0:
                            signal = 'BEARISH_CONFIRMATION'
                        elif obv_change > 0 and latest['price_change'] < 0:
                            signal = 'BULLISH_DIVERGENCE'
                        elif obv_change < 0 and latest['price_change'] > 0:
                            signal = 'BEARISH_DIVERGENCE'
                        else:
                            signal = 'NEUTRAL'
                        
                        results.append({
                            'code': code,
                            'date': latest['date'],
                            'close': float(latest['close']),
                            'obv': obv_value,
                            'obv_change': obv_change,
                            'signal': signal
                        })
            
            if results:
                return {
                    "success": True,
                    "indicator": "OBV",
                    "results": results,
                    "summary": {
                        "tested_stocks": len(results),
                        "bullish_conf_count": sum(1 for r in results if r['signal'] == 'BULLISH_CONFIRMATION'),
                        "bearish_conf_count": sum(1 for r in results if r['signal'] == 'BEARISH_CONFIRMATION'),
                        "bullish_div_count": sum(1 for r in results if r['signal'] == 'BULLISH_DIVERGENCE'),
                        "bearish_div_count": sum(1 for r in results if r['signal'] == 'BEARISH_DIVERGENCE'),
                        "neutral_count": sum(1 for r in results if r['signal'] == 'NEUTRAL'),
                        "avg_obv_change": sum(r['obv_change'] for r in results) / len(results)
                    }
                }
            else:
                return {"success": False, "error": "无法计算OBV指标"}
                
        except Exception as e:
            logger.error(f"❌ OBV指标验证失败: {e}")
            return {"success": False, "error": str(e)}
    
    def validate_all_indicators(self, indicators: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        批量验证多个指标
        
        Args:
            indicators: 要验证的指标列表，如果为None则验证所有支持的指标
            
        Returns:
            验证结果汇总
        """
        start_time = datetime.now()
        logger.info("🚀 开始批量指标验证")
        
        # 默认验证的指标列表
        if indicators is None:
            indicators = ['MA', 'RSI', 'MACD', 'BOLL', 'KDJ', 'CCI', 'WR', 'STOCH', 'ROC', 'ATR', 'OBV']
        
        try:
            # 1. 获取股票池
            stock_pool = self._get_stock_pool()
            
            # 2. 一次性加载股票数据
            stock_data = self._load_stock_data(stock_pool)
            
            if stock_data.empty:
                return {
                    "success": False,
                    "error": "无法获取股票数据",
                    "elapsed_time": (datetime.now() - start_time).total_seconds()
                }
            
            # 3. 使用同一份数据验证所有指标
            validation_results = {}
            
            for indicator in indicators:
                logger.info(f"🔍 验证指标: {indicator}")
                
                if indicator.upper() == 'MA':
                    result = self._validate_ma_indicator(stock_data, period=5)
                elif indicator.upper() == 'RSI':
                    result = self._validate_rsi_indicator(stock_data, period=14)
                elif indicator.upper() == 'MACD':
                    result = self._validate_macd_indicator(stock_data)
                elif indicator.upper() == 'BOLL':
                    result = self._validate_bollinger_bands(stock_data)
                elif indicator.upper() == 'KDJ':
                    result = self._validate_kdj_indicator(stock_data)
                elif indicator.upper() == 'CCI':
                    result = self._validate_cci_indicator(stock_data)
                elif indicator.upper() == 'WR':
                    result = self._validate_wr_indicator(stock_data)
                elif indicator.upper() == 'STOCH':
                    result = self._validate_stoch_indicator(stock_data)
                elif indicator.upper() == 'ROC':
                    result = self._validate_roc_indicator(stock_data)
                elif indicator.upper() == 'ATR':
                    result = self._validate_atr_indicator(stock_data)
                elif indicator.upper() == 'OBV':
                    result = self._validate_obv_indicator(stock_data)
                else:
                    result = {
                        "success": False,
                        "error": f"不支持的指标: {indicator}"
                    }
                
                validation_results[indicator] = result
                
                if result.get('success'):
                    logger.info(f"✅ {indicator} 验证成功")
                else:
                    logger.error(f"❌ {indicator} 验证失败: {result.get('error')}")
            
            # 4. 汇总结果
            elapsed_time = (datetime.now() - start_time).total_seconds()
            successful_indicators = [k for k, v in validation_results.items() if v.get('success')]
            
            summary = {
                "success": True,
                "total_indicators": len(indicators),
                "successful_indicators": len(successful_indicators),
                "failed_indicators": len(indicators) - len(successful_indicators),
                "success_rate": len(successful_indicators) / len(indicators) * 100,
                "stock_pool_size": len(stock_pool),
                "data_records": len(stock_data),
                "elapsed_time": elapsed_time,
                "indicators": validation_results
            }
            
            logger.info(f"🎉 批量验证完成，成功率: {summary['success_rate']:.1f}%，耗时: {elapsed_time:.2f}秒")
            return summary
            
        except Exception as e:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ 批量验证失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "elapsed_time": elapsed_time
            }
    
    def _print_validation_summary(self, results: Dict[str, Any]):
        """打印验证结果汇总"""
        print("\n" + "="*80)
        print("📊 批量指标验证结果汇总")
        print("="*80)
        
        if results.get('success'):
            print(f"✅ 验证状态: 成功")
            print(f"📈 股票池大小: {results.get('stock_pool_size', 'N/A')}")
            print(f"📋 数据记录数: {results.get('data_records', 'N/A')}")
            print(f"⏱️  总执行时间: {results.get('elapsed_time', 0):.2f}秒")
            print(f"📊 指标总数: {results.get('total_indicators', 0)}")
            print(f"✅ 成功指标: {results.get('successful_indicators', 0)}")
            print(f"❌ 失败指标: {results.get('failed_indicators', 0)}")
            print(f"📈 成功率: {results.get('success_rate', 0):.1f}%")
            
            print("\n" + "-"*80)
            print("📋 各指标验证详情:")
            print("-"*80)
            
            for indicator, result in results.get('indicators', {}).items():
                if result.get('success'):
                    print(f"✅ {indicator}:")
                    summary = result.get('summary', {})
                    if 'tested_stocks' in summary:
                        print(f"   📊 测试股票数: {summary['tested_stocks']}")
                    
                    # 显示特定指标的统计信息
                    if 'buy_signals' in summary:
                        print(f"   🔼 买入信号: {summary['buy_signals']}")
                    if 'oversold_count' in summary:
                        print(f"   📉 超卖: {summary['oversold_count']}")
                    if 'overbought_count' in summary:
                        print(f"   📈 超买: {summary['overbought_count']}")
                    if 'normal_count' in summary:
                        print(f"   ⚖️  正常: {summary['normal_count']}")
                    if 'avg_ma' in summary:
                        print(f"   📊 平均MA: {summary['avg_ma']:.2f}")
                    if 'avg_rsi' in summary:
                        print(f"   📊 平均RSI: {summary['avg_rsi']:.2f}")
                    if 'avg_macd' in summary:
                        print(f"   📊 平均MACD: {summary['avg_macd']:.4f}")
                    if 'avg_k' in summary:
                        print(f"   📊 平均K: {summary['avg_k']:.2f}")
                    if 'avg_d' in summary:
                        print(f"   📊 平均D: {summary['avg_d']:.2f}")
                    if 'avg_cci' in summary:
                        print(f"   📊 平均CCI: {summary['avg_cci']:.2f}")
                    if 'avg_wr' in summary:
                        print(f"   📊 平均WR: {summary['avg_wr']:.2f}")
                    
                    # 显示部分详细结果
                    results_data = result.get('results', [])
                    if results_data:
                        print(f"   📋 详细结果:")
                        for i, stock_result in enumerate(results_data[:3]):  # 显示前3个详细结果
                            code = stock_result['code']
                            signal = stock_result.get('signal', 'N/A')
                            date = stock_result.get('date', 'N/A')
                            close_price = stock_result.get('close_price', 0)
                            
                            print(f"      📈 {code} ({date}): {signal}")
                            print(f"         💰 收盘价: {close_price:.2f}")
                            
                            # 根据指标类型显示不同的数值
                            if 'ma_value' in stock_result:
                                ma_val = stock_result['ma_value']
                                price_vs_ma = stock_result.get('price_vs_ma', 'N/A')
                                print(f"         📊 MA值: {ma_val:.2f} ({price_vs_ma})")
                            elif 'rsi_value' in stock_result:
                                rsi_val = stock_result['rsi_value']
                                rsi_level = stock_result.get('rsi_level', 'N/A')
                                print(f"         📊 RSI值: {rsi_val:.1f} ({rsi_level})")
                            elif 'macd_value' in stock_result:
                                macd_val = stock_result['macd_value']
                                signal_line = stock_result.get('signal_line', 0)
                                histogram = stock_result.get('histogram', 0)
                                print(f"         📊 MACD: {macd_val:.4f}, 信号线: {signal_line:.4f}")
                                print(f"         📊 柱状图: {histogram:.4f}")
                            elif 'upper_band' in stock_result:
                                upper = stock_result['upper_band']
                                lower = stock_result['lower_band']
                                middle = stock_result['middle_band']
                                print(f"         📊 布林带: 上轨{upper:.2f}, 中轨{middle:.2f}, 下轨{lower:.2f}")
                            elif 'k_value' in stock_result:
                                k_val = stock_result['k_value']
                                d_val = stock_result['d_value']
                                j_val = stock_result.get('j_value', 0)
                                print(f"         📊 KDJ: K={k_val:.1f}, D={d_val:.1f}, J={j_val:.1f}")
                            elif 'cci_value' in stock_result:
                                cci_val = stock_result['cci_value']
                                print(f"         📊 CCI值: {cci_val:.2f}")
                            elif 'wr_value' in stock_result:
                                wr_val = stock_result['wr_value']
                                print(f"         📊 WR值: {wr_val:.2f}")
                            elif 'roc_value' in stock_result:
                                roc_val = stock_result['roc_value']
                                print(f"         📊 ROC值: {roc_val:.2f}%")
                            elif 'atr_value' in stock_result:
                                atr_val = stock_result['atr_value']
                                print(f"         📊 ATR值: {atr_val:.4f}")
                            elif 'obv_value' in stock_result:
                                obv_val = stock_result['obv_value']
                                obv_change = stock_result.get('obv_change', 0)
                                print(f"         📊 OBV值: {obv_val:.0f} (变化: {obv_change:+.2f}%)")
                            
                            print()  # 空行分隔
                else:
                    print(f"❌ {indicator}: {result.get('error', '未知错误')}")
                print()
        else:
            print("❌ 验证状态: 失败")
            print(f"❗ 错误信息: {results.get('error', '未知错误')}")
            print(f"⏱️  执行时间: {results.get('elapsed_time', 0):.2f}秒")
        
        print("="*80)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量指标验证器')
    parser.add_argument('--indicators', '-i', type=str, nargs='*',
                       help='要验证的指标列表 (如: MA RSI MACD BOLL)，不指定则验证所有支持的指标')
    
    args = parser.parse_args()
    
    # 创建验证器
    validator = BatchIndicatorValidator()
    
    # 执行批量验证
    results = validator.validate_all_indicators(args.indicators)
    
    # 打印结果
    validator._print_validation_summary(results)
    
    logger.info("🎉 批量验证完成")

if __name__ == "__main__":
    main() 