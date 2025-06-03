# utils/market_analysis.py
"""
市场分析工具模块（占位文件，防止import错误）
""" 

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

class MarketAnalyzer:
    """市场分析器"""
    
    def __init__(self, db_connector=None):
        """
        初始化市场分析器
        
        Args:
            db_connector: 数据库连接器
        """
        self.db = db_connector
        self.cache = {}
        
    def get_market_status(self, date: str) -> Dict[str, Any]:
        """
        获取市场状态
        
        Args:
            date: 日期
            
        Returns:
            市场状态字典
        """
        if date in self.cache:
            return self.cache[date]
            
        status = {
            'trend': self._analyze_market_trend(date),
            'volatility': self._analyze_market_volatility(date),
            'breadth': self._analyze_market_breadth(date),
            'sentiment': self._analyze_market_sentiment(date)
        }
        
        self.cache[date] = status
        return status
        
    def _analyze_market_trend(self, date: str) -> Dict[str, Any]:
        """分析市场趋势"""
        # 获取指数数据
        indices = ['000001.SH', '399001.SZ', '399006.SZ']  # 上证、深证、创业板
        trends = {}
        
        for index in indices:
            # 获取最近20个交易日数据
            end_date = datetime.strptime(date, '%Y-%m-%d')
            start_date = end_date - timedelta(days=40)
            
            df = self.db.get_stock_data(index, start_date.strftime('%Y-%m-%d'), date, 'daily')
            if df is None or df.empty:
                continue
                
            # 计算20日均线
            df['ma20'] = df['close'].rolling(window=20).mean()
            
            # 判断趋势
            current_price = df['close'].iloc[-1]
            ma20 = df['ma20'].iloc[-1]
            
            if current_price > ma20:
                trend = 'up'
            elif current_price < ma20:
                trend = 'down'
            else:
                trend = 'neutral'
                
            trends[index] = {
                'trend': trend,
                'strength': abs(current_price - ma20) / ma20
            }
            
        return trends
        
    def _analyze_market_volatility(self, date: str) -> Dict[str, Any]:
        """分析市场波动性"""
        # 获取指数数据
        index = '000001.SH'  # 上证指数
        
        # 获取最近20个交易日数据
        end_date = datetime.strptime(date, '%Y-%m-%d')
        start_date = end_date - timedelta(days=40)
        
        df = self.db.get_stock_data(index, start_date.strftime('%Y-%m-%d'), date, 'daily')
        if df is None or df.empty:
            return {'volatility': 'unknown', 'level': 0.0}
            
        # 计算波动率
        returns = df['close'].pct_change()
        volatility = returns.std() * np.sqrt(252)  # 年化波动率
        
        # 判断波动水平
        if volatility < 0.15:
            level = 'low'
        elif volatility < 0.25:
            level = 'medium'
        else:
            level = 'high'
            
        return {
            'volatility': level,
            'value': float(volatility)
        }
        
    def _analyze_market_breadth(self, date: str) -> Dict[str, Any]:
        """分析市场宽度"""
        # 获取所有A股数据
        stocks = self.db.get_all_stocks()
        if not stocks:
            return {'breadth': 'unknown', 'ratio': 0.0}
            
        # 统计上涨和下跌股票数量
        up_count = 0
        down_count = 0
        
        for stock in stocks:
            df = self.db.get_stock_data(stock, date, date, 'daily')
            if df is None or df.empty:
                continue
                
            if df['close'].iloc[-1] > df['open'].iloc[-1]:
                up_count += 1
            elif df['close'].iloc[-1] < df['open'].iloc[-1]:
                down_count += 1
                
        total = up_count + down_count
        if total == 0:
            return {'breadth': 'unknown', 'ratio': 0.0}
            
        ratio = up_count / total
        
        # 判断市场宽度
        if ratio > 0.7:
            breadth = 'strong'
        elif ratio > 0.5:
            breadth = 'moderate'
        else:
            breadth = 'weak'
            
        return {
            'breadth': breadth,
            'ratio': ratio,
            'up_count': up_count,
            'down_count': down_count
        }
        
    def _analyze_market_sentiment(self, date: str) -> Dict[str, Any]:
        """分析市场情绪"""
        # 获取指数数据
        index = '000001.SH'  # 上证指数
        
        # 获取最近20个交易日数据
        end_date = datetime.strptime(date, '%Y-%m-%d')
        start_date = end_date - timedelta(days=40)
        
        df = self.db.get_stock_data(index, start_date.strftime('%Y-%m-%d'), date, 'daily')
        if df is None or df.empty:
            return {'sentiment': 'unknown', 'score': 0.0}
            
        # 计算RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # 计算MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        
        # 计算情绪得分
        rsi_score = (rsi.iloc[-1] - 50) / 50  # 归一化到[-1, 1]
        macd_score = (macd.iloc[-1] - signal.iloc[-1]) / abs(signal.iloc[-1]) if signal.iloc[-1] != 0 else 0
        
        sentiment_score = (rsi_score + macd_score) / 2
        
        # 判断情绪
        if sentiment_score > 0.3:
            sentiment = 'bullish'
        elif sentiment_score < -0.3:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'
            
        return {
            'sentiment': sentiment,
            'score': float(sentiment_score),
            'rsi': float(rsi.iloc[-1]),
            'macd': float(macd.iloc[-1])
        } 