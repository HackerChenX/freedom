"""
多周期选股策略

基于增强策略基类实现的多周期选股策略，支持：
1. 多周期数据分析（日线、周线、小时线）
2. 多指标组合筛选
3. 默认使用最新数据时间
4. 详细的筛选结果输出
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

from strategy.enhanced_base_strategy import Enhanced_base_strategy
# from strategy.enhanced_base_strategy import Indicator_condition  # 暂时注释掉不存在的导入
from utils.logger import getLogger

logger = getLogger(__name__)


class MultiPeriodStrategy(Enhanced_base_strategy):
    """
    多周期选股策略
    
    结合日线、周线、小时线数据进行选股
    """
    
    def __init__(self):
        super().__init__(
            name="多周期技术指标选股策略",
            description="基于多周期技术指标的综合选股策略",
            default_period="1d"
        )
        
        # 初始化策略参数
        self._init_strategy_parameters()
        
        # 添加指标条件
        self._add_indicator_conditions()
    
    def _init_strategy_parameters(self):
        """初始化策略参数"""
        self.set_parameters({
            'lookback_days': 60,  # 回看天数
            'min_volume': 500000,  # 最小成交量
            'max_results': 50,     # 最大结果数
            'score_threshold': 60,  # 评分阈值
            'enable_multi_period': True,  # 启用多周期分析
            'enable_volume_filter': True,  # 启用成交量过滤
            'enable_trend_filter': True   # 启用趋势过滤
        })
    
    def _add_indicator_conditions(self):
        """添加指标条件"""
        
        # 日线指标条件
        daily_conditions = [
            Indicator_condition(
                indicator_name='MA',
                period='1d',
                parameters={'period': 20},
                condition='close > ma20',
                signal_type='BUY',
                weight=1.0
            ),
            Indicator_condition(
                indicator_name='RSI',
                period='1d',
                parameters={'period': 14},
                condition='30 < rsi < 70',
                signal_type='NORMAL',
                weight=0.8
            ),
            Indicator_condition(
                indicator_name='MACD',
                period='1d',
                parameters={'fast': 12, 'slow': 26, 'signal': 9},
                condition='macd > signal',
                signal_type='BUY',
                weight=1.2
            ),
            Indicator_condition(
                indicator_name='BOLL',
                period='1d',
                parameters={'period': 20, 'std_dev': 2.0},
                condition='lower_band < close < upper_band',
                signal_type='NORMAL',
                weight=0.6
            ),
            Indicator_condition(
                indicator_name='VOL',
                period='1d',
                parameters={'ma_period': 20},
                condition='volume > volume_ma * 1.2',
                signal_type='HIGH_VOLUME',
                weight=0.5
            )
        ]
        
        # 周线指标条件
        weekly_conditions = [
            Indicator_condition(
                indicator_name='MA',
                period='1w',
                parameters={'period': 10},
                condition='close > ma10',
                signal_type='BUY',
                weight=1.5
            ),
            Indicator_condition(
                indicator_name='RSI',
                period='1w',
                parameters={'period': 14},
                condition='rsi > 50',
                signal_type='BULLISH',
                weight=1.0
            ),
            Indicator_condition(
                indicator_name='KDJ',
                period='1w',
                parameters={'period': 9, 'k_period': 3, 'd_period': 3},
                condition='k > d and j > 50',
                signal_type='BUY',
                weight=0.8
            )
        ]
        
        # 小时线指标条件（短期动量）
        hourly_conditions = [
            Indicator_condition(
                indicator_name='RSI',
                period='1h',
                parameters={'period': 14},
                condition='rsi > 45',
                signal_type='BULLISH',
                weight=0.5
            ),
            Indicator_condition(
                indicator_name='BOLL',
                period='1h',
                parameters={'period': 20, 'std_dev': 2.0},
                condition='close > middle_band',
                signal_type='BUY',
                weight=0.4
            )
        ]
        
        # 添加所有条件
        for condition in daily_conditions + weekly_conditions + hourly_conditions:
            self.add_indicator_condition(condition)
    
    def select_strategy(self, universe: Optional[List[str]] = None, *args, **kwargs) -> pd.DataFrame:
        """
        执行多周期选股
        
        Args:
            universe: 股票代码列表
            
        Returns:
            pd.DataFrame: 选股结果
        """
        logger.info("🚀 开始执行多周期选股策略")
        
        # 获取有效日期范围
        start_date, end_date = self.get_effective_date_range()
        
        # 如果没有指定股票池，使用默认股票池
        if universe is None:
            universe = self.get_stock_universe()
        
        if not universe:
            logger.warning("股票池为空，无法执行选股")
            return pd.DataFrame()
        
        logger.info(f"股票池大小: {len(universe)}只股票")
        logger.info(f"分析时间范围: {start_date} 到 {end_date}")
        
        # 多周期分析
        results = []
        
        for i, stock_code in enumerate(universe):
            try:
                logger.info(f"分析股票 {i+1}/{len(universe)}: {stock_code}")
                
                # 分析单只股票
                stock_result = self._analyze_stock(stock_code, start_date, end_date)
                
                if stock_result:
                    results.append(stock_result)
                    
            except Exception as e:
                logger.error(f"分析股票 {stock_code} 时出错: {e}")
                continue
        
        if not results:
            logger.warning("没有股票通过筛选条件")
            return pd.DataFrame()
        
        # 转换为DataFrame并排序
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('total_score', ascending=False)
        
        # 应用评分阈值
        score_threshold = self._parameters.get('score_threshold', 60)
        df_results = df_results[df_results['total_score'] >= score_threshold]
        
        # 限制结果数量
        max_results = self._parameters.get('max_results', 50)
        df_results = df_results.head(max_results)
        
        logger.info(f"✅ 选股完成，共选出 {len(df_results)} 只股票")
        
        return df_results
    
    def _analyze_stock(self, stock_code: str, start_date: str, end_date: str) -> Optional[Dict[str, Any]]:
        """
        分析单只股票
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Optional[Dict[str, Any]]: 分析结果
        """
        try:
            # 多周期数据收集
            period_data = {}
            period_scores = {}
            
            # 获取各周期数据
            periods_to_analyze = ['1d', '1w', '1h']
            
            for period in periods_to_analyze:
                data = self.get_stock_data(stock_code, period, start_date, end_date)
                if data is not None and not data.empty:
                    period_data[period] = data
                    
                    # 计算该周期的评分
                    period_score = self._calculate_period_score(stock_code, period, data)
                    period_scores[period] = period_score
            
            # 检查是否有足够的数据
            if not period_data:
                return None
            
            # 获取最新数据（使用日线数据）
            daily_data = period_data.get('1d')
            if daily_data is None or daily_data.empty:
                return None
            
            latest_data = daily_data.iloc[-1]
            
            # 计算总评分
            total_score = self._calculate_total_score(period_scores)
            
            # 如果评分太低，直接过滤
            if total_score < 30:  # 预筛选阈值
                return None
            
            # 构建结果
            result = {
                'code': stock_code,
                'name': latest_data.get('name', ''),
                'close_price': float(latest_data['close']),
                'volume': float(latest_data['volume']),
                'date': latest_data.name.strftime('%Y-%m-%d') if hasattr(latest_data.name, 'strftime') else str(latest_data.name),
                'total_score': round(total_score, 2),
                'daily_score': round(period_scores.get('1d', 0), 2),
                'weekly_score': round(period_scores.get('1w', 0), 2),
                'hourly_score': round(period_scores.get('1h', 0), 2),
                'signal_summary': self._generate_signal_summary(period_scores),
                'analysis_periods': list(period_data.keys())
            }
            
            # 添加技术指标详情
            result.update(self._calculate_technical_details(daily_data))
            
            return result
            
        except Exception as e:
            logger.error(f"分析股票 {stock_code} 时出错: {e}")
            return None
    
    def _calculate_period_score(self, stock_code: str, period: str, data: pd.DataFrame) -> float:
        """
        计算单个周期的评分
        
        Args:
            stock_code: 股票代码
            period: 周期
            data: 股票数据
            
        Returns:
            float: 周期评分
        """
        try:
            # 获取该周期的条件
            conditions = self.get_conditions_by_period(period)
            
            if not conditions:
                return 0.0
            
            total_weight = sum(c.weight for c in conditions)
            if total_weight == 0:
                return 0.0
            
            score = 0.0
            
            for condition in conditions:
                # 模拟指标计算和条件判断
                condition_score = self._evaluate_condition(condition, data)
                weighted_score = condition_score * condition.weight
                score += weighted_score
            
            # 标准化评分到0-100
            normalized_score = (score / total_weight) * 100
            
            return max(0.0, min(100.0, normalized_score))
            
        except Exception as e:
            logger.error(f"计算{period}周期评分时出错: {e}")
            return 0.0
    
    def _evaluate_condition(self, condition: Dict[str, Any], data: pd.DataFrame) -> float:
        """
        评估单个条件
        
        Args:
            condition: 指标条件
            data: 股票数据
            
        Returns:
            float: 条件评分（0-1）
        """
        try:
            if data.empty:
                return 0.0
            
            # 根据指标类型进行简化计算
            if condition.indicator_name == 'MA':
                return self._evaluate_ma_condition(condition, data)
            elif condition.indicator_name == 'RSI':
                return self._evaluate_rsi_condition(condition, data)
            elif condition.indicator_name == 'MACD':
                return self._evaluate_macd_condition(condition, data)
            elif condition.indicator_name == 'BOLL':
                return self._evaluate_boll_condition(condition, data)
            elif condition.indicator_name == 'VOL':
                return self._evaluate_volume_condition(condition, data)
            elif condition.indicator_name == 'KDJ':
                return self._evaluate_kdj_condition(condition, data)
            else:
                # 默认随机评分（实际应用中应该实现具体的指标计算）
                return np.random.uniform(0.3, 0.9)
                
        except Exception as e:
            logger.error(f"评估条件 {condition.indicator_name} 时出错: {e}")
            return 0.0
    
    def _evaluate_ma_condition(self, condition: Dict[str, Any], data: pd.DataFrame) -> float:
        """评估MA条件"""
        try:
            period = condition.parameters.get('period', 20)
            if len(data) < period:
                return 0.0
            
            ma = data['close'].rolling(window=period).mean()
            latest_close = data['close'].iloc[-1]
            latest_ma = ma.iloc[-1]
            
            if pd.isna(latest_ma):
                return 0.0
            
            # 价格高于均线得分更高
            if latest_close > latest_ma:
                ratio = (latest_close - latest_ma) / latest_ma
                return min(1.0, 0.5 + ratio * 10)  # 基础分0.5，超出部分按比例加分
            else:
                ratio = (latest_ma - latest_close) / latest_ma
                return max(0.0, 0.5 - ratio * 10)  # 低于均线扣分
                
        except Exception as e:
            logger.error(f"评估MA条件时出错: {e}")
            return 0.0
    
    def _evaluate_rsi_condition(self, condition: Dict[str, Any], data: pd.DataFrame) -> float:
        """评估RSI条件"""
        try:
            period = condition.parameters.get('period', 14)
            if len(data) < period + 1:
                return 0.0
            
            # 计算RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            latest_rsi = rsi.iloc[-1]
            
            if pd.isna(latest_rsi):
                return 0.0
            
            # RSI在30-70之间得分较高
            if 30 <= latest_rsi <= 70:
                return 0.8
            elif 20 <= latest_rsi < 30 or 70 < latest_rsi <= 80:
                return 0.6
            else:
                return 0.3
                
        except Exception as e:
            logger.error(f"评估RSI条件时出错: {e}")
            return 0.0
    
    def _evaluate_macd_condition(self, condition: Dict[str, Any], data: pd.DataFrame) -> float:
        """评估MACD条件"""
        try:
            fast = condition.parameters.get('fast', 12)
            slow = condition.parameters.get('slow', 26)
            signal = condition.parameters.get('signal', 9)
            
            if len(data) < slow + signal:
                return 0.0
            
            # 计算MACD
            ema_fast = data['close'].ewm(span=fast).mean()
            ema_slow = data['close'].ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal).mean()
            
            latest_macd = macd.iloc[-1]
            latest_signal = signal_line.iloc[-1]
            
            if pd.isna(latest_macd) or pd.isna(latest_signal):
                return 0.0
            
            # MACD高于信号线得分更高
            if latest_macd > latest_signal:
                return 0.8
            else:
                return 0.3
                
        except Exception as e:
            logger.error(f"评估MACD条件时出错: {e}")
            return 0.0
    
    def _evaluate_boll_condition(self, condition: Dict[str, Any], data: pd.DataFrame) -> float:
        """评估布林带条件"""
        try:
            period = condition.parameters.get('period', 20)
            std_dev = condition.parameters.get('std_dev', 2.0)
            
            if len(data) < period:
                return 0.0
            
            # 计算布林带
            middle_band = data['close'].rolling(window=period).mean()
            std = data['close'].rolling(window=period).std()
            upper_band = middle_band + (std * std_dev)
            lower_band = middle_band - (std * std_dev)
            
            latest_close = data['close'].iloc[-1]
            latest_upper = upper_band.iloc[-1]
            latest_lower = lower_band.iloc[-1]
            latest_middle = middle_band.iloc[-1]
            
            if pd.isna(latest_upper) or pd.isna(latest_lower):
                return 0.0
            
            # 价格在布林带中间区域得分较高
            if latest_lower < latest_close < latest_upper:
                # 计算在布林带中的位置
                position = (latest_close - latest_lower) / (latest_upper - latest_lower)
                if 0.3 <= position <= 0.7:  # 中间区域
                    return 0.8
                else:
                    return 0.6
            else:
                return 0.2  # 超出布林带
                
        except Exception as e:
            logger.error(f"评估BOLL条件时出错: {e}")
            return 0.0
    
    def _evaluate_volume_condition(self, condition: Dict[str, Any], data: pd.DataFrame) -> float:
        """评估成交量条件"""
        try:
            ma_period = condition.parameters.get('ma_period', 20)
            
            if len(data) < ma_period:
                return 0.0
            
            volume_ma = data['volume'].rolling(window=ma_period).mean()
            latest_volume = data['volume'].iloc[-1]
            latest_volume_ma = volume_ma.iloc[-1]
            
            if pd.isna(latest_volume_ma) or latest_volume_ma == 0:
                return 0.0
            
            # 成交量高于均线得分更高
            ratio = latest_volume / latest_volume_ma
            if ratio >= 1.5:
                return 1.0
            elif ratio >= 1.2:
                return 0.8
            elif ratio >= 1.0:
                return 0.6
            else:
                return 0.3
                
        except Exception as e:
            logger.error(f"评估VOL条件时出错: {e}")
            return 0.0
    
    def _evaluate_kdj_condition(self, condition: Dict[str, Any], data: pd.DataFrame) -> float:
        """评估KDJ条件"""
        try:
            period = condition.parameters.get('period', 9)
            k_period = condition.parameters.get('k_period', 3)
            d_period = condition.parameters.get('d_period', 3)
            
            if len(data) < period + max(k_period, d_period):
                return 0.0
            
            # 计算KDJ（简化版本）
            high_n = data['high'].rolling(window=period).max()
            low_n = data['low'].rolling(window=period).min()
            rsv = (data['close'] - low_n) / (high_n - low_n) * 100
            
            k = rsv.ewm(alpha=1/k_period).mean()
            d = k.ewm(alpha=1/d_period).mean()
            j = 3 * k - 2 * d
            
            latest_k = k.iloc[-1]
            latest_d = d.iloc[-1]
            latest_j = j.iloc[-1]
            
            if pd.isna(latest_k) or pd.isna(latest_d):
                return 0.0
            
            # K > D 且 J > 50 得分较高
            if latest_k > latest_d and latest_j > 50:
                return 0.8
            elif latest_k > latest_d:
                return 0.6
            else:
                return 0.3
                
        except Exception as e:
            logger.error(f"评估KDJ条件时出错: {e}")
            return 0.0
    
    def _calculate_total_score(self, period_scores: Dict[str, float]) -> float:
        """
        计算总评分
        
        Args:
            period_scores: 各周期评分
            
        Returns:
            float: 总评分
        """
        # 权重配置
        weights = {
            '1d': 0.5,   # 日线权重50%
            '1w': 0.3,   # 周线权重30%
            '1h': 0.2    # 小时线权重20%
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for period, score in period_scores.items():
            if period in weights:
                weight = weights[period]
                total_score += score * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return total_score / total_weight * 100
    
    def _generate_signal_summary(self, period_scores: Dict[str, float]) -> str:
        """生成信号汇总"""
        signals = []
        
        for period, score in period_scores.items():
            if score >= 80:
                signals.append(f"{period}强势")
            elif score >= 60:
                signals.append(f"{period}看好")
            elif score >= 40:
                signals.append(f"{period}中性")
            else:
                signals.append(f"{period}偏弱")
        
        return ", ".join(signals)
    
    def _calculate_technical_details(self, data: pd.DataFrame) -> Dict[str, Any]:
        """计算技术指标详情"""
        try:
            if data.empty:
                return {}
            
            latest = data.iloc[-1]
            
            # 计算简单的技术指标
            ma5 = data['close'].rolling(window=5).mean().iloc[-1] if len(data) >= 5 else None
            ma20 = data['close'].rolling(window=20).mean().iloc[-1] if len(data) >= 20 else None
            
            # 计算涨跌幅
            if len(data) >= 2:
                prev_close = data['close'].iloc[-2]
                change_pct = ((latest['close'] - prev_close) / prev_close * 100)
            else:
                change_pct = 0.0
            
            return {
                'ma5': round(float(ma5), 2) if ma5 and not pd.isna(ma5) else None,
                'ma20': round(float(ma20), 2) if ma20 and not pd.isna(ma20) else None,
                'change_pct': round(change_pct, 2),
                'high': float(latest['high']),
                'low': float(latest['low']),
                'volume_ratio': round(float(latest['volume']) / data['volume'].mean(), 2) if len(data) > 1 else 1.0
            }
            
        except Exception as e:
            logger.error(f"计算技术指标详情时出错: {e}")
            return {} 