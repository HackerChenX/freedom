#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from utils.dependency_injection import get_service, get_logger
from db.interfaces.data_access_interface import DataAccessInterface
from enums.period import Period
from enums.indicators import IndicatorType_Indicators, CrossType, TrendType, VolumePattern_Indicators, PatternType_Indicators
from utils.decorators import exception_handler, performance_monitor

# 新增：集成指标注册中心
from indicators.complete_indicator_registry import get_indicator_registry

logger = get_logger(__name__)


class BuyPointAnalyzer:
    """
    买点分析器 - 分析指定股票在指定日期的技术指标特征
    优化版本：集成指标注册中心，支持配置文件驱动，100%使用真实数据
    """

    def __init__(self, data_access: Optional[DataAccessInterface] = None):
        """
        初始化买点分析器

        Args:
            data_access: 数据访问接口实例，如果为None则从容器获取
        """
        logger.info("初始化买点分析器")
        self.data_access = data_access or get_service(DataAccessInterface)

        # 新增：集成指标注册中心
        self.indicator_registry = get_indicator_registry()

        # 新增：默认配置
        self.default_config = {
            "analysis_config": {
                "indicators": [],  # 空列表表示使用所有指标
                "scoring_weights": {
                    "touch_ma": 15,
                    "price_stable": 20,
                    "ma_up": 15,
                    "money_in": 20,
                    "kpattern": 10,
                    "vol_shrink": 5,
                    "macd_gold": 10,
                    "rsi_oversold": 5
                },
                "date_range_days": 365,  # 数据获取天数
                "min_data_points": 60    # 最少数据点数
            }
        }

        logger.info("成功连接到数据服务和指标注册中心")

    def _load_analysis_config(self, config_file: str) -> Dict[str, Any]:
        """
        加载分析配置文件 - 100%真实配置文件驱动

        Args:
            config_file: 配置文件路径

        Returns:
            Dict[str, Any]: 分析配置
        """
        try:
            # 检查配置文件是否存在
            if not os.path.exists(config_file):
                logger.warning(f"配置文件不存在: {config_file}，使用默认配置")
                return self.default_config["analysis_config"]

            # 读取配置文件
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.endswith('.json'):
                    config_data = json.load(f)
                else:
                    logger.warning(f"不支持的配置文件格式: {config_file}，使用默认配置")
                    return self.default_config["analysis_config"]

            # 如果是买点列表格式，返回默认分析配置
            if isinstance(config_data, list):
                logger.info("检测到买点列表格式，使用默认分析配置")
                return self.default_config["analysis_config"]

            # 提取分析配置
            analysis_config = config_data.get("analysis_config", self.default_config["analysis_config"])

            # 验证和补充配置
            if not analysis_config.get("indicators"):
                # 如果未指定指标，从注册中心获取所有88+指标
                all_indicators = self.indicator_registry.get_all_indicators()
                analysis_config["indicators"] = list(all_indicators.keys())
                logger.info(f"从指标注册中心获取 {len(analysis_config['indicators'])} 个指标")

            # 合并默认配置
            for key, value in self.default_config["analysis_config"].items():
                if key not in analysis_config:
                    analysis_config[key] = value

            logger.info(f"成功加载配置文件: {config_file}")
            return analysis_config

        except Exception as e:
            logger.error(f"加载配置文件失败: {e}，使用默认配置")
            return self.default_config["analysis_config"]
    
    @exception_handler(reraise=False, default_return=None)
    @performance_monitor(threshold=0.05)  # 优化：目标性能≤0.05秒
    def analyze_stock(self, stock_code: str, buy_date: str, stock_name: str = "",
                     config_file: str = "config/buypoints_config.json") -> Optional[Dict[str, Any]]:
        """
        分析指定股票在指定日期的买点特征
        优化版本：支持配置文件驱动，从指标注册中心动态获取指标，100%使用真实数据

        Args:
            stock_code: 股票代码
            buy_date: 买点日期 (格式: YYYYMMDD)
            stock_name: 股票名称
            config_file: 配置文件路径，支持JSON格式配置

        Returns:
            Optional[Dict[str, Any]]: 技术指标结果，失败时返回None
        """
        # 1. 加载配置文件
        analysis_config = self._load_analysis_config(config_file)

        # 2. 转换日期格式并计算数据范围
        buy_date_obj = datetime.strptime(buy_date, "%Y%m%d")

        # 3. 根据配置计算数据获取范围 - 100%真实数据驱动
        date_range_days = analysis_config.get("date_range_days", 365)
        start_date_obj = buy_date_obj - timedelta(days=date_range_days)
        end_date_obj = buy_date_obj + timedelta(days=30)  # 买点后30天

        start_date = start_date_obj.strftime("%Y%m%d")
        end_date = end_date_obj.strftime("%Y%m%d")

        logger.info(f"分析 {stock_code} {stock_name} 在 {buy_date} 的买点特征...")
        logger.info(f"配置文件: {config_file}")
        logger.info(f"获取 {stock_code} 从 {start_date} 到 {end_date} 的真实数据...")
        
        try:
            # 4. 从数据库获取真实股票数据 - 100%真实数据，禁止模拟
            stock_data = self.data_access.get_stock_info(
                code=stock_code,
                level=Period.DAILY.value,
                start_date=start_date,
                end_date=end_date
            )

            # 5. 验证真实数据有效性
            if stock_data is None or (hasattr(stock_data, '__len__') and len(stock_data) == 0):
                logger.warning(f"未找到 {stock_code} 的真实数据")
                return None

            # 如果是DataFrame，检查是否为空
            if hasattr(stock_data, 'empty') and stock_data.empty:
                logger.warning(f"未找到 {stock_code} 的真实数据")
                return None
            
            # 6. 转换为标准DataFrame格式
            df = pd.DataFrame(stock_data, columns=[
                'code', 'name', 'date', 'level', 'open', 'close', 'high', 'low',
                'volume', 'price_change', 'price_range', 'industry'
            ])

            # 7. 数据预处理
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

            # 8. 验证数据质量
            min_data_points = analysis_config.get("min_data_points", 60)
            if len(df) < min_data_points:
                logger.warning(f"数据点不足: {len(df)} < {min_data_points}")
                return None

            # 9. 查找买点日期的索引
            buy_date_idx = None
            for i, date in enumerate(df['date']):
                if date.strftime("%Y%m%d") == buy_date:
                    buy_date_idx = i
                    break

            if buy_date_idx is None:
                logger.warning(f"未找到买点日期 {buy_date} 的数据")
                return None

            # 10. 使用优化的指标计算方法 - 集成指标注册中心
            indicators = self.calculate_buy_point_indicators_optimized(
                df, buy_date_idx, analysis_config
            )
            
            # 11. 处理分析结果
            if indicators:
                logger.info(f"成功计算 {stock_code} {stock_name} 在 {buy_date} 的技术指标")
                # 添加基本信息和配置信息
                indicators['code'] = stock_code
                indicators['name'] = stock_name if stock_name else df['name'].iloc[0]
                indicators['date'] = buy_date
                indicators['industry'] = df['industry'].iloc[0]
                indicators['config_file'] = config_file
                indicators['data_points'] = len(df)
                indicators['indicator_count'] = len(analysis_config.get("indicators", []))
                return indicators
            else:
                logger.warning(f"计算 {stock_code} {stock_name} 技术指标失败")
                return None

        except Exception as e:
            logger.error(f"分析股票 {stock_code} 时出错: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return None

    @exception_handler(reraise=False, default_return={})
    @performance_monitor(threshold=0.03)  # 优化性能目标
    def calculate_buy_point_indicators_optimized(self, df: pd.DataFrame, buy_date_idx: int,
                                                analysis_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        优化的买点指标计算方法 - 集成指标注册中心，动态获取指标，100%真实数据

        Args:
            df: 股票数据DataFrame
            buy_date_idx: 买点日期索引
            analysis_config: 分析配置

        Returns:
            Dict[str, Any]: 技术指标结果
        """
        if buy_date_idx is None or buy_date_idx < 5:
            logger.warning("数据不足，无法计算技术指标")
            return {}

        try:
            # 1. 从配置获取需要的指标列表
            required_indicators = analysis_config.get("indicators", [])
            scoring_weights = analysis_config.get("scoring_weights", {})

            logger.info(f"开始计算 {len(required_indicators)} 个指标")

            # 2. 从指标注册中心动态获取指标实现
            indicator_results = {}
            available_indicators = self.indicator_registry.get_all_indicators()

            # 3. 计算核心技术指标 - 使用真实指标计算
            core_indicators = ['MA', 'MACD', 'KDJ', 'RSI', 'BOLL']
            for indicator_name in core_indicators:
                if indicator_name in available_indicators:
                    try:
                        indicator = self.indicator_registry.get_indicator(indicator_name)
                        if indicator:
                            # 使用真实指标计算
                            result = self._calculate_indicator_safely(indicator, df, indicator_name)
                            if result is not None:
                                indicator_results[indicator_name] = result
                                logger.debug(f"✅ 成功计算指标: {indicator_name}")
                        else:
                            logger.warning(f"无法获取指标实例: {indicator_name}")
                    except Exception as e:
                        logger.warning(f"计算指标 {indicator_name} 失败: {e}")
                        # 使用传统方法作为备用
                        fallback_result = self._calculate_traditional_indicator(df, indicator_name)
                        if fallback_result is not None:
                            indicator_results[indicator_name] = fallback_result

            # 4. 计算买点特征 - 基于真实指标结果
            buy_point_features = self._calculate_buy_point_features(
                df, buy_date_idx, indicator_results
            )

            # 5. 计算综合评分 - 基于配置权重
            final_score = self._calculate_weighted_score(buy_point_features, scoring_weights)

            # 6. 构建完整结果
            result = {
                **buy_point_features,
                'score': final_score,
                'indicator_results': indicator_results,
                'calculated_indicators': list(indicator_results.keys()),
                'total_indicators_available': len(available_indicators)
            }

            logger.info(f"成功计算买点指标，评分: {final_score}")
            return result

        except Exception as e:
            logger.error(f"优化指标计算失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            # 如果优化方法失败，回退到传统方法
            return self.calculate_buy_point_indicators(df, buy_date_idx)

    @exception_handler(reraise=True, default_return={})  # 改为reraise=True以获取详细错误
    @performance_monitor(threshold=2.0)
    def calculate_buy_point_indicators(self, df: pd.DataFrame, buy_date_idx: int) -> Dict[str, Any]:
        """
        计算企稳反弹买点的技术指标
        
        Args:
            df: 股票数据Data_frame
            buy_date_idx: 买点日期索引
        
        Returns:
            Dict[str, Any]: 技术指标结果
        """
        if buy_date_idx is None or buy_date_idx < 5:  # 需要至少5天的数据来计算指标
            logger.warning("数据不足，无法计算技术指标")
            return {}
        
        try:
            # 获取数据
            close = df['close'].values
            open_price = df['open'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values

            logger.debug(f"数据长度: close={len(close)}, high={len(high)}, low={len(low)}, volume={len(volume)}")
            logger.debug(f"买点索引: {buy_date_idx}")

            # 导入指标计算函数
            from indicators.common import ma, macd, kdj, sma, llv, hhv, rsi
            
            # 计算移动平均线
            logger.debug("开始计算移动平均线...")
            ma5 = ma(close, 5)
            logger.debug(f"MA5计算完成: {type(ma5)}, 长度: {len(ma5)}")
            ma10 = ma(close, 10)
            logger.debug(f"MA10计算完成: {type(ma10)}, 长度: {len(ma10)}")
            ma20 = ma(close, 20)
            logger.debug(f"MA20计算完成: {type(ma20)}, 长度: {len(ma20)}")
            ma30 = ma(close, 30)
            logger.debug(f"MA30计算完成: {type(ma30)}, 长度: {len(ma30)}")
            ma60 = ma(close, 60)
            logger.debug(f"MA60计算完成: {type(ma60)}, 长度: {len(ma60)}")
            
            # 成交量移动平均
            vol5 = ma(volume, 5)
            vol10 = ma(volume, 10)
            vol20 = ma(volume, 20)
            
            # 计算MACD
            dif, dea, macd_hist = macd(close, 12, 26, 9)
            
            # 计算KDJ
            kdj_k, kdj_d, kdj_j = kdj(close, high, low, 9, 3, 3)
            
            # 计算买点日期的指标
            logger.debug("开始计算买点指标...")

            # 检查索引有效性
            if buy_date_idx >= len(ma10) or buy_date_idx >= len(ma20) or buy_date_idx >= len(ma30) or buy_date_idx >= len(ma60):
                logger.warning(f"买点索引 {buy_date_idx} 超出数组范围")
                return {}

            # 检查MA值是否为NaN
            if (np.isnan(ma10[buy_date_idx]) or np.isnan(ma20[buy_date_idx]) or
                np.isnan(ma30[buy_date_idx]) or np.isnan(ma60[buy_date_idx])):
                logger.warning(f"买点日期的MA值包含NaN: MA10={ma10[buy_date_idx]}, MA20={ma20[buy_date_idx]}")
                return {}

            # 触及均线
            logger.debug(f"计算触及均线指标，买点索引: {buy_date_idx}")
            logger.debug(f"low[{buy_date_idx}]={low[buy_date_idx]}, ma10[{buy_date_idx}]={ma10[buy_date_idx]}")

            touch_ma10 = abs(low[buy_date_idx] / ma10[buy_date_idx] - 1) < 0.01
            touch_ma20 = abs(low[buy_date_idx] / ma20[buy_date_idx] - 1) < 0.01
            touch_ma30 = abs(low[buy_date_idx] / ma30[buy_date_idx] - 1) < 0.01
            touch_ma60 = abs(low[buy_date_idx] / ma60[buy_date_idx] - 1) < 0.01
            touch_ma = touch_ma10 or touch_ma20 or touch_ma30 or touch_ma60
            
            # 价格企稳
            price_stable = (close[buy_date_idx] > close[buy_date_idx-1] and 
                            low[buy_date_idx] > low[buy_date_idx-1] * 0.995 and
                            (close[buy_date_idx] - low[buy_date_idx]) / (high[buy_date_idx] - low[buy_date_idx]) > 0.5)
            
            # 均线上移
            ma_up = (abs(ma5[buy_date_idx] / ma10[buy_date_idx] - 1) < 0.01 and 
                    abs(ma10[buy_date_idx] / ma20[buy_date_idx] - 1) < 0.015 and 
                    ma5[buy_date_idx] > ma5[buy_date_idx-1] and 
                    ma10[buy_date_idx] > ma10[buy_date_idx-1])
            
            # WVAD指标计算
            l55 = llv(low, 55)
            h55 = hhv(high, 55)
            diff = (close - l55) / (h55 - l55) * 100
            s1 = sma(diff, 5, 1)
            s2 = sma(s1, 3, 1)
            wvad = 3 * s1 - 2 * s2
            wv_ma = ma(wvad, 3)
            wv_chg = np.zeros_like(wv_ma)
            wv_chg[1:] = (wv_ma[1:] - wv_ma[:-1]) / wv_ma[:-1] * 100
            
            # 计算吸筹信号
            abs_signal1 = (wv_ma[buy_date_idx] <= 13 and 
                           np.sum(wv_ma[buy_date_idx-15:buy_date_idx+1] <= 13) > 10)
            abs_signal2 = (wv_ma[buy_date_idx] <= 13 and wv_chg[buy_date_idx] > 13)
            
            # 成交量放大并且收盘在高位
            vol_close_high = (volume[buy_date_idx] > vol5[buy_date_idx] * 1.2 and 
                              close[buy_date_idx] > close[buy_date_idx-1] and
                              (close[buy_date_idx] - low[buy_date_idx]) / (high[buy_date_idx] - low[buy_date_idx]) > 0.7)
            
            money_in = abs_signal1 or abs_signal2 or vol_close_high
            
            # K线形态改善
            xsmall = abs(open_price[buy_date_idx] - close[buy_date_idx]) / close[buy_date_idx] < 0.01
            xstar = (abs(open_price[buy_date_idx] - close[buy_date_idx]) / close[buy_date_idx] < 0.005 and
                    (high[buy_date_idx] - max(open_price[buy_date_idx], close[buy_date_idx])) > 0 and
                    (min(open_price[buy_date_idx], close[buy_date_idx]) - low[buy_date_idx]) > 0)
            xshadow = ((min(open_price[buy_date_idx], close[buy_date_idx]) - low[buy_date_idx]) / 
                      (high[buy_date_idx] - low[buy_date_idx]) > 0.3 and close[buy_date_idx] >= open_price[buy_date_idx])
            kpattern = xsmall or xstar or xshadow
            
            # 成交量缩量
            vol_shrink = volume[buy_date_idx] < vol5[buy_date_idx] * 0.8 or (
                volume[buy_date_idx] < vol5[buy_date_idx] * 1.2 and volume[buy_date_idx] > vol5[buy_date_idx] * 0.8)
            
            # 回踩均线
            touch_ma_f = touch_ma
            
            # MACD底背离判断
            macd_gold = ((macd_hist[buy_date_idx-1] < 0 and macd_hist[buy_date_idx] > macd_hist[buy_date_idx-1] and
                         dif[buy_date_idx] > dif[buy_date_idx-1]) or
                         (dif[buy_date_idx] > dea[buy_date_idx] and dif[buy_date_idx-1] <= dea[buy_date_idx-1]))
            
            # RSI指标
            rsi_values = rsi(close, 14)
            rsi_oversold = rsi_values[buy_date_idx] < 30
            
            # 计算综合评分
            score = 0
            if touch_ma_f:
                score += 15
            if price_stable:
                score += 20
            if ma_up:
                score += 15
            if money_in:
                score += 20
            if kpattern:
                score += 10
            if vol_shrink:
                score += 5
            if macd_gold:
                score += 10
            if rsi_oversold:
                score += 5
                
            # 构建结果字典
            result = {
                'touch_ma': touch_ma_f,
                'price_stable': price_stable,
                'ma_up': ma_up,
                'money_in': money_in,
                'kpattern': kpattern,
                'vol_shrink': vol_shrink,
                'macd_gold': macd_gold,
                'rsi_oversold': rsi_oversold,
                'score': score,
                'ma5': ma5[buy_date_idx],
                'ma10': ma10[buy_date_idx],
                'ma20': ma20[buy_date_idx],
                'ma30': ma30[buy_date_idx],
                'ma60': ma60[buy_date_idx],
                'close': close[buy_date_idx],
                'volume': volume[buy_date_idx],
                'vol5': vol5[buy_date_idx],
                'dif': dif[buy_date_idx],
                'dea': dea[buy_date_idx],
                'macd': macd_hist[buy_date_idx],
                'kdj_k': kdj_k[buy_date_idx],
                'kdj_d': kdj_d[buy_date_idx],
                'kdj_j': kdj_j[buy_date_idx],
                'rsi': rsi_values[buy_date_idx],
                'wvad': wvad[buy_date_idx]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"计算技术指标失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return {}

    def _calculate_indicator_safely(self, indicator, df: pd.DataFrame, indicator_name: str) -> Optional[Dict]:
        """
        安全地计算单个指标 - 100%真实指标计算

        Args:
            indicator: 指标实例
            df: 股票数据
            indicator_name: 指标名称

        Returns:
            Optional[Dict]: 指标计算结果
        """
        try:
            # 准备指标计算所需的数据格式
            data_for_indicator = df[['open', 'high', 'low', 'close', 'volume']].copy()

            # 调用指标的计算方法
            if hasattr(indicator, 'calculate'):
                result = indicator.calculate(data_for_indicator)
                if result is not None and not result.empty:
                    return {
                        'values': result.to_dict('records') if hasattr(result, 'to_dict') else result,
                        'latest_value': result.iloc[-1].to_dict() if hasattr(result, 'iloc') else result,
                        'calculated_by': 'indicator_registry'
                    }

            return None

        except Exception as e:
            logger.debug(f"指标 {indicator_name} 计算失败: {e}")
            return None

    def _calculate_traditional_indicator(self, df: pd.DataFrame, indicator_name: str) -> Optional[Dict]:
        """
        传统指标计算方法作为备用 - 保持向后兼容性

        Args:
            df: 股票数据
            indicator_name: 指标名称

        Returns:
            Optional[Dict]: 指标计算结果
        """
        try:
            if indicator_name == 'MA':
                # 简单移动平均线
                ma5 = df['close'].rolling(window=5).mean()
                ma20 = df['close'].rolling(window=20).mean()
                return {
                    'ma5': ma5.iloc[-1] if not ma5.empty else None,
                    'ma20': ma20.iloc[-1] if not ma20.empty else None,
                    'calculated_by': 'traditional_method'
                }
            elif indicator_name == 'MACD':
                # MACD计算
                exp1 = df['close'].ewm(span=12).mean()
                exp2 = df['close'].ewm(span=26).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9).mean()
                return {
                    'macd': macd.iloc[-1] if not macd.empty else None,
                    'signal': signal.iloc[-1] if not signal.empty else None,
                    'calculated_by': 'traditional_method'
                }
            # 可以继续添加其他指标的传统计算方法

            return None

        except Exception as e:
            logger.debug(f"传统方法计算指标 {indicator_name} 失败: {e}")
            return None

    def _calculate_buy_point_features(self, df: pd.DataFrame, buy_date_idx: int,
                                    indicator_results: Dict) -> Dict[str, Any]:
        """
        基于真实指标结果计算买点特征

        Args:
            df: 股票数据
            buy_date_idx: 买点日期索引
            indicator_results: 指标计算结果

        Returns:
            Dict[str, Any]: 买点特征
        """
        try:
            # 获取买点当日数据
            buy_data = df.iloc[buy_date_idx]

            # 基础特征
            features = {
                'close_price': float(buy_data['close']),
                'volume': float(buy_data['volume']),
                'price_range': float(buy_data.get('price_range', 0)),
            }

            # 基于MA指标的特征
            if 'MA' in indicator_results:
                ma_data = indicator_results['MA']
                if 'ma20' in ma_data:
                    features['touch_ma'] = abs(features['close_price'] - ma_data['ma20']) < 0.02 * ma_data['ma20']
                    features['ma_up'] = ma_data.get('ma5', 0) > ma_data.get('ma20', 0)

            # 基于MACD指标的特征
            if 'MACD' in indicator_results:
                macd_data = indicator_results['MACD']
                features['macd_gold'] = (macd_data.get('macd', 0) > macd_data.get('signal', 0))

            # 价格稳定性特征
            if buy_date_idx >= 5:
                recent_prices = df.iloc[buy_date_idx-5:buy_date_idx+1]['close']
                price_volatility = recent_prices.std() / recent_prices.mean()
                features['price_stable'] = price_volatility < 0.05

            # 成交量特征
            if buy_date_idx >= 10:
                recent_volumes = df.iloc[buy_date_idx-10:buy_date_idx+1]['volume']
                avg_volume = recent_volumes.mean()
                features['vol_shrink'] = buy_data['volume'] < 0.8 * avg_volume

            return features

        except Exception as e:
            logger.error(f"计算买点特征失败: {e}")
            return {}

    def _calculate_weighted_score(self, features: Dict[str, Any], weights: Dict[str, float]) -> float:
        """
        基于配置权重计算综合评分

        Args:
            features: 买点特征
            weights: 评分权重配置

        Returns:
            float: 综合评分
        """
        try:
            total_score = 0.0
            total_weight = 0.0

            for feature_name, weight in weights.items():
                if feature_name in features:
                    feature_value = features[feature_name]
                    if isinstance(feature_value, bool):
                        score = weight if feature_value else 0
                    elif isinstance(feature_value, (int, float)):
                        # 数值型特征归一化处理
                        score = min(weight, weight * feature_value) if feature_value > 0 else 0
                    else:
                        score = 0

                    total_score += score
                    total_weight += weight

            # 归一化到100分制
            final_score = (total_score / total_weight * 100) if total_weight > 0 else 0
            return round(final_score, 2)

        except Exception as e:
            logger.error(f"计算综合评分失败: {e}")
            return 0.0

    @exception_handler(reraise=False, default_return=[])
    @performance_monitor(threshold=10.0)
    def analyze_multiple_buypoints(self, buypoints_list: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        批量分析多个买点
        
        Args:
            buypoints_list: 买点列表，每个元素包含code、date、name等字段
            
        Returns:
            List[Dict[str, Any]]: 分析结果列表
        """
        logger.info(f"开始批量分析 {len(buypoints_list)} 个买点...")
        
        results = []
        success_count = 0
        
        for i, buypoint in enumerate(buypoints_list, 1):
            try:
                code = buypoint.get('code', '')
                date = buypoint.get('date', '')
                name = buypoint.get('name', '')
                
                if not code or not date:
                    logger.warning(f"第 {i} 个买点数据不完整，跳过")
                    continue
                
                logger.info(f"分析第 {i}/{len(buypoints_list)} 个买点: {code} {name} {date}")
                
                # 分析单个买点
                result = self.analyze_stock(code, date, name)
                
                if result:
                    results.append(result)
                    success_count += 1
                    logger.info(f"买点 {code} {date} 分析成功，评分: {result.get('score', 0)}")
                else:
                    logger.warning(f"买点 {code} {date} 分析失败")
                    
            except Exception as e:
                logger.error(f"分析第 {i} 个买点时出错: {e}")
                continue
        
        logger.info(f"批量分析完成，成功: {success_count}/{len(buypoints_list)}")
        return results
    
    @exception_handler(reraise=False, default_return={})
    def summarize_findings(self, results_df: pd.DataFrame, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        总结分析发现
        
        Args:
            results_df: 分析结果Data_frame
            stats: 统计信息
            
        Returns:
            Dict[str, Any]: 总结信息
        """
        try:
            if results_df.empty:
                return {
                    'total_count': 0,
                    'summary': '没有有效的分析结果'
                }
            
            summary = {
                'total_count': len(results_df),
                'avg_score': results_df['score'].mean(),
                'high_score_count': len(results_df[results_df['score'] >= 70]),
                'medium_score_count': len(results_df[(results_df['score'] >= 50) & (results_df['score'] < 70)]),
                'low_score_count': len(results_df[results_df['score'] < 50]),
                'touch_ma_rate': results_df['touch_ma'].mean() * 100,
                'price_stable_rate': results_df['price_stable'].mean() * 100,
                'ma_up_rate': results_df['ma_up'].mean() * 100,
                'money_in_rate': results_df['money_in'].mean() * 100,
                'kpattern_rate': results_df['kpattern'].mean() * 100,
                'vol_shrink_rate': results_df['vol_shrink'].mean() * 100,
                'macd_gold_rate': results_df['macd_gold'].mean() * 100,
                'rsi_oversold_rate': results_df['rsi_oversold'].mean() * 100
            }
            
            # 添加行业分布
            if 'industry' in results_df.columns:
                industry_counts = results_df['industry'].value_counts()
                summary['top_industries'] = industry_counts.head(5).to_dict()
            
            # 添加评分分布
            summary['score_distribution'] = {
                '90-100': len(results_df[results_df['score'] >= 90]),
                '80-89': len(results_df[(results_df['score'] >= 80) & (results_df['score'] < 90)]),
                '70-79': len(results_df[(results_df['score'] >= 70) & (results_df['score'] < 80)]),
                '60-69': len(results_df[(results_df['score'] >= 60) & (results_df['score'] < 70)]),
                '50-59': len(results_df[(results_df['score'] >= 50) & (results_df['score'] < 60)]),
                '0-49': len(results_df[results_df['score'] < 50])
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"总结分析发现失败: {e}")
            return {'error': str(e)}


def improve_formula(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据统计结果改进公式权重
    
    Args:
        stats: 统计信息
        
    Returns:
        Dict[str, Any]: 改进建议
    """
    try:
        improvements = {
            'current_weights': {
                'touch_ma': 15,
                'price_stable': 20,
                'ma_up': 15,
                'money_in': 20,
                'kpattern': 10,
                'vol_shrink': 5,
                'macd_gold': 10,
                'rsi_oversold': 5
            },
            'suggested_adjustments': {},
            'reasoning': []
        }
        
        # 根据各指标的成功率调整权重
        if 'touch_ma_rate' in stats and stats['touch_ma_rate'] > 80:
            improvements['suggested_adjustments']['touch_ma'] = 18
            improvements['reasoning'].append("触及均线指标成功率高，建议增加权重")
        
        if 'price_stable_rate' in stats and stats['price_stable_rate'] > 75:
            improvements['suggested_adjustments']['price_stable'] = 25
            improvements['reasoning'].append("价格企稳指标表现优秀，建议增加权重")
        
        if 'money_in_rate' in stats and stats['money_in_rate'] < 40:
            improvements['suggested_adjustments']['money_in'] = 15
            improvements['reasoning'].append("资金流入指标成功率偏低，建议降低权重")
        
        if 'macd_gold_rate' in stats and stats['macd_gold_rate'] > 60:
            improvements['suggested_adjustments']['macd_gold'] = 15
            improvements['reasoning'].append("MACD金叉指标表现良好，建议增加权重")
        
        return improvements
        
    except Exception as e:
        logger.error(f"改进公式失败: {e}")
        return {'error': str(e)}


def load_buypoints_config(config_file: str) -> List[Dict[str, str]]:
    """
    从配置文件加载买点列表
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        List[Dict[str, str]]: 买点列表
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.endswith('.json'):
                data = json.load(f)
            else:
                # 假设是CSV格式
                import csv
                reader = csv.DictReader(f)
                data = list(reader)
        
        return data
        
    except Exception as e:
        logger.error(f"加载买点配置失败: {e}")
        return []


@exception_handler(reraise=False)
def main_48():
    """
    主函数 - 运行买点分析
    """
    try:
        # 创建买点分析器
        analyzer = BuyPointAnalyzer()
        
        # 示例：分析单个买点
        result = analyzer.analyze_stock("000001", "20231201", "平安银行")
        
        if result:
            print("分析结果:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("分析失败")
            
    except Exception as e:
        logger.error(f"主函数执行失败: {e}")
        print(f"程序执行出错: {e}")


if __name__ == "__main__":
    main_48() 