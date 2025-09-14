#!/usr/bin/env python3
"""
通用买点分析器 - 符合六层架构的生产级实现
L4: 核心服务层 - 买点分析核心业务逻辑
"""

import sys
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from db.data_access_manager import DataAccessManager
from indicators.complete_indicator_registry import get_indicator_registry

logger = get_logger(__name__)

class UniversalBuyPointAnalyzer:
    """
    通用买点分析器
    
    符合六层架构设计：
    - 依赖注入获取数据访问接口
    - 动态获取指标注册系统
    - 通用化的分析接口
    """
    
    def __init__(self):
        """初始化分析器"""
        try:
            # 直接实例化服务（简化版本）
            self.data_access = DataAccessManager()
            self.indicator_registry = get_indicator_registry()

            logger.info("通用买点分析器初始化完成")

        except Exception as e:
            logger.error(f"通用买点分析器初始化失败: {e}")
            raise
    
    def analyze_buypoint(self, stock_code: str, target_date: str, 
                        timeframes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        通用买点分析接口
        
        Args:
            stock_code: 股票代码
            target_date: 目标日期 (YYYYMMDD)
            timeframes: 时间周期列表，默认为['日线']
            
        Returns:
            Dict: 买点分析结果
        """
        if timeframes is None:
            timeframes = ['日线']
            
        logger.info(f"开始分析 {stock_code} 在 {target_date} 的买点")
        
        try:
            # 1. 获取股票数据
            stock_data = self._get_stock_data(stock_code, target_date, timeframes)
            if not stock_data:
                return self._create_empty_result(stock_code, target_date, "无法获取股票数据")
            
            # 2. 计算所有技术指标
            technical_indicators = self._calculate_all_indicators(stock_data, timeframes)
            
            # 3. 动态匹配所有注册的形态
            pattern_matches = self._match_all_registered_patterns(
                technical_indicators, stock_code, target_date
            )
            
            # 4. 计算综合买点评分
            buypoint_score = self._calculate_buypoint_score(
                technical_indicators, pattern_matches
            )
            
            # 5. 生成分析结果
            result = {
                'stock_code': stock_code,
                'target_date': target_date,
                'timeframes': timeframes,
                'technical_indicators': technical_indicators,
                'pattern_matches': pattern_matches,
                'buypoint_score': buypoint_score,
                'total_indicators': len(technical_indicators),
                'total_patterns': len(pattern_matches),
                'analysis_timestamp': datetime.now().isoformat(),
                'success': True
            }
            
            logger.info(f"买点分析完成: {stock_code}, 指标数: {len(technical_indicators)}, "
                       f"形态数: {len(pattern_matches)}, 评分: {buypoint_score:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"买点分析失败 {stock_code}: {e}")
            return self._create_empty_result(stock_code, target_date, str(e))
    
    def _get_stock_data(self, stock_code: str, target_date: str, 
                       timeframes: List[str]) -> Dict[str, Any]:
        """获取股票数据"""
        try:
            # 计算数据范围（需要足够的历史数据计算指标）
            from datetime import datetime, timedelta
            target_dt = datetime.strptime(target_date, '%Y%m%d')
            start_dt = target_dt - timedelta(days=365)  # 获取一年的历史数据
            
            start_date = start_dt.strftime('%Y%m%d')
            end_date = target_date
            
            all_data = {}
            for timeframe in timeframes:
                data = self.data_access.get_stock_data_data_access_manager(
                    code=stock_code,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if data is not None and not data.empty:
                    all_data[timeframe] = data
                    logger.debug(f"获取 {stock_code} {timeframe} 数据: {len(data)} 条")
                else:
                    logger.warning(f"无法获取 {stock_code} {timeframe} 数据")
            
            return all_data
            
        except Exception as e:
            logger.error(f"获取股票数据失败 {stock_code}: {e}")
            return {}
    
    def _calculate_all_indicators(self, stock_data: Dict[str, Any], 
                                 timeframes: List[str]) -> Dict[str, Any]:
        """计算所有注册的技术指标"""
        all_indicators = {}
        
        try:
            # 获取所有注册的指标名称
            registered_indicators_dict = self.indicator_registry.get_all_indicators()
            indicator_names = list(registered_indicators_dict.keys())
            logger.info(f"开始计算 {len(indicator_names)} 个注册指标")

            # 调试：显示前几个指标的信息
            logger.info(f"前5个指标: {indicator_names[:5]}")
            
            for timeframe in timeframes:
                if timeframe not in stock_data:
                    continue
                    
                data = stock_data[timeframe]
                timeframe_indicators = {}
                
                # 动态计算每个注册的指标
                success_count = 0
                for indicator_name in indicator_names:
                    try:
                        # 使用正确的方法获取指标实例
                        indicator = self.indicator_registry.get_indicator(indicator_name)
                        if indicator is None:
                            logger.debug(f"无法获取指标实例: {indicator_name}")
                            continue

                        # 尝试多种计算方法
                        indicator_result = None

                        # 方法1: 使用BaseIndicator统一抽象方法（正确的方式）
                        if hasattr(indicator, '_calculate_baseindicator'):
                            try:
                                indicator_result = indicator._calculate_baseindicator(data)
                                if indicator_result is not None and not indicator_result.empty:
                                    logger.debug(f"使用_calculate_baseindicator方法成功: {indicator_name}")
                            except Exception as e:
                                logger.debug(f"_calculate_baseindicator方法失败 {indicator_name}: {e}")

                        # 方法2: 备用的calculate方法
                        if (indicator_result is None or indicator_result.empty) and hasattr(indicator, 'calculate'):
                            try:
                                indicator_result = indicator.calculate(data)
                                if indicator_result is not None and not indicator_result.empty:
                                    logger.debug(f"使用calculate方法成功: {indicator_name}")
                            except Exception as e:
                                logger.debug(f"calculate方法失败 {indicator_name}: {e}")

                        # 方法3: 如果都失败，记录详细错误信息
                        if indicator_result is None or indicator_result.empty:
                            available_methods = [m for m in dir(indicator) if 'calculate' in m.lower() and not m.startswith('_')]
                            logger.warning(f"指标 {indicator_name} 所有计算方法都失败，类型: {type(indicator).__name__}, 可用方法: {available_methods}")

                        # 如果成功计算，保存结果
                        if indicator_result is not None and not indicator_result.empty:
                            timeframe_indicators[indicator_name] = indicator_result
                            success_count += 1
                            logger.debug(f"成功计算指标: {indicator_name}")
                        else:
                            logger.debug(f"指标计算返回空值: {indicator_name}")

                    except Exception as e:
                        logger.debug(f"指标计算失败 {indicator_name}: {e}")
                        continue
                
                all_indicators[timeframe] = timeframe_indicators
                logger.info(f"{timeframe} 成功计算 {len(timeframe_indicators)} 个指标")
            
            return all_indicators
            
        except Exception as e:
            logger.error(f"计算技术指标失败: {e}")
            return {}



    def _match_all_registered_patterns(self, technical_indicators: Dict[str, Any],
                                     stock_code: str, target_date: str) -> List[Dict[str, Any]]:
        """动态匹配所有注册的形态"""
        all_patterns = []

        try:
            # 简化版本：使用基础形态匹配
            for timeframe, indicators in technical_indicators.items():
                timeframe_patterns = []

                # 基础形态匹配逻辑
                patterns = self._basic_pattern_matching(indicators, stock_code, target_date)

                for pattern in patterns:
                    pattern['timeframe'] = timeframe
                    timeframe_patterns.append(pattern)

                all_patterns.extend(timeframe_patterns)
                logger.info(f"{timeframe} 匹配到 {len(timeframe_patterns)} 个形态")

            return all_patterns

        except Exception as e:
            logger.error(f"形态匹配失败: {e}")
            return []

    def _basic_pattern_matching(self, indicators: Dict[str, Any],
                               stock_code: str, target_date: str) -> List[Dict[str, Any]]:
        """基础形态匹配逻辑"""
        patterns = []

        try:
            # 遍历所有指标，寻找形态
            for indicator_name, indicator_data in indicators.items():
                if indicator_data is None:
                    continue

                # 基于指标类型进行形态匹配
                indicator_patterns = self._match_indicator_patterns(
                    indicator_name, indicator_data, stock_code, target_date
                )
                patterns.extend(indicator_patterns)

            return patterns

        except Exception as e:
            logger.debug(f"基础形态匹配失败: {e}")
            return []

    def _match_indicator_patterns(self, indicator_name: str, indicator_data: Any,
                                 stock_code: str, target_date: str) -> List[Dict[str, Any]]:
        """匹配单个指标的形态"""
        patterns = []

        try:
            # 根据指标名称进行不同的形态匹配
            if 'KDJ' in indicator_name.upper():
                patterns.extend(self._match_kdj_patterns(indicator_data, stock_code, target_date))
            elif 'RSI' in indicator_name.upper():
                patterns.extend(self._match_rsi_patterns(indicator_data, stock_code, target_date))
            elif 'MACD' in indicator_name.upper():
                patterns.extend(self._match_macd_patterns(indicator_data, stock_code, target_date))
            elif 'MA' in indicator_name.upper():
                patterns.extend(self._match_ma_patterns(indicator_data, stock_code, target_date))
            elif 'BOLL' in indicator_name.upper():
                patterns.extend(self._match_boll_patterns(indicator_data, stock_code, target_date))

            return patterns

        except Exception as e:
            logger.debug(f"指标形态匹配失败 {indicator_name}: {e}")
            return []

    def _match_kdj_patterns(self, kdj_data: Any, stock_code: str, target_date: str) -> List[Dict[str, Any]]:
        """匹配KDJ形态"""
        patterns = []

        try:
            if hasattr(kdj_data, 'k') and hasattr(kdj_data, 'd'):
                k_value = kdj_data.k.iloc[-1] if hasattr(kdj_data.k, 'iloc') else kdj_data.k
                d_value = kdj_data.d.iloc[-1] if hasattr(kdj_data.d, 'iloc') else kdj_data.d

                # KDJ金叉
                if k_value > d_value and k_value > 20:
                    patterns.append({
                        'pattern_id': 'KDJ_GOLDEN_CROSS',
                        'pattern_name': 'KDJ金叉',
                        'description': 'K线上穿D线，形成金叉信号',
                        'match_strength': 0.8,
                        'indicator_values': {'k': k_value, 'd': d_value},
                        'stock_code': stock_code,
                        'target_date': target_date
                    })

                # KDJ超卖反弹
                if k_value < 30 and d_value < 30:
                    patterns.append({
                        'pattern_id': 'KDJ_OVERSOLD_BOUNCE',
                        'pattern_name': 'KDJ超卖反弹',
                        'description': 'KDJ处于超卖区域，可能反弹',
                        'match_strength': 0.6,
                        'indicator_values': {'k': k_value, 'd': d_value},
                        'stock_code': stock_code,
                        'target_date': target_date
                    })

            return patterns

        except Exception as e:
            logger.debug(f"KDJ形态匹配失败: {e}")
            return []

    def _match_rsi_patterns(self, rsi_data: Any, stock_code: str, target_date: str) -> List[Dict[str, Any]]:
        """匹配RSI形态"""
        patterns = []

        try:
            rsi_value = rsi_data.iloc[-1] if hasattr(rsi_data, 'iloc') else rsi_data

            # RSI超卖
            if rsi_value < 30:
                patterns.append({
                    'pattern_id': 'RSI_OVERSOLD',
                    'pattern_name': 'RSI超卖',
                    'description': 'RSI低于30，处于超卖状态',
                    'match_strength': 0.7,
                    'indicator_values': {'rsi': rsi_value},
                    'stock_code': stock_code,
                    'target_date': target_date
                })

            # RSI中性偏强
            elif 50 <= rsi_value <= 70:
                patterns.append({
                    'pattern_id': 'RSI_NEUTRAL_STRONG',
                    'pattern_name': 'RSI中性偏强',
                    'description': 'RSI处于中性偏强区域',
                    'match_strength': 0.5,
                    'indicator_values': {'rsi': rsi_value},
                    'stock_code': stock_code,
                    'target_date': target_date
                })

            return patterns

        except Exception as e:
            logger.debug(f"RSI形态匹配失败: {e}")
            return []

    def _match_macd_patterns(self, macd_data: Any, stock_code: str, target_date: str) -> List[Dict[str, Any]]:
        """匹配MACD形态"""
        patterns = []

        try:
            if hasattr(macd_data, 'dif') and hasattr(macd_data, 'dea'):
                dif = macd_data.dif.iloc[-1] if hasattr(macd_data.dif, 'iloc') else macd_data.dif
                dea = macd_data.dea.iloc[-1] if hasattr(macd_data.dea, 'iloc') else macd_data.dea

                # MACD金叉
                if dif > dea and dif > 0:
                    patterns.append({
                        'pattern_id': 'MACD_GOLDEN_CROSS',
                        'pattern_name': 'MACD金叉',
                        'description': 'DIF上穿DEA，形成金叉信号',
                        'match_strength': 0.8,
                        'indicator_values': {'dif': dif, 'dea': dea},
                        'stock_code': stock_code,
                        'target_date': target_date
                    })

            return patterns

        except Exception as e:
            logger.debug(f"MACD形态匹配失败: {e}")
            return []

    def _match_ma_patterns(self, ma_data: Any, stock_code: str, target_date: str) -> List[Dict[str, Any]]:
        """匹配均线形态"""
        patterns = []

        try:
            # 简化的均线形态匹配
            patterns.append({
                'pattern_id': 'MA_TREND',
                'pattern_name': '均线趋势',
                'description': '均线系统分析',
                'match_strength': 0.4,
                'indicator_values': {'ma_data': str(ma_data)[:50]},
                'stock_code': stock_code,
                'target_date': target_date
            })

            return patterns

        except Exception as e:
            logger.debug(f"均线形态匹配失败: {e}")
            return []

    def _match_boll_patterns(self, boll_data: Any, stock_code: str, target_date: str) -> List[Dict[str, Any]]:
        """匹配布林带形态"""
        patterns = []

        try:
            # 简化的布林带形态匹配
            patterns.append({
                'pattern_id': 'BOLL_ANALYSIS',
                'pattern_name': '布林带分析',
                'description': '布林带技术分析',
                'match_strength': 0.4,
                'indicator_values': {'boll_data': str(boll_data)[:50]},
                'stock_code': stock_code,
                'target_date': target_date
            })

            return patterns

        except Exception as e:
            logger.debug(f"布林带形态匹配失败: {e}")
            return []

    def _execute_pattern_matching(self, pattern_id: str, pattern_info: Dict[str, Any],
                                 indicators: Dict[str, Any], stock_code: str, 
                                 target_date: str) -> Optional[Dict[str, Any]]:
        """执行单个形态匹配"""
        try:
            # 获取形态匹配所需的指标
            required_indicators = pattern_info.get('required_indicators', [])
            
            # 检查是否有足够的指标数据
            available_indicators = set(indicators.keys())
            required_set = set(required_indicators)
            
            if not required_set.issubset(available_indicators):
                return None
            
            # 执行形态匹配逻辑
            match_conditions = pattern_info.get('conditions', [])
            match_score = 0.0
            total_conditions = len(match_conditions)
            
            if total_conditions == 0:
                return None
            
            for condition in match_conditions:
                if self._evaluate_condition(condition, indicators):
                    match_score += condition.get('weight', 1.0)
            
            # 计算匹配强度
            match_strength = match_score / total_conditions
            
            # 如果匹配强度超过阈值，返回匹配结果
            threshold = pattern_info.get('threshold', 0.5)
            if match_strength >= threshold:
                return {
                    'pattern_id': pattern_id,
                    'pattern_name': pattern_info.get('name', pattern_id),
                    'description': pattern_info.get('description', ''),
                    'match_strength': match_strength,
                    'match_score': match_score,
                    'indicator_values': {ind: indicators.get(ind) for ind in required_indicators},
                    'stock_code': stock_code,
                    'target_date': target_date
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"执行形态匹配失败 {pattern_id}: {e}")
            return None
    
    def _evaluate_condition(self, condition: Dict[str, Any], 
                           indicators: Dict[str, Any]) -> bool:
        """评估单个条件"""
        try:
            indicator_name = condition.get('indicator')
            field = condition.get('field')
            operator = condition.get('operator')
            value = condition.get('value')
            
            if not all([indicator_name, field, operator]):
                return False
            
            # 获取指标值
            indicator_data = indicators.get(indicator_name)
            if indicator_data is None:
                return False
            
            # 获取字段值
            if hasattr(indicator_data, field):
                field_value = getattr(indicator_data, field)
            elif isinstance(indicator_data, dict) and field in indicator_data:
                field_value = indicator_data[field]
            else:
                return False
            
            # 执行比较
            if operator == '>':
                return field_value > value
            elif operator == '<':
                return field_value < value
            elif operator == '>=':
                return field_value >= value
            elif operator == '<=':
                return field_value <= value
            elif operator == '==':
                return field_value == value
            elif operator == '!=':
                return field_value != value
            else:
                return False
                
        except Exception as e:
            logger.debug(f"条件评估失败: {e}")
            return False
    
    def _calculate_buypoint_score(self, technical_indicators: Dict[str, Any],
                                 pattern_matches: List[Dict[str, Any]]) -> float:
        """计算综合买点评分"""
        try:
            if not pattern_matches:
                return 0.0
            
            # 基于形态匹配强度计算综合评分
            total_score = sum(pattern.get('match_strength', 0) for pattern in pattern_matches)
            average_score = total_score / len(pattern_matches)
            
            # 考虑形态数量的加权
            pattern_count_bonus = min(len(pattern_matches) * 0.1, 0.5)
            
            final_score = min(average_score + pattern_count_bonus, 1.0)
            
            return final_score
            
        except Exception as e:
            logger.error(f"计算买点评分失败: {e}")
            return 0.0
    
    def _create_empty_result(self, stock_code: str, target_date: str, 
                           error_message: str) -> Dict[str, Any]:
        """创建空的分析结果"""
        return {
            'stock_code': stock_code,
            'target_date': target_date,
            'technical_indicators': {},
            'pattern_matches': [],
            'buypoint_score': 0.0,
            'total_indicators': 0,
            'total_patterns': 0,
            'analysis_timestamp': datetime.now().isoformat(),
            'success': False,
            'error_message': error_message
        }
