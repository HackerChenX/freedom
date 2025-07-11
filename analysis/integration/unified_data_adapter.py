from db.query_executor import get_query_executor
from db.sql_manager import QueryType
#!/usr/bin/env python3
"""
统一数据适配器

实现买点分析和策略选股系统之间的数据格式统一，
提供双向数据转换和格式标准化功能。
"""

import sys
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import pandas as pd
import threading

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from utils.logger import getLogger
from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import DataAccessInterface
from utils.dependency_injection import get_service

logger = getLogger(__name__)


class UnifiedDataAdapter:
    """统一数据适配器"""
    
    def __init__(self):
        """初始化适配器"""
        self.data_manager = get_service(DataAccessInterface)
        
        # 定义标准数据格式规范
        self.standard_format = {
            'required_fields': [
                'stock_code',      # 股票代码
                'stock_name',      # 股票名称  
                'industry',        # 行业信息
                'price',           # 当前价格
                'change_pct',      # 涨跌幅
                'score',           # 综合评分
                'selection_date'   # 选股/分析日期
            ],
            'optional_fields': [
                'match_details',   # 匹配详情
                'recommendation',  # 推荐等级
                'market_cap',      # 市值
                'pe_ratio',        # 市盈率
                'pb_ratio',        # 市净率
                'volume',          # 成交量
                'turnover_rate',   # 换手率
                'source_system'    # 数据来源系统
            ],
            'data_types': {
                'stock_code': str,
                'stock_name': str,
                'industry': str,
                'price': float,
                'change_pct': float,
                'score': float,
                'selection_date': str,
                'match_details': dict,
                'recommendation': str,
                'market_cap': float,
                'pe_ratio': float,
                'pb_ratio': float,
                'volume': int,
                'turnover_rate': float,
                'source_system': str
            }
        }
        
        # 指标映射表：买点分析指标 -> 标准指标
        self.indicator_mapping = {
            'macd_gold': 'MACD_BULLISH',
            'macd_death': 'MACD_BEARISH',
            'dif_up': 'MACD_DIF_UP',
            'dea_up': 'MACD_DEA_UP',
            'k_up': 'KDJ_K_UP',
            'd_up': 'KDJ_D_UP',
            'j_up': 'KDJ_J_UP',
            'kdj_gold': 'KDJ_BULLISH',
            'kdj_death': 'KDJ_BEARISH',
            'touch_ma': 'MA_TOUCH',
            'touch_ma5': 'MA5_TOUCH',
            'touch_ma10': 'MA10_TOUCH',
            'touch_ma20': 'MA20_TOUCH',
            'touch_ma30': 'MA30_TOUCH',
            'touch_ma60': 'MA60_TOUCH',
            'ma_up': 'MA_TREND_UP',
            'ma_down': 'MA_TREND_DOWN',
            'vol_shrink': 'VOLUME_SHRINK',
            'vol_expand': 'VOLUME_EXPAND',
            'vol_close_high': 'VOLUME_HIGH',
            'money_in': 'MONEY_FLOW_IN',
            'money_out': 'MONEY_FLOW_OUT',
            'abs_signal1': 'ABSORPTION_SIGNAL_1',
            'abs_signal2': 'ABSORPTION_SIGNAL_2',
            'xc': 'ABSORPTION_MAIN',
            'price_stable': 'PRICE_STABILITY',
            'kpattern': 'CANDLESTICK_PATTERN',
            'xsmall': 'SMALL_BODY_CANDLE',
            'xstar': 'STAR_PATTERN',
            'xshadow': 'SHADOW_PATTERN',
            'rsi_oversold': 'RSI_OVERSOLD',
            'rsi_overbought': 'RSI_OVERBOUGHT',
            'boll_break_up': 'BOLL_BREAKOUT_UP',
            'boll_break_down': 'BOLL_BREAKOUT_DOWN'
        }
        
        # 评分权重配置
        self.scoring_weights = {
            'technical_indicators': 0.40,   # 技术指标 40%
            'trend_analysis': 0.25,         # 趋势分析 25%
            'volume_analysis': 0.15,        # 成交量分析 15%
            'pattern_recognition': 0.10,    # 形态识别 10%
            'market_sentiment': 0.10        # 市场情绪 10%
        }
        
        logger.info("统一数据适配器初始化完成")
    
    def convert_buypoint_to_standard(self, buypoint_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        将买点分析结果转换为标准格式
        
        Args:
            buypoint_result: 买点分析结果
            
        Returns:
            Optional[Dict[str, Any]]: 标准格式数据，转换失败返回None
        """
        try:
            if not buypoint_result or 'stock_code' not in buypoint_result:
                logger.warning("买点分析结果格式无效")
                return None
            
            stock_code = buypoint_result['stock_code']
            analysis_date = buypoint_result.get('buypoint_date', datetime.now().strftime('%Y%m%d'))
            
            # 获取股票基本信息
            stock_info = self._get_stock_basic_info_Unified_Data_Adapter(stock_code, analysis_date)
            if not stock_info:
                logger.warning(f"无法获取股票 {stock_code} 的基本信息")
                return None
            
            # 转换指标结果
            match_details = self._convert_indicator_results(buypoint_result.get('indicator_results', {}))
            
            # 计算综合评分
            score = self._calculate_unified_score(buypoint_result, match_details)
            
            # 生成推荐等级
            recommendation = self._generate_recommendation_Unified_Data_Adapter(score)
            
            # 构建标准格式数据
            standard_data = {
                # 必需字段
                'stock_code': stock_code,
                'stock_name': stock_info['name'],
                'industry': stock_info['industry'],
                'price': stock_info['price'],
                'change_pct': stock_info['change_pct'],
                'score': score,
                'selection_date': analysis_date,
                
                # 可选字段
                'match_details': match_details,
                'recommendation': recommendation,
                'market_cap': stock_info.get('market_cap'),
                'pe_ratio': stock_info.get('pe_ratio'),
                'pb_ratio': stock_info.get('pb_ratio'),
                'volume': stock_info.get('volume'),
                'turnover_rate': stock_info.get('turnover_rate'),
                'source_system': 'buypoint_analysis'
            }
            
            # 验证数据格式
            if self._validate_standard_format(standard_data):
                logger.info(f"成功转换买点分析结果: {stock_code}")
                return standard_data
            else:
                logger.error(f"转换后的数据格式验证失败: {stock_code}")
                return None
                
        except Exception as e:
            logger.error(f"转换买点分析结果时出错: {e}")
            return None
    
    def convert_strategy_to_standard(self, strategy_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        将策略选股结果转换为标准格式
        
        Args:
            strategy_result: 策略选股结果
            
        Returns:
            Optional[Dict[str, Any]]: 标准格式数据，转换失败返回None
        """
        try:
            if not strategy_result or 'stock_code' not in strategy_result:
                logger.warning("策略选股结果格式无效")
                return None
            
            # 策略选股结果通常已经接近标准格式，主要是补全缺失字段
            standard_data = strategy_result.copy()
            
            # 确保必需字段存在
            required_fields = self.standard_format['required_fields']
            for field in required_fields:
                if field not in standard_data:
                    # 尝试从其他字段推导或设置默认值
                    if field == 'selection_date' and 'date' in standard_data:
                        standard_data['selection_date'] = standard_data['date']
                    elif field == 'score' and 'final_score' in standard_data:
                        standard_data['score'] = standard_data['final_score']
                    else:
                        logger.warning(f"策略选股结果缺少必需字段: {field}")
                        return None
            
            # 添加来源标识
            standard_data['source_system'] = 'strategy_selection'
            
            # 验证数据格式
            if self._validate_standard_format(standard_data):
                logger.info(f"成功转换策略选股结果: {standard_data['stock_code']}")
                return standard_data
            else:
                logger.error(f"转换后的数据格式验证失败: {standard_data['stock_code']}")
                return None
                
        except Exception as e:
            logger.error(f"转换策略选股结果时出错: {e}")
            return None
    
    def batch_convert_to_standard(self, results: List[Dict[str, Any]], source_type: str) -> List[Dict[str, Any]]:
        """
        批量转换为标准格式
        
        Args:
            results: 结果列表
            source_type: 来源类型 ('buypoint' 或 'strategy')
            
        Returns:
            List[Dict[str, Any]]: 标准格式数据列表
        """
        standard_results = []
        
        for result in results:
            try:
                if source_type == 'buypoint':
                    standard_result = self.convert_buypoint_to_standard(result)
                elif source_type == 'strategy':
                    standard_result = self.convert_strategy_to_standard(result)
                else:
                    logger.error(f"不支持的来源类型: {source_type}")
                    continue
                
                if standard_result:
                    standard_results.append(standard_result)
                    
            except Exception as e:
                logger.error(f"批量转换时出错: {e}")
                continue
        
        logger.info(f"批量转换完成: {len(standard_results)}/{len(results)}")
        return standard_results
    
    def merge_analysis_results(self, 
                             buypoint_results: List[Dict[str, Any]], 
                             strategy_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        合并买点分析和策略选股结果
        
        Args:
            buypoint_results: 买点分析结果列表
            strategy_results: 策略选股结果列表
            
        Returns:
            List[Dict[str, Any]]: 合并后的标准格式结果列表
        """
        try:
            # 转换为标准格式
            standard_buypoint = self.batch_convert_to_standard(buypoint_results, 'buypoint')
            standard_strategy = self.batch_convert_to_standard(strategy_results, 'strategy')
            
            # 按股票代码合并
            merged_results = {}
            
            # 处理买点分析结果
            for result in standard_buypoint:
                stock_code = result['stock_code']
                merged_results[stock_code] = result
                merged_results[stock_code]['analysis_sources'] = ['buypoint']
            
            # 处理策略选股结果
            for result in standard_strategy:
                stock_code = result['stock_code']
                if stock_code in merged_results:
                    # 合并结果
                    merged_result = self._merge_single_stock_results(
                        merged_results[stock_code], result
                    )
                    merged_results[stock_code] = merged_result
                    merged_results[stock_code]['analysis_sources'].append('strategy')
                else:
                    merged_results[stock_code] = result
                    merged_results[stock_code]['analysis_sources'] = ['strategy']
            
            # 转换为列表并按评分排序
            final_results = list(merged_results.values())
            final_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            logger.info(f"合并分析结果完成: {len(final_results)} 只股票")
            return final_results
            
        except Exception as e:
            logger.error(f"合并分析结果时出错: {e}")
            return []
    
    def _get_stock_basic_info_Unified_Data_Adapter(self, stock_code: str, date: str) -> Optional[Dict[str, Any]]:
        """获取股票基本信息"""
        try:
            # 构建查询语句
            query = f"""
            SELECT 
                name,
                industry,
                close as price,
                change_pct,
                market_cap,
                pe_ratio,
                pb_ratio,
                volume,
                turnover_rate
            FROM stock_info WHERE 1=1
            WHERE code = '{stock_code}' 
            AND date <= '{date}'
            ORDER BY date DESC 
            LIMIT 1
            """
            
            result = self.data_manager.execute_query(query)
            
            if result is not None and not result.empty:
                row = result.iloc[0]
                return {
                    'name': row.get('name', f'股票{stock_code}'),
                    'industry': row.get('industry', '未知行业'),
                    'price': float(row.get('price', 0)),
                    'change_pct': float(row.get('change_pct', 0)),
                    'market_cap': float(row.get('market_cap', 0)) if row.get('market_cap') else None,
                    'pe_ratio': float(row.get('pe_ratio', 0)) if row.get('pe_ratio') else None,
                    'pb_ratio': float(row.get('pb_ratio', 0)) if row.get('pb_ratio') else None,
                    'volume': int(row.get('volume', 0)) if row.get('volume') else None,
                    'turnover_rate': float(row.get('turnover_rate', 0)) if row.get('turnover_rate') else None
                }
            else:
                # 返回默认信息
                return {
                    'name': f'股票{stock_code}',
                    'industry': '未知行业',
                    'price': 0.0,
                    'change_pct': 0.0,
                    'market_cap': None,
                    'pe_ratio': None,
                    'pb_ratio': None,
                    'volume': None,
                    'turnover_rate': None
                }
                
        except Exception as e:
            logger.error(f"获取股票基本信息失败 {stock_code}: {e}")
            return None
    
    def _convert_indicator_results(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """转换指标结果为标准格式"""
        match_details = {
            'passing_indicators': [],
            'failing_indicators': [],
            'indicator_scores': {},
            'pattern_matches': {},
            'technical_summary': {}
        }
        
        try:
            # 处理多周期指标结果
            for period, period_results in indicator_results.items():
                if not isinstance(period_results, dict):
                    continue
                
                for indicator_name, indicator_data in period_results.items():
                    if not isinstance(indicator_data, dict):
                        continue
                    
                    # 处理指标的各种形态
                    for pattern_name, pattern_result in indicator_data.items():
                        if isinstance(pattern_result, dict) and pattern_result.get('detected', False):
                            # 映射到标准指标名称
                            standard_name = self.indicator_mapping.get(pattern_name, pattern_name)
                            
                            # 记录通过的指标
                            if standard_name not in match_details['passing_indicators']:
                                match_details['passing_indicators'].append(standard_name)
                            
                            # 记录指标评分
                            confidence = pattern_result.get('confidence', 0.5)
                            match_details['indicator_scores'][standard_name] = confidence
                            
                            # 记录形态匹配详情
                            match_details['pattern_matches'][standard_name] = {
                                'period': period,
                                'indicator': indicator_name,
                                'pattern': pattern_name,
                                'confidence': confidence,
                                'description': pattern_result.get('description', '')
                            }
            
            # 生成技术摘要
            match_details['technical_summary'] = {
                'total_indicators': len(match_details['indicator_scores']),
                'passing_count': len(match_details['passing_indicators']),
                'average_confidence': sum(match_details['indicator_scores'].values()) / len(match_details['indicator_scores']) if match_details['indicator_scores'] else 0,
                'strongest_signals': sorted(match_details['indicator_scores'].items(), key=lambda x: x[1], reverse=True)[:5]
            }
            
        except Exception as e:
            logger.error(f"转换指标结果时出错: {e}")
        
        return match_details
    
    def _calculate_unified_score(self, buypoint_result: Dict[str, Any], match_details: Dict[str, Any]) -> float:
        """计算统一评分"""
        try:
            # 基础评分
            base_score = 0.0
            
            # 技术指标评分
            technical_score = 0.0
            if match_details['indicator_scores']:
                technical_score = sum(match_details['indicator_scores'].values()) / len(match_details['indicator_scores']) * 100
            
            # 趋势分析评分（基于原始评分）
            trend_score = buypoint_result.get('summary', {}).get('overall_score', 0)
            
            # 成交量分析评分
            volume_score = 50.0  # 默认中性评分
            if 'VOLUME_SHRINK' in match_details['passing_indicators']:
                volume_score += 20
            if 'VOLUME_EXPAND' in match_details['passing_indicators']:
                volume_score += 15
            
            # 形态识别评分
            pattern_score = 50.0  # 默认中性评分
            pattern_indicators = [name for name in match_details['passing_indicators'] if 'PATTERN' in name or 'CANDLE' in name]
            pattern_score += len(pattern_indicators) * 10
            
            # 市场情绪评分（基于通过指标数量）
            sentiment_score = min(len(match_details['passing_indicators']) * 5, 100)
            
            # 加权计算最终评分
            final_score = (
                technical_score * self.scoring_weights['technical_indicators'] +
                trend_score * self.scoring_weights['trend_analysis'] +
                volume_score * self.scoring_weights['volume_analysis'] +
                pattern_score * self.scoring_weights['pattern_recognition'] +
                sentiment_score * self.scoring_weights['market_sentiment']
            )
            
            # 确保评分在0-100范围内
            final_score = max(0, min(100, final_score))
            
            return round(final_score, 2)
            
        except Exception as e:
            logger.error(f"计算统一评分时出错: {e}")
            return 0.0
    
    def _generate_recommendation_Unified_Data_Adapter(self, score: float) -> str:
        """根据评分生成推荐等级"""
        if score >= 80:
            return "强烈买入"
        elif score >= 70:
            return "买入"
        elif score >= 60:
            return "谨慎买入"
        elif score >= 50:
            return "观望"
        elif score >= 40:
            return "谨慎观望"
        else:
            return "回避"
    
    def _validate_standard_format(self, data: Dict[str, Any]) -> bool:
        """验证数据是否符合标准格式"""
        try:
            # 检查必需字段
            required_fields = self.standard_format['required_fields']
            for field in required_fields:
                if field not in data:
                    logger.error(f"缺少必需字段: {field}")
                    return False
            
            # 检查数据类型
            data_types = self.standard_format['data_types']
            for field, expected_type in data_types.items():
                if field in data and data[field] is not None:
                    if not isinstance(data[field], expected_type):
                        # 尝试类型转换
                        try:
                            data[field] = expected_type(data[field])
                        except (ValueError, TypeError):
                            logger.error(f"字段 {field} 类型不匹配，期望 {expected_type}，实际 {type(data[field])}")
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"验证标准格式时出错: {e}")
            return False
    
    def _merge_single_stock_results(self, result1: Dict[str, Any], result2: Dict[str, Any]) -> Dict[str, Any]:
        """合并单只股票的分析结果"""
        try:
            merged = result1.copy()
            
            # 合并评分（取平均值）
            score1 = result1.get('score', 0)
            score2 = result2.get('score', 0)
            merged['score'] = round((score1 + score2) / 2, 2)
            
            # 合并匹配详情
            if 'match_details' in result2:
                if 'match_details' not in merged:
                    merged['match_details'] = {}
                
                details1 = merged['match_details']
                details2 = result2['match_details']
                
                # 合并通过的指标
                passing1 = set(details1.get('passing_indicators', []))
                passing2 = set(details2.get('passing_indicators', []))
                merged['match_details']['passing_indicators'] = list(passing1.union(passing2))
                
                # 合并指标评分
                scores1 = details1.get('indicator_scores', {})
                scores2 = details2.get('indicator_scores', {})
                merged_scores = scores1.copy()
                for indicator, score in scores2.items():
                    if indicator in merged_scores:
                        merged_scores[indicator] = (merged_scores[indicator] + score) / 2
                    else:
                        merged_scores[indicator] = score
                merged['match_details']['indicator_scores'] = merged_scores
            
            # 更新推荐等级
            merged['recommendation'] = self._generate_recommendation_Unified_Data_Adapter(merged['score'])
            
            return merged
            
        except Exception as e:
            logger.error(f"合并单只股票结果时出错: {e}")
            return result1


# ===== 依赖注入和兼容性接口 =====

def create_unified_data_adapter() -> UnifiedDataAdapter:
    """创建统一数据适配器实例（兼容性方法）"""
    return UnifiedDataAdapter()


def get_unified_data_adapter() -> UnifiedDataAdapter:
    """
    获取统一数据适配器实例（依赖注入方式）
    
    Returns:
        UnifiedDataAdapter: 统一数据适配器实例
    """
    try:
        from utils.dependency_injection import get_container
        container = get_container()
        return container.resolve(UnifiedDataAdapter)
    except Exception as e:
        logger.warning(f"从依赖注入容器获取UnifiedDataAdapter失败，创建新实例: {e}")
        return UnifiedDataAdapter()


def get_legacy_unified_data_adapter() -> UnifiedDataAdapter:
    """获取统一数据适配器实例（向后兼容）"""
    return get_unified_data_adapter()


# 注册到依赖注入容器
try:
    from utils.dependency_injection import get_container
    container = get_container()
    if not container.is_registered(UnifiedDataAdapter):
        container.register_singleton(UnifiedDataAdapter, UnifiedDataAdapter)
        logger.info("UnifiedDataAdapter已注册到依赖注入容器")
except Exception as e:
    logger.warning(f"注册UnifiedDataAdapter到依赖注入容器失败: {e}")
