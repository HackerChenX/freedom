#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
买点分析到选股策略的数据适配器

将买点分析系统的输出格式转换为选股策略系统期望的输入格式，
确保两个系统的无缝集成。
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

from db.unified_data_manager import get_unified_data_manager
from utils.logger import get_logger

logger = get_logger(__name__)


class BuyPointToStrategyAdapter:
    """买点分析结果到选股策略格式的适配器"""
    
    def __init__(self):
        """初始化适配器"""
        self.data_manager = get_unified_data_manager()
        
        # 指标映射表：买点分析指标 -> 选股策略指标
        self.indicator_mapping = {
            'macd_gold': 'MACD',
            'dif_up': 'MACD',
            'dea_up': 'MACD',
            'k_up': 'KDJ',
            'd_up': 'KDJ', 
            'j_up': 'KDJ',
            'touch_ma': 'MA',
            'touch_ma10': 'MA',
            'touch_ma20': 'MA',
            'touch_ma30': 'MA',
            'touch_ma60': 'MA',
            'ma_up': 'MA',
            'vol_shrink': 'VOL',
            'vol_close_high': 'VOL',
            'money_in': 'WVAD',
            'abs_signal1': 'CUSTOM_ABSORPTION',
            'abs_signal2': 'CUSTOM_ABSORPTION',
            'xc': 'CUSTOM_ABSORPTION',
            'price_stable': 'CUSTOM_STABILITY',
            'kpattern': 'CUSTOM_PATTERN',
            'xsmall': 'CUSTOM_PATTERN',
            'xstar': 'CUSTOM_PATTERN',
            'xshadow': 'CUSTOM_PATTERN'
        }
        
        # 选股策略指标权重配置
        self.strategy_weights = {
            "MACD": 1.0,
            "KDJ": 0.9,
            "MA": 0.7,
            "VOL": 0.7,
            "WVAD": 0.5,
            "CUSTOM_ABSORPTION": 0.8,  # 吸筹信号权重较高
            "CUSTOM_STABILITY": 0.6,   # 价格稳定性
            "CUSTOM_PATTERN": 0.5      # K线形态
        }
        
        # 指标评分配置
        self.indicator_scores = {
            "MACD": 75.0,
            "KDJ": 70.0,
            "MA": 60.0,
            "VOL": 65.0,
            "WVAD": 55.0,
            "CUSTOM_ABSORPTION": 80.0,  # 吸筹信号评分较高
            "CUSTOM_STABILITY": 65.0,
            "CUSTOM_PATTERN": 60.0
        }
        
        logger.info("买点分析到选股策略适配器初始化完成")
    
    def convert_buypoint_result(self, buypoint_result: Dict) -> Optional[Dict]:
        """
        将买点分析结果转换为选股策略格式
        
        Args:
            buypoint_result: 买点分析结果
            
        Returns:
            Optional[Dict]: 选股策略格式的数据，转换失败返回None
        """
        try:
            if not buypoint_result or 'stock_code' not in buypoint_result:
                logger.warning("买点分析结果格式无效")
                return None
            
            stock_code = buypoint_result['stock_code']
            buypoint_date = buypoint_result.get('buypoint_date', datetime.now().strftime('%Y%m%d'))
            
            # 获取股票基本信息
            stock_info = self._get_stock_basic_info(stock_code, buypoint_date)
            if not stock_info:
                logger.warning(f"无法获取股票 {stock_code} 的基本信息")
                return None
            
            # 转换评分
            score = self._convert_score(buypoint_result)
            
            # 提取技术指标匹配详情
            match_details = self._extract_match_details(buypoint_result)
            
            # 生成推荐等级
            recommendation = self._generate_recommendation(score, match_details)
            
            # 构建选股策略兼容格式
            strategy_result = {
                'stock_code': stock_code,
                'stock_name': stock_info['name'],
                'industry': stock_info['industry'],
                'price': stock_info['price'],
                'change_pct': stock_info['change_pct'],
                'score': score,
                'recommendation': recommendation,
                'match_details': match_details,
                'selection_date': buypoint_date,
                'source': 'buypoint_analysis',
                'original_result': buypoint_result  # 保留原始结果用于调试
            }
            
            logger.debug(f"成功转换股票 {stock_code} 的买点分析结果")
            return strategy_result
            
        except Exception as e:
            logger.error(f"转换买点分析结果时出错: {e}")
            return None
    
    def convert_batch_results(self, buypoint_results: List[Dict]) -> pd.DataFrame:
        """
        批量转换买点分析结果
        
        Args:
            buypoint_results: 买点分析结果列表
            
        Returns:
            pd.DataFrame: 选股策略格式的结果DataFrame
        """
        converted_results = []
        
        for result in buypoint_results:
            converted = self.convert_buypoint_result(result)
            if converted:
                converted_results.append(converted)
        
        if not converted_results:
            logger.warning("没有成功转换的买点分析结果")
            return pd.DataFrame()
        
        # 转换为DataFrame
        df = pd.DataFrame(converted_results)
        
        # 按评分排序
        if 'score' in df.columns:
            df = df.sort_values(by='score', ascending=False)
        
        logger.info(f"成功转换 {len(df)} 个买点分析结果")
        return df
    
    def _get_stock_basic_info(self, stock_code: str, date: str) -> Optional[Dict]:
        """
        获取股票基本信息
        
        Args:
            stock_code: 股票代码
            date: 日期
            
        Returns:
            Optional[Dict]: 股票基本信息
        """
        try:
            # 获取股票数据
            stock_data = self.data_manager.get_stock_data(
                stock_code=stock_code,
                period='daily',
                limit=1
            )
            
            if stock_data is None or stock_data.empty:
                return None
            
            latest_data = stock_data.iloc[-1]
            
            # 获取股票名称和行业
            stock_name = self.data_manager.get_stock_name(stock_code) or stock_code
            industry = self.data_manager.get_stock_industry(stock_code) or "未知行业"
            
            return {
                'name': stock_name,
                'industry': industry,
                'price': float(latest_data.get('close', 0)),
                'change_pct': float(latest_data.get('pct_chg', 0))
            }
            
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 基本信息时出错: {e}")
            return None
    
    def _convert_score(self, buypoint_result: Dict) -> float:
        """
        转换评分到选股策略格式
        
        Args:
            buypoint_result: 买点分析结果
            
        Returns:
            float: 统一评分 (0-100)
        """
        try:
            # 尝试从summary中获取overall_score
            summary = buypoint_result.get('summary', {})
            if 'overall_score' in summary:
                base_score = float(summary['overall_score'])
            else:
                # 如果没有overall_score，基于指标结果计算
                base_score = self._calculate_score_from_indicators(buypoint_result)
            
            # 应用买点分析特有的加权调整
            adjusted_score = self._apply_buypoint_adjustments(base_score, buypoint_result)
            
            # 确保分数在合理范围内
            final_score = max(0, min(100, adjusted_score))
            
            return round(final_score, 1)
            
        except Exception as e:
            logger.error(f"转换评分时出错: {e}")
            return 50.0  # 返回默认评分
    
    def _calculate_score_from_indicators(self, buypoint_result: Dict) -> float:
        """
        基于指标结果计算评分
        
        Args:
            buypoint_result: 买点分析结果
            
        Returns:
            float: 计算得出的评分
        """
        try:
            # 从indicator_results中提取指标信息
            indicator_results = buypoint_result.get('indicator_results', {})
            
            total_score = 0.0
            total_weight = 0.0
            
            # 遍历技术分析结果
            technical_analysis = indicator_results.get('technical_analysis', {})
            for indicator_name, indicator_data in technical_analysis.items():
                if isinstance(indicator_data, dict):
                    # 获取指标强度或值
                    strength = indicator_data.get('strength', 0.5)
                    signal = indicator_data.get('signal', 'neutral')
                    
                    # 映射到选股策略指标
                    strategy_indicator = self.indicator_mapping.get(indicator_name, 'UNKNOWN')
                    weight = self.strategy_weights.get(strategy_indicator, 0.5)
                    score = self.indicator_scores.get(strategy_indicator, 60.0)
                    
                    # 根据信号调整评分
                    if signal in ['buy', 'positive', 'bullish']:
                        adjusted_score = score * (0.8 + 0.4 * strength)
                    elif signal in ['sell', 'negative', 'bearish']:
                        adjusted_score = score * (0.2 + 0.3 * strength)
                    else:
                        adjusted_score = score * (0.5 + 0.3 * strength)
                    
                    total_score += adjusted_score * weight
                    total_weight += weight
            
            # 计算加权平均分
            if total_weight > 0:
                return total_score / total_weight
            else:
                return 50.0  # 默认评分
                
        except Exception as e:
            logger.error(f"基于指标计算评分时出错: {e}")
            return 50.0
    
    def _apply_buypoint_adjustments(self, base_score: float, buypoint_result: Dict) -> float:
        """
        应用买点分析特有的评分调整
        
        Args:
            base_score: 基础评分
            buypoint_result: 买点分析结果
            
        Returns:
            float: 调整后的评分
        """
        try:
            adjusted_score = base_score
            
            # 获取形态分析结果
            pattern_results = buypoint_result.get('pattern_results', {})
            
            # 买点特有的加分项
            buypoint_bonuses = 0.0
            
            # 吸筹信号加分
            if any(pattern.get('detected', False) for pattern_name, pattern in pattern_results.items() 
                   if 'absorption' in pattern_name.lower() or 'xc' in pattern_name.lower()):
                buypoint_bonuses += 5.0
                logger.debug("检测到吸筹信号，加分5.0")
            
            # 价格稳定加分
            if any(pattern.get('detected', False) for pattern_name, pattern in pattern_results.items()
                   if 'stable' in pattern_name.lower() or 'price_stable' in pattern_name.lower()):
                buypoint_bonuses += 3.0
                logger.debug("检测到价格稳定信号，加分3.0")
            
            # 均线支撑加分
            if any(pattern.get('detected', False) for pattern_name, pattern in pattern_results.items()
                   if 'ma' in pattern_name.lower() and 'touch' in pattern_name.lower()):
                buypoint_bonuses += 4.0
                logger.debug("检测到均线支撑信号，加分4.0")
            
            # 成交量配合加分
            if any(pattern.get('detected', False) for pattern_name, pattern in pattern_results.items()
                   if 'vol' in pattern_name.lower() and 'shrink' in pattern_name.lower()):
                buypoint_bonuses += 2.0
                logger.debug("检测到成交量配合信号，加分2.0")
            
            # 应用加分，但限制最大加分幅度
            max_bonus = 15.0  # 最大加分15分
            final_bonus = min(buypoint_bonuses, max_bonus)
            adjusted_score += final_bonus
            
            return adjusted_score
            
        except Exception as e:
            logger.error(f"应用买点调整时出错: {e}")
            return base_score
    
    def _extract_match_details(self, buypoint_result: Dict) -> Dict:
        """
        提取技术指标匹配详情
        
        Args:
            buypoint_result: 买点分析结果
            
        Returns:
            Dict: 匹配详情
        """
        try:
            match_details = {
                "passing_indicators": [],
                "failing_indicators": [],
                "condition_results": {},
                "buypoint_specific": {}
            }
            
            # 从pattern_results中提取通过和失败的指标
            pattern_results = buypoint_result.get('pattern_results', {})
            
            for pattern_name, pattern_data in pattern_results.items():
                if isinstance(pattern_data, dict):
                    detected = pattern_data.get('detected', False)
                    confidence = pattern_data.get('confidence', 0.0)
                    
                    # 映射到选股策略指标
                    strategy_indicator = self.indicator_mapping.get(pattern_name, pattern_name)
                    
                    if detected:
                        match_details["passing_indicators"].append(strategy_indicator)
                    else:
                        match_details["failing_indicators"].append(strategy_indicator)
                    
                    # 记录详细结果
                    match_details["condition_results"][pattern_name] = {
                        "type": "pattern",
                        "indicator_id": strategy_indicator,
                        "result": detected,
                        "confidence": confidence
                    }
            
            # 添加买点特有信息
            summary = buypoint_result.get('summary', {})
            match_details["buypoint_specific"] = {
                "total_indicators": summary.get('total_indicators', 0),
                "positive_signals": summary.get('positive_signals', 0),
                "negative_signals": summary.get('negative_signals', 0),
                "buypoint_date": buypoint_result.get('buypoint_date', ''),
                "analysis_periods": list(buypoint_result.get('indicator_results', {}).get('period_data', {}).keys())
            }
            
            return match_details
            
        except Exception as e:
            logger.error(f"提取匹配详情时出错: {e}")
            return {
                "passing_indicators": [],
                "failing_indicators": [],
                "condition_results": {},
                "buypoint_specific": {}
            }
    
    def _generate_recommendation(self, score: float, match_details: Dict) -> str:
        """
        生成推荐等级
        
        Args:
            score: 评分
            match_details: 匹配详情
            
        Returns:
            str: 推荐等级 (buy/hold/sell)
        """
        try:
            passing_count = len(match_details.get("passing_indicators", []))
            total_count = passing_count + len(match_details.get("failing_indicators", []))
            
            # 基于评分的推荐
            if score >= 75:
                base_recommendation = "buy"
            elif score >= 60:
                base_recommendation = "hold"
            else:
                base_recommendation = "sell"
            
            # 基于通过指标比例的调整
            if total_count > 0:
                pass_ratio = passing_count / total_count
                if pass_ratio >= 0.7 and score >= 65:
                    return "buy"
                elif pass_ratio >= 0.5 and score >= 55:
                    return "hold"
                else:
                    return "sell"
            
            return base_recommendation
            
        except Exception as e:
            logger.error(f"生成推荐等级时出错: {e}")
            return "hold"  # 默认推荐


def get_buypoint_strategy_adapter() -> BuyPointToStrategyAdapter:
    """获取买点分析到选股策略适配器实例"""
    return BuyPointToStrategyAdapter()


# 使用示例
if __name__ == "__main__":
    # 创建适配器
    adapter = get_buypoint_strategy_adapter()
    
    # 模拟买点分析结果
    sample_buypoint_result = {
        'stock_code': '000001',
        'buypoint_date': '20240601',
        'summary': {
            'total_indicators': 10,
            'positive_signals': 7,
            'negative_signals': 3,
            'overall_score': 72.5
        },
        'pattern_results': {
            'macd_gold': {'detected': True, 'confidence': 0.8},
            'touch_ma': {'detected': True, 'confidence': 0.7},
            'price_stable': {'detected': True, 'confidence': 0.6},
            'xc': {'detected': True, 'confidence': 0.9}
        }
    }
    
    # 转换结果
    strategy_result = adapter.convert_buypoint_result(sample_buypoint_result)
    
    if strategy_result:
        print("转换成功:")
        print(f"股票: {strategy_result['stock_name']} ({strategy_result['stock_code']})")
        print(f"评分: {strategy_result['score']}")
        print(f"推荐: {strategy_result['recommendation']}")
        print(f"通过指标: {strategy_result['match_details']['passing_indicators']}")
    else:
        print("转换失败")
