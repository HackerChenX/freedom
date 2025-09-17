"""
停牌处理器

专门处理股票停牌、复牌相关的数据质量问题。
提供停牌检测、停牌期间数据过滤、复牌影响分析等功能。

Author: System
Date: 2025-01-15
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from utils.logger import get_logger
from utils.decorators import exception_handler, performance_monitor
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


class HaltType(Enum):
    """停牌类型"""
    TEMPORARY = "临时停牌"
    REORGANIZATION = "重组停牌"
    ST_SUSPENSION = "ST停牌"
    DELISTING = "退市停牌"
    UNKNOWN = "未知停牌"


class HaltStatus(Enum):
    """停牌状态"""
    NORMAL = "正常交易"
    HALTED = "停牌中"
    RESUMED = "刚复牌"
    SUSPECTED = "疑似停牌"


@dataclass
class HaltPeriod:
    """停牌期间"""
    start_date: str
    end_date: str
    halt_type: HaltType
    duration_days: int
    confidence: float
    affected_data_points: int
    volume_evidence: bool
    price_evidence: bool


@dataclass
class HaltAnalysisResult:
    """停牌分析结果"""
    stock_code: str
    total_halt_periods: int
    total_halt_days: int
    halt_periods: List[HaltPeriod]
    data_quality_impact: str
    recommended_actions: List[str]
    analysis_date: str


class TradingHaltProcessor:
    """
    停牌处理器
    
    核心功能：
    1. 停牌检测 - 基于成交量、价格变化等指标
    2. 停牌分类 - 区分不同类型的停牌
    3. 数据过滤 - 过滤停牌期间的无效数据
    4. 复牌分析 - 分析复牌对后续数据的影响
    5. 停牌修正 - 对停牌期间数据进行合理处理
    """
    
    def __init__(self):
        # 停牌检测阈值
        self.halt_detection_config = {
            'zero_volume_threshold': 1000,        # 成交量接近零的阈值
            'min_halt_days': 2,                   # 最小停牌天数
            _threshold': 0.001,      # 价格变化阈值
            'volume_ratio_threshold': 0.01,       # 成交量比例阈值
            'turnover_rate_threshold': 0.0001     # 换手率阈值
        }
        
        # 停牌类型判断规则
        self.halt_type_rules = {
            'temporary': {'max_days': 7, 'volume_pattern': 'zero'},
            'reorganization': {'min_days': 30, 'volume_pattern': 'zero'},
            'st_suspension': {'min_days': 1, 'max_days': 10, 'price_pattern': 'unchanged'},
            'delisting': {'min_days': 60, 'volume_pattern': 'zero'}
        }
        
        # 统计信息
        self.processing_stats = {
            'total_processed': 0,
            'halts_detected': 0,
            'data_points_filtered': 0,
            'halt_types_distribution': {}
        }
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=3.0)
    def analyze_trading_halts(
        self, 
        stock_code: str, 
        stock_data: pd.DataFrame
    ) -> HaltAnalysisResult:
        """
        分析交易停牌情况
        
        Args:
            stock_code: 股票代码
            stock_data: 股票数据
            
        Returns:
            HaltAnalysisResult: 停牌分析结果
        """
        logger.info(f"🔍 开始停牌分析: {stock_code}")
        
        if stock_data.empty or len(stock_data) < 5:
            return self._create_empty_analysis(stock_code)
        
        # 检测停牌期间
        halt_periods = self._detect_halt_periods(stock_data)
        
        # 分类停牌类型
        classified_periods = self._classify_halt_types(halt_periods, stock_data)
        
        # 评估数据质量影响
        quality_impact = self._assess_quality_impact(classified_periods, len(stock_data))
        
        # 生成处理建议
        recommendations = self._generate_recommendations(classified_periods)
        
        # 更新统计
        self._update_stats(classified_periods)
        
        result = HaltAnalysisResult(
            stock_code=stock_code,
            total_halt_periods=len(classified_periods),
            total_halt_days=sum(p.duration_days for p in classified_periods),
            halt_periods=classified_periods,
            data_quality_impact=quality_impact,
            recommended_actions=recommendations,
            analysis_date=datetime.now().isoformat()
        )
        
        logger.info(f"✅ 停牌分析完成: {stock_code}, "
                   f"检测到 {len(classified_periods)} 个停牌期间, "
                   f"总计 {result.total_halt_days} 天")
        
        return result
    
    def _detect_halt_periods(self, stock_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """检测停牌期间"""
        if 'volume' not in stock_data.columns:
            return []
        
        stock_data = stock_data.copy()
        
        # 停牌条件1: 成交量为零或极低
        volume_condition = stock_data['volume'] <= self.halt_detection_config['zero_volume_threshold']
        
        # 停牌条件2: 价格基本不变（如果有价格数据）
        price_condition = pd.Series([False] * len(stock_data))
        if 'close' in stock_data.columns:
            s = stock_data['close'].diff().abs()
            price_condition = s <= self.halt_detection_config[_threshold']
        
        # 停牌条件3: 换手率极低（如果有换手率数据）
        turnover_rate_condition = pd.Series([False] * len(stock_data))
        if 'turnover_rate' in stock_data.columns:
            turnover_rate_condition = stock_data['turnover_rate'] <= self.halt_detection_config['turnover_rate_threshold']
        
        # 综合判断停牌
        halt_condition = volume_condition & (price_condition | turnover_rate_condition)
        
        # 找到连续的停牌期间
        halt_periods = []
        start_idx = None
        
        for i, is_halt in enumerate(halt_condition):
            if is_halt and start_idx is None:
                start_idx = i
            elif not is_halt and start_idx is not None:
                # 停牌期间结束
                duration = i - start_idx
                if duration >= self.halt_detection_config['min_halt_days']:
                    halt_periods.append({
                        'start_idx': start_idx,
                        'end_idx': i,
                        'start_date': stock_data.iloc[start_idx]['date'] if 'date' in stock_data.columns else f"index_{start_idx}",
                        'end_date': stock_data.iloc[i-1]['date'] if 'date' in stock_data.columns else f"index_{i-1}",
                        'duration_days': duration,
                        'volume_evidence': volume_condition.iloc[start_idx:i].all(),
                        'price_evidence': price_condition.iloc[start_idx:i].all() if 'close' in stock_data.columns else False
                    })
                start_idx = None
        
        # 处理数据末尾的停牌
        if start_idx is not None:
            duration = len(stock_data) - start_idx
            if duration >= self.halt_detection_config['min_halt_days']:
                halt_periods.append({
                    'start_idx': start_idx,
                    'end_idx': len(stock_data),
                    'start_date': stock_data.iloc[start_idx]['date'] if 'date' in stock_data.columns else f"index_{start_idx}",
                    'end_date': stock_data.iloc[-1]['date'] if 'date' in stock_data.columns else f"index_{len(stock_data)-1}",
                    'duration_days': duration,
                    'volume_evidence': volume_condition.iloc[start_idx:].all(),
                    'price_evidence': price_condition.iloc[start_idx:].all() if 'close' in stock_data.columns else False
                })
        
        return halt_periods
    
    def _classify_halt_types(self, halt_periods: List[Dict[str, Any]], stock_data: pd.DataFrame) -> List[HaltPeriod]:
        """对停牌进行分类"""
        classified_periods = []
        
        for period in halt_periods:
            duration = period['duration_days']
            
            # 根据持续时间和特征判断停牌类型
            if duration <= 7:
                halt_type = HaltType.TEMPORARY
                confidence = 0.8
            elif duration >= 60:
                halt_type = HaltType.DELISTING
                confidence = 0.7
            elif duration >= 30:
                halt_type = HaltType.REORGANIZATION
                confidence = 0.6
            elif 1 <= duration <= 10:
                halt_type = HaltType.ST_SUSPENSION
                confidence = 0.5
            else:
                halt_type = HaltType.UNKNOWN
                confidence = 0.3
            
            # 提高置信度基于更多证据
            if period['volume_evidence'] and period['price_evidence']:
                confidence = min(1.0, confidence + 0.2)
            
            classified_period = HaltPeriod(
                start_date=period['start_date'],
                end_date=period['end_date'],
                halt_type=halt_type,
                duration_days=duration,
                confidence=confidence,
                affected_data_points=period['end_idx'] - period['start_idx'],
                volume_evidence=period['volume_evidence'],
                price_evidence=period['price_evidence']
            )
            
            classified_periods.append(classified_period)
        
        return classified_periods
    
    def _assess_quality_impact(self, halt_periods: List[HaltPeriod], total_data_points: int) -> str:
        """评估停牌对数据质量的影响"""
        if not halt_periods:
            return "无停牌影响，数据质量良好"
        
        total_halt_days = sum(p.duration_days for p in halt_periods)
        halt_ratio = total_halt_days / total_data_points
        
        if halt_ratio < 0.05:
            return "停牌影响轻微，对数据质量影响较小"
        elif halt_ratio < 0.15:
            return "停牌影响中等，需要在分析中考虑停牌期间"
        elif halt_ratio < 0.3:
            return "停牌影响较大，建议过滤停牌期间数据"
        else:
            return "停牌影响严重，数据质量受到重大影响"
    
    def _generate_recommendations(self, halt_periods: List[HaltPeriod]) -> List[str]:
        """生成处理建议"""
        recommendations = []
        
        if not halt_periods:
            recommendations.append("无停牌检测，数据可正常使用")
            return recommendations
        
        # 基于停牌情况生成建议
        total_halt_days = sum(p.duration_days for p in halt_periods)
        long_halts = [p for p in halt_periods if p.duration_days > 30]
        
        if total_halt_days > 0:
            recommendations.append(f"检测到 {len(halt_periods)} 个停牌期间，总计 {total_halt_days} 天")
        
        if long_halts:
            recommendations.append("存在长期停牌，建议排除这些期间的数据进行分析")
        
        # 按停牌类型给出建议
        type_counts = {}
        for period in halt_periods:
            type_counts[period.halt_type] = type_counts.get(period.halt_type, 0) + 1
        
        if HaltType.REORGANIZATION in type_counts:
            recommendations.append("检测到重组停牌，需要特别关注复牌后的价格变化")
        
        if HaltType.ST_SUSPENSION in type_counts:
            recommendations.append("检测到ST相关停牌，建议验证公司基本面信息")
        
        if HaltType.DELISTING in type_counts:
            recommendations.append("检测到疑似退市停牌，建议核实股票状态")
        
        # 数据处理建议
        recommendations.append("建议使用 filter_halt_periods() 方法过滤停牌期间数据")
        
        return recommendations
    
    def filter_halt_periods(
        self, 
        stock_data: pd.DataFrame, 
        halt_periods: List[HaltPeriod],
        filter_mode: str = "exclude"
    ) -> pd.DataFrame:
        """
        过滤停牌期间数据
        
        Args:
            stock_data: 股票数据
            halt_periods: 停牌期间列表
            filter_mode: 过滤模式 ("exclude": 排除停牌期间, "include": 仅保留停牌期间, "mark": 标记停牌期间)
            
        Returns:
            pd.DataFrame: 过滤后的数据
        """
        if not halt_periods or stock_data.empty:
            return stock_data.copy()
        
        filtered_data = stock_data.copy()
        
        if 'date' not in stock_data.columns:
            logger.warning("数据中缺少日期列，无法过滤停牌期间")
            return filtered_data
        
        # 创建停牌标记
        halt_mask = pd.Series([False] * len(filtered_data))
        
        for period in halt_periods:
            period_mask = (
                (filtered_data['date'] >= period.start_date) & 
                (filtered_data['date'] <= period.end_date)
            )
            halt_mask |= period_mask
        
        if filter_mode == "exclude":
            # 排除停牌期间
            filtered_data = filtered_data[~halt_mask].copy()
            logger.info(f"排除停牌期间数据: {halt_mask.sum()} 条记录")
            
        elif filter_mode == "include":
            # 仅保留停牌期间
            filtered_data = filtered_data[halt_mask].copy()
            logger.info(f"仅保留停牌期间数据: {halt_mask.sum()} 条记录")
            
        elif filter_mode == "mark":
            # 标记停牌期间
            filtered_data['is_halted'] = halt_mask
            filtered_data['halt_type'] = 'normal'
            
            for period in halt_periods:
                period_mask = (
                    (filtered_data['date'] >= period.start_date) & 
                    (filtered_data['date'] <= period.end_date)
                )
                filtered_data.loc[period_mask, 'halt_type'] = period.halt_type.value
            
            logger.info(f"标记停牌期间数据: {halt_mask.sum()} 条记录")
        
        return filtered_data
    
    def _create_empty_analysis(self, stock_code: str) -> HaltAnalysisResult:
        """创建空分析结果"""
        return HaltAnalysisResult(
            stock_code=stock_code,
            total_halt_periods=0,
            total_halt_days=0,
            halt_periods=[],
            data_quality_impact="数据不足，无法分析",
            recommended_actions=["获取更多历史数据"],
            analysis_date=datetime.now().isoformat()
        )
    
    def _update_stats(self, halt_periods: List[HaltPeriod]):
        """更新统计信息"""
        self.processing_stats['total_processed'] += 1
        self.processing_stats['halts_detected'] += len(halt_periods)
        
        for period in halt_periods:
            halt_type = period.halt_type.value
            self.processing_stats['halt_types_distribution'][halt_type] = \
                self.processing_stats['halt_types_distribution'].get(halt_type, 0) + 1
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """获取处理统计摘要"""
        return {
            'total_stocks_processed': self.processing_stats['total_processed'],
            'total_halts_detected': self.processing_stats['halts_detected'],
            'average_halts_per_stock': self.processing_stats['halts_detected'] / max(1, self.processing_stats['total_processed']),
            'halt_types_distribution': self.processing_stats['halt_types_distribution']
        }
    
    @exception_handler(reraise=False, default_return=pd.DataFrame())
    def clean_resumption_data(
        self, 
        stock_data: pd.DataFrame, 
        halt_periods: List[HaltPeriod],
        resumption_window: int = 5
    ) -> pd.DataFrame:
        """
        清洗复牌数据
        
        Args:
            stock_data: 股票数据
            halt_periods: 停牌期间列表
            resumption_window: 复牌后观察窗口（天数）
            
        Returns:
            pd.DataFrame: 清洗后的数据
        """
        if not halt_periods or stock_data.empty:
            return stock_data.copy()
        
        cleaned_data = stock_data.copy()
        
        for period in halt_periods:
            # 找到复牌后的数据
            resumption_mask = (
                cleaned_data['date'] > period.end_date
            )
            
            if resumption_mask.any():
                resumption_start_idx = cleaned_data[resumption_mask].index[0]
                resumption_end_idx = min(
                    resumption_start_idx + resumption_window,
                    len(cleaned_data)
                )
                
                # 标记复牌期间
                cleaned_data.loc[resumption_start_idx:resumption_end_idx, 'is_post_resumption'] = True
                
                # 可以在这里添加复牌数据的特殊处理逻辑
                # 例如：价格异常值处理、成交量异常处理等
        
        return cleaned_data


# 便捷函数
@exception_handler(reraise=False, default_return=None)
def detect_trading_halts(stock_code: str, stock_data: pd.DataFrame) -> Optional[HaltAnalysisResult]:
    """
    检测交易停牌
    
    Args:
        stock_code: 股票代码
        stock_data: 股票数据
        
    Returns:
        Optional[HaltAnalysisResult]: 停牌分析结果
    """
    processor = TradingHaltProcessor()
    return processor.analyze_trading_halts(stock_code, stock_data)


if __name__ == "__main__":
    # 测试示例
    import numpy as np
    
    # 创建包含停牌的测试数据
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    test_data = pd.DataFrame({
        'date': dates,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 100),
        'turnover_rate': np.random.random(100) * 0.1
    })
    
    # 模拟停牌期间（第20-30天）
    test_data.loc[20:30, 'volume'] = 0
    test_data.loc[20:30, 'turnover_rate'] = 0
    test_data.loc[20:30, 'close'] = test_data.loc[19, 'close']  # 价格不变
    
    # 模拟另一个短期停牌（第50-52天）
    test_data.loc[50:52, 'volume'] = 100
    test_data.loc[50:52, 'turnover_rate'] = 0.0001
    
    # 执行停牌检测
    result = detect_trading_halts('000001.SZ', test_data)
    
    if result:
        print(f"停牌分析完成:")
        print(f"  股票代码: {result.stock_code}")
        print(f"  停牌期间数: {result.total_halt_periods}")
        print(f"  总停牌天数: {result.total_halt_days}")
        print(f"  数据质量影响: {result.data_quality_impact}")
        
        print(f"\n检测到的停牌期间:")
        for i, period in enumerate(result.halt_periods, 1):
            print(f"  {i}. {period.start_date} 至 {period.end_date}")
            print(f"     类型: {period.halt_type.value}, 天数: {period.duration_days}")
            print(f"     置信度: {period.confidence:.2f}")
        
        print(f"\n处理建议:")
        for i, recommendation in enumerate(result.recommended_actions, 1):
            print(f"  {i}. {recommendation}")
        
        # 测试数据过滤
        processor = TradingHaltProcessor()
        filtered_data = processor.filter_halt_periods(
            test_data, result.halt_periods, filter_mode="exclude"
        )
        print(f"\n数据过滤结果:")
        print(f"  原始数据: {len(test_data)} 条")
        print(f"  过滤后数据: {len(filtered_data)} 条")
        print(f"  过滤比例: {(len(test_data) - len(filtered_data)) / len(test_data):.1%}")
    
    else:
        print("停牌检测失败") 