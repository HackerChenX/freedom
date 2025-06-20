"""
数据质量验证器

负责验证多时间周期数据的一致性、完整性和准确性
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from utils.logger import get_logger
from db.data_manager import DataManager

logger = get_logger(__name__)


class DataQualityValidator:
    """数据质量验证器"""
    
    def __init__(self):
        """初始化验证器"""
        self.data_manager = DataManager()
        
    def validate_multi_period_data(self, stock_code: str, date: str) -> Dict[str, Any]:
        """
        验证多时间周期数据的一致性
        
        Args:
            stock_code: 股票代码
            date: 验证日期
            
        Returns:
            Dict: 验证结果
        """
        validation_results = {
            'stock_code': stock_code,
            'validation_date': date,
            'overall_quality': 'unknown',
            'period_results': {},
            'consistency_checks': {},
            'issues': []
        }
        
        try:
            # 获取多周期数据
            periods = ['15min', '30min', '60min', 'daily', 'weekly', 'monthly']
            period_data = {}
            
            for period in periods:
                try:
                    data = self.data_manager.get_stock_data(
                        stock_code=stock_code,
                        period=period,
                        start_date=self._get_start_date(date, period),
                        end_date=date
                    )
                    period_data[period] = data
                    
                    # 验证单个周期数据质量
                    period_quality = self._validate_single_period_data(data, period)
                    validation_results['period_results'][period] = period_quality
                    
                except Exception as e:
                    logger.warning(f"获取{period}数据失败: {e}")
                    validation_results['period_results'][period] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            # 验证周期间数据一致性
            consistency_results = self._validate_period_consistency(period_data)
            validation_results['consistency_checks'] = consistency_results
            
            # 汇总质量评估
            validation_results['overall_quality'] = self._assess_overall_quality(
                validation_results['period_results'],
                validation_results['consistency_checks']
            )
            
        except Exception as e:
            logger.error(f"验证数据质量时出错: {e}")
            validation_results['issues'].append(f"验证过程出错: {str(e)}")
            validation_results['overall_quality'] = 'error'
        
        return validation_results
    
    def _validate_single_period_data(self, data: pd.DataFrame, period: str) -> Dict[str, Any]:
        """验证单个周期数据质量"""
        if data is None or data.empty:
            return {
                'status': 'empty',
                'data_count': 0,
                'issues': ['数据为空']
            }
        
        issues = []
        
        # 检查必需列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            issues.append(f"缺少必需列: {missing_columns}")
        
        # 检查数据完整性
        null_counts = data[required_columns].isnull().sum()
        if null_counts.any():
            issues.append(f"存在空值: {null_counts.to_dict()}")
        
        # 检查数据逻辑性
        if (data['high'] < data['low']).any():
            issues.append("存在最高价低于最低价的异常数据")
        
        if (data['high'] < data['close']).any():
            issues.append("存在最高价低于收盘价的异常数据")
        
        if (data['low'] > data['close']).any():
            issues.append("存在最低价高于收盘价的异常数据")
        
        # 检查价格异常波动
        if len(data) > 1:
            price_changes = data['close'].pct_change().abs()
            extreme_changes = price_changes > 0.2  # 20%以上变化
            if extreme_changes.any():
                issues.append(f"存在异常价格波动: {extreme_changes.sum()}个")
        
        # 检查成交量异常
        if (data['volume'] <= 0).any():
            issues.append("存在零成交量或负成交量")
        
        return {
            'status': 'valid' if not issues else 'warning',
            'data_count': len(data),
            'issues': issues,
            'date_range': {
                'start': data.index.min().strftime('%Y-%m-%d') if not data.empty else None,
                'end': data.index.max().strftime('%Y-%m-%d') if not data.empty else None
            }
        }
    
    def _validate_period_consistency(self, period_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """验证周期间数据一致性"""
        consistency_results = {
            'time_alignment': {},
            'price_consistency': {},
            'volume_consistency': {},
            'overall_consistency': 'unknown'
        }
        
        try:
            # 检查时间对齐
            if '15min' in period_data and '30min' in period_data:
                alignment_check = self._check_time_alignment(
                    period_data['15min'], period_data['30min'], '15min', '30min')
                consistency_results['time_alignment']['15min_30min'] = alignment_check
            
            # 检查价格一致性（日线与分钟线）
            if 'daily' in period_data and '15min' in period_data:
                price_check = self._check_price_consistency(
                    period_data['daily'], period_data['15min'])
                consistency_results['price_consistency']['daily_15min'] = price_check
            
            # 评估整体一致性
            consistency_results['overall_consistency'] = self._assess_consistency_quality(
                consistency_results)
            
        except Exception as e:
            logger.error(f"验证数据一致性时出错: {e}")
            consistency_results['overall_consistency'] = 'error'
        
        return consistency_results
    
    def _check_time_alignment(self, data1: pd.DataFrame, data2: pd.DataFrame, 
                            period1: str, period2: str) -> Dict[str, Any]:
        """检查两个周期数据的时间对齐"""
        if data1.empty or data2.empty:
            return {'status': 'no_data', 'message': '数据为空，无法检查对齐'}
        
        # 获取共同的交易日
        dates1 = set(data1.index.date)
        dates2 = set(data2.index.date)
        common_dates = dates1 & dates2
        
        if not common_dates:
            return {
                'status': 'no_overlap',
                'message': '没有共同的交易日',
                'dates1_count': len(dates1),
                'dates2_count': len(dates2)
            }
        
        alignment_ratio = len(common_dates) / max(len(dates1), len(dates2))
        
        return {
            'status': 'aligned' if alignment_ratio > 0.8 else 'misaligned',
            'alignment_ratio': alignment_ratio,
            'common_dates_count': len(common_dates),
            'total_dates1': len(dates1),
            'total_dates2': len(dates2)
        }
    
    def _check_price_consistency(self, daily_data: pd.DataFrame, 
                               minute_data: pd.DataFrame) -> Dict[str, Any]:
        """检查日线与分钟线价格一致性"""
        if daily_data.empty or minute_data.empty:
            return {'status': 'no_data', 'message': '数据为空，无法检查一致性'}
        
        inconsistencies = []
        
        # 按日期分组检查
        for date in daily_data.index:
            date_str = date.strftime('%Y-%m-%d')
            daily_row = daily_data.loc[date]
            
            # 获取对应日期的分钟数据
            minute_day_data = minute_data[minute_data.index.date == date.date()]
            
            if minute_day_data.empty:
                continue
            
            # 检查开盘价
            minute_open = minute_day_data.iloc[0]['open']
            if abs(daily_row['open'] - minute_open) / daily_row['open'] > 0.01:  # 1%误差
                inconsistencies.append(f"{date_str}: 开盘价不一致")
            
            # 检查收盘价
            minute_close = minute_day_data.iloc[-1]['close']
            if abs(daily_row['close'] - minute_close) / daily_row['close'] > 0.01:
                inconsistencies.append(f"{date_str}: 收盘价不一致")
            
            # 检查最高价和最低价
            minute_high = minute_day_data['high'].max()
            minute_low = minute_day_data['low'].min()
            
            if abs(daily_row['high'] - minute_high) / daily_row['high'] > 0.01:
                inconsistencies.append(f"{date_str}: 最高价不一致")
            
            if abs(daily_row['low'] - minute_low) / daily_row['low'] > 0.01:
                inconsistencies.append(f"{date_str}: 最低价不一致")
        
        return {
            'status': 'consistent' if not inconsistencies else 'inconsistent',
            'inconsistency_count': len(inconsistencies),
            'inconsistencies': inconsistencies[:10]  # 只返回前10个
        }
    
    def _assess_overall_quality(self, period_results: Dict, consistency_results: Dict) -> str:
        """评估整体数据质量"""
        # 检查各周期数据状态
        valid_periods = sum(1 for result in period_results.values() 
                          if result.get('status') == 'valid')
        total_periods = len(period_results)
        
        if valid_periods == 0:
            return 'poor'
        elif valid_periods / total_periods < 0.5:
            return 'fair'
        elif valid_periods / total_periods < 0.8:
            return 'good'
        else:
            # 检查一致性
            consistency_status = consistency_results.get('overall_consistency', 'unknown')
            if consistency_status in ['good', 'excellent']:
                return 'excellent'
            else:
                return 'good'
    
    def _assess_consistency_quality(self, consistency_results: Dict) -> str:
        """评估一致性质量"""
        # 简单的一致性评估逻辑
        alignment_results = consistency_results.get('time_alignment', {})
        price_results = consistency_results.get('price_consistency', {})
        
        good_alignments = sum(1 for result in alignment_results.values()
                            if result.get('status') == 'aligned')
        good_prices = sum(1 for result in price_results.values()
                        if result.get('status') == 'consistent')
        
        total_checks = len(alignment_results) + len(price_results)
        
        if total_checks == 0:
            return 'unknown'
        
        good_ratio = (good_alignments + good_prices) / total_checks
        
        if good_ratio >= 0.9:
            return 'excellent'
        elif good_ratio >= 0.7:
            return 'good'
        elif good_ratio >= 0.5:
            return 'fair'
        else:
            return 'poor'
    
    def _get_start_date(self, end_date: str, period: str) -> str:
        """根据周期获取合适的开始日期"""
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        if period in ['15min', '30min', '60min']:
            # 分钟数据，往前30天
            start_dt = end_dt - timedelta(days=30)
        elif period == 'daily':
            # 日线数据，往前120天
            start_dt = end_dt - timedelta(days=120)
        elif period == 'weekly':
            # 周线数据，往前1年
            start_dt = end_dt - timedelta(days=365)
        else:  # monthly
            # 月线数据，往前3年
            start_dt = end_dt - timedelta(days=365*3)
        
        return start_dt.strftime('%Y-%m-%d')
