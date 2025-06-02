#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
评分系统验证模块

用于验证技术指标评分系统的有效性和准确性
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import seaborn as sns

from utils.logger import get_logger
from utils.decorators import performance_monitor

logger = get_logger(__name__)


class ScoringValidator:
    """
    评分系统验证器类
    
    验证技术指标评分系统的有效性和准确性
    """
    
    def __init__(self, lookforward_days: int = 5, score_thresholds: Dict[str, float] = None):
        """
        初始化评分系统验证器
        
        Args:
            lookforward_days: 向前看的天数，用于验证评分与未来价格表现的关系
            score_thresholds: 评分阈值字典，用于分类不同类型的信号
                例如：{'bullish': 70, 'bearish': 30}
        """
        self.lookforward_days = lookforward_days
        self.score_thresholds = score_thresholds or {'bullish': 70, 'bearish': 30}
        self.results = {}
    
    @performance_monitor()
    def validate(self, 
                scores: pd.Series, 
                price_data: pd.DataFrame,
                label_func: Optional[Callable] = None) -> Dict[str, Any]:
        """
        验证评分系统
        
        Args:
            scores: 评分序列
            price_data: 价格数据DataFrame，必须包含'close'列
            label_func: 可选的标签生成函数，用于自定义未来价格表现的判断逻辑
                        如果未提供，则使用默认的价格变化百分比逻辑
        
        Returns:
            Dict[str, Any]: 验证结果字典
        """
        if 'close' not in price_data.columns:
            raise ValueError("价格数据必须包含'close'列")
            
        if len(scores) != len(price_data):
            raise ValueError("评分序列和价格数据长度必须一致")
            
        # 确保索引对齐
        scores = scores.copy()
        price_data = price_data.copy()
        
        if not label_func:
            # 默认标签生成函数：使用未来价格变化百分比
            def default_label_func(prices):
                future_returns = prices['close'].pct_change(self.lookforward_days).shift(-self.lookforward_days)
                # 1表示上涨，-1表示下跌，0表示持平
                return np.where(future_returns > 0.01, 1, np.where(future_returns < -0.01, -1, 0))
            
            label_func = default_label_func
        
        # 生成实际标签
        actual_labels = label_func(price_data)
        
        # 根据评分阈值生成预测标签
        bullish_threshold = self.score_thresholds.get('bullish', 70)
        bearish_threshold = self.score_thresholds.get('bearish', 30)
        
        predicted_labels = np.where(scores > bullish_threshold, 1, 
                                    np.where(scores < bearish_threshold, -1, 0))
        
        # 转换为DataFrame便于分析
        validation_df = pd.DataFrame({
            'score': scores,
            'predicted': predicted_labels,
            'actual': actual_labels,
            'close': price_data['close']
        })
        
        # 去除包含NaN的行
        validation_df = validation_df.dropna()
        
        # 如果数据不足，返回空结果
        if len(validation_df) < 10:
            logger.warning("有效数据点不足，无法进行验证")
            return {}
        
        # 计算准确率
        # 只考虑有明确预测（非0）和明确结果（非0）的样本
        valid_samples = validation_df[(validation_df['predicted'] != 0) & (validation_df['actual'] != 0)]
        
        if len(valid_samples) > 0:
            accuracy = (valid_samples['predicted'] == valid_samples['actual']).mean()
        else:
            accuracy = np.nan
            
        # 计算混淆矩阵
        # 将标签转换为分类形式
        y_true = np.where(validation_df['actual'] > 0, 'bullish', 
                         np.where(validation_df['actual'] < 0, 'bearish', 'neutral'))
        y_pred = np.where(validation_df['predicted'] > 0, 'bullish', 
                         np.where(validation_df['predicted'] < 0, 'bearish', 'neutral'))
                         
        cm = confusion_matrix(y_true, y_pred, labels=['bullish', 'neutral', 'bearish'])
        
        # 计算精确率和召回率（针对看涨信号）
        bullish_precision = cm[0, 0] / cm[:, 0].sum() if cm[:, 0].sum() > 0 else 0
        bullish_recall = cm[0, 0] / cm[0, :].sum() if cm[0, :].sum() > 0 else 0
        
        # 计算精确率和召回率（针对看跌信号）
        bearish_precision = cm[2, 2] / cm[:, 2].sum() if cm[:, 2].sum() > 0 else 0
        bearish_recall = cm[2, 2] / cm[2, :].sum() if cm[2, :].sum() > 0 else 0
        
        # 按评分分组分析未来回报
        score_bins = [0, 20, 40, 50, 60, 80, 100]
        validation_df['score_group'] = pd.cut(validation_df['score'], bins=score_bins)
        
        # 计算每个分数组的未来回报
        future_returns = price_data['close'].pct_change(self.lookforward_days).shift(-self.lookforward_days)
        validation_df['future_return'] = future_returns
        
        score_group_performance = validation_df.groupby('score_group')['future_return'].agg(['mean', 'count']).reset_index()
        
        # 存储结果
        result = {
            'accuracy': accuracy,
            'bullish_precision': bullish_precision,
            'bullish_recall': bullish_recall,
            'bearish_precision': bearish_precision,
            'bearish_recall': bearish_recall,
            'confusion_matrix': cm,
            'score_group_performance': score_group_performance,
            'validation_data': validation_df
        }
        
        self.results = result
        return result
    
    def optimize_thresholds(self, 
                          scores: pd.Series, 
                          price_data: pd.DataFrame,
                          threshold_range: Tuple[int, int, int] = (50, 80, 5)) -> Dict[str, float]:
        """
        优化评分阈值
        
        Args:
            scores: 评分序列
            price_data: 价格数据DataFrame
            threshold_range: 阈值范围元组 (最小值, 最大值, 步长)
                             用于确定bullish阈值，bearish阈值自动设置为100-bullish
        
        Returns:
            Dict[str, float]: 优化后的阈值字典
        """
        best_accuracy = 0
        best_thresholds = self.score_thresholds.copy()
        
        min_thresh, max_thresh, step = threshold_range
        
        for bullish_threshold in range(min_thresh, max_thresh + 1, step):
            # bearish阈值对称设置
            bearish_threshold = 100 - bullish_threshold
            
            test_thresholds = {'bullish': bullish_threshold, 'bearish': bearish_threshold}
            
            # 使用当前阈值进行验证
            self.score_thresholds = test_thresholds
            result = self.validate(scores, price_data)
            
            # 如果结果有效且准确率更高，则更新最佳阈值
            if result and 'accuracy' in result and not np.isnan(result['accuracy']):
                if result['accuracy'] > best_accuracy:
                    best_accuracy = result['accuracy']
                    best_thresholds = test_thresholds.copy()
        
        # 恢复最佳阈值
        self.score_thresholds = best_thresholds
        
        logger.info(f"优化后的阈值: {best_thresholds}, 准确率: {best_accuracy:.4f}")
        return best_thresholds
    
    def plot_validation_results(self, output_file: Optional[str] = None):
        """
        绘制验证结果图表
        
        Args:
            output_file: 输出文件路径，如果为None则显示图表
            
        Returns:
            None
        """
        if not self.results:
            logger.warning("没有验证结果可供绘制")
            return
            
        validation_df = self.results.get('validation_data')
        if validation_df is None or len(validation_df) == 0:
            logger.warning("验证数据为空")
            return
            
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 混淆矩阵热力图
        cm = self.results.get('confusion_matrix')
        if cm is not None:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['看涨', '中性', '看跌'],
                       yticklabels=['看涨', '中性', '看跌'],
                       ax=axes[0, 0])
            axes[0, 0].set_title('混淆矩阵')
            axes[0, 0].set_xlabel('预测标签')
            axes[0, 0].set_ylabel('实际标签')
        
        # 2. 评分分布图
        validation_df['score'].hist(bins=20, ax=axes[0, 1])
        axes[0, 1].set_title('评分分布')
        axes[0, 1].set_xlabel('评分')
        axes[0, 1].set_ylabel('频率')
        
        # 3. 评分组与未来回报的关系
        score_perf = self.results.get('score_group_performance')
        if score_perf is not None:
            score_perf.plot(x='score_group', y='mean', kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title(f'评分组与{self.lookforward_days}日未来回报的关系')
            axes[1, 0].set_xlabel('评分组')
            axes[1, 0].set_ylabel('平均未来回报')
        
        # 4. ROC曲线（针对看涨预测）
        # 将问题转换为二分类：是否为看涨信号
        y_true_binary = (validation_df['actual'] > 0).astype(int)
        
        # 使用评分作为预测概率
        y_score = validation_df['score'] / 100  # 归一化到0-1
        
        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)
        
        axes[1, 1].plot(fpr, tpr, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        axes[1, 1].plot([0, 1], [0, 1], 'k--')
        axes[1, 1].set_xlim([0.0, 1.0])
        axes[1, 1].set_ylim([0.0, 1.05])
        axes[1, 1].set_xlabel('假正例率')
        axes[1, 1].set_ylabel('真正例率')
        axes[1, 1].set_title('看涨信号的ROC曲线')
        axes[1, 1].legend(loc="lower right")
        
        # 调整布局
        plt.tight_layout()
        
        # 保存或显示图表
        if output_file:
            plt.savefig(output_file, dpi=300)
            plt.close(fig)
            logger.info(f"验证结果图表已保存至: {output_file}")
        else:
            plt.show()
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """
        获取性能统计信息
        
        Returns:
            Dict[str, Any]: 性能统计字典
        """
        if not self.results:
            logger.warning("没有验证结果可供分析")
            return {}
            
        validation_df = self.results.get('validation_data')
        if validation_df is None or len(validation_df) == 0:
            logger.warning("验证数据为空")
            return {}
            
        # 1. 总体准确率
        accuracy = self.results.get('accuracy', np.nan)
        
        # 2. 各信号类型的精确率和召回率
        bullish_precision = self.results.get('bullish_precision', np.nan)
        bullish_recall = self.results.get('bullish_recall', np.nan)
        bearish_precision = self.results.get('bearish_precision', np.nan)
        bearish_recall = self.results.get('bearish_recall', np.nan)
        
        # 3. 各评分组的样本数和平均回报
        score_perf = self.results.get('score_group_performance')
        
        # 4. 评分与未来回报的相关性
        if 'score' in validation_df.columns and 'future_return' in validation_df.columns:
            correlation = validation_df[['score', 'future_return']].corr().iloc[0, 1]
        else:
            correlation = np.nan
            
        # 5. 计算信号盈亏比
        # 看涨信号的平均回报
        bullish_returns = validation_df[validation_df['predicted'] > 0]['future_return']
        avg_bullish_return = bullish_returns.mean() if len(bullish_returns) > 0 else np.nan
        
        # 看跌信号的平均回报（取反，使正值表示正确）
        bearish_returns = validation_df[validation_df['predicted'] < 0]['future_return']
        avg_bearish_return = -bearish_returns.mean() if len(bearish_returns) > 0 else np.nan
        
        # 计算盈亏比（胜率和赔率）
        bullish_win_rate = (bullish_returns > 0).mean() if len(bullish_returns) > 0 else np.nan
        bearish_win_rate = (bearish_returns < 0).mean() if len(bearish_returns) > 0 else np.nan
        
        # 计算平均盈利和平均亏损
        avg_bullish_profit = bullish_returns[bullish_returns > 0].mean() if len(bullish_returns[bullish_returns > 0]) > 0 else np.nan
        avg_bullish_loss = bullish_returns[bullish_returns < 0].mean() if len(bullish_returns[bullish_returns < 0]) > 0 else np.nan
        
        if not np.isnan(avg_bullish_profit) and not np.isnan(avg_bullish_loss) and avg_bullish_loss != 0:
            bullish_profit_loss_ratio = abs(avg_bullish_profit / avg_bullish_loss)
        else:
            bullish_profit_loss_ratio = np.nan
            
        # 收集性能统计结果
        stats = {
            'accuracy': accuracy,
            'bullish_precision': bullish_precision,
            'bullish_recall': bullish_recall,
            'bearish_precision': bearish_precision,
            'bearish_recall': bearish_recall,
            'correlation': correlation,
            'avg_bullish_return': avg_bullish_return,
            'avg_bearish_return': avg_bearish_return,
            'bullish_win_rate': bullish_win_rate,
            'bearish_win_rate': bearish_win_rate,
            'bullish_profit_loss_ratio': bullish_profit_loss_ratio
        }
        
        return stats 