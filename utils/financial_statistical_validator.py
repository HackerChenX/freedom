#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
金融量化策略统计显著性验证器

专业量化交易系统的统计显著性验证模块，确保策略生成符合金融统计标准：
- 统计显著性检验（t检验、卡方检验、Mann-Whitney U检验）
- 样本量充足性验证和效力分析
- 策略泛化能力评估（交叉验证、样本外测试）
- 多重假设检验校正（Bonferroni、FDR）
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from scipy import stats
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from statsmodels.stats.power import ttest_power, chisquare_power
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
import warnings
from utils.dependency_injection import get_logger

logger = get_logger(__name__)

class StatisticalTestType(Enum):
    """统计检验类型"""
    T_TEST = "t_test"                    # t检验
    PAIRED_T_TEST = "paired_t_test"      # 配对t检验
    MANN_WHITNEY = "mann_whitney"        # Mann-Whitney U检验
    WILCOXON = "wilcoxon"               # Wilcoxon符号秩检验
    CHI_SQUARE = "chi_square"           # 卡方检验
    FISHER_EXACT = "fisher_exact"       # Fisher精确检验
    KOLMOGOROV_SMIRNOV = "ks_test"     # K-S检验

class EffectSizeType(Enum):
    """效应量类型"""
    COHEN_D = "cohen_d"                 # Cohen's d
    HEDGES_G = "hedges_g"               # Hedges' g
    GLASS_DELTA = "glass_delta"         # Glass's Δ
    CLIFF_DELTA = "cliff_delta"         # Cliff's δ
    CRAMERS_V = "cramers_v"            # Cramér's V
    ETA_SQUARED = "eta_squared"         # η²

@dataclass
class StatisticalTestResult:
    """统计检验结果"""
    test_type: StatisticalTestType
    statistic: float
    p_value: float
    critical_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    effect_size: Optional[float] = None
    effect_size_type: Optional[EffectSizeType] = None
    power: Optional[float] = None
    interpretation: str = ""

    def is_significant(self, alpha: float = 0.05) -> bool:
        """判断是否统计显著"""
        return self.p_value < alpha

@dataclass
class SampleSizeAnalysis:
    """样本量分析结果"""
    current_sample_size: int
    recommended_min_size: int
    power_achieved: float
    effect_size_detectable: float
    adequacy_status: str  # "ADEQUATE", "MARGINAL", "INADEQUATE"

class FinancialStatisticalValidator:
    """金融量化策略统计显著性验证器"""

    def __init__(self,
                 alpha_level: float = 0.05,
                 min_power: float = 0.8,
                 min_effect_size: float = 0.2):
        """
        初始化统计验证器

        Args:
            alpha_level: 显著性水平（默认0.05）
            min_power: 最小统计功效（默认0.8）
            min_effect_size: 最小效应量（默认0.2，小效应）
        """
        self.alpha_level = alpha_level
        self.min_power = min_power
        self.min_effect_size = min_effect_size

        # 金融量化特定参数
        self.min_return_periods = 20  # 最小回报观察期数
        self.min_strategies_for_comparison = 5  # 策略比较最小数量

    def validate_strategy_statistical_significance(self,
                                                 returns_data: List[float],
                                                 benchmark_returns: Optional[List[float]] = None,
                                                 control_group_returns: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        验证策略的统计显著性

        Args:
            returns_data: 策略收益数据
            benchmark_returns: 基准收益数据（可选）
            control_group_returns: 对照组收益数据（可选）

        Returns:
            Dict: 完整的统计验证结果
        """
        results = {
            'sample_size_analysis': None,
            'normality_tests': {},
            'significance_tests': {},
            'effect_size_analysis': {},
            'multiple_testing_correction': {},
            'overall_assessment': {}
        }

        try:
            # 1. 样本量充足性验证
            results['sample_size_analysis'] = self._analyze_sample_adequacy(returns_data)

            # 2. 数据分布正态性检验
            results['normality_tests'] = self._test_normality(returns_data)

            # 3. 核心统计显著性检验
            if benchmark_returns:
                # 与基准比较
                results['significance_tests']['vs_benchmark'] = self._compare_with_benchmark(
                    returns_data, benchmark_returns, results['normality_tests']
                )

            if control_group_returns:
                # 与对照组比较
                results['significance_tests']['vs_control'] = self._compare_with_control(
                    returns_data, control_group_returns, results['normality_tests']
                )

            # 单样本检验（是否显著异于零）
            results['significance_tests']['single_sample'] = self._single_sample_test(
                returns_data, results['normality_tests']
            )

            # 4. 效应量分析
            results['effect_size_analysis'] = self._calculate_effect_sizes(
                returns_data, benchmark_returns, control_group_returns
            )

            # 5. 多重假设检验校正
            if len(results['significance_tests']) > 1:
                results['multiple_testing_correction'] = self._correct_multiple_testing(
                    results['significance_tests']
                )

            # 6. 综合评估
            results['overall_assessment'] = self._generate_overall_assessment(results)

        except Exception as e:
            logger.error(f"统计显著性验证失败: {e}")
            results['error'] = str(e)

        return results

    def validate_pattern_statistical_robustness(self,
                                              pattern_features: List[Dict[str, Any]],
                                              success_outcomes: List[bool]) -> Dict[str, Any]:
        """
        验证技术形态模式的统计稳健性

        Args:
            pattern_features: 技术形态特征数据
            success_outcomes: 成功结果（True/False）

        Returns:
            Dict: 模式统计稳健性分析结果
        """
        results = {
            'feature_significance_tests': {},
            'pattern_association_tests': {},
            'cross_validation_results': {},
            'generalization_analysis': {}
        }

        try:
            # 1. 特征与成功率的关联显著性
            for feature_name in pattern_features[0].keys():
                if feature_name not in ['stock_code', 'buypoint_date', 'extraction_time']:
                    results['feature_significance_tests'][feature_name] = \
                        self._test_feature_significance(pattern_features, success_outcomes, feature_name)

            # 2. 模式关联检验
            results['pattern_association_tests'] = self._test_pattern_associations(
                pattern_features, success_outcomes
            )

            # 3. 交叉验证分析
            results['cross_validation_results'] = self._perform_cross_validation(
                pattern_features, success_outcomes
            )

            # 4. 泛化能力分析
            results['generalization_analysis'] = self._analyze_generalization_capability(
                pattern_features, success_outcomes
            )

        except Exception as e:
            logger.error(f"模式统计稳健性验证失败: {e}")
            results['error'] = str(e)

        return results

    def _analyze_sample_adequacy(self, returns_data: List[float]) -> SampleSizeAnalysis:
        """分析样本量充足性"""
        current_size = len(returns_data)

        # 基于效应量估算所需样本量
        estimated_effect_size = abs(np.mean(returns_data) / np.std(returns_data)) if np.std(returns_data) > 0 else 0

        # 计算达到期望功效所需的样本量
        if estimated_effect_size > 0:
            recommended_size = int(ttest_power(
                effect_size=max(estimated_effect_size, self.min_effect_size),
                power=self.min_power,
                alpha=self.alpha_level,
                alternative='two-sided'
            )) if estimated_effect_size > 0 else 30
        else:
            recommended_size = 30  # 金融数据最小建议样本量

        # 计算当前样本量的统计功效
        if estimated_effect_size > 0 and current_size > 2:
            try:
                achieved_power = ttest_power(
                    effect_size=estimated_effect_size,
                    nobs=current_size,
                    alpha=self.alpha_level,
                    alternative='two-sided'
                )
            except:
                achieved_power = 0.5
        else:
            achieved_power = 0.5

        # 当前样本量能检测到的最小效应量
        if current_size > 2:
            try:
                min_detectable_effect = ttest_power(
                    power=self.min_power,
                    nobs=current_size,
                    alpha=self.alpha_level,
                    alternative='two-sided'
                )
            except:
                min_detectable_effect = 0.5
        else:
            min_detectable_effect = 1.0

        # 判断充足性状态
        if current_size >= recommended_size and achieved_power >= self.min_power:
            adequacy_status = "ADEQUATE"
        elif current_size >= recommended_size * 0.7 and achieved_power >= self.min_power * 0.8:
            adequacy_status = "MARGINAL"
        else:
            adequacy_status = "INADEQUATE"

        return SampleSizeAnalysis(
            current_sample_size=current_size,
            recommended_min_size=max(recommended_size, self.min_return_periods),
            power_achieved=achieved_power,
            effect_size_detectable=min_detectable_effect,
            adequacy_status=adequacy_status
        )

    def _test_normality(self, data: List[float]) -> Dict[str, StatisticalTestResult]:
        """检验数据正态性"""
        results = {}

        if len(data) < 3:
            return results

        data_array = np.array(data)
        data_clean = data_array[~np.isnan(data_array)]

        if len(data_clean) < 3:
            return results

        # Shapiro-Wilk检验（小样本）
        if len(data_clean) <= 5000:
            try:
                statistic, p_value = stats.shapiro(data_clean)
                results['shapiro_wilk'] = StatisticalTestResult(
                    test_type=StatisticalTestType.T_TEST,  # 使用已定义的枚举
                    statistic=statistic,
                    p_value=p_value,
                    interpretation="正态性检验：H0=数据服从正态分布"
                )
            except Exception as e:
                logger.warning(f"Shapiro-Wilk检验失败: {e}")

        # Kolmogorov-Smirnov检验
        try:
            statistic, p_value = stats.kstest(data_clean, 'norm',
                                             args=(np.mean(data_clean), np.std(data_clean)))
            results['ks_test'] = StatisticalTestResult(
                test_type=StatisticalTestType.KOLMOGOROV_SMIRNOV,
                statistic=statistic,
                p_value=p_value,
                interpretation="K-S正态性检验：H0=数据服从正态分布"
            )
        except Exception as e:
            logger.warning(f"K-S检验失败: {e}")

        # Jarque-Bera检验
        try:
            statistic, p_value = stats.jarque_bera(data_clean)
            results['jarque_bera'] = StatisticalTestResult(
                test_type=StatisticalTestType.T_TEST,  # 使用已定义的枚举
                statistic=statistic,
                p_value=p_value,
                interpretation="Jarque-Bera正态性检验：H0=数据服从正态分布"
            )
        except Exception as e:
            logger.warning(f"Jarque-Bera检验失败: {e}")

        return results

    def _compare_with_benchmark(self,
                               strategy_returns: List[float],
                               benchmark_returns: List[float],
                               normality_results: Dict) -> Dict[str, StatisticalTestResult]:
        """与基准进行比较检验"""
        results = {}

        strategy_array = np.array(strategy_returns)
        benchmark_array = np.array(benchmark_returns)

        # 确保数据长度一致
        min_length = min(len(strategy_array), len(benchmark_array))
        strategy_clean = strategy_array[:min_length]
        benchmark_clean = benchmark_array[:min_length]

        if len(strategy_clean) < 3:
            return results

        # 基于正态性选择合适的检验方法
        is_normal = self._is_data_normal(normality_results)

        if is_normal:
            # 配对t检验
            try:
                statistic, p_value = stats.ttest_rel(strategy_clean, benchmark_clean)
                results['paired_t_test'] = StatisticalTestResult(
                    test_type=StatisticalTestType.PAIRED_T_TEST,
                    statistic=statistic,
                    p_value=p_value,
                    interpretation="配对t检验：H0=策略收益与基准收益无显著差异"
                )
            except Exception as e:
                logger.warning(f"配对t检验失败: {e}")
        else:
            # Wilcoxon符号秩检验（非参数）
            try:
                statistic, p_value = stats.wilcoxon(strategy_clean, benchmark_clean)
                results['wilcoxon'] = StatisticalTestResult(
                    test_type=StatisticalTestType.WILCOXON,
                    statistic=statistic,
                    p_value=p_value,
                    interpretation="Wilcoxon符号秩检验：H0=策略收益与基准收益分布无显著差异"
                )
            except Exception as e:
                logger.warning(f"Wilcoxon检验失败: {e}")

        return results

    def _compare_with_control(self,
                             strategy_returns: List[float],
                             control_returns: List[float],
                             normality_results: Dict) -> Dict[str, StatisticalTestResult]:
        """与对照组进行比较检验"""
        results = {}

        strategy_array = np.array(strategy_returns)
        control_array = np.array(control_returns)

        if len(strategy_array) < 3 or len(control_array) < 3:
            return results

        is_normal = self._is_data_normal(normality_results)

        if is_normal:
            # 独立样本t检验
            try:
                statistic, p_value = stats.ttest_ind(strategy_array, control_array)
                results['independent_t_test'] = StatisticalTestResult(
                    test_type=StatisticalTestType.T_TEST,
                    statistic=statistic,
                    p_value=p_value,
                    interpretation="独立样本t检验：H0=策略与对照组收益均值无显著差异"
                )
            except Exception as e:
                logger.warning(f"独立样本t检验失败: {e}")
        else:
            # Mann-Whitney U检验（非参数）
            try:
                statistic, p_value = stats.mannwhitneyu(strategy_array, control_array)
                results['mann_whitney'] = StatisticalTestResult(
                    test_type=StatisticalTestType.MANN_WHITNEY,
                    statistic=statistic,
                    p_value=p_value,
                    interpretation="Mann-Whitney U检验：H0=策略与对照组收益分布无显著差异"
                )
            except Exception as e:
                logger.warning(f"Mann-Whitney U检验失败: {e}")

        return results

    def _single_sample_test(self,
                           returns_data: List[float],
                           normality_results: Dict) -> Dict[str, StatisticalTestResult]:
        """单样本检验（检验是否显著异于零）"""
        results = {}

        data_array = np.array(returns_data)
        data_clean = data_array[~np.isnan(data_array)]

        if len(data_clean) < 3:
            return results

        is_normal = self._is_data_normal(normality_results)

        if is_normal:
            # 单样本t检验
            try:
                statistic, p_value = stats.ttest_1samp(data_clean, 0)
                results['one_sample_t'] = StatisticalTestResult(
                    test_type=StatisticalTestType.T_TEST,
                    statistic=statistic,
                    p_value=p_value,
                    interpretation="单样本t检验：H0=策略平均收益等于零"
                )
            except Exception as e:
                logger.warning(f"单样本t检验失败: {e}")
        else:
            # Wilcoxon符号秩检验
            try:
                statistic, p_value = stats.wilcoxon(data_clean)
                results['one_sample_wilcoxon'] = StatisticalTestResult(
                    test_type=StatisticalTestType.WILCOXON,
                    statistic=statistic,
                    p_value=p_value,
                    interpretation="单样本Wilcoxon检验：H0=策略收益中位数等于零"
                )
            except Exception as e:
                logger.warning(f"单样本Wilcoxon检验失败: {e}")

        return results

    def _calculate_effect_sizes(self,
                               strategy_returns: List[float],
                               benchmark_returns: Optional[List[float]] = None,
                               control_returns: Optional[List[float]] = None) -> Dict[str, float]:
        """计算效应量"""
        effect_sizes = {}

        strategy_array = np.array(strategy_returns)
        strategy_mean = np.mean(strategy_array)
        strategy_std = np.std(strategy_array, ddof=1)

        # 与基准比较的效应量
        if benchmark_returns:
            benchmark_array = np.array(benchmark_returns)
            min_length = min(len(strategy_array), len(benchmark_array))

            diff = strategy_array[:min_length] - benchmark_array[:min_length]
            diff_mean = np.mean(diff)
            diff_std = np.std(diff, ddof=1)

            if diff_std > 0:
                effect_sizes['cohen_d_vs_benchmark'] = diff_mean / diff_std

        # 与对照组比较的效应量
        if control_returns:
            control_array = np.array(control_returns)
            control_mean = np.mean(control_array)
            control_std = np.std(control_array, ddof=1)

            if strategy_std > 0 and control_std > 0:
                pooled_std = np.sqrt(((len(strategy_array)-1)*strategy_std**2 +
                                     (len(control_array)-1)*control_std**2) /
                                    (len(strategy_array) + len(control_array) - 2))
                if pooled_std > 0:
                    effect_sizes['cohen_d_vs_control'] = (strategy_mean - control_mean) / pooled_std

        # 单样本效应量（与零比较）
        if strategy_std > 0:
            effect_sizes['cohen_d_vs_zero'] = strategy_mean / strategy_std

        return effect_sizes

    def _correct_multiple_testing(self, significance_tests: Dict) -> Dict[str, Any]:
        """多重假设检验校正"""
        correction_results = {}

        # 收集所有p值
        p_values = []
        test_names = []

        for test_group, tests in significance_tests.items():
            for test_name, result in tests.items():
                if hasattr(result, 'p_value'):
                    p_values.append(result.p_value)
                    test_names.append(f"{test_group}_{test_name}")

        if len(p_values) > 1:
            # Bonferroni校正
            try:
                bonferroni_corrected = multipletests(p_values, method='bonferroni')
                correction_results['bonferroni'] = {
                    'corrected_p_values': bonferroni_corrected[1].tolist(),
                    'rejected_hypotheses': bonferroni_corrected[0].tolist(),
                    'test_names': test_names
                }
            except Exception as e:
                logger.warning(f"Bonferroni校正失败: {e}")

            # FDR校正（Benjamini-Hochberg）
            try:
                fdr_corrected = multipletests(p_values, method='fdr_bh')
                correction_results['fdr'] = {
                    'corrected_p_values': fdr_corrected[1].tolist(),
                    'rejected_hypotheses': fdr_corrected[0].tolist(),
                    'test_names': test_names
                }
            except Exception as e:
                logger.warning(f"FDR校正失败: {e}")

        return correction_results

    def _test_feature_significance(self,
                                  pattern_features: List[Dict[str, Any]],
                                  success_outcomes: List[bool],
                                  feature_name: str) -> Optional[StatisticalTestResult]:
        """检验特征与成功率的关联显著性"""
        try:
            # 提取特征值
            feature_values = []
            outcome_values = []

            for i, features in enumerate(pattern_features):
                if feature_name in features and i < len(success_outcomes):
                    value = features[feature_name]
                    if isinstance(value, (int, float)) and not pd.isna(value):
                        feature_values.append(value)
                        outcome_values.append(success_outcomes[i])

            if len(feature_values) < 5:  # 最小样本量要求
                return None

            # 将成功结果分组
            success_group = [fv for fv, ov in zip(feature_values, outcome_values) if ov]
            failure_group = [fv for fv, ov in zip(feature_values, outcome_values) if not ov]

            if len(success_group) < 2 or len(failure_group) < 2:
                return None

            # 进行Mann-Whitney U检验（非参数，适用于大多数金融数据）
            statistic, p_value = stats.mannwhitneyu(success_group, failure_group,
                                                   alternative='two-sided')

            return StatisticalTestResult(
                test_type=StatisticalTestType.MANN_WHITNEY,
                statistic=statistic,
                p_value=p_value,
                interpretation=f"特征{feature_name}与成功率关联检验：H0=成功组与失败组该特征分布无显著差异"
            )

        except Exception as e:
            logger.warning(f"特征{feature_name}显著性检验失败: {e}")
            return None

    def _test_pattern_associations(self,
                                  pattern_features: List[Dict[str, Any]],
                                  success_outcomes: List[bool]) -> Dict[str, Any]:
        """测试模式关联性"""
        association_results = {}

        try:
            # 计算总体成功率
            overall_success_rate = np.mean(success_outcomes)

            # 卡方检验
            success_count = sum(success_outcomes)
            failure_count = len(success_outcomes) - success_count

            if success_count > 5 and failure_count > 5:
                # 构建列联表
                observed = np.array([[success_count, failure_count]])
                expected_success = len(success_outcomes) * 0.5  # 假设随机成功率50%
                expected = np.array([[expected_success, len(success_outcomes) - expected_success]])

                try:
                    chi2_stat, p_value = stats.chisquare(observed.flatten(), expected.flatten())
                    association_results['chi_square_goodness_of_fit'] = StatisticalTestResult(
                        test_type=StatisticalTestType.CHI_SQUARE,
                        statistic=chi2_stat,
                        p_value=p_value,
                        interpretation="卡方拟合优度检验：H0=成功率等于50%（随机水平）"
                    )
                except Exception as e:
                    logger.warning(f"卡方检验失败: {e}")

            # 二项检验
            try:
                p_value = stats.binom_test(success_count, len(success_outcomes), 0.5)
                association_results['binomial_test'] = StatisticalTestResult(
                    test_type=StatisticalTestType.T_TEST,  # 使用已定义的枚举
                    statistic=success_count / len(success_outcomes),
                    p_value=p_value,
                    interpretation="二项检验：H0=成功率等于50%"
                )
            except Exception as e:
                logger.warning(f"二项检验失败: {e}")

        except Exception as e:
            logger.error(f"模式关联检验失败: {e}")

        return association_results

    def _perform_cross_validation(self,
                                 pattern_features: List[Dict[str, Any]],
                                 success_outcomes: List[bool]) -> Dict[str, Any]:
        """执行交叉验证分析"""
        cv_results = {}

        try:
            if len(pattern_features) < 10:  # 最小样本量要求
                cv_results['error'] = "样本量不足以进行交叉验证"
                return cv_results

            # 时间序列分割（适用于金融数据）
            tscv = TimeSeriesSplit(n_splits=min(5, len(pattern_features) // 5))

            # 简单的成功率预测模型
            X = np.arange(len(pattern_features)).reshape(-1, 1)  # 简化特征
            y = np.array(success_outcomes)

            cv_scores = []
            fold_results = []

            for fold, (train_index, test_index) in enumerate(tscv.split(X)):
                train_y, test_y = y[train_index], y[test_index]

                # 简单基线模型：使用训练集的平均成功率
                train_success_rate = np.mean(train_y)
                predicted = np.full(len(test_y), train_success_rate > 0.5)

                # 计算准确率
                accuracy = accuracy_score(test_y, predicted)
                cv_scores.append(accuracy)

                fold_results.append({
                    'fold': fold + 1,
                    'train_size': len(train_index),
                    'test_size': len(test_index),
                    'train_success_rate': train_success_rate,
                    'test_accuracy': accuracy
                })

            cv_results['cross_validation_scores'] = cv_scores
            cv_results['mean_cv_score'] = np.mean(cv_scores)
            cv_results['std_cv_score'] = np.std(cv_scores)
            cv_results['fold_details'] = fold_results

            # 稳定性分析
            score_cv = np.std(cv_scores) / np.mean(cv_scores) if np.mean(cv_scores) > 0 else float('inf')
            cv_results['stability_coefficient'] = score_cv
            cv_results['stability_assessment'] = 'STABLE' if score_cv < 0.2 else ('MODERATE' if score_cv < 0.5 else 'UNSTABLE')

        except Exception as e:
            logger.error(f"交叉验证失败: {e}")
            cv_results['error'] = str(e)

        return cv_results

    def _analyze_generalization_capability(self,
                                          pattern_features: List[Dict[str, Any]],
                                          success_outcomes: List[bool]) -> Dict[str, Any]:
        """分析泛化能力"""
        generalization_results = {}

        try:
            # 样本分布分析
            feature_values = {}
            for features in pattern_features:
                for key, value in features.items():
                    if isinstance(value, (int, float)) and not pd.isna(value):
                        if key not in feature_values:
                            feature_values[key] = []
                        feature_values[key].append(value)

            # 特征分布稳定性
            feature_stability = {}
            for feature_name, values in feature_values.items():
                if len(values) > 5:
                    cv = np.std(values) / abs(np.mean(values)) if np.mean(values) != 0 else float('inf')
                    feature_stability[feature_name] = {
                        'coefficient_of_variation': cv,
                        'stability': 'HIGH' if cv < 0.3 else ('MEDIUM' if cv < 0.7 else 'LOW')
                    }

            generalization_results['feature_stability'] = feature_stability

            # 成功率时间稳定性（如果有时间信息）
            if len(success_outcomes) > 10:
                # 分段分析
                segment_size = len(success_outcomes) // 3
                segments = [
                    success_outcomes[:segment_size],
                    success_outcomes[segment_size:2*segment_size],
                    success_outcomes[2*segment_size:]
                ]

                segment_success_rates = [np.mean(seg) for seg in segments if len(seg) > 0]

                if len(segment_success_rates) > 1:
                    rate_cv = np.std(segment_success_rates) / np.mean(segment_success_rates) \
                             if np.mean(segment_success_rates) > 0 else float('inf')

                    generalization_results['temporal_stability'] = {
                        'segment_success_rates': segment_success_rates,
                        'coefficient_of_variation': rate_cv,
                        'assessment': 'STABLE' if rate_cv < 0.3 else ('MODERATE' if rate_cv < 0.6 else 'UNSTABLE')
                    }

        except Exception as e:
            logger.error(f"泛化能力分析失败: {e}")
            generalization_results['error'] = str(e)

        return generalization_results

    def _is_data_normal(self, normality_results: Dict) -> bool:
        """判断数据是否符合正态分布"""
        if not normality_results:
            return False

        # 如果多个正态性检验中有任何一个显著，认为非正态
        for test_name, result in normality_results.items():
            if hasattr(result, 'p_value') and result.p_value < self.alpha_level:
                return False

        return True

    def _generate_overall_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成综合评估"""
        assessment = {
            'statistical_significance_level': 'UNKNOWN',
            'sample_adequacy': 'UNKNOWN',
            'effect_size_magnitude': 'UNKNOWN',
            'generalization_confidence': 'UNKNOWN',
            'overall_recommendation': 'UNKNOWN',
            'key_concerns': [],
            'strengths': []
        }

        try:
            # 样本充足性评估
            if 'sample_size_analysis' in results and results['sample_size_analysis']:
                sample_analysis = results['sample_size_analysis']
                assessment['sample_adequacy'] = sample_analysis.adequacy_status

                if sample_analysis.adequacy_status == 'INADEQUATE':
                    assessment['key_concerns'].append(f"样本量不足：当前{sample_analysis.current_sample_size}，建议至少{sample_analysis.recommended_min_size}")
                elif sample_analysis.adequacy_status == 'ADEQUATE':
                    assessment['strengths'].append(f"样本量充足：{sample_analysis.current_sample_size}个观测值")

            # 统计显著性评估
            significant_tests = 0
            total_tests = 0

            if 'significance_tests' in results:
                for test_group, tests in results['significance_tests'].items():
                    for test_name, result in tests.items():
                        if hasattr(result, 'p_value'):
                            total_tests += 1
                            if result.is_significant(self.alpha_level):
                                significant_tests += 1

            if total_tests > 0:
                significance_rate = significant_tests / total_tests
                if significance_rate >= 0.7:
                    assessment['statistical_significance_level'] = 'HIGH'
                    assessment['strengths'].append(f"{significant_tests}/{total_tests}项检验达到统计显著性")
                elif significance_rate >= 0.4:
                    assessment['statistical_significance_level'] = 'MODERATE'
                else:
                    assessment['statistical_significance_level'] = 'LOW'
                    assessment['key_concerns'].append(f"统计显著性不足：仅{significant_tests}/{total_tests}项检验显著")

            # 效应量评估
            if 'effect_size_analysis' in results and results['effect_size_analysis']:
                effect_sizes = results['effect_size_analysis']
                max_effect_size = max([abs(es) for es in effect_sizes.values() if isinstance(es, (int, float))], default=0)

                if max_effect_size >= 0.8:
                    assessment['effect_size_magnitude'] = 'LARGE'
                    assessment['strengths'].append(f"效应量大：最大效应量{max_effect_size:.3f}")
                elif max_effect_size >= 0.5:
                    assessment['effect_size_magnitude'] = 'MEDIUM'
                elif max_effect_size >= 0.2:
                    assessment['effect_size_magnitude'] = 'SMALL'
                else:
                    assessment['effect_size_magnitude'] = 'NEGLIGIBLE'
                    assessment['key_concerns'].append(f"效应量可忽略：最大效应量{max_effect_size:.3f}")

            # 泛化能力评估
            if 'cross_validation_results' in results and 'stability_assessment' in results['cross_validation_results']:
                cv_stability = results['cross_validation_results']['stability_assessment']
                if cv_stability == 'STABLE':
                    assessment['generalization_confidence'] = 'HIGH'
                    assessment['strengths'].append("交叉验证结果稳定")
                elif cv_stability == 'MODERATE':
                    assessment['generalization_confidence'] = 'MODERATE'
                else:
                    assessment['generalization_confidence'] = 'LOW'
                    assessment['key_concerns'].append("交叉验证结果不稳定，泛化能力存疑")

            # 综合建议
            high_quality_indicators = [
                assessment['sample_adequacy'] == 'ADEQUATE',
                assessment['statistical_significance_level'] == 'HIGH',
                assessment['effect_size_magnitude'] in ['LARGE', 'MEDIUM'],
                assessment['generalization_confidence'] == 'HIGH'
            ]

            quality_score = sum(high_quality_indicators)

            if quality_score >= 3:
                assessment['overall_recommendation'] = 'ACCEPT'
            elif quality_score >= 2:
                assessment['overall_recommendation'] = 'CONDITIONAL_ACCEPT'
            else:
                assessment['overall_recommendation'] = 'REJECT'

        except Exception as e:
            logger.error(f"生成综合评估失败: {e}")
            assessment['error'] = str(e)

        return assessment

# 创建全局实例
_validator = FinancialStatisticalValidator()

def get_statistical_validator() -> FinancialStatisticalValidator:
    """获取统计验证器实例"""
    return _validator