from utils.container import container
#!/usr/bin/env python
from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType
# -*- coding: utf-8 -*-  # TODO: 将魔法数字提取到配置中

"""
均线多空指标(BIAS_Bias)

(收盘价-MA)/MA×100%
"""

import pandas as pd
from typing import Dict, Any
import numpy as np
from typing import List

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


class BiasBias(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    均线多空指标(BIAS_Bias) (BIAS_Bias)

    分类：趋势类指标
    描述：(收盘价-MA)/MA×100%
    """

    def __init__(self, name: str = "BIAS", description: str = "均线多空指标",
                 period: int = 14, periods: List[int] = None, **kwargs):  # TODO: 将魔法数字提取到配置中
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化均线多空指标(BIAS_Bias)指标
        """
        super().__init__()
        self.name = name
        self.description = description
        self.periods = periods if periods is not None else [6, 12, 24]  # BIAS常用周期  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        self.indicator_type = "BIAS"
        self.REQUIRED_COLUMNS = ['close']  # 添加必需列定义
        
    @property
    def minimum_periods(self) -> int:
        """返回计算指标所需的最小周期数"""
        return max(self.periods) if self.periods else 6  # TODO: 将魔法数字提取到配置中

    def set_parameters_Bias_Bias_Bias_bias(self, period: int = 14, **kwargs):  # TODO: 将魔法数字提取到配置中
        """
        设置BIAS指标的参数
        """
        self.periods = kwargs.get('periods', [period])

    def _validate_dataframe_bias(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        验证Data_frame是否包含所需的列
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame缺少必要的列: {', '.join(missing_columns)}")
    
    def _calculate_bias(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算均线多空指标(BIAS_Bias)指标
        """
        if data.empty:

            return pd.DataFrame()

            
        self._validate_dataframe_bias(data, ['close'])
        
        # 创建一个临时的DataFrame来存储新计算的列
        result_df = pd.DataFrame(index=data.index)
        
        # 计算所有周期的BIAS
        for p in self.periods:
            ma = data['close'].rolling(window=p, min_periods=1).mean()
            result_df[f'BIAS_Bias{p}'] = (data['close'] - ma) / ma * 100
        
        # 为主周期创建 'BIAS_Bias' 和 'BIAS_MA' 列，以供形态识别使用
        if self.periods:
            main_period = self.periods[0]
            main_bias_col = f'BIAS_Bias{main_period}'
            if main_bias_col in result_df:
                result_df['BIAS_Bias'] = result_df[main_bias_col]
                result_df['BIAS_MA'] = result_df['BIAS_Bias'].rolling(window=main_period, min_periods=1).mean()

        # 只返回计算出的指标列，不包含原始数据列
        
        # 添加形态识别和信号生成
        result_df = self.add_pattern_detection(result_df)
        result_df = self.add_signal_generation(result_df)

        # 重写信号生成逻辑（BIAS指标特定逻辑）
        result_df = self._apply_bias_signal_logic(result_df)

        return result_df

    def _apply_bias_signal_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用BIAS指标特定的信号生成逻辑
        基于乖离率的正负值生成信号
        """
        try:
            # 获取BIAS值
            if 'BIAS_Bias' not in df.columns:
                # 如果没有BIAS值，使用默认信号
                return df

            bias_value = df['BIAS_Bias']

            # BIAS信号生成逻辑：
            # BUY: BIAS值为正且上升（价格高于均线且乖离增大）
            # SELL: BIAS值为负且下降（价格低于均线且乖离增大）
            # HOLD: BIAS值接近零或趋势不明确

            # 计算BIAS的变化
            bias_positive = bias_value > 0
            bias_negative = bias_value < 0
            bias_rising = bias_value > bias_value.shift(1)
            bias_falling = bias_value < bias_value.shift(1)

            # 生成信号
            df.loc[:, 'buy_signal'] = bias_positive & bias_rising
            df.loc[:, 'sell_signal'] = bias_negative & bias_falling
            df.loc[:, 'hold_signal'] = ~(df['buy_signal'] | df['sell_signal'])

            # 确保信号类型为布尔值
            df['buy_signal'] = df['buy_signal'].astype(bool)
            df['sell_signal'] = df['sell_signal'].astype(bool)
            df['hold_signal'] = df['hold_signal'].astype(bool)

        except Exception as e:
            logger.warning(f"BIAS信号生成失败: {e}")
            # 如果出错，使用默认信号
            df.loc[:, 'buy_signal'] = False
            df.loc[:, 'sell_signal'] = False
            df.loc[:, 'hold_signal'] = True

        return df

    def get_patterns_Bias(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        识别所有已注册的BIAS相关形态
        """
        # 首先，调用calculate来获取所有需要的列
        calculated_data = self._calculate_bias(data)

        # 验证必要的列是否存在
        required_cols = ['BIAS_Bias', 'BIAS_MA']
        if not all(col in calculated_data.columns for col in required_cols):
             logger.warning(f"BIAS指标在形态识别时缺少必要的计算列: {required_cols}")
             # 返回一个空的DataFrame，但保留索引
             return pd.DataFrame(index=data.index)

        # 实现BIAS形态识别逻辑
        bias_values = calculated_data['BIAS_Bias']

        # 创建形态识别结果DataFrame，只包含形态列
        patterns_df = pd.DataFrame(index=data.index)

        # 1. BIAS极值形态
        patterns_df['BIAS_EXTREME_HIGH'] = bias_values > 15.0  # TODO: 将魔法数字提取到配置中
        patterns_df['BIAS_EXTREME_LOW'] = bias_values < -15.0  # TODO: 将魔法数字提取到配置中

        # 2. BIAS中度偏离形态
        patterns_df['BIAS_MODERATE_HIGH'] = (bias_values > 5.0) & (bias_values <= 15.0)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        patterns_df['BIAS_MODERATE_LOW'] = (bias_values < -5.0) & (bias_values >= -15.0)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 3. BIAS中性形态  # TODO: 将魔法数字提取到配置中
        patterns_df['BIAS_NEUTRAL'] = (bias_values >= -5.0) & (bias_values <= 5.0)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 4. BIAS背离形态（简化版本）  # TODO: 将魔法数字提取到配置中
        if len(bias_values) >= 10:
            # 计算价格和BIAS的相关性来检测背离
            # 使用原始数据中的close列
            price_trend = data['close'].diff(5)  # 5日价格变化  # TODO: 将魔法数字提取到配置中
            bias_trend = bias_values.diff(5)  # 5日BIAS变化  # TODO: 将魔法数字提取到配置中

            # 背离：价格上涨但BIAS下降，或价格下跌但BIAS上升
            bullish_divergence = (price_trend < 0) & (bias_trend > 0)
            bearish_divergence = (price_trend > 0) & (bias_trend < 0)

            patterns_df['BIAS_BULLISH_DIVERGENCE'] = bullish_divergence
            patterns_df['BIAS_BEARISH_DIVERGENCE'] = bearish_divergence
            patterns_df['BIAS_DIVERGENCE'] = bullish_divergence | bearish_divergence
        else:
            patterns_df['BIAS_BULLISH_DIVERGENCE'] = False
            patterns_df['BIAS_BEARISH_DIVERGENCE'] = False
            patterns_df['BIAS_DIVERGENCE'] = False

        # 确保所有列都是布尔类型，填充NaN为False
        for col in patterns_df.columns:
            patterns_df[col] = patterns_df[col].fillna(False).astype(bool)

        return patterns_df

    def calculate_raw_score_Bias(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算BIAS指标的原始评分 (0-100分)

        评分逻辑：
        - BIAS值越接近0，评分越接近50（中性）
        - BIAS值为正且较大时，评分偏高（超买）
        - BIAS值为负且较大时，评分偏低（超卖）
        """
        # 首先计算指标值
        calculated_data = self._calculate_bias(data)

        if 'BIAS_Bias' not in calculated_data.columns:
            return pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中

        bias_values = calculated_data['BIAS_Bias']

        # 计算评分
        # BIAS在-10到+10之间为正常范围，对应40-60分
        # BIAS超过+10为超买，对应60-100分
        # BIAS低于-10为超卖，对应0-40分
        scores = pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中

        # 处理有效值
        valid_mask = bias_values.notna()
        valid_bias = bias_values[valid_mask]

        if len(valid_bias) > 0:
            # 标准化BIAS值到评分
            # 使用sigmoid函数进行平滑转换
            normalized_bias = valid_bias / 10.0  # 将BIAS值标准化
            sigmoid_scores = 50 + 40 * (2 / (1 + pd.Series(np.exp(-normalized_bias), index=valid_bias.index)) - 1)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            scores[valid_mask] = sigmoid_scores.clip(0, 100)

        return scores

    def calculate_confidence_Bias(self, score: pd.Series, patterns: List[str], signals: dict) -> float:
        """
        计算BIAS指标的置信度

        Args:
            score: 得分序列
            patterns: 检测到的形态列表
            signals: 生成的信号字典

        Returns:
            float: 置信度分数 (0-1)
        """
        if score.empty:
            return 0.5  # TODO: 将魔法数字提取到配置中

        # 基础置信度
        confidence = 0.5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 1. 基于评分的置信度
        last_score = score.iloc[-1]

        # 极端评分（超买/超卖）置信度较高
        if last_score > 70 or last_score < 30:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            confidence += 0.2
        # 中性评分置信度中等
        elif 40 <= last_score <= 60:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            confidence += 0.1
        else:
            confidence += 0.15  # TODO: 将魔法数字提取到配置中

        # 2. 基于形态的置信度
        if isinstance(patterns, (list, pd.DataFrame)):
            if isinstance(patterns, pd.DataFrame):
                # 统计最近几个周期的形态数量
                try:
                    # 只统计数值列的形态
                    numeric_cols = patterns.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        recent_data = patterns[numeric_cols].iloc[-5:] if len(patterns) >= 5 else patterns[numeric_cols]  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        recent_patterns = recent_data.sum().sum()
                    else:
                        recent_patterns = 0
                except:
                    recent_patterns = 0
            else:
                recent_patterns = len(patterns)

            if recent_patterns > 0:
                confidence += min(recent_patterns * 0.05, 0.2)  # TODO: 将魔法数字提取到配置中

        # 3. 基于评分稳定性的置信度  # TODO: 将魔法数字提取到配置中
        if len(score) >= 5:  # TODO: 将魔法数字提取到配置中
            recent_scores = score.iloc[-5:]  # TODO: 将魔法数字提取到配置中
            score_stability = 1.0 - (recent_scores.std() / 50.0)  # 标准差越小，稳定性越高  # TODO: 将魔法数字提取到配置中
            confidence += score_stability * 0.1

        return min(confidence, 1.0)

    def register_patterns_Bias(self):
        """
        注册BIAS指标的技术形态
        """
        # 注册BIAS极值形态
        self.register_pattern_to_registry(
            pattern_id="BIAS_EXTREME_HIGH",
            display_name="BIAS极高值",
            description="BIAS值超过+15%，表示严重超买",  # TODO: 将魔法数字提取到配置中
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="BIAS_EXTREME_LOW",
            display_name="BIAS极低值",
            description="BIAS值低于-15%，表示严重超卖",  # TODO: 将魔法数字提取到配置中
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE"
        )

        # 注册BIAS中度偏离形态
        self.register_pattern_to_registry(
            pattern_id="BIAS_MODERATE_HIGH",
            display_name="BIAS中度偏高",
            description="BIAS值在+5%到+15%之间，表示轻度超买",  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-10.0,
            polarity="NEGATIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="BIAS_MODERATE_LOW",
            display_name="BIAS中度偏低",
            description="BIAS值在-15%到-5%之间，表示轻度超卖",  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=10.0,
            polarity="POSITIVE"
        )

        # 注册BIAS背离形态
        self.register_pattern_to_registry(
            pattern_id="BIAS_BULLISH_DIVERGENCE",
            display_name="BIAS底背离",
            description="价格创新低但BIAS未创新低，表明下跌动能减弱",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=15.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="BIAS_BEARISH_DIVERGENCE",
            display_name="BIAS顶背离",
            description="价格创新高但BIAS未创新高，表明上涨动能减弱",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-15.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE"
        )

        # 注册BIAS中性形态
        self.register_pattern_to_registry(
            pattern_id="BIAS_NEUTRAL",
            display_name="BIAS中性",
            description="BIAS值在-5%到+5%之间，表示价格相对均衡",  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            pattern_type="NEUTRAL",
            default_strength="WEAK",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

    def get_pattern_info_Bias(self, pattern_id: str) -> dict:
        """
        获取指定形态的详细信息

        Args:
            pattern_id: 形态ID

        Returns:
            dict: 形态信息字典
        """
        pattern_info_map = {
            'BIAS_EXTREME_HIGH': {
                'name': 'BIAS极高值',
                'description': 'BIAS值超过+15%，表示严重超买',  # TODO: 将魔法数字提取到配置中
                'strength': 'strong',
                'type': 'bearish'
            },
            'BIAS_EXTREME_LOW': {
                'name': 'BIAS极低值',
                'description': 'BIAS值低于-15%，表示严重超卖',  # TODO: 将魔法数字提取到配置中
                'strength': 'strong',
                'type': 'bullish'
            },
            'BIAS_DIVERGENCE': {
                'name': 'BIAS背离',
                'description': '价格与BIAS指标出现背离',
                'strength': 'medium',
                'type': 'neutral'
            },
            'BIAS_MODERATE_HIGH': {
                'name': 'BIAS中度偏高',
                'description': 'BIAS值在+5%到+15%之间，表示轻度超买',  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                'strength': 'medium',
                'type': 'bearish'
            },
            'BIAS_MODERATE_LOW': {
                'name': 'BIAS中度偏低',
                'description': 'BIAS值在-15%到-5%之间，表示轻度超卖',  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                'strength': 'medium',
                'type': 'bullish'
            },
            'BIAS_NEUTRAL': {
                'name': 'BIAS中性',
                'description': 'BIAS值在-5%到+5%之间，表示价格相对均衡',  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                'strength': 'weak',
                'type': 'neutral'
            }
        }

        return pattern_info_map.get(pattern_id, {
            'name': pattern_id,
            'description': f'BIAS形态: {pattern_id}',
            'strength': 'medium',
            'type': 'neutral'
        })

    def _get_default_parameters_bias(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {'periods': [6, 12, 24]}  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
    
    def set_parameters_Bias_Bias_Bias_bias_duplicate(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数字典
        """
        # 验证参数
        try:
            from utils.indicator_parameter_validator import IndicatorParameterValidator
            validator = IndicatorParameterValidator()
            
            # 合并默认参数和用户参数
            params = self._default_parameters.copy()
            params.update(kwargs)
            
            # 验证参数
            is_valid, errors = validator.validate_indicator_parameters('BIAS_Bias', params)
            if not is_valid:
                from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType
                logger = get_logger(__name__)
                logger.warning(f"BIAS参数验证失败: {'; '.join(errors)}")
                # 使用默认参数
                params = self._default_parameters.copy()
            
            # 设置参数（保持向后兼容）
            for key, value in params.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    
        except Exception:
            # 如果验证失败，静默处理
            pass

    # ==================== 抽象方法实现 ====================

    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """抽象基类要求的计算方法"""
        return self._calculate_bias(data, **kwargs)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """抽象基类要求的评分方法"""
        return self.calculate_raw_score_Bias(data, **kwargs)

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """抽象基类要求的形态方法"""
        return self.get_patterns_Bias(data, **kwargs)

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """抽象基类要求的参数设置方法"""
        return self.set_parameters_Bias_Bias_Bias_bias(**kwargs)

    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """抽象基类要求的置信度计算方法"""
        return self.calculate_confidence_Bias(score, patterns, signals)

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """统一的计算接口"""
        return self._calculate_bias(data, **kwargs)

    # ==================== 兼容性方法 - 真实实现 ====================

    def get_patterns(self, data: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        """真实实现：获取BIAS形态"""
        if data is None or data.empty:
            return pd.DataFrame()

        # 首先计算BIAS指标
        bias_data = self._calculate_bias(data)

        # 创建形态DataFrame
        patterns_df = pd.DataFrame(index=data.index)

        # 为每个周期检测形态
        for period in self.periods:
            bias_col = f'BIAS{period}'
            if bias_col in bias_data.columns:
                bias_values = bias_data[bias_col]

                # 1. 正偏离形态 (BIAS > 3%)  # TODO: 将魔法数字提取到配置中
                patterns_df[f'BIAS{period}_POSITIVE'] = bias_values > 3  # TODO: 将魔法数字提取到配置中

                # 2. 负偏离形态 (BIAS < -3%)  # TODO: 将魔法数字提取到配置中
                patterns_df[f'BIAS{period}_NEGATIVE'] = bias_values < -3  # TODO: 将魔法数字提取到配置中

                # 3. 强正偏离形态 (BIAS > 6%)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                patterns_df[f'BIAS{period}_STRONG_POSITIVE'] = bias_values > 6  # TODO: 将魔法数字提取到配置中

                # 4. 强负偏离形态 (BIAS < -6%)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                patterns_df[f'BIAS{period}_STRONG_NEGATIVE'] = bias_values < -6  # TODO: 将魔法数字提取到配置中

                # 5. 零轴上穿形态  # TODO: 将魔法数字提取到配置中
                patterns_df[f'BIAS{period}_ZERO_CROSS_UP'] = (bias_values > 0) & (bias_values.shift(1) <= 0)

                # 6. 零轴下穿形态  # TODO: 将魔法数字提取到配置中
                patterns_df[f'BIAS{period}_ZERO_CROSS_DOWN'] = (bias_values < 0) & (bias_values.shift(1) >= 0)

                # 7. 收敛形态（BIAS接近0）  # TODO: 将魔法数字提取到配置中
                patterns_df[f'BIAS{period}_CONVERGENCE'] = np.abs(bias_values) < 1

                # 8. 发散形态（BIAS远离0）  # TODO: 将魔法数字提取到配置中
                patterns_df[f'BIAS{period}_DIVERGENCE'] = np.abs(bias_values) > 5  # TODO: 将魔法数字提取到配置中

        # 9. 多周期共振形态  # TODO: 将魔法数字提取到配置中
        if len(self.periods) >= 2:
            # 所有周期都为正
            all_positive = True
            all_negative = True
            for period in self.periods:
                bias_col = f'BIAS{period}'
                if bias_col in bias_data.columns:
                    all_positive &= (bias_data[bias_col] > 0)
                    all_negative &= (bias_data[bias_col] < 0)

            patterns_df['BIAS_ALL_POSITIVE'] = all_positive
            patterns_df['BIAS_ALL_NEGATIVE'] = all_negative

        return patterns_df

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """真实实现：计算BIAS原始评分"""
        if data.empty:
            return pd.Series(dtype=float)

        # 计算BIAS指标
        bias_data = self._calculate_bias(data)

        # 初始化评分
        score = pd.Series(50.0, index=data.index)  # 基础分50分  # TODO: 将魔法数字提取到配置中

        # 为每个周期计算评分
        for period in self.periods:
            bias_col = f'BIAS{period}'
            if bias_col in bias_data.columns:
                bias_values = bias_data[bias_col]

                # 1. 基于BIAS位置的评分
                # 负偏离加分（超卖）
                negative_condition = bias_values < -3  # TODO: 将魔法数字提取到配置中
                score += negative_condition * (10 / len(self.periods))

                # 强负偏离加分
                strong_negative_condition = bias_values < -6  # TODO: 将魔法数字提取到配置中
                score += strong_negative_condition * (15 / len(self.periods))  # TODO: 将魔法数字提取到配置中

                # 正偏离减分（超买）
                positive_condition = bias_values > 3  # TODO: 将魔法数字提取到配置中
                score -= positive_condition * (10 / len(self.periods))

                # 强正偏离减分
                strong_positive_condition = bias_values > 6  # TODO: 将魔法数字提取到配置中
                score -= strong_positive_condition * (15 / len(self.periods))  # TODO: 将魔法数字提取到配置中

                # 2. 基于零轴交叉的评分
                zero_cross_up = (bias_values > 0) & (bias_values.shift(1) <= 0)
                zero_cross_down = (bias_values < 0) & (bias_values.shift(1) >= 0)

                # 零轴上穿加分
                score += zero_cross_up * (8 / len(self.periods))  # TODO: 将魔法数字提取到配置中

                # 零轴下穿减分
                score -= zero_cross_down * (8 / len(self.periods))  # TODO: 将魔法数字提取到配置中

                # 3. 基于BIAS趋势的评分  # TODO: 将魔法数字提取到配置中
                # BIAS上升趋势加分
                bias_rising = bias_values > bias_values.shift(1)
                score += bias_rising * (3 / len(self.periods))  # TODO: 将魔法数字提取到配置中

                # BIAS下降趋势减分
                bias_falling = bias_values < bias_values.shift(1)
                score -= bias_falling * (3 / len(self.periods))  # TODO: 将魔法数字提取到配置中

        # 4. 多周期共振奖励  # TODO: 将魔法数字提取到配置中
        if len(self.periods) >= 2:
            all_negative = True
            all_positive = True
            for period in self.periods:
                bias_col = f'BIAS{period}'
                if bias_col in bias_data.columns:
                    all_negative &= (bias_data[bias_col] < 0)
                    all_positive &= (bias_data[bias_col] > 0)

            # 所有周期负偏离（强烈超卖）
            score += all_negative * 20  # TODO: 将魔法数字提取到配置中

            # 所有周期正偏离（强烈超买）
            score -= all_positive * 20  # TODO: 将魔法数字提取到配置中

        # 限制评分在0-100之间
        return score.clip(0, 100)

    def get_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """真实实现：生成BIAS交易信号"""
        if data.empty:
            return pd.DataFrame()

        # 计算BIAS指标
        bias_data = self._calculate_bias(data)
        result_df = data.copy()

        # 合并BIAS数据
        for col in bias_data.columns:
            result_df[col] = bias_data[col]

        # 初始化信号列
        result_df['bias_signal'] = 0
        result_df['bias_strength'] = 0.0
        result_df['bias_confidence'] = 0.0

        # 为每个周期生成信号
        for period in self.periods:
            bias_col = f'BIAS{period}'
            if bias_col in bias_data.columns:
                bias_values = bias_data[bias_col]

                # 1. 负偏离反弹买入信号
                negative_bounce = (bias_values > -3) & (bias_values.shift(1) <= -3)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                result_df.loc[negative_bounce, 'bias_signal'] = 1
                result_df.loc[negative_bounce, 'bias_strength'] = 0.7  # TODO: 将魔法数字提取到配置中
                result_df.loc[negative_bounce, 'bias_confidence'] = 0.8  # TODO: 将魔法数字提取到配置中

                # 2. 正偏离回落卖出信号
                positive_fall = (bias_values < 3) & (bias_values.shift(1) >= 3)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                result_df.loc[positive_fall, 'bias_signal'] = -1
                result_df.loc[positive_fall, 'bias_strength'] = 0.7  # TODO: 将魔法数字提取到配置中
                result_df.loc[positive_fall, 'bias_confidence'] = 0.8  # TODO: 将魔法数字提取到配置中

                # 3. 零轴突破信号  # TODO: 将魔法数字提取到配置中
                zero_cross_up = (bias_values > 0) & (bias_values.shift(1) <= 0)
                result_df.loc[zero_cross_up, 'bias_signal'] = 1
                result_df.loc[zero_cross_up, 'bias_strength'] = 0.5  # TODO: 将魔法数字提取到配置中
                result_df.loc[zero_cross_up, 'bias_confidence'] = 0.6  # TODO: 将魔法数字提取到配置中

                zero_cross_down = (bias_values < 0) & (bias_values.shift(1) >= 0)
                result_df.loc[zero_cross_down, 'bias_signal'] = -1
                result_df.loc[zero_cross_down, 'bias_strength'] = 0.5  # TODO: 将魔法数字提取到配置中
                result_df.loc[zero_cross_down, 'bias_confidence'] = 0.6  # TODO: 将魔法数字提取到配置中

                # 4. 强偏离信号  # TODO: 将魔法数字提取到配置中
                strong_negative = bias_values < -6  # TODO: 将魔法数字提取到配置中
                result_df.loc[strong_negative, 'bias_signal'] = 1
                result_df.loc[strong_negative, 'bias_strength'] = 0.9  # TODO: 将魔法数字提取到配置中
                result_df.loc[strong_negative, 'bias_confidence'] = 0.9  # TODO: 将魔法数字提取到配置中

                strong_positive = bias_values > 6  # TODO: 将魔法数字提取到配置中
                result_df.loc[strong_positive, 'bias_signal'] = -1
                result_df.loc[strong_positive, 'bias_strength'] = 0.9  # TODO: 将魔法数字提取到配置中
                result_df.loc[strong_positive, 'bias_confidence'] = 0.9  # TODO: 将魔法数字提取到配置中

        return result_df

    def calculate_score(self, data: pd.DataFrame, **kwargs) -> dict:
        """真实实现：计算BIAS综合评分"""
        if data.empty:
            return {'score': 50.0, 'confidence': 0.0, 'signals': {}}  # TODO: 将魔法数字提取到配置中

        # 计算原始评分
        raw_score = self.calculate_raw_score(data, **kwargs)

        # 获取形态
        patterns = self.get_patterns(data, **kwargs)

        # 计算最终评分
        final_score = raw_score.iloc[-1] if not raw_score.empty else 50.0  # TODO: 将魔法数字提取到配置中

        # 基于形态调整评分
        if not patterns.empty:
            latest_patterns = patterns.iloc[-1]

            # 正面形态加分
            for period in self.periods:
                if latest_patterns.get(f'BIAS{period}_NEGATIVE', False):
                    final_score += 10
                if latest_patterns.get(f'BIAS{period}_STRONG_NEGATIVE', False):
                    final_score += 15  # TODO: 将魔法数字提取到配置中
                if latest_patterns.get(f'BIAS{period}_ZERO_CROSS_UP', False):
                    final_score += 8  # TODO: 将魔法数字提取到配置中

                # 负面形态减分
                if latest_patterns.get(f'BIAS{period}_POSITIVE', False):
                    final_score -= 10
                if latest_patterns.get(f'BIAS{period}_STRONG_POSITIVE', False):
                    final_score -= 15  # TODO: 将魔法数字提取到配置中
                if latest_patterns.get(f'BIAS{period}_ZERO_CROSS_DOWN', False):
                    final_score -= 8  # TODO: 将魔法数字提取到配置中

            # 多周期共振
            if latest_patterns.get('BIAS_ALL_NEGATIVE', False):
                final_score += 20  # TODO: 将魔法数字提取到配置中
            if latest_patterns.get('BIAS_ALL_POSITIVE', False):
                final_score -= 20  # TODO: 将魔法数字提取到配置中

        # 计算置信度
        bias_data = self._calculate_bias(data)

        # 基于BIAS值的分布计算置信度
        confidence = 0.5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        for period in self.periods:
            bias_col = f'BIAS{period}'
            if bias_col in bias_data.columns:
                bias_value = bias_data[bias_col].iloc[-1] if len(bias_data[bias_col]) > 0 else 0
                bias_abs = abs(bias_value)

                if bias_abs > 6:  # TODO: 将魔法数字提取到配置中
                    confidence += 0.3  # TODO: 将魔法数字提取到配置中 / len(self.periods)  # TODO: 将魔法数字提取到配置中
                elif bias_abs > 3:  # TODO: 将魔法数字提取到配置中
                    confidence += 0.2 / len(self.periods)
                elif bias_abs > 1:
                    confidence += 0.1 / len(self.periods)

        # 限制评分范围
        final_score = max(0, min(100, final_score))
        confidence = max(0.0, min(1.0, confidence))

        return {
            'score': final_score,
            'confidence': confidence,
            'signals': {
                'bias_values': {f'BIAS{period}': bias_data.get(f'BIAS{period}', pd.Series([0])).iloc[-1]
                               if f'BIAS{period}' in bias_data.columns else 0 for period in self.periods},
                'trend': 'up' if final_score > 60 else 'down' if final_score < 40 else 'neutral'  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            }
        }

    def set_parameters(self, **kwargs):
        """真实实现：设置BIAS参数"""
        # 验证并设置periods参数
        if 'periods' in kwargs:
            periods = kwargs['periods']
            if isinstance(periods, list) and all(isinstance(p, int) and 1 <= p <= 100 for p in periods):
                self.periods = periods
            else:
                logger.warning(f"无效的periods参数: {periods}, 保持原值")

        # 验证并设置单个period参数
        if 'period' in kwargs:
            period = kwargs['period']
            if isinstance(period, int) and 1 <= period <= 100:
                if period not in self.periods:
                    self.periods.append(period)
            else:
                logger.warning(f"无效的period参数: {period}, 保持原值")

        # 记录参数变更
        logger.info(f"BIAS参数已更新: periods={self.periods}")

    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """真实实现：生成BIAS交易信号"""
        return self.get_signals(data, **kwargs)

    def compute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """真实实现：计算BIAS指标"""
        return self._calculate_bias(data, **kwargs)

    def calculate_confidence_Bias(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """真实实现：计算BIAS置信度"""
        if score.empty:
            return 0.3  # TODO: 将魔法数字提取到配置中

        # 基础置信度
        confidence = 0.5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 基于评分的置信度调整
        latest_score = score.iloc[-1] if not score.empty else 50.0  # TODO: 将魔法数字提取到配置中

        # 极端评分提高置信度
        if latest_score > 80 or latest_score < 20:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            confidence += 0.3  # TODO: 将魔法数字提取到配置中
        elif latest_score > 70 or latest_score < 30:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            confidence += 0.2
        elif latest_score > 60 or latest_score < 40:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            confidence += 0.1

        # 基于形态的置信度调整
        if not patterns.empty:
            latest_patterns = patterns.iloc[-1]

            # 强势形态提高置信度
            for period in self.periods:
                if latest_patterns.get(f'BIAS{period}_STRONG_NEGATIVE', False):
                    confidence += 0.15  # TODO: 将魔法数字提取到配置中 / len(self.periods)
                if latest_patterns.get(f'BIAS{period}_STRONG_POSITIVE', False):
                    confidence += 0.15  # TODO: 将魔法数字提取到配置中 / len(self.periods)

            # 多周期共振提高置信度
            if latest_patterns.get('BIAS_ALL_NEGATIVE', False):
                confidence += 0.2
            if latest_patterns.get('BIAS_ALL_POSITIVE', False):
                confidence += 0.2

        # 基于信号的置信度调整
        if signals:
            signal_strength = signals.get('strength', 0)
            confidence += signal_strength * 0.1

        # 限制置信度在0-1范围内
        return max(0.0, min(1.0, confidence))

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """兼容性方法：计算置信度"""
        return self.calculate_confidence_Bias(score, patterns, signals)

    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """兼容性方法：识别形态"""
        return self.get_patterns(data, **kwargs)

    def calculate_raw_score_bias(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """兼容性方法：计算原始评分"""
        return self.calculate_raw_score(data, **kwargs)


# 为了兼容指标注册表，创建别名
BIAS = BiasBias
