from utils.container import container
#!/usr/bin/env python3
"""
ZXM_WASHPLATE 指标

基于ZXM体系教程的真实洗盘形态识别算法
实现横盘震荡洗盘,回调洗盘,假突破洗盘,时间洗盘,连续阴线洗盘等形态识别
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from enum import Enum

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.logger import get_logger

logger = get_logger(__name__)


class ZxmWashplate(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    ZXM_WASHPLATE 指标
    
    自动生成的最小化实现,支持参数标准化
    """
    
    def __init__(self, **kwargs):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化ZXM_WASHPLATE指标
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "ZXM_WASHPLATE"
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters_zxmwashplate()
        
        # 应用用户参数
        self.set_parameters_Washplate(**kwargs)
    
    def _get_default_parameters_zxmwashplate(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14}  # TODO: 将魔法数字提取到配置中
    
    def set_parameters_Washplate(self, **kwargs):
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
            is_valid, errors = validator.validate_indicator_parameters('ZXM_WASHPLATE', params)
            if not is_valid:
                # 静默处理验证失败,避免过多警告
                pass
                # 使用默认参数
                params = self._default_parameters.copy()
            
            # 设置参数
            self.period = params.get('period', 14)  # TODO: 将魔法数字提取到配置中
                    
        except Exception:
            # 如果验证失败,静默处理,保持向后兼容
            self.period = 14  # TODO: 将魔法数字提取到配置中
    
    def calculate_Washplate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算ZXM_WASHPLATE指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了ZXM_WASHPLATE指标的Data_frame
        """
        result = self._calculate_zxmwashplate(data, **kwargs)
        self._result = result
        return result
    
    def _calculate_zxmwashplate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算ZXM_WASHPLATE指标 - 基于ZXM体系教程的真实洗盘算法

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            添加了ZXM_WASHPLATE指标的DataFrame
        """
        df = data.copy()

        # 确保有足够的数据
        if len(df) < 60:  # TODO: 将魔法数字提取到配置中
            # 数据不足时返回空结果
            for wash_type in WashPlateType:
                df[wash_type.value] = False
            df['ZXM_WASHPLATE_SIGNAL'] = 0
            df['ZXM_WASHPLATE_STRENGTH'] = 0.0
            self._result = df  # 缓存结果
            return df

        # 计算技术指标用于洗盘识别
        df = self._calculate_technical_indicators(df)

        # 识别各种洗盘形态
        df = self._identify_shock_wash(df)          # 横盘震荡洗盘
        df = self._identify_pullback_wash(df)       # 回调洗盘
        df = self._identify_false_break_wash(df)    # 假突破洗盘
        df = self._identify_time_wash(df)           # 时间洗盘
        df = self._identify_continuous_yin_wash(df) # 连续阴线洗盘

        # 生成综合洗盘信号
        df = self._generate_washplate_signals(df)

        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        # 缓存结果
        self._result = df
        return df

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算洗盘识别所需的技术指标"""
        # 移动平均线
        df['MA5'] = df['close'].rolling(window=5).mean()  # TODO: 将魔法数字提取到配置中
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()  # TODO: 将魔法数字提取到配置中
        df['MA60'] = df['close'].rolling(window=60).mean()  # TODO: 将魔法数字提取到配置中

        # 成交量移动平均
        df['VOL_MA5'] = df['volume'].rolling(window=5).mean()  # TODO: 将魔法数字提取到配置中
        df['VOL_MA20'] = df['volume'].rolling(window=20).mean()  # TODO: 将魔法数字提取到配置中

        # 价格波动率
        df['PRICE_VOLATILITY'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 最高价和最低价
        df['HIGH_20'] = df['high'].rolling(window=20).max()  # TODO: 将魔法数字提取到配置中
        df['LOW_20'] = df['low'].rolling(window=20).min()  # TODO: 将魔法数字提取到配置中

        # K线实体大小
        df['BODY_SIZE'] = abs(df['close'] - df['open']) / df['open']

        # 上下影线长度
        df['UPPER_SHADOW'] = df['high'] - np.maximum(df['open'], df['close'])
        df['LOWER_SHADOW'] = np.minimum(df['open'], df['close']) - df['low']

        return df

    def _identify_shock_wash(self, df: pd.DataFrame) -> pd.DataFrame:
        """识别横盘震荡洗盘 - 基于ZXM体系教程"""
        shock_wash = pd.Series(False, index=df.index)

        for i in range(20, len(df)):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            # 获取最近20天的数据
            recent_data = df.iloc[i-19:i+1]  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

            # 条件1: 价格在5-10%区间内震荡
            = (recent_data['high'].max() - recent_data['low'].min()) / recent_data['close'].iloc[0]
            range_condition = 0.05 <= <= 0.10  # TODO: 将魔法数字提取到配置中

            # 条件2: 成交量忽大忽小
            vol_std = recent_data['volume'].std()
            vol_mean = recent_data['volume'].mean()
            vol_condition = vol_std / vol_mean > 0.5 if vol_mean > 0 else False  # TODO: 将魔法数字提取到配置中

            # 条件3: 区间下轨有支撑
            low_support = recent_data['low'].min()
            support_tests = (recent_data['low'] <= low_support * 1.02).sum()
            support_condition = support_tests >= 3  # TODO: 将魔法数字提取到配置中

            # 条件4: 持续时间1-3周
            duration_condition = True  # 已通过20天窗口控制

            shock_wash.iloc[i] = range_condition and vol_condition and support_condition and duration_condition

        df[WashPlateType.SHOCK_WASH.value] = shock_wash
        return df

    def _identify_pullback_wash(self, df: pd.DataFrame) -> pd.DataFrame:
        """识别回调洗盘 - 基于ZXM体系教程"""
        pullback_wash = pd.Series(False, index=df.index)

        for i in range(30, len(df)):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            # 获取最近30天的数据
            recent_data = df.iloc[i-29:i+1]  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

            # 条件1: 前期有涨幅
            prev_high = recent_data['high'].iloc[:15].max()  # TODO: 将魔法数字提取到配置中
            prev_low = recent_data['low'].iloc[:15].min()  # TODO: 将魔法数字提取到配置中
            prev_gain = (prev_high - prev_low) / prev_low if prev_low > 0 else 0
            gain_condition = prev_gain > 0.15  # 前期涨幅超过15%  # TODO: 将魔法数字提取到配置中

            # 条件2: 回调幅度为前期涨幅的1/3到1/2
            current_high = recent_data['high'].max()
            current_low = recent_data['low'].iloc[-10:].min()  # 最近10天最低点
            pullback_ratio = (current_high - current_low) / (current_high - prev_low) if current_high > prev_low else 0
            pullback_condition = 0.33 <= pullback_ratio <= 0.5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

            # 条件3: 成交量逐步萎缩
            early_vol = recent_data['volume'].iloc[:10].mean()
            late_vol = recent_data['volume'].iloc[-10:].mean()
            vol_shrink_condition = late_vol < early_vol * 0.7 if early_vol > 0 else False  # TODO: 将魔法数字提取到配置中

            # 条件4: 在重要支撑位止跌
            ma20_support = abs(current_low - recent_data['MA20'].iloc[-1]) / recent_data['MA20'].iloc[-1] < 0.03  # TODO: 将魔法数字提取到配置中

            pullback_wash.iloc[i] = gain_condition and pullback_condition and vol_shrink_condition and ma20_support

        df[WashPlateType.PULLBACK_WASH.value] = pullback_wash
        return df

    def calculate_raw_score_Washplate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算原始评分"""
        if not self.has_result():
            self.calculate_Washplate(data, **kwargs)

        # 基于洗盘信号强度计算评分
        result = self._result if hasattr(self, '_result') and self._result is not None else data

        if 'ZXM_WASHPLATE_STRENGTH' in result.columns:
            # 洗盘强度转换为评分 (0-100)
            base_score = 50.0  # 基础分  # TODO: 将魔法数字提取到配置中
            strength_bonus = result['ZXM_WASHPLATE_STRENGTH'] * 30.0  # 强度加分  # TODO: 将魔法数字提取到配置中
            return pd.Series(base_score + strength_bonus, index=data.index)
        else:
            return pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中
    
    def _identify_false_break_wash(self, df: pd.DataFrame) -> pd.DataFrame:
        """识别假突破洗盘 - 基于ZXM体系教程"""
        false_break_wash = pd.Series(False, index=df.index)

        for i in range(20, len(df)):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            # 获取最近20天的数据
            recent_data = df.iloc[i-19:i+1]  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

            # 条件1: 向下突破重要支撑位
            support_level = recent_data['MA20'].iloc[-5]  # MA20作为支撑  # TODO: 将魔法数字提取到配置中
            break_down = recent_data['low'].iloc[-3:].min() < support_level * 0.97  # 突破3%  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

            # 条件2: 快速收复
            current_close = recent_data['close'].iloc[-1]
            quick_recovery = current_close > support_level * 0.99  # 快速收复到支撑位附近  # TODO: 将魔法数字提取到配置中

            # 条件3: 突破时量能放大,收复时量能更大
            break_vol = recent_data['volume'].iloc[-3:].max()  # TODO: 将魔法数字提取到配置中
            recovery_vol = recent_data['volume'].iloc[-1]
            vol_condition = recovery_vol > break_vol * 1.2

            false_break_wash.iloc[i] = break_down and quick_recovery and vol_condition

        df[WashPlateType.FALSE_BREAK_WASH.value] = false_break_wash
        return df

    def _identify_time_wash(self, df: pd.DataFrame) -> pd.DataFrame:
        """识别时间洗盘 - 基于ZXM体系教程"""
        time_wash = pd.Series(False, index=df.index)

        for i in range(30, len(df)):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            # 获取最近30天的数据
            recent_data = df.iloc[i-29:i+1]  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

            # 条件1: 价格小幅波动
            price_volatility = recent_data['PRICE_VOLATILITY'].iloc[-1]
            low_volatility = price_volatility < 0.02  # 波动率小于2%

            # 条件2: 持续时间较长 (30天窗口)
            duration_condition = True

            # 条件3: 成交量整体萎缩
            early_vol = recent_data['volume'].iloc[:10].mean()
            late_vol = recent_data['volume'].iloc[-10:].mean()
            vol_shrink = late_vol < early_vol * 0.6 if early_vol > 0 else False  # TODO: 将魔法数字提取到配置中

            # 条件4: 偶有放量试盘
            vol_spikes = (recent_data['volume'] > recent_data['VOL_MA20'] * 1.5).sum()  # TODO: 将魔法数字提取到配置中
            spike_condition = 1 <= vol_spikes <= 3  # TODO: 将魔法数字提取到配置中

            time_wash.iloc[i] = low_volatility and duration_condition and vol_shrink and spike_condition

        df[WashPlateType.TIME_WASH.value] = time_wash
        return df

    def _identify_continuous_yin_wash(self, df: pd.DataFrame) -> pd.DataFrame:
        """识别连续阴线洗盘 - 基于ZXM体系教程"""
        continuous_yin_wash = pd.Series(False, index=df.index)

        for i in range(10, len(df)):
            # 获取最近10天的数据
            recent_data = df.iloc[i-9:i+1]  # TODO: 将魔法数字提取到配置中

            # 条件1: 连续3-5根阴线
            yin_lines = (recent_data['close'] < recent_data['open']).sum()
            yin_condition = 3 <= yin_lines <= 5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

            # 条件2: 实体不断缩小
            body_sizes = recent_data['BODY_SIZE'].iloc[-5:]  # TODO: 将魔法数字提取到配置中
            body_shrink = body_sizes.iloc[-1] < body_sizes.iloc[0] * 0.7  # TODO: 将魔法数字提取到配置中

            # 条件3: 下影线增多
            lower_shadows = recent_data['LOWER_SHADOW'].iloc[-5:]  # TODO: 将魔法数字提取到配置中
            shadow_increase = lower_shadows.iloc[-3:].mean() > lower_shadows.iloc[:2].mean()  # TODO: 将魔法数字提取到配置中

            # 条件4: 量能逐步萎缩
            vol_trend = recent_data['volume'].iloc[-5:]  # TODO: 将魔法数字提取到配置中
            vol_shrink = vol_trend.iloc[-1] < vol_trend.iloc[0] * 0.8  # TODO: 将魔法数字提取到配置中

            continuous_yin_wash.iloc[i] = yin_condition and body_shrink and shadow_increase and vol_shrink

        df[WashPlateType.CONTINUOUS_YIN_WASH.value] = continuous_yin_wash
        return df

    def _generate_washplate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成综合洗盘信号"""
        # 计算洗盘信号强度
        washplate_strength = pd.Series(0.0, index=df.index)

        for wash_type in WashPlateType:
            if wash_type.value in df.columns:
                # 不同洗盘类型的权重
                weights = {
                    WashPlateType.SHOCK_WASH.value: 0.25,  # TODO: 将魔法数字提取到配置中
                    WashPlateType.PULLBACK_WASH.value: 0.30,  # TODO: 将魔法数字提取到配置中
                    WashPlateType.FALSE_BREAK_WASH.value: 0.20,  # TODO: 将魔法数字提取到配置中
                    WashPlateType.TIME_WASH.value: 0.15,  # TODO: 将魔法数字提取到配置中
                    WashPlateType.CONTINUOUS_YIN_WASH.value: 0.10
                }
                weight = weights.get(wash_type.value, 0.2)
                washplate_strength += df[wash_type.value].astype(float) * weight

        df['ZXM_WASHPLATE_STRENGTH'] = washplate_strength
        df['ZXM_WASHPLATE_SIGNAL'] = (washplate_strength > 0.05).astype(int)  # 降低阈值提高敏感度  # TODO: 将魔法数字提取到配置中

        return df

    def calculate_confidence_Washplate(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        if hasattr(self, '_result') and self._result is not None and 'ZXM_WASHPLATE_STRENGTH' in self._result.columns:
            avg_strength = self._result['ZXM_WASHPLATE_STRENGTH'].mean()
            return min(0.9, max(0.1, avg_strength))  # TODO: 将魔法数字提取到配置中
        return 0.5  # TODO: 将魔法数字提取到配置中

    def get_patterns_Washplate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取形态"""
        if not self.has_result():
            self.calculate_Washplate(data, **kwargs)

        if hasattr(self, '_result') and self._result is not None:
            # 返回洗盘形态列
            pattern_columns = [wash_type.value for wash_type in WashPlateType]
            available_columns = [col for col in pattern_columns if col in self._result.columns]
            if available_columns:
                return self._result[available_columns].copy()

        return pd.DataFrame(index=data.index)

    # 实现BaseIndicator的抽象方法
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """计算ZXM洗盘指标"""
        return self.calculate_Washplate(data, **kwargs)

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取洗盘形态"""
        return self.get_patterns_Washplate(data, **kwargs)

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算原始评分"""
        return self.calculate_raw_score_Washplate(data, **kwargs)

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        return self.calculate_confidence_Washplate(score, patterns, signals)

    def set_parameters(self, **kwargs):
        """设置参数"""
        return self.set_parameters_Washplate(**kwargs)

    def _get_default_parameters(self) -> Dict[str, Any]:
        """获取默认参数"""
        return self._get_default_parameters_zxmwashplate()

    def register_patterns(self):
        """注册形态到全局注册表"""
        try:
            from utils.dependency_injection import get_container
from db.sql_manager import SQLManager, QueryType
            container = get_container()
            if container.has('pattern_registry'):
                pattern_registry = container.get('pattern_registry')

                # 注册各种洗盘形态
                for wash_type in WashPlateType:
                    pattern_registry.register_pattern(
                        pattern_id=f"ZXM_{wash_type.name}",
                        display_name=wash_type.value,
                        description=f"ZXM体系{wash_type.value}识别",
                        pattern_type="NEUTRAL",
                        default_strength="MEDIUM",
                        score_impact=0.0,
                        polarity="NEUTRAL"
                    )
        except Exception:
            # 静默处理注册失败
            pass

    # 实现BaseIndicator的其他抽象方法
    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """BaseIndicator的抽象方法实现"""
        return self.calculate_Washplate(data, **kwargs)

    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """BaseIndicator的抽象方法实现"""
        return self.calculate_confidence_Washplate(score, patterns, signals)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """BaseIndicator的抽象方法实现"""
        return self.calculate_raw_score_Washplate(data, **kwargs)

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """BaseIndicator的抽象方法实现"""
        return self.get_patterns_Washplate(data, **kwargs)

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """BaseIndicator的抽象方法实现"""
        return self.set_parameters_Washplate(**kwargs)

    def has_result(self) -> bool:
        """检查是否有计算结果"""
        return hasattr(self, '_result') and self._result is not None and not self._result.empty


    @property
    def minimum_periods(self) -> int:
        """
        ZxmWashplate指标所需的最少数据周期数

        计算逻辑:洗盘识别需要足够的历史数据来分析形态

        Returns:
            int: 最少需要的数据周期数
        """
        return 60  # 洗盘形态识别需要更多历史数据  # TODO: 将魔法数字提取到配置中


# 为了向后兼容,创建别名
zxmwash_plate = ZxmWashplate
ZXM_WASHPLATE = ZxmWashplate  # 🔧 Ultra Think修复:添加缺失的别名

class WashPlateType(Enum):
    """洗盘形态类型 - 基于ZXM体系教程"""
    SHOCK_WASH = "横盘震荡洗盘"
    PULLBACK_WASH = "回调洗盘"
    FALSE_BREAK_WASH = "假突破洗盘"
    TIME_WASH = "时间洗盘"
    CONTINUOUS_YIN_WASH = "连续阴线洗盘"
