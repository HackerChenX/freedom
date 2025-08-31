#!/usr/bin/env python
from utils.dependency_injection import get_logger
# -*- coding: utf-8 -*-

"""
趋向指标(DMI)

判断趋势强度与方向
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any
from functools import lru_cache


from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from indicators.common import crossover, crossunder
from utils.dependency_injection import get_logger
from indicators.pattern_registry import PatternRegistry, PatternTypePatternRegistry, PatternStrengthPatternRegistry
logger = get_logger(__name__)

class DirectionalMovementIndex(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    趋向指标(DMI) (DMI)
    
    分类：趋势类指标
    描述：判断趋势强度与方向
    """
    
    def __init__(self, **kwargs):
        """
        初始化趋向指标(DMI)指标

        Args:
            **kwargs: 指标参数，支持period、adx_threshold等
        """
        super().__init__()
        self.REQUIRED_COLUMNS = ['high', 'low', 'close']
        self.name = "DMI"
        self.description = "趋向指标，判断趋势强度与方向"

        # 先设置默认值，确保属性存在
        self.period = 14
        self.adx_threshold = 25.0
        self.adx_period = self.period
        # minimum_periods 是 @property，不需要手动设置

        # 设置默认参数
        self._default_parameters = self._get_default_parameters()

        # 应用用户参数
        self.set_parameters(**kwargs)

    def _get_default_parameters_dmi(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14, "adx_threshold": 25.0}

    def _get_default_parameters(self) -> Dict[str, Any]:
        """获取默认参数（标准接口）"""
        return self._get_default_parameters_dmi()
    
    def set_parameters(self, **kwargs):
        """
        设置指标参数（标准接口）
        """
        return self.set_parameters_Dmi(**kwargs)

    def set_parameters_Dmi(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典，支持以下参数：
                - period: 计算周期
                - adx_threshold: ADX趋势强度阈值
        """
        # 验证参数
        from utils.indicator_parameter_validator import IndicatorParameterValidator
        validator = IndicatorParameterValidator()

        # 合并默认参数和用户参数
        params = self._default_parameters.copy()
        params.update(kwargs)        # 验证参数
        try:
            from utils.indicator_parameter_validator import IndicatorParameterValidator
            validator = IndicatorParameterValidator()
            
            # 验证参数
            is_valid, errors = validator.validate_indicator_parameters('DMI', params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass
                
        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            pass

        # 设置参数
        self.period = params.get('period', 14)
        self.adx_threshold = params.get('adx_threshold', 25.0)

        # 保持向后兼容性
        self.adx_period = self.period  # 为了兼容现有代码
    
    def _validate_dataframe_dmi(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        验证Data_frame是否包含所需的列
        
        Args:
            df: 要验证的Data_frame
            required_columns: 所需的列名列表
            
        Raises:
            ValueError: 如果缺少必要的列
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame缺少必要的列: {', '.join(missing_columns)}")
    
    def _calculate_dmi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算趋向指标(DMI)指标
        
        Args:
            df: 包含OHLCV数据的Data_frame
                必须包含以下列：
                - close: 收盘价
                - high: 最高价
                - low: 最低价
                
        Returns:
            添加了DMI指标列的Data_frame
        """
        if df.empty:
            return pd.DataFrame()

        # 确保数据包含必要的列

        required_columns = ['close', 'high', 'low']
        self._validate_dataframe_dmi(df, required_columns)
        
        df_copy = df.copy()
        
        # 计算真实波幅TR
        df_copy['high_low'] = df_copy['high'] - df_copy['low']
        df_copy['high_close'] = abs(df_copy['high'] - df_copy['close'].shift(1))
        df_copy['low_close'] = abs(df_copy['low'] - df_copy['close'].shift(1))
        df_copy['TR'] = df_copy[['high_low', 'high_close', 'low_close']].max(axis=1)
        
        # 计算方向线DM
        df_copy['up_move'] = df_copy['high'] - df_copy['high'].shift(1)
        df_copy['down_move'] = df_copy['low'].shift(1) - df_copy['low']
        
        # 计算+DM和-DM
        df_copy['+DM'] = np.where((df_copy['up_move'] > df_copy['down_move']) & (df_copy['up_move'] > 0), 
                                df_copy['up_move'], 0)
        df_copy['-DM'] = np.where((df_copy['down_move'] > df_copy['up_move']) & (df_copy['down_move'] > 0), 
                                df_copy['down_move'], 0)
        
        # 计算平滑后的TR、+DM和-DM
        df_copy['TR_' + str(self.period)] = df_copy['TR'].rolling(window=self.period).sum()
        df_copy['+DM_' + str(self.period)] = df_copy['+DM'].rolling(window=self.period).sum()
        df_copy['-DM_' + str(self.period)] = df_copy['-DM'].rolling(window=self.period).sum()
        
        # 计算+DI和-DI
        df_copy['PDI'] = 100 * df_copy['+DM_' + str(self.period)] / df_copy['TR_' + str(self.period)]
        df_copy['MDI'] = 100 * df_copy['-DM_' + str(self.period)] / df_copy['TR_' + str(self.period)]
        
        # 计算方向指数DX
        df_copy['DX'] = 100 * abs(df_copy['PDI'] - df_copy['MDI']) / (df_copy['PDI'] + df_copy['MDI'])
        
        # 计算平均方向指数ADX
        df_copy['ADX'] = df_copy['DX'].rolling(window=self.adx_period).mean()
        
        # 计算平均方向指数评估ADXR
        df_copy['ADXR'] = (df_copy['ADX'] + df_copy['ADX'].shift(self.adx_period)) / 2
        
        # 删除中间计算列
        df_copy.drop(['high_low', 'high_close', 'low_close', 'TR', 'up_move', 'down_move',
                     '+DM', '-DM', 'TR_' + str(self.period), '+DM_' + str(self.period), 
                     '-DM_' + str(self.period), 'DX'], axis=1, inplace=True)
        
        # 添加形态识别和信号生成
        df_copy = self.add_pattern_detection(df_copy)
        df_copy = self.add_signal_generation(df_copy)

        # 存储结果
        self._result = df_copy

        return df_copy

    def calculate_raw_score_Dmi(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算DMI原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算DMI
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        pdi = self._result['PDI']
        mdi = self._result['MDI']
        adx = self._result['ADX']
        adxr = self._result['ADXR']
        
        # 1. DI线交叉评分
        di_cross_score = self._calculate_di_cross_score(pdi, mdi)
        score += di_cross_score
        
        # 2. ADX趋势强度评分
        adx_strength_score = self._calculate_adx_strength_score(adx)
        score += adx_strength_score
        
        # 3. ADX趋势评分
        adx_trend_score = self._calculate_adx_trend_score(adx)
        score += adx_trend_score
        
        # 4. DI线位置评分
        di_position_score = self._calculate_di_position_score(pdi, mdi)
        score += di_position_score
        
        # 5. ADXR确认评分
        adxr_confirm_score = self._calculate_adxr_confirm_score(adx, adxr)
        score += adxr_confirm_score
        
        # 6. DMI综合形态评分
        pattern_score = self._calculate_dmi_pattern_score(pdi, mdi, adx)
        score += pattern_score
        
        return np.clip(score, 0, 100)
    
    def identify_patterns_Dmi(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别DMI技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算DMI
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return patterns
        
        pdi = self._result['PDI']
        mdi = self._result['MDI']
        adx = self._result['ADX']
        adxr = self._result['ADXR']
        
        # 检查最近的信号
        recent_periods = min(10, len(pdi))
        if recent_periods == 0:
            return patterns
        
        recent_pdi = pdi.tail(recent_periods)
        recent_mdi = mdi.tail(recent_periods)
        recent_adx = adx.tail(recent_periods)
        recent_adxr = adxr.tail(recent_periods)
        
        # 检查是否有有效数据
        if recent_pdi.isna().all() or recent_mdi.isna().all() or recent_adx.isna().all():
            return patterns
        
        # 获取最后一个有效值，增加更好的边界检查
        valid_pdi = recent_pdi.dropna()
        valid_mdi = recent_mdi.dropna()
        valid_adx = recent_adx.dropna()
        
        if len(valid_pdi) == 0 or len(valid_mdi) == 0 or len(valid_adx) == 0:
            return patterns
        
        try:
            current_pdi = valid_pdi.iloc[-1]
            current_mdi = valid_mdi.iloc[-1]
            current_adx = valid_adx.iloc[-1]
            
            # 1. DI线交叉形态
            if len(recent_pdi) >= 2 and len(recent_mdi) >= 2:
                if crossover(recent_pdi, recent_mdi).any():
                    patterns.append("DMI多头交叉")
                if crossunder(recent_pdi, recent_mdi).any():
                    patterns.append("DMI空头交叉")
            
            # 2. ADX趋势强度形态
            adx_strength = self._classify_adx_strength(current_adx)
            patterns.append(f"ADX{adx_strength}")
            
            # 3. ADX趋势方向形态
            if len(recent_adx) >= 5:
                adx_trend = self._detect_adx_trend(recent_adx)
                if adx_trend:
                    patterns.append(f"ADX{adx_trend}")
            
            # 4. DI线优势形态
            if current_pdi > current_mdi + 5:
                patterns.append("DMI多头优势")
            elif current_mdi > current_pdi + 5:
                patterns.append("DMI空头优势")
            else:
                patterns.append("DMI多空均衡")
            
            # 5. DMI极值形态
            if current_pdi > 40:
                patterns.append("PDI极强")
            elif current_pdi < 10:
                patterns.append("PDI极弱")
            
            if current_mdi > 40:
                patterns.append("MDI极强")
            elif current_mdi < 10:
                patterns.append("MDI极弱")
            
            # 6. DMI背离形态
            if len(data) >= 20 and len(recent_pdi) >= 10 and len(recent_mdi) >= 10:
                divergence_type = self._detect_dmi_divergence_pattern(data['close'], recent_pdi, recent_mdi)
                if divergence_type:
                    patterns.append(f"DMI{divergence_type}")
            
            # 7. ADX钝化形态
            if len(recent_adx) >= 5:
                if self._detect_adx_stagnation(recent_adx, threshold=25, periods=5):
                    patterns.append("ADX高位钝化")
                    
        except (IndexError, KeyError) as e:
            # 如果出现索引错误，返回空列表
            logger.warning(f"DMI形态识别出现索引错误: {e}")
            return []
        
        return patterns
    
    def _calculate_di_cross_score(self, pdi: pd.Series, mdi: pd.Series) -> pd.Series:
        """
        计算DI线交叉评分
        
        Args:
            pdi: +DI序列
            mdi: -DI序列
            
        Returns:
            pd.Series: 交叉评分
        """
        cross_score = pd.Series(0.0, index=pdi.index)
        
        # +DI上穿-DI（多头信号）+25分
        pdi_cross_up_mdi = crossover(pdi, mdi)
        cross_score += pdi_cross_up_mdi * 25

        # -DI上穿+DI（空头信号）-25分
        mdi_cross_up_pdi = crossover(mdi, pdi)
        cross_score -= mdi_cross_up_pdi * 25
        
        # DI线差距评分
        di_diff = pdi - mdi
        
        # 多头优势+10分
        strong_bullish = di_diff > 10
        cross_score += strong_bullish * 10
        
        # 空头优势-10分
        strong_bearish = di_diff < -10
        cross_score -= strong_bearish * 10
        
        return cross_score
    
    def _calculate_adx_strength_score(self, adx: pd.Series) -> pd.Series:
        """
        计算ADX趋势强度评分
        
        Args:
            adx: ADX序列
            
        Returns:
            pd.Series: 强度评分
        """
        strength_score = pd.Series(0.0, index=adx.index)
        
        # ADX > 25（强趋势）+20分
        strong_trend = adx > 25
        strength_score += strong_trend * 20
        
        # ADX > 40（极强趋势）+25分
        very_strong_trend = adx > 40
        strength_score += very_strong_trend * 25
        
        # ADX < 20（弱趋势）-10分
        weak_trend = adx < 20
        strength_score -= weak_trend * 10
        
        # ADX < 15（无趋势）-15分
        no_trend = adx < 15
        strength_score -= no_trend * 15
        
        return strength_score
    
    def _calculate_adx_trend_score(self, adx: pd.Series) -> pd.Series:
        """
        计算ADX趋势评分
        
        Args:
            adx: ADX序列
            
        Returns:
            pd.Series: 趋势评分
        """
        trend_score = pd.Series(0.0, index=adx.index)
        
        if len(adx) < 5:
            return trend_score
        
        # ADX上升趋势+15分
        adx_rising = adx > adx.shift(3)
        trend_score += adx_rising * 15
        
        # ADX下降趋势-10分
        adx_falling = adx < adx.shift(3)
        trend_score -= adx_falling * 10
        
        # ADX加速上升+20分
        if len(adx) >= 6:
            adx_accelerating = (adx.diff(3) > adx.shift(3).diff(3))
            trend_score += adx_accelerating * 20
        
        return trend_score
    
    def _calculate_di_position_score(self, pdi: pd.Series, mdi: pd.Series) -> pd.Series:
        """
        计算DI线位置评分
        
        Args:
            pdi: +DI序列
            mdi: -DI序列
            
        Returns:
            pd.Series: 位置评分
        """
        position_score = pd.Series(0.0, index=pdi.index)
        
        # +DI在上方+10分
        pdi_above = pdi > mdi
        position_score += pdi_above * 10
        
        # -DI在上方-10分
        mdi_above = mdi > pdi
        position_score -= mdi_above * 10
        
        # DI线极值评分
        # +DI > 40+15分
        pdi_extreme_high = pdi > 40
        position_score += pdi_extreme_high * 15
        
        # -DI > 40-15分
        mdi_extreme_high = mdi > 40
        position_score -= mdi_extreme_high * 15
        
        return position_score
    
    def _calculate_adxr_confirm_score(self, adx: pd.Series, adxr: pd.Series) -> pd.Series:
        """
        计算ADXR确认评分
        
        Args:
            adx: ADX序列
            adxr: ADXR序列
            
        Returns:
            pd.Series: 确认评分
        """
        confirm_score = pd.Series(0.0, index=adx.index)
        
        # ADX > ADXR（趋势加强）+10分
        adx_above_adxr = adx > adxr
        confirm_score += adx_above_adxr * 10
        
        # ADX < ADXR（趋势减弱）-5分
        adx_below_adxr = adx < adxr
        confirm_score -= adx_below_adxr * 5
        
        # ADX与ADXR差距评分
        adx_adxr_diff = abs(adx - adxr)
        
        # 差距较大+5分
        large_diff = adx_adxr_diff > 5
        confirm_score += large_diff * 5
        
        return confirm_score
    
    def _calculate_dmi_pattern_score(self, pdi: pd.Series, mdi: pd.Series, adx: pd.Series) -> pd.Series:
        """
        计算DMI综合形态评分
        
        Args:
            pdi: +DI序列
            mdi: -DI序列
            adx: ADX序列
            
        Returns:
            pd.Series: 形态评分
        """
        pattern_score = pd.Series(0.0, index=pdi.index)
        
        # 强势多头形态：+DI > -DI且ADX > 25
        strong_bullish = (pdi > mdi) & (adx > 25)
        pattern_score += strong_bullish * 20
        
        # 强势空头形态：-DI > +DI且ADX > 25
        strong_bearish = (mdi > pdi) & (adx > 25)
        pattern_score -= strong_bearish * 20
        
        # 震荡形态：ADX < 20且DI线接近
        sideways = (adx < 20) & (abs(pdi - mdi) < 5)
        pattern_score -= sideways * 10
        
        return pattern_score
    
    def _classify_adx_strength(self, adx_value: float) -> str:
        """
        分类ADX强度
        
        Args:
            adx_value: ADX值
            
        Returns:
            str: 强度分类
        """
        if adx_value > 50:
            return "极强趋势"
        elif adx_value > 40:
            return "很强趋势"
        elif adx_value > 25:
            return "强趋势"
        elif adx_value > 20:
            return "中等趋势"
        elif adx_value > 15:
            return "弱趋势"
        else:
            return "无趋势"
    
    def _detect_adx_trend(self, adx: pd.Series) -> Optional[str]:
        """
        检测ADX趋势
        
        Args:
            adx: ADX序列
            
        Returns:
            Optional[str]: 趋势类型或None
        """
        if len(adx) < 5:
            return None
        
        # 计算ADX趋势
        recent_adx = adx.tail(5)
        
        # 连续上升
        if all(recent_adx.iloc[i] >= recent_adx.iloc[i-1] for i in range(1, len(recent_adx))):
            return "持续上升"
        
        # 连续下降
        if all(recent_adx.iloc[i] <= recent_adx.iloc[i-1] for i in range(1, len(recent_adx))):
            return "持续下降"
        
        # 整体上升趋势
        if recent_adx.iloc[-1] > recent_adx.iloc[0] * 1.1:
            return "上升趋势"
        
        # 整体下降趋势
        if recent_adx.iloc[-1] < recent_adx.iloc[0] * 0.9:
            return "下降趋势"
        
        return None
    
    def _detect_dmi_divergence_pattern(self, price: pd.Series, pdi: pd.Series, mdi: pd.Series) -> Optional[str]:
        """
        检测DMI背离形态
        
        Args:
            price: 价格序列
            pdi: +DI序列
            mdi: -DI序列
            
        Returns:
            Optional[str]: 背离类型或None
        """
        if len(price) < 20:
            return None
        
        # 寻找最近的峰值和谷值
        recent_price = price.tail(20)
        recent_pdi = pdi.tail(20)
        recent_mdi = mdi.tail(20)
        
        price_extremes = []
        pdi_extremes = []
        mdi_extremes = []
        
        # 简化的极值检测
        for i in range(2, len(recent_price) - 2):
            if (recent_price.iloc[i] > recent_price.iloc[i-1] and 
                recent_price.iloc[i] > recent_price.iloc[i+1]):
                price_extremes.append(recent_price.iloc[i])
                pdi_extremes.append(recent_pdi.iloc[i])
                mdi_extremes.append(recent_mdi.iloc[i])
            elif (recent_price.iloc[i] < recent_price.iloc[i-1] and 
                  recent_price.iloc[i] < recent_price.iloc[i+1]):
                price_extremes.append(recent_price.iloc[i])
                pdi_extremes.append(recent_pdi.iloc[i])
                mdi_extremes.append(recent_mdi.iloc[i])
        
        if len(price_extremes) >= 2:
            price_trend = price_extremes[-1] - price_extremes[-2]
            pdi_trend = pdi_extremes[-1] - pdi_extremes[-2]
            mdi_trend = mdi_extremes[-1] - mdi_extremes[-2]
            
            # 正背离：价格创新低但+DI未创新低
            if price_trend < -0.01 and pdi_trend > 2:
                return "正背离"
            # 负背离：价格创新高但+DI未创新高
            elif price_trend > 0.01 and pdi_trend < -2:
                return "负背离"
        
        return None
    
    def _detect_adx_stagnation(self, adx: pd.Series, threshold: float, periods: int) -> bool:
        """
        检测ADX钝化
        
        Args:
            adx: ADX序列
            threshold: 阈值
            periods: 检测周期数
            
        Returns:
            bool: 是否钝化
        """
        if len(adx) < periods:
            return False
        
        recent_adx = adx.tail(periods)
        
        # 高位钝化：ADX在高位且变化很小
        return (recent_adx > threshold).all() and (recent_adx.max() - recent_adx.min()) < 5
        
    def get_signals_Dmi(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成趋向指标(DMI)指标交易信号
        
        Args:
            df: 包含价格数据和DMI指标的Data_frame
            **kwargs: 额外参数
                
        Returns:
            添加了信号列的Data_frame:
            - dmi_signal: 1=买入信号, -1=卖出信号, 0=无信号
        """
        if df.empty:

            return pd.DataFrame()

            
            return df
            
        # 检查必要的指标列是否存在
        required_columns = ['PDI', 'MDI', 'ADX']
        self._validate_dataframe_dmi(df, required_columns)
        
        df_copy = df.copy()
        
        # 初始化信号列
        df_copy['dmi_signal'] = 0
        
        # +DI上穿-DI为买入信号
        df_copy.loc[crossover(df_copy['PDI'], df_copy['MDI']), 'dmi_signal'] = 1
        
        # -DI上穿+DI为卖出信号
        df_copy.loc[crossover(df_copy['MDI'], df_copy['PDI']), 'dmi_signal'] = -1
        
        # 强化信号：ADX > 25表示趋势显著
        df_copy.loc[(df_copy['dmi_signal'] == 1) & (df_copy['ADX'] < 25), 'dmi_signal'] = 0
        df_copy.loc[(df_copy['dmi_signal'] == -1) & (df_copy['ADX'] < 25), 'dmi_signal'] = 0
        
        return df_copy

    def generate_signals_Dmi(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成DMI指标标准化交易信号
        
        Args:
            data: 输入数据，包含OHLCV数据
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            pd.DataFrame: 信号结果Data_frame，包含标准化信号
        """
        # 确保已计算DMI指标
        if not self.has_result():
            self.calculate(data)
        
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['buy_signal'] = False
        signals['sell_signal'] = False
        signals['neutral_signal'] = True  # 默认为中性信号
        signals['trend'] = 0  # 0表示中性
        signals['score'] = 50.0  # 默认评分50分
        signals['signal_type'] = None
        signals['signal_desc'] = None
        signals['confidence'] = 50.0
        signals['risk_level'] = '中'
        signals['position_size'] = 0.0
        signals['stop_loss'] = None
        signals['market_env'] = 'sideways_market'
        signals['volume_confirmation'] = False
        
        # 计算评分
        score = self.calculate_raw_score_Dmi(data, **kwargs)
        signals['score'] = score
        
        # 检测形态
        patterns = self.identify_patterns_Dmi(data, **kwargs)
        
        # 获取DMI数据
        pdi = self._result['PDI']
        mdi = self._result['MDI']
        adx = self._result['ADX']
        adxr = self._result['ADXR']
        
        # 1. +DI上穿-DI，买入信号
        di_crossover = crossover(pdi, mdi)
        signals.loc[di_crossover, 'buy_signal'] = True
        signals.loc[di_crossover, 'neutral_signal'] = False
        signals.loc[di_crossover, 'trend'] = 1
        signals.loc[di_crossover, 'signal_type'] = 'DI金叉'
        signals.loc[di_crossover, 'signal_desc'] = '+DI上穿-DI，多头趋势确立'
        signals.loc[di_crossover, 'confidence'] = 70.0
        signals.loc[di_crossover, 'position_size'] = 0.4
        signals.loc[di_crossover, 'risk_level'] = '中'
        
        # 2. -DI上穿+DI，卖出信号
        di_crossunder = crossover(mdi, pdi)
        signals.loc[di_crossunder, 'sell_signal'] = True
        signals.loc[di_crossunder, 'neutral_signal'] = False
        signals.loc[di_crossunder, 'trend'] = -1
        signals.loc[di_crossunder, 'signal_type'] = 'DI死叉'
        signals.loc[di_crossunder, 'signal_desc'] = '-DI上穿+DI，空头趋势确立'
        signals.loc[di_crossunder, 'confidence'] = 70.0
        signals.loc[di_crossunder, 'position_size'] = 0.4
        signals.loc[di_crossunder, 'risk_level'] = '中'
        
        # 3. ADX上升且大于阈值，趋势增强信号
        adx_rising = (adx > adx.shift(1)) & (adx > 25)
        
        # 强多头趋势信号：ADX上升且+DI>-DI
        strong_uptrend = adx_rising & (pdi > mdi)
        signals.loc[strong_uptrend, 'buy_signal'] = True
        signals.loc[strong_uptrend, 'neutral_signal'] = False
        signals.loc[strong_uptrend, 'trend'] = 1
        signals.loc[strong_uptrend, 'signal_type'] = '强多头趋势'
        signals.loc[strong_uptrend, 'signal_desc'] = 'ADX上升且+DI>-DI，多头趋势增强'
        signals.loc[strong_uptrend, 'confidence'] = 80.0
        signals.loc[strong_uptrend, 'position_size'] = 0.5
        signals.loc[strong_uptrend, 'risk_level'] = '低'
        
        # 强空头趋势信号：ADX上升且-DI>+DI
        strong_downtrend = adx_rising & (mdi > pdi)
        signals.loc[strong_downtrend, 'sell_signal'] = True
        signals.loc[strong_downtrend, 'neutral_signal'] = False
        signals.loc[strong_downtrend, 'trend'] = -1
        signals.loc[strong_downtrend, 'signal_type'] = '强空头趋势'
        signals.loc[strong_downtrend, 'signal_desc'] = 'ADX上升且-DI>+DI，空头趋势增强'
        signals.loc[strong_downtrend, 'confidence'] = 80.0
        signals.loc[strong_downtrend, 'position_size'] = 0.5
        signals.loc[strong_downtrend, 'risk_level'] = '低'
        
        # 4. ADX下降，趋势减弱信号
        adx_falling = (adx < adx.shift(1)) & (adx > 20)
        
        # 多头趋势减弱信号：ADX下降且+DI>-DI
        weakening_uptrend = adx_falling & (pdi > mdi)
        signals.loc[weakening_uptrend, 'buy_signal'] = True
        signals.loc[weakening_uptrend, 'neutral_signal'] = False
        signals.loc[weakening_uptrend, 'trend'] = 1
        signals.loc[weakening_uptrend, 'signal_type'] = '减弱多头趋势'
        signals.loc[weakening_uptrend, 'signal_desc'] = 'ADX下降且+DI>-DI，多头趋势减弱'
        signals.loc[weakening_uptrend, 'confidence'] = 60.0
        signals.loc[weakening_uptrend, 'position_size'] = 0.3
        signals.loc[weakening_uptrend, 'risk_level'] = '中'
        
        # 空头趋势减弱信号：ADX下降且-DI>+DI
        weakening_downtrend = adx_falling & (mdi > pdi)
        signals.loc[weakening_downtrend, 'sell_signal'] = True
        signals.loc[weakening_downtrend, 'neutral_signal'] = False
        signals.loc[weakening_downtrend, 'trend'] = -1
        signals.loc[weakening_downtrend, 'signal_type'] = '减弱空头趋势'
        signals.loc[weakening_downtrend, 'signal_desc'] = 'ADX下降且-DI>+DI，空头趋势减弱'
        signals.loc[weakening_downtrend, 'confidence'] = 60.0
        signals.loc[weakening_downtrend, 'position_size'] = 0.3
        signals.loc[weakening_downtrend, 'risk_level'] = '中'
        
        # 5. ADX非常低，无趋势信号
        no_trend = adx < 15
        signals.loc[no_trend, 'neutral_signal'] = True
        signals.loc[no_trend, 'buy_signal'] = False
        signals.loc[no_trend, 'sell_signal'] = False
        signals.loc[no_trend, 'trend'] = 0
        signals.loc[no_trend, 'signal_type'] = '无趋势'
        signals.loc[no_trend, 'signal_desc'] = 'ADX低于15，市场处于无趋势震荡状态'
        signals.loc[no_trend, 'confidence'] = 60.0
        signals.loc[no_trend, 'position_size'] = 0.0
        signals.loc[no_trend, 'risk_level'] = '中'
        
        # 6. ADX非常高，趋势过热信号
        extreme_trend = adx > 50
        
        # 根据DI判断是多头还是空头过热
        extreme_uptrend = extreme_trend & (pdi > mdi)
        signals.loc[extreme_uptrend, 'buy_signal'] = True
        signals.loc[extreme_uptrend, 'neutral_signal'] = False
        signals.loc[extreme_uptrend, 'trend'] = 1
        signals.loc[extreme_uptrend, 'signal_type'] = '极端多头趋势'
        signals.loc[extreme_uptrend, 'signal_desc'] = 'ADX极高且+DI>-DI，多头趋势过热'
        signals.loc[extreme_uptrend, 'confidence'] = 65.0
        signals.loc[extreme_uptrend, 'position_size'] = 0.3
        signals.loc[extreme_uptrend, 'risk_level'] = '高'
        
        extreme_downtrend = extreme_trend & (mdi > pdi)
        signals.loc[extreme_downtrend, 'sell_signal'] = True
        signals.loc[extreme_downtrend, 'neutral_signal'] = False
        signals.loc[extreme_downtrend, 'trend'] = -1
        signals.loc[extreme_downtrend, 'signal_type'] = '极端空头趋势'
        signals.loc[extreme_downtrend, 'signal_desc'] = 'ADX极高且-DI>+DI，空头趋势过热'
        signals.loc[extreme_downtrend, 'confidence'] = 65.0
        signals.loc[extreme_downtrend, 'position_size'] = 0.3
        signals.loc[extreme_downtrend, 'risk_level'] = '高'
        
        # 7. ADXR确认信号
        adxr_confirming_adx = (adxr > adxr.shift(1)) & (adx > adx.shift(1))
        
        # ADXR确认的多头趋势
        adxr_confirmed_uptrend = adxr_confirming_adx & (pdi > mdi)
        signals.loc[adxr_confirmed_uptrend, 'buy_signal'] = True
        signals.loc[adxr_confirmed_uptrend, 'neutral_signal'] = False
        signals.loc[adxr_confirmed_uptrend, 'trend'] = 1
        signals.loc[adxr_confirmed_uptrend, 'signal_type'] = 'ADXR确认多头'
        signals.loc[adxr_confirmed_uptrend, 'signal_desc'] = 'ADXR确认多头趋势持续'
        signals.loc[adxr_confirmed_uptrend, 'confidence'] = 75.0
        signals.loc[adxr_confirmed_uptrend, 'position_size'] = 0.4
        signals.loc[adxr_confirmed_uptrend, 'risk_level'] = '中'
        
        # ADXR确认的空头趋势
        adxr_confirmed_downtrend = adxr_confirming_adx & (mdi > pdi)
        signals.loc[adxr_confirmed_downtrend, 'sell_signal'] = True
        signals.loc[adxr_confirmed_downtrend, 'neutral_signal'] = False
        signals.loc[adxr_confirmed_downtrend, 'trend'] = -1
        signals.loc[adxr_confirmed_downtrend, 'signal_type'] = 'ADXR确认空头'
        signals.loc[adxr_confirmed_downtrend, 'signal_desc'] = 'ADXR确认空头趋势持续'
        signals.loc[adxr_confirmed_downtrend, 'confidence'] = 75.0
        signals.loc[adxr_confirmed_downtrend, 'position_size'] = 0.4
        signals.loc[adxr_confirmed_downtrend, 'risk_level'] = '中'
        
        # 8. 根据形态设置更多信号
        for pattern in patterns:
            pattern_idx = signals.index[-5:]  # 假设形态影响最近5个周期
            
            if '多头趋势' in pattern or '看涨' in pattern:
                signals.loc[pattern_idx, 'buy_signal'] = True
                signals.loc[pattern_idx, 'neutral_signal'] = False
                signals.loc[pattern_idx, 'trend'] = 1
                signals.loc[pattern_idx, 'signal_type'] = 'DMI趋势信号'
                signals.loc[pattern_idx, 'signal_desc'] = pattern
                signals.loc[pattern_idx, 'confidence'] = 70.0
                signals.loc[pattern_idx, 'position_size'] = 0.4
                signals.loc[pattern_idx, 'risk_level'] = '中'
            
            elif '空头趋势' in pattern or '看跌' in pattern:
                signals.loc[pattern_idx, 'sell_signal'] = True
                signals.loc[pattern_idx, 'neutral_signal'] = False
                signals.loc[pattern_idx, 'trend'] = -1
                signals.loc[pattern_idx, 'signal_type'] = 'DMI趋势信号'
                signals.loc[pattern_idx, 'signal_desc'] = pattern
                signals.loc[pattern_idx, 'confidence'] = 70.0
                signals.loc[pattern_idx, 'position_size'] = 0.4
                signals.loc[pattern_idx, 'risk_level'] = '中'
            
            elif '无趋势' in pattern or '横盘' in pattern:
                signals.loc[pattern_idx, 'neutral_signal'] = True
                signals.loc[pattern_idx, 'buy_signal'] = False
                signals.loc[pattern_idx, 'sell_signal'] = False
                signals.loc[pattern_idx, 'trend'] = 0
                signals.loc[pattern_idx, 'signal_type'] = 'DMI震荡信号'
                signals.loc[pattern_idx, 'signal_desc'] = pattern
                signals.loc[pattern_idx, 'confidence'] = 60.0
                signals.loc[pattern_idx, 'position_size'] = 0.0
                signals.loc[pattern_idx, 'risk_level'] = '低'
        
        # 设置止损价格
        if 'low' in data.columns and 'high' in data.columns:
            # 买入信号的止损设为最近的低点
            buy_indices = signals[signals['buy_signal']].index
            if not buy_indices.empty:
                for idx in buy_indices:
                    if idx > data.index[10]:  # 确保有足够的历史数据
                        lookback = 5
                        # 使用最近低点作为止损
                        recent_low = data.loc[idx-lookback:idx, 'low'].min()
                        signals.loc[idx, 'stop_loss'] = recent_low
        
            # 卖出信号的止损设为最近的高点
            sell_indices = signals[signals['sell_signal']].index
            if not sell_indices.empty:
                for idx in sell_indices:
                    if idx > data.index[10]:  # 确保有足够的历史数据
                        lookback = 5
                        # 使用最近高点作为止损
                        recent_high = data.loc[idx-lookback:idx, 'high'].max()
                        signals.loc[idx, 'stop_loss'] = recent_high
        
        # 根据ADX值判断市场环境
        signals['market_env'] = 'sideways_market'  # 默认震荡市场
        
        # ADX高且+DI>-DI，上升趋势市场
        uptrend_market = (adx > 25) & (pdi > mdi)
        signals.loc[uptrend_market, 'market_env'] = 'bull_market'
        
        # ADX高且-DI>+DI，下降趋势市场
        downtrend_market = (adx > 25) & (mdi > pdi)
        signals.loc[downtrend_market, 'market_env'] = 'bear_market'
        
        # ADX低，震荡市场
        strong_sideways = adx < 15
        signals.loc[strong_sideways, 'market_env'] = 'sideways_market'
        
        # 设置成交量确认
        if 'volume' in data.columns:
            # 如果有成交量数据，检查成交量是否支持当前信号
            vol = data['volume']
            vol_avg = vol.rolling(window=20).mean()
            
            # 成交量大于20日均量1.5倍为放量
            vol_increase = vol > vol_avg * 1.5
            
            # 买入信号且成交量放大，确认信号
            signals.loc[signals['buy_signal'] & vol_increase, 'volume_confirmation'] = True
            
            # 卖出信号且成交量放大，确认信号
            signals.loc[signals['sell_signal'] & vol_increase, 'volume_confirmation'] = True
        
        return signals
    
    def get_patterns_Dmi(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        识别所有已定义的DMI形态，并以Data_frame形式返回

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含所有形态信号的Data_frame
        """
        if not self.has_result():
            self.calculate(data, **kwargs)
            
        result = self._result
        if result is None:
            return pd.DataFrame(index=data.index)

        patterns_df = pd.DataFrame(index=result.index)
        
        pdi = result['PDI']
        mdi = result['MDI']
        adx = result['ADX']

        # DI线金叉/死叉
        patterns_df['DMI_GOLDEN_CROSS'] = crossover(pdi, mdi)
        patterns_df['DMI_DEATH_CROSS'] = crossunder(pdi, mdi)

        # ADX趋势强度
        patterns_df['ADX_STRONG_TREND'] = adx > 25
        patterns_df['ADX_WEAK_TREND'] = adx < 20
        
        # ADX上升/下降
        patterns_df['ADX_RISING'] = adx > adx.shift(1)
        patterns_df['ADX_FALLING'] = adx < adx.shift(1)

        return patterns_df

    def calculate_confidence_Dmi(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算DMI指标的置信度

        Args:
            score: 得分序列
            patterns: 检测到的形态Data_frame
            signals: 生成的信号字典

        Returns:
            float: 置信度分数 (0-1)
        """
        if score.empty:
            return 0.5

        # 基础置信度
        confidence = 0.5

        # 1. 基于评分的置信度
        last_score = score.iloc[-1]

        # 极端评分置信度较高
        if last_score > 80 or last_score < 20:
            confidence += 0.25
        # 中性评分置信度中等
        elif 40 <= last_score <= 60:
            confidence += 0.1
        else:
            confidence += 0.15

        # 2. 基于形态的置信度
        if isinstance(patterns, pd.DataFrame) and not patterns.empty:
            try:
                # 统计最近几个周期的形态数量
                numeric_cols = patterns.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    recent_data = patterns[numeric_cols].iloc[-5:] if len(patterns) >= 5 else patterns[numeric_cols]
                    recent_patterns = recent_data.sum().sum()
                    if recent_patterns > 0:
                        confidence += min(recent_patterns * 0.05, 0.2)
            except:
                pass

        # 3. 基于ADX强度的置信度
        if hasattr(self, '_result') and self._result is not None and 'ADX' in self._result.columns:
            try:
                adx_values = self._result['ADX'].dropna()
                if len(adx_values) > 0:
                    last_adx = adx_values.iloc[-1]
                    # ADX越高，置信度越高
                    if last_adx > 40:
                        confidence += 0.2
                    elif last_adx > 25:
                        confidence += 0.15
                    elif last_adx < 15:
                        confidence -= 0.1
            except:
                pass

        # 4. 基于评分稳定性的置信度
        if len(score) >= 5:
            recent_scores = score.iloc[-5:]
            score_stability = 1.0 - (recent_scores.std() / 50.0)
            confidence += score_stability * 0.1

        return min(confidence, 1.0)

    def register_patterns_Dmi(self):
        """
        注册DMI指标的技术形态
        """
        # 注册DMI交叉形态
        self.register_pattern_to_registry(
            pattern_id="DMI_GOLDEN_CROSS",
            display_name="DMI金叉",
            description="+DI上穿-DI，显示多头趋势开始",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=30.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="DMI_DEATH_CROSS",
            display_name="DMI死叉",
            description="-DI上穿+DI，显示空头趋势开始",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-30.0,
            polarity="NEGATIVE"
        )

        # 注册ADX趋势强度形态
        self.register_pattern_to_registry(
            pattern_id="ADX_STRONG_TREND",
            display_name="ADX强趋势",
            description="ADX大于25，表示趋势强劲",
            pattern_type="NEUTRAL",
            default_strength="MEDIUM",
            score_impact=20.0,
            polarity="NEUTRAL"
        )

        self.register_pattern_to_registry(
            pattern_id="ADX_WEAK_TREND",
            display_name="ADX弱趋势",
            description="ADX小于20，表示趋势疲弱，可能进入盘整",
            pattern_type="NEUTRAL",
            default_strength="WEAK",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

        # 注册ADX趋势方向形态
        self.register_pattern_to_registry(
            pattern_id="ADX_RISING",
            display_name="ADX上升",
            description="ADX上升，趋势强度增强",
            pattern_type="NEUTRAL",
            default_strength="MEDIUM",
            score_impact=15.0,
            polarity="NEUTRAL"
        )

        self.register_pattern_to_registry(
            pattern_id="ADX_FALLING",
            display_name="ADX下降",
            description="ADX下降，趋势强度减弱，可能预示反转或盘整",
            pattern_type="NEUTRAL",
            default_strength="WEAK",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

    def get_pattern_info_Dmi(self, pattern_id: str) -> dict:
        """
        获取指定形态的详细信息

        Args:
            pattern_id: 形态ID

        Returns:
            dict: 形态信息字典
        """
        pattern_info_map = {
            'DMI_GOLDEN_CROSS': {
                'name': 'DMI金叉',
                'description': '+DI上穿-DI，显示多头趋势开始',
                'strength': 'strong',
                'type': 'bullish'
            },
            'DMI_DEATH_CROSS': {
                'name': 'DMI死叉',
                'description': '-DI上穿+DI，显示空头趋势开始',
                'strength': 'strong',
                'type': 'bearish'
            },
            'ADX_STRONG_TREND': {
                'name': 'ADX强趋势',
                'description': 'ADX大于25，表示趋势强劲',
                'strength': 'medium',
                'type': 'neutral'
            },
            'ADX_WEAK_TREND': {
                'name': 'ADX弱趋势',
                'description': 'ADX小于20，表示趋势疲弱',
                'strength': 'weak',
                'type': 'neutral'
            },
            'ADX_RISING': {
                'name': 'ADX上升',
                'description': 'ADX上升，趋势强度增强',
                'strength': 'medium',
                'type': 'neutral'
            },
            'ADX_FALLING': {
                'name': 'ADX下降',
                'description': 'ADX下降，趋势强度减弱',
                'strength': 'weak',
                'type': 'neutral'
            },
            'DMI_STRONG_UPTREND': {
                'name': 'DMI强多头趋势',
                'description': '+DI明显高于-DI且ADX上升',
                'strength': 'strong',
                'type': 'bullish'
            },
            'DMI_STRONG_DOWNTREND': {
                'name': 'DMI强空头趋势',
                'description': '-DI明显高于+DI且ADX上升',
                'strength': 'strong',
                'type': 'bearish'
            },
            'DMI_TREND_WEAKENING': {
                'name': 'DMI趋势减弱',
                'description': 'ADX下降，当前趋势强度减弱',
                'strength': 'medium',
                'type': 'neutral'
            },
            'DMI_NO_TREND': {
                'name': 'DMI无趋势',
                'description': 'ADX低于15，市场处于无趋势状态',
                'strength': 'weak',
                'type': 'neutral'
            }
        }

        return pattern_info_map.get(pattern_id, {
            'name': pattern_id,
            'description': f'DMI形态: {pattern_id}',
            'strength': 'medium',
            'type': 'neutral'
        })

    # ==================== 抽象方法实现 ====================

    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """抽象基类要求的计算方法"""
        return self._calculate_dmi(data, **kwargs)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """抽象基类要求的评分方法"""
        return self.calculate_raw_score_Dmi(data, **kwargs)

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """抽象基类要求的形态方法"""
        return self.get_patterns_Dmi(data, **kwargs)

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """抽象基类要求的参数设置方法"""
        return self.set_parameters_Dmi(**kwargs)

    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """抽象基类要求的置信度计算方法"""
        return self.calculate_confidence_Dmi(score, patterns, signals)

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """统一的计算接口"""
        return self._calculate_dmi(data, **kwargs)

    # ==================== 兼容性方法 - 真实实现 ====================

    def get_patterns(self, data: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        """真实实现：获取DMI形态"""
        if data is None or data.empty:
            return pd.DataFrame()

        # 首先计算DMI指标
        dmi_data = self._calculate_dmi(data)

        # 创建形态DataFrame
        patterns_df = pd.DataFrame(index=data.index)

        # 获取DMI数据
        pdi = dmi_data.get('PDI', pd.Series(index=data.index))
        mdi = dmi_data.get('MDI', pd.Series(index=data.index))
        adx = dmi_data.get('ADX', pd.Series(index=data.index))

        # 1. 强趋势形态 (ADX > 25)
        patterns_df['DMI_STRONG_TREND'] = adx > 25

        # 2. 弱趋势形态 (ADX < 20)
        patterns_df['DMI_WEAK_TREND'] = adx < 20

        # 3. 上升趋势形态 (PDI > MDI and ADX > 20)
        patterns_df['DMI_UPTREND'] = (pdi > mdi) & (adx > 20)

        # 4. 下降趋势形态 (MDI > PDI and ADX > 20)
        patterns_df['DMI_DOWNTREND'] = (mdi > pdi) & (adx > 20)

        # 5. PDI上穿MDI形态
        patterns_df['DMI_PDI_CROSS_UP'] = (pdi > mdi) & (pdi.shift(1) <= mdi.shift(1))

        # 6. MDI上穿PDI形态
        patterns_df['DMI_MDI_CROSS_UP'] = (mdi > pdi) & (mdi.shift(1) <= pdi.shift(1))

        # 7. ADX上升形态
        patterns_df['DMI_ADX_RISING'] = adx > adx.shift(1)

        # 8. ADX下降形态
        patterns_df['DMI_ADX_FALLING'] = adx < adx.shift(1)

        # 9. 极强趋势形态 (ADX > 40)
        patterns_df['DMI_VERY_STRONG_TREND'] = adx > 40

        # 10. 无趋势形态 (ADX < 15)
        patterns_df['DMI_NO_TREND'] = adx < 15

        # 11. 趋势转换形态
        patterns_df['DMI_TREND_CHANGE'] = (
            ((pdi > mdi) & (pdi.shift(1) <= mdi.shift(1))) |
            ((mdi > pdi) & (mdi.shift(1) <= pdi.shift(1)))
        )

        return patterns_df

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """真实实现：计算DMI原始评分"""
        if data.empty:
            return pd.Series(dtype=float)

        # 计算DMI指标
        dmi_data = self._calculate_dmi(data)

        # 初始化评分
        score = pd.Series(50.0, index=data.index)  # 基础分50分

        # 获取DMI数据
        pdi = dmi_data.get('PDI', pd.Series(index=data.index))
        mdi = dmi_data.get('MDI', pd.Series(index=data.index))
        adx = dmi_data.get('ADX', pd.Series(index=data.index))

        # 1. 基于趋势方向的评分
        # PDI > MDI 加分（上升趋势）
        uptrend_condition = pdi > mdi
        score += uptrend_condition * 15

        # MDI > PDI 减分（下降趋势）
        downtrend_condition = mdi > pdi
        score -= downtrend_condition * 15

        # 2. 基于趋势强度的评分
        # 强趋势加分
        strong_trend = adx > 25
        score += strong_trend * 10

        # 极强趋势额外加分
        very_strong_trend = adx > 40
        score += very_strong_trend * 10

        # 弱趋势减分
        weak_trend = adx < 20
        score -= weak_trend * 10

        # 无趋势大幅减分
        no_trend = adx < 15
        score -= no_trend * 15

        # 3. 基于交叉信号的评分
        # PDI上穿MDI加分
        pdi_cross_up = (pdi > mdi) & (pdi.shift(1) <= mdi.shift(1))
        score += pdi_cross_up * 12

        # MDI上穿PDI减分
        mdi_cross_up = (mdi > pdi) & (mdi.shift(1) <= pdi.shift(1))
        score -= mdi_cross_up * 12

        # 4. 基于ADX趋势的评分
        # ADX上升加分（趋势加强）
        adx_rising = adx > adx.shift(1)
        score += adx_rising * 5

        # ADX下降减分（趋势减弱）
        adx_falling = adx < adx.shift(1)
        score -= adx_falling * 5

        # 5. 基于PDI和MDI差值的评分
        di_diff = np.abs(pdi - mdi)
        # 差值越大，趋势越明确
        diff_bonus = np.minimum(di_diff / 2, 10)  # 最多10分奖励
        score += diff_bonus

        # 限制评分在0-100之间
        return score.clip(0, 100)

    def get_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """真实实现：生成DMI交易信号"""
        if data.empty:
            return pd.DataFrame()

        # 计算DMI指标
        dmi_data = self._calculate_dmi(data)
        result_df = data.copy()

        # 合并DMI数据
        for col in dmi_data.columns:
            result_df[col] = dmi_data[col]

        # 初始化信号列
        result_df['dmi_signal'] = 0
        result_df['dmi_strength'] = 0.0
        result_df['dmi_confidence'] = 0.0

        # 获取DMI数据
        pdi = dmi_data.get('PDI', pd.Series(index=data.index))
        mdi = dmi_data.get('MDI', pd.Series(index=data.index))
        adx = dmi_data.get('ADX', pd.Series(index=data.index))

        # 1. PDI上穿MDI买入信号
        pdi_cross_up = (pdi > mdi) & (pdi.shift(1) <= mdi.shift(1))
        strong_trend_up = pdi_cross_up & (adx > 20)
        result_df.loc[strong_trend_up, 'dmi_signal'] = 1
        result_df.loc[strong_trend_up, 'dmi_strength'] = 0.8
        result_df.loc[strong_trend_up, 'dmi_confidence'] = 0.9

        # 2. MDI上穿PDI卖出信号
        mdi_cross_up = (mdi > pdi) & (mdi.shift(1) <= pdi.shift(1))
        strong_trend_down = mdi_cross_up & (adx > 20)
        result_df.loc[strong_trend_down, 'dmi_signal'] = -1
        result_df.loc[strong_trend_down, 'dmi_strength'] = 0.8
        result_df.loc[strong_trend_down, 'dmi_confidence'] = 0.9

        # 3. 强趋势确认信号
        very_strong_up = (pdi > mdi) & (adx > 40)
        result_df.loc[very_strong_up, 'dmi_signal'] = 1
        result_df.loc[very_strong_up, 'dmi_strength'] = 0.9
        result_df.loc[very_strong_up, 'dmi_confidence'] = 0.95

        very_strong_down = (mdi > pdi) & (adx > 40)
        result_df.loc[very_strong_down, 'dmi_signal'] = -1
        result_df.loc[very_strong_down, 'dmi_strength'] = 0.9
        result_df.loc[very_strong_down, 'dmi_confidence'] = 0.95

        # 4. 趋势减弱警告信号
        trend_weakening = (adx < adx.shift(1)) & (adx.shift(1) > 30)
        result_df.loc[trend_weakening, 'dmi_signal'] = 0
        result_df.loc[trend_weakening, 'dmi_strength'] = 0.3
        result_df.loc[trend_weakening, 'dmi_confidence'] = 0.6

        return result_df

    def calculate_score(self, data: pd.DataFrame, **kwargs) -> dict:
        """真实实现：计算DMI综合评分"""
        if data.empty:
            return {'score': 50.0, 'confidence': 0.0, 'signals': {}}

        # 计算原始评分
        raw_score = self.calculate_raw_score(data, **kwargs)

        # 获取形态
        patterns = self.get_patterns(data, **kwargs)

        # 计算最终评分
        final_score = raw_score.iloc[-1] if not raw_score.empty else 50.0

        # 基于形态调整评分
        if not patterns.empty:
            latest_patterns = patterns.iloc[-1]

            # 正面形态加分
            if latest_patterns.get('DMI_UPTREND', False):
                final_score += 15
            if latest_patterns.get('DMI_STRONG_TREND', False):
                final_score += 12
            if latest_patterns.get('DMI_VERY_STRONG_TREND', False):
                final_score += 20
            if latest_patterns.get('DMI_PDI_CROSS_UP', False):
                final_score += 10
            if latest_patterns.get('DMI_ADX_RISING', False):
                final_score += 8

            # 负面形态减分
            if latest_patterns.get('DMI_DOWNTREND', False):
                final_score -= 15
            if latest_patterns.get('DMI_WEAK_TREND', False):
                final_score -= 12
            if latest_patterns.get('DMI_NO_TREND', False):
                final_score -= 20
            if latest_patterns.get('DMI_MDI_CROSS_UP', False):
                final_score -= 10
            if latest_patterns.get('DMI_ADX_FALLING', False):
                final_score -= 8

        # 计算置信度
        dmi_data = self._calculate_dmi(data)

        pdi = dmi_data.get('PDI', pd.Series([0])).iloc[-1] if len(dmi_data.get('PDI', pd.Series([0]))) > 0 else 0
        mdi = dmi_data.get('MDI', pd.Series([0])).iloc[-1] if len(dmi_data.get('MDI', pd.Series([0]))) > 0 else 0
        adx = dmi_data.get('ADX', pd.Series([0])).iloc[-1] if len(dmi_data.get('ADX', pd.Series([0]))) > 0 else 0

        # 基于ADX强度计算置信度
        if adx > 40:
            confidence = 0.95  # 极强趋势
        elif adx > 25:
            confidence = 0.8   # 强趋势
        elif adx > 20:
            confidence = 0.6   # 中等趋势
        elif adx > 15:
            confidence = 0.4   # 弱趋势
        else:
            confidence = 0.2   # 无趋势

        # 基于PDI和MDI差值调整置信度
        di_diff = abs(pdi - mdi)
        if di_diff > 20:
            confidence += 0.1
        elif di_diff > 10:
            confidence += 0.05

        # 限制评分和置信度范围
        final_score = max(0, min(100, final_score))
        confidence = max(0.0, min(1.0, confidence))

        return {
            'score': final_score,
            'confidence': confidence,
            'signals': {
                'pdi': pdi,
                'mdi': mdi,
                'adx': adx,
                'trend': 'up' if pdi > mdi else 'down' if mdi > pdi else 'neutral',
                'strength': 'strong' if adx > 25 else 'weak' if adx < 20 else 'medium'
            }
        }

    def set_parameters(self, **kwargs):
        """真实实现：设置DMI参数"""
        # 验证并设置period参数
        if 'period' in kwargs:
            period = kwargs['period']
            if isinstance(period, int) and 5 <= period <= 50:
                self.period = period
            else:
                logger.warning(f"无效的period参数: {period}, 保持原值")

        # 验证并设置adx_period参数
        if 'adx_period' in kwargs:
            adx_period = kwargs['adx_period']
            if isinstance(adx_period, int) and 5 <= adx_period <= 50:
                self.adx_period = adx_period
            else:
                logger.warning(f"无效的adx_period参数: {adx_period}, 保持原值")

        # 验证并设置adx_threshold参数
        if 'adx_threshold' in kwargs:
            adx_threshold = kwargs['adx_threshold']
            if isinstance(adx_threshold, (int, float)) and 10 <= adx_threshold <= 50:
                self.adx_threshold = adx_threshold
            else:
                logger.warning(f"无效的adx_threshold参数: {adx_threshold}, 保持原值")

        # 记录参数变更
        logger.info(f"DMI参数已更新")

    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """真实实现：生成DMI交易信号"""
        return self.get_signals(data, **kwargs)

    def compute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """真实实现：计算DMI指标"""
        return self._calculate_dmi(data, **kwargs)

    def calculate_confidence_Dmi(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """真实实现：计算DMI置信度"""
        if score.empty:
            return 0.3

        # 基础置信度
        confidence = 0.5

        # 基于评分的置信度调整
        latest_score = score.iloc[-1] if not score.empty else 50.0

        # 极端评分提高置信度
        if latest_score > 80 or latest_score < 20:
            confidence += 0.3
        elif latest_score > 70 or latest_score < 30:
            confidence += 0.2
        elif latest_score > 60 or latest_score < 40:
            confidence += 0.1

        # 基于形态的置信度调整
        if not patterns.empty:
            latest_patterns = patterns.iloc[-1]

            # 强势形态提高置信度
            if latest_patterns.get('DMI_VERY_STRONG_TREND', False):
                confidence += 0.2
            if latest_patterns.get('DMI_STRONG_TREND', False):
                confidence += 0.15
            if latest_patterns.get('DMI_UPTREND', False) or latest_patterns.get('DMI_DOWNTREND', False):
                confidence += 0.1

            # 弱势形态降低置信度
            if latest_patterns.get('DMI_NO_TREND', False):
                confidence -= 0.2
            if latest_patterns.get('DMI_WEAK_TREND', False):
                confidence -= 0.1

        # 基于信号的置信度调整
        if signals:
            signal_strength = signals.get('strength', 0)
            confidence += signal_strength * 0.1

        # 限制置信度在0-1范围内
        return max(0.0, min(1.0, confidence))

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """兼容性方法：计算置信度"""
        return self.calculate_confidence_Dmi(score, patterns, signals)

    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """兼容性方法：识别形态"""
        return self.get_patterns(data, **kwargs)

    def calculate_raw_score_dmi(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """兼容性方法：计算原始评分"""
        return self.calculate_raw_score(data, **kwargs)

    def _classify_adx_strength(self, adx_value: float) -> str:
        """真实实现：分类ADX强度"""
        if adx_value > 40:
            return "very_strong"
        elif adx_value > 25:
            return "strong"
        elif adx_value > 20:
            return "medium"
        elif adx_value > 15:
            return "weak"
        else:
            return "no_trend"

    @property
    def minimum_periods(self) -> int:
        """
        DirectionalMovementIndex指标所需的最少数据周期数
        
        计算逻辑：使用默认值
        
        Returns:
            int: 最少需要的数据周期数
        """
        return 30

def get_directionalmovementindex():
    """获取DirectionalMovementIndex实例（通过依赖注入）"""
    try:
        container = get_service_container()
        if not container.is_registered(DirectionalMovementIndex):
            container.register_singleton(DirectionalMovementIndex)
        return container.get_service(DirectionalMovementIndex)
    except Exception:
        # 降级处理：如果依赖注入失败，直接创建实例
        return DirectionalMovementIndex()


def get_dmi_indicator(**kwargs):
    """获取DMI指标实例"""
    return DirectionalMovementIndex(**kwargs)


# 为了兼容指标注册表，创建别名
DMI = DirectionalMovementIndex
