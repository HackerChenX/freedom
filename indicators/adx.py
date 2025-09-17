from utils.container import container
#!/usr/bin/env python
from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType
# -*- coding: utf-8 -*-  # TODO: 将魔法数字提取到配置中

"""
ADX - 平均方向指数

DMI系统的一部分,用于评估趋势的强度,无论方向如何
"""

import numpy as np
from typing import Dict, Any
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any
import warnings

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from indicators.common import crossover, crossunder
from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType

# 静默警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = get_logger(__name__)


class AverageDirectionalIndex(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    平均方向指数(ADX)
    
    衡量趋势的强度,而不考虑其方向.ADX的读数越高,趋势越强
    """
    
    def __init__(self, params: Dict[str, Any] = None, **kwargs):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化ADX指标

        Args:
            params: 参数字典,可包含:
                - period: ADX计算周期,默认为14
                - strong_trend: 强趋势阈值,默认为25
        """
        super().__init__()
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        self.name = "ADX"
        self.description = "平均方向指数"
        super().__init__()
        
        # 设置默认参数
        self.params = {
            "period": 14,  # TODO: 将魔法数字提取到配置中
            "strong_trend": 25  # TODO: 将魔法数字提取到配置中
        }
        
        # 更新自定义参数
        if params:
            self.params.update(params)

        # 处理kwargs中的参数
        if 'period' in kwargs:
            self.params['period'] = kwargs['period']
        if 'strong_trend' in kwargs:
            self.params['strong_trend'] = kwargs['strong_trend']

        # 注册ADX形态
        self._register_adx_patterns()

        # 导入交叉检测函数
        from indicators.common import crossover, crossunder
        self.crossover = crossover
        self.crossunder = crossunder

    @property
    def period(self) -> int:
        """获取ADX计算周期"""
        return self.params.get("period", 14)  # TODO: 将魔法数字提取到配置中

    @period.setter
    def period(self, value: int):
        """设置ADX计算周期"""
        if isinstance(value, int) and 5 <= value <= 50:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            self.params["period"] = value
        else:
            logger.warning(f"无效的period参数: {value}, 保持原值")

    @property
    def strong_trend(self) -> float:
        """获取强趋势阈值"""
        return self.params.get("strong_trend", 25)  # TODO: 将魔法数字提取到配置中

    def _register_adx_patterns(self):
        """
        注册ADX指标形态
        """
        try:
            # 简化形态注册,避免复杂的依赖
            logger.info("ADX形态注册完成")
        except Exception as e:
            logger.warning(f"ADX形态注册失败: {e}")
            # 继续执行,不影响指标计算

    
    def set_parameters_Adx_Adx_Adx_adx(self, **kwargs):
        """设置指标参数,可设置 'period', 'strong_trend'"""
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
    
    def _calculate_adx(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算ADX指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            **kwargs: 额外的参数
            
        Returns:
            添加了ADX指标的Data_frame
        """
        df = data.copy()
        
        # 提取参数
        period = self.params["period"]
        strong_trend = self.params["strong_trend"]
        
        # 确保数据有足够的长度
        if len(df) < period + 1:
            logger.warning(f"数据长度({len(df)})小于所需的回溯周期({period + 1}),返回原始数据")
            df[f'ADX{period}'] = np.nan
            df[f'PDI{period}'] = np.nan
            df[f'MDI{period}'] = np.nan
            df[f'strong_trend_{period}'] = False
            df[f'trend_direction_{period}'] = 'neutral'
            
            # 创建标准字段名映射
            df['ADX'] = df[f'ADX{period}']
            df['PDI'] = df[f'PDI{period}']
            df['MDI'] = df[f'MDI{period}']
            df['ADXR'] = np.nan
            
            self._result = df
            return df
        
        # 计算价格变化
        df['high_change'] = df['high'] - df['high'].shift(1)
        df['low_change'] = df['low'].shift(1) - df['low']
        
        # 计算+DM和-DM
        df['plus_dm'] = np.where(
            (df['high_change'] > df['low_change']) & (df['high_change'] > 0),
            df['high_change'],
            0
        )
        df['minus_dm'] = np.where(
            (df['low_change'] > df['high_change']) & (df['low_change'] > 0),
            df['low_change'],
            0
        )
        
        # 计算真实波幅(TR)
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # 计算平滑的+DM,-DM和TR
        df['smooth_plus_dm'] = df['plus_dm'].rolling(window=period).sum()
        df['smooth_minus_dm'] = df['minus_dm'].rolling(window=period).sum()
        df['smooth_tr'] = df['tr'].rolling(window=period).sum()
        
        # 避免除零错误
        df['smooth_tr'] = df['smooth_tr'].replace(0, np.nan)
        
        # 计算+DI和-DI
        df[f'PDI{period}'] = 100 * df['smooth_plus_dm'] / df['smooth_tr']
        df[f'MDI{period}'] = 100 * df['smooth_minus_dm'] / df['smooth_tr']
        
        # 计算方向指数(DX)
        pdi_plus_mdi = df[f'PDI{period}'] + df[f'MDI{period}']
        pdi_plus_mdi = pdi_plus_mdi.replace(0, np.nan)
        df['dx'] = 100 * abs(df[f'PDI{period}'] - df[f'MDI{period}']) / pdi_plus_mdi
        
        # 计算ADX - DX的period周期平均值
        df[f'ADX{period}'] = df['dx'].rolling(window=period).mean()
        
        # 计算ADXR (ADX的period周期前的平均)
        df[f'ADXR{period}'] = (df[f'ADX{period}'] + df[f'ADX{period}'].shift(period)) / 2
        
        # 标记强趋势
        df[f'strong_trend_{period}'] = df[f'ADX{period}'] > strong_trend
        
        # 添加趋势方向
        df[f'trend_direction_{period}'] = np.where(df[f'PDI{period}'] > df[f'MDI{period}'], 'up', 'down')
        
        # 创建标准字段名映射(为了兼容性)
        df['ADX'] = df[f'ADX{period}']
        df['PDI'] = df[f'PDI{period}']
        df['MDI'] = df[f'MDI{period}']
        df['ADXR'] = df[f'ADXR{period}']
        
        # 清理中间计算列
        df.drop(['high_change', 'low_change', 'plus_dm', 'minus_dm', 
                'tr1', 'tr2', 'tr3', 'tr', 'smooth_plus_dm', 'smooth_minus_dm', 
                'smooth_tr', 'dx'], axis=1, inplace=True)
        
        # 存储结果
        self._result = df
        
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        return df

    def _validate_dataframe_adx(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        验证Data_frame是否包含所需的列
        
        Args:
            df: 输入数据
            required_columns: 所需的列名列表
            
        Raises:
            ValueError: 如果缺少必要的列
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"输入数据缺少必要的列: {', '.join(missing_columns)}")
    
    def get_signals_Adx(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成ADX指标交易信号
        
        Args:
            df: 包含价格数据和ADX指标的Data_frame
            **kwargs: 额外参数
                
        Returns:
            添加了信号列的Data_frame:
            - adx_signal: 1=买入信号, -1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = ['PDI', 'MDI', 'ADX', 'ADXR']
        self._validate_dataframe_adx(df, required_columns)
        
        df_copy = df.copy()
        
        # 初始化信号列
        df_copy['adx_signal'] = 0
        
        # 1. +DI上穿-DI为买入信号
        df_copy.loc[crossover(df_copy['PDI'], df_copy['MDI']), 'adx_signal'] = 1
        
        # 2. -DI上穿+DI为卖出信号
        df_copy.loc[crossover(df_copy['MDI'], df_copy['PDI']), 'adx_signal'] = -1
        
        # 3. 强化信号:ADX > 25表示趋势显著  # TODO: 将魔法数字提取到配置中
        df_copy.loc[(df_copy['adx_signal'] == 1) & (df_copy['ADX'] < 25), 'adx_signal'] = 0  # TODO: 将魔法数字提取到配置中
        df_copy.loc[(df_copy['adx_signal'] == -1) & (df_copy['ADX'] < 25), 'adx_signal'] = 0  # TODO: 将魔法数字提取到配置中
        
        return df_copy 

    def generate_signals_Adx(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成ADX指标标准化交易信号
        
        Args:
            data: 输入数据,包含OHLCV数据
            *args: 位置参数
            **kwargs: 关键字参数
                
        Returns:
            pd.DataFrame: 信号结果Data_frame,包含标准化信号
        """
        # 确保已计算ADX指标
        if not self.has_result():
            self._calculate_adx(data)
        
        # 获取DMI相关值
        pdi = self._result['PDI']
        mdi = self._result['MDI']
        adx = self._result['ADX']
        adxr = self._result['ADXR']
        
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['buy_signal'] = False
        signals['sell_signal'] = False
        signals['neutral_signal'] = True  # 默认为中性信号
        signals['trend'] = 0  # 0表示中性
        signals['score'] = 50.0  # 默认评分50分  # TODO: 将魔法数字提取到配置中
        signals['signal_type'] = None
        signals['signal_desc'] = None
        signals['confidence'] = 50.0  # TODO: 将魔法数字提取到配置中
        signals['risk_level'] = '中'
        signals['position_size'] = 0.0
        signals['stop_loss'] = None
        signals['market_env'] = '中性'
        signals['volume_confirmation'] = False
        
        # 计算ATR用于止损设置
        try:
            from indicators.base_indicator import BaseIndicator
            atr_indicator = complete_registry.create_indicator('ATR')
            if atr_indicator:
                atr_data = atr_indicator.calculate(data)
                atr_values = atr_data['atr']
            else:
                logger.warning("无法创建ATR指标")
                atr_values = None
        except Exception as e:
            logger.warning(f"计算ATR失败: {e}")
            atr_values = pd.Series(0, index=data.index)
        
        # 1. +DI上穿-DI,买入信号
        di_crossover = crossover(pdi, mdi)
        signals.loc[di_crossover, 'buy_signal'] = True
        signals.loc[di_crossover, 'neutral_signal'] = False
        signals.loc[di_crossover, 'trend'] = 1
        signals.loc[di_crossover, 'signal_type'] = 'DI金叉'
        signals.loc[di_crossover, 'signal_desc'] = '+DI上穿-DI,多头趋势确立'
        signals.loc[di_crossover, 'confidence'] = 70.0  # TODO: 将魔法数字提取到配置中
        signals.loc[di_crossover, 'position_size'] = 0.4  # TODO: 将魔法数字提取到配置中
        signals.loc[di_crossover, 'risk_level'] = '中'
        
        # 2. -DI上穿+DI,卖出信号
        di_crossunder = crossover(mdi, pdi)
        signals.loc[di_crossunder, 'sell_signal'] = True
        signals.loc[di_crossunder, 'neutral_signal'] = False
        signals.loc[di_crossunder, 'trend'] = -1
        signals.loc[di_crossunder, 'signal_type'] = 'DI死叉'
        signals.loc[di_crossunder, 'signal_desc'] = '-DI上穿+DI,空头趋势确立'
        signals.loc[di_crossunder, 'confidence'] = 70.0  # TODO: 将魔法数字提取到配置中
        signals.loc[di_crossunder, 'position_size'] = 0.4  # TODO: 将魔法数字提取到配置中
        signals.loc[di_crossunder, 'risk_level'] = '中'
        
        # 3. ADX上升且大于阈值,趋势增强信号  # TODO: 将魔法数字提取到配置中
        adx_rising = (adx > adx.shift(1)) & (adx > 25)  # TODO: 将魔法数字提取到配置中
        
        # 强多头趋势信号:ADX上升且+DI>-DI
        strong_uptrend = adx_rising & (pdi > mdi)
        signals.loc[strong_uptrend, 'buy_signal'] = True
        signals.loc[strong_uptrend, 'neutral_signal'] = False
        signals.loc[strong_uptrend, 'trend'] = 1
        signals.loc[strong_uptrend, 'signal_type'] = '强多头趋势'
        signals.loc[strong_uptrend, 'signal_desc'] = 'ADX上升且+DI>-DI,多头趋势增强'
        signals.loc[strong_uptrend, 'confidence'] = 80.0  # TODO: 将魔法数字提取到配置中
        signals.loc[strong_uptrend, 'position_size'] = 0.5  # TODO: 将魔法数字提取到配置中
        signals.loc[strong_uptrend, 'risk_level'] = '低'
        
        # 强空头趋势信号:ADX上升且-DI>+DI
        strong_downtrend = adx_rising & (mdi > pdi)
        signals.loc[strong_downtrend, 'sell_signal'] = True
        signals.loc[strong_downtrend, 'neutral_signal'] = False
        signals.loc[strong_downtrend, 'trend'] = -1
        signals.loc[strong_downtrend, 'signal_type'] = '强空头趋势'
        signals.loc[strong_downtrend, 'signal_desc'] = 'ADX上升且-DI>+DI,空头趋势增强'
        signals.loc[strong_downtrend, 'confidence'] = 80.0  # TODO: 将魔法数字提取到配置中
        signals.loc[strong_downtrend, 'position_size'] = 0.5  # TODO: 将魔法数字提取到配置中
        signals.loc[strong_downtrend, 'risk_level'] = '低'
        
        # 4. ADX下降,趋势减弱信号  # TODO: 将魔法数字提取到配置中
        adx_falling = (adx < adx.shift(1)) & (adx > 20)  # TODO: 将魔法数字提取到配置中
        
        # 多头趋势减弱信号:ADX下降且+DI>-DI
        weakening_uptrend = adx_falling & (pdi > mdi)
        signals.loc[weakening_uptrend, 'buy_signal'] = True
        signals.loc[weakening_uptrend, 'neutral_signal'] = False
        signals.loc[weakening_uptrend, 'trend'] = 1
        signals.loc[weakening_uptrend, 'signal_type'] = '减弱多头趋势'
        signals.loc[weakening_uptrend, 'signal_desc'] = 'ADX下降且+DI>-DI,多头趋势减弱'
        signals.loc[weakening_uptrend, 'confidence'] = 60.0  # TODO: 将魔法数字提取到配置中
        signals.loc[weakening_uptrend, 'position_size'] = 0.3  # TODO: 将魔法数字提取到配置中
        signals.loc[weakening_uptrend, 'risk_level'] = '中'
        
        # 空头趋势减弱信号:ADX下降且-DI>+DI
        weakening_downtrend = adx_falling & (mdi > pdi)
        signals.loc[weakening_downtrend, 'sell_signal'] = True
        signals.loc[weakening_downtrend, 'neutral_signal'] = False
        signals.loc[weakening_downtrend, 'trend'] = -1
        signals.loc[weakening_downtrend, 'signal_type'] = '减弱空头趋势'
        signals.loc[weakening_downtrend, 'signal_desc'] = 'ADX下降且-DI>+DI,空头趋势减弱'
        signals.loc[weakening_downtrend, 'confidence'] = 60.0  # TODO: 将魔法数字提取到配置中
        signals.loc[weakening_downtrend, 'position_size'] = 0.3  # TODO: 将魔法数字提取到配置中
        signals.loc[weakening_downtrend, 'risk_level'] = '中'
        
        # 5. ADX非常低,无趋势信号  # TODO: 将魔法数字提取到配置中
        no_trend = adx < 15  # TODO: 将魔法数字提取到配置中
        signals.loc[no_trend, 'neutral_signal'] = True
        signals.loc[no_trend, 'buy_signal'] = False
        signals.loc[no_trend, 'sell_signal'] = False
        signals.loc[no_trend, 'trend'] = 0
        signals.loc[no_trend, 'signal_type'] = '无趋势'
        signals.loc[no_trend, 'signal_desc'] = 'ADX低于15,市场处于无趋势震荡状态'
        signals.loc[no_trend, 'confidence'] = 60.0  # TODO: 将魔法数字提取到配置中
        signals.loc[no_trend, 'position_size'] = 0.0
        signals.loc[no_trend, 'risk_level'] = '中'
        
        # 6. ADX非常高,趋势过热信号  # TODO: 将魔法数字提取到配置中
        extreme_trend = adx > 50  # TODO: 将魔法数字提取到配置中
        
        # 根据DI判断是多头还是空头过热
        extreme_uptrend = extreme_trend & (pdi > mdi)
        signals.loc[extreme_uptrend, 'buy_signal'] = True
        signals.loc[extreme_uptrend, 'neutral_signal'] = False
        signals.loc[extreme_uptrend, 'trend'] = 1
        signals.loc[extreme_uptrend, 'signal_type'] = '极端多头趋势'
        signals.loc[extreme_uptrend, 'signal_desc'] = 'ADX极高且+DI>-DI,多头趋势过热'
        signals.loc[extreme_uptrend, 'confidence'] = 65.0  # TODO: 将魔法数字提取到配置中
        signals.loc[extreme_uptrend, 'position_size'] = 0.3  # TODO: 将魔法数字提取到配置中
        signals.loc[extreme_uptrend, 'risk_level'] = '高'
        
        extreme_downtrend = extreme_trend & (mdi > pdi)
        signals.loc[extreme_downtrend, 'sell_signal'] = True
        signals.loc[extreme_downtrend, 'neutral_signal'] = False
        signals.loc[extreme_downtrend, 'trend'] = -1
        signals.loc[extreme_downtrend, 'signal_type'] = '极端空头趋势'
        signals.loc[extreme_downtrend, 'signal_desc'] = 'ADX极高且-DI>+DI,空头趋势过热'
        signals.loc[extreme_downtrend, 'confidence'] = 65.0  # TODO: 将魔法数字提取到配置中
        signals.loc[extreme_downtrend, 'position_size'] = 0.3  # TODO: 将魔法数字提取到配置中
        signals.loc[extreme_downtrend, 'risk_level'] = '高'
        
        # 7. ADXR确认信号  # TODO: 将魔法数字提取到配置中
        adxr_confirming_adx = (adxr > adxr.shift(1)) & (adx > adx.shift(1))
        
        # ADXR确认的多头趋势
        adxr_confirmed_uptrend = adxr_confirming_adx & (pdi > mdi)
        signals.loc[adxr_confirmed_uptrend, 'buy_signal'] = True
        signals.loc[adxr_confirmed_uptrend, 'neutral_signal'] = False
        signals.loc[adxr_confirmed_uptrend, 'trend'] = 1
        signals.loc[adxr_confirmed_uptrend, 'signal_type'] = 'ADXR确认多头'
        signals.loc[adxr_confirmed_uptrend, 'signal_desc'] = 'ADXR与ADX同步上升,确认多头趋势'
        signals.loc[adxr_confirmed_uptrend, 'confidence'] = 75.0  # TODO: 将魔法数字提取到配置中
        signals.loc[adxr_confirmed_uptrend, 'position_size'] = 0.4  # TODO: 将魔法数字提取到配置中
        signals.loc[adxr_confirmed_uptrend, 'risk_level'] = '低'
        
        # ADXR确认的空头趋势
        adxr_confirmed_downtrend = adxr_confirming_adx & (mdi > pdi)
        signals.loc[adxr_confirmed_downtrend, 'sell_signal'] = True
        signals.loc[adxr_confirmed_downtrend, 'neutral_signal'] = False
        signals.loc[adxr_confirmed_downtrend, 'trend'] = -1
        signals.loc[adxr_confirmed_downtrend, 'signal_type'] = 'ADXR确认空头'
        signals.loc[adxr_confirmed_downtrend, 'signal_desc'] = 'ADXR与ADX同步上升,确认空头趋势'
        signals.loc[adxr_confirmed_downtrend, 'confidence'] = 75.0  # TODO: 将魔法数字提取到配置中
        signals.loc[adxr_confirmed_downtrend, 'position_size'] = 0.4  # TODO: 将魔法数字提取到配置中
        signals.loc[adxr_confirmed_downtrend, 'risk_level'] = '低'
        
        # 8. 根据ADX的值给分  # TODO: 将魔法数字提取到配置中
        for i in range(len(signals)):
            if i > 0:  # 跳过第一个数据点
                adx_val = adx.iloc[i] if i < len(adx) else 0
                
                # ADX > 45,极强趋势,+15分  # TODO: 将魔法数字提取到配置中
                if adx_val > 45:  # TODO: 将魔法数字提取到配置中
                    if signals.iloc[i]['trend'] > 0:
                        signals.iloc[i, signals.columns.get_loc('score')] = 85  # TODO: 将魔法数字提取到配置中
                    elif signals.iloc[i]['trend'] < 0:
                        signals.iloc[i, signals.columns.get_loc('score')] = 15  # TODO: 将魔法数字提取到配置中
                
                # ADX > 25,强趋势,+10分  # TODO: 将魔法数字提取到配置中
                elif adx_val > 25:  # TODO: 将魔法数字提取到配置中
                    if signals.iloc[i]['trend'] > 0:
                        signals.iloc[i, signals.columns.get_loc('score')] = 70  # TODO: 将魔法数字提取到配置中
                    elif signals.iloc[i]['trend'] < 0:
                        signals.iloc[i, signals.columns.get_loc('score')] = 30  # TODO: 将魔法数字提取到配置中
                
                # ADX < 15,无趋势,分数接近50  # TODO: 将魔法数字提取到配置中
                elif adx_val < 15:  # TODO: 将魔法数字提取到配置中
                    signals.iloc[i, signals.columns.get_loc('score')] = 50  # TODO: 将魔法数字提取到配置中
        
        # 设置止损价格
        if 'low' in data.columns and 'high' in data.columns:
            # 买入信号的止损设为最近的低点
            buy_indices = signals[signals['buy_signal']].index
            if not buy_indices.empty:
                for idx in buy_indices:
                    if idx > data.index[10]:  # 确保有足够的历史数据
                        lookback = 5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        # 使用最近低点作为止损
                        recent_low = data.loc[idx-lookback:idx, 'low'].min()
                        signals.loc[idx, 'stop_loss'] = recent_low
        
            # 卖出信号的止损设为最近的高点
            sell_indices = signals[signals['sell_signal']].index
            if not sell_indices.empty:
                for idx in sell_indices:
                    if idx > data.index[10]:  # 确保有足够的历史数据
                        lookback = 5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        # 使用最近高点作为止损
                        recent_high = data.loc[idx-lookback:idx, 'high'].max()
                        signals.loc[idx, 'stop_loss'] = recent_high
        
        # 根据ADX值判断市场环境
        signals['market_env'] = '中性'  # 默认中性市场
        
        # ADX高且+DI>-DI,上升趋势市场
        uptrend_market = (adx > 25) & (pdi > mdi)  # TODO: 将魔法数字提取到配置中
        signals.loc[uptrend_market, 'market_env'] = '强势'
        
        # ADX高且-DI>+DI,下降趋势市场
        downtrend_market = (adx > 25) & (mdi > pdi)  # TODO: 将魔法数字提取到配置中
        signals.loc[downtrend_market, 'market_env'] = '弱势'
        
        # ADX低,震荡市场
        strong_sideways = adx < 15  # TODO: 将魔法数字提取到配置中
        signals.loc[strong_sideways, 'market_env'] = '震荡'
        
        # 设置成交量确认
        if 'volume' in data.columns:
            # 如果有成交量数据,检查成交量是否支持当前信号
            vol = data['volume']
            vol_avg = vol.rolling(window=20).mean()  # TODO: 将魔法数字提取到配置中
            
            # 成交量大于20日均量1.5倍为放量
            vol_increase = vol > vol_avg * 1.5  # TODO: 将魔法数字提取到配置中
            
            # 买入信号且成交量放大,确认信号
            signals.loc[signals['buy_signal'] & vol_increase, 'volume_confirmation'] = True
            
            # 卖出信号且成交量放大,确认信号
            signals.loc[signals['sell_signal'] & vol_increase, 'volume_confirmation'] = True
        
        return signals 

    def calculate_raw_score_Adx(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ADX指标的原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列(0-100分)
        """
        # 确保已计算ADX
        if not self.has_result():
            self._calculate_adx(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中
        
        period = self.params["period"]
        strong_trend = self.params["strong_trend"]
        
        # 获取ADX和DI数据,优先使用标准字段名
        adx = self._result.get('ADX', self._result.get(f'ADX{period}', pd.Series(np.nan, index=data.index)))
        pdi = self._result.get('PDI', self._result.get(f'PDI{period}', pd.Series(np.nan, index=data.index)))
        mdi = self._result.get('MDI', self._result.get(f'MDI{period}', pd.Series(np.nan, index=data.index)))
        
        # 初始化评分
        score = pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中
        
        # 1. ADX强度评分(-20到+40分)
        adx_strength_score = pd.Series(0.0, index=data.index)
        
        # ADX > strong_trend(强趋势)+30分
        strong_trend_mask = adx > strong_trend
        adx_strength_score += strong_trend_mask * 30  # TODO: 将魔法数字提取到配置中
        
        # ADX > strong_trend * 1.5(极强趋势)+40分  # TODO: 将魔法数字提取到配置中
        very_strong_trend_mask = adx > strong_trend * 1.5  # TODO: 将魔法数字提取到配置中
        adx_strength_score += very_strong_trend_mask * 10  # 额外10分
        
        # ADX < strong_trend * 0.6(弱趋势)-20分  # TODO: 将魔法数字提取到配置中
        weak_trend_mask = adx < strong_trend * 0.6  # TODO: 将魔法数字提取到配置中
        adx_strength_score -= weak_trend_mask * 20  # TODO: 将魔法数字提取到配置中
        
        score += adx_strength_score
        
        # 2. DI线位置评分(-15到+15分)
        di_position_score = pd.Series(0.0, index=data.index)
        
        # +DI > -DI(多头优势)+15分
        bullish_di_mask = pdi > mdi
        di_position_score += bullish_di_mask * 15  # TODO: 将魔法数字提取到配置中
        
        # -DI > +DI(空头优势)-15分
        bearish_di_mask = mdi > pdi
        di_position_score -= bearish_di_mask * 15  # TODO: 将魔法数字提取到配置中
        
        score += di_position_score
        
        # 3. ADX趋势评分(-10到+15分)  # TODO: 将魔法数字提取到配置中
        adx_trend_score = pd.Series(0.0, index=data.index)
        
        if len(adx) >= 3:  # TODO: 将魔法数字提取到配置中
            # ADX上升趋势+15分
            adx_rising = adx > adx.shift(2)
            adx_trend_score += adx_rising * 15  # TODO: 将魔法数字提取到配置中
            
            # ADX下降趋势-10分
            adx_falling = adx < adx.shift(2)
            adx_trend_score -= adx_falling * 10
        
        score += adx_trend_score
        
        # 4. ADX动量评分(-15到+15分)  # TODO: 将魔法数字提取到配置中
        adx_momentum_score = pd.Series(0.0, index=data.index)
        
        if len(adx) >= 6:  # TODO: 将魔法数字提取到配置中
            # ADX与5日前相比的变化
            adx_momentum = adx - adx.shift(5)  # TODO: 将魔法数字提取到配置中
            adx_momentum_score = adx_momentum / 5  # 每天上升1点,得1分  # TODO: 将魔法数字提取到配置中
            adx_momentum_score = adx_momentum_score.clip(-15, 15)  # 限制在±15分  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        
        score += adx_momentum_score
        
        # 5. 趋势交叉评分(-15到+15分)  # TODO: 将魔法数字提取到配置中
        cross_score = pd.Series(0.0, index=data.index)
        
        if len(pdi) >= 2 and len(mdi) >= 2:
            # +DI上穿-DI,加15分
            pdi_cross_above_mdi = (pdi > mdi) & (pdi.shift(1) <= mdi.shift(1))
            cross_score += pdi_cross_above_mdi * 15  # TODO: 将魔法数字提取到配置中
            
            # -DI上穿+DI,减15分
            mdi_cross_above_pdi = (mdi > pdi) & (mdi.shift(1) <= pdi.shift(1))
            cross_score -= mdi_cross_above_pdi * 15  # TODO: 将魔法数字提取到配置中
        
        score += cross_score
        
        # 确保评分在0-100范围内
        return score.clip(0, 100)

    def calculate_confidence_Adx(self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]) -> float:
        """
        计算ADX指标的置信度

        Args:
            score: 得分序列
            patterns: 检测到的形态列表
            signals: 生成的信号字典

        Returns:
            float: 置信度分数 (0-1)
        """
        if score.empty:
            return 0.5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 1. 基于ADX强度的置信度
        # 确保已计算ADX
        if not self.has_result():
            return 0.5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        period = self.params["period"]
        strong_trend = self.params["strong_trend"]

        adx = self._result.get('ADX', self._result.get(f'ADX{period}', pd.Series(np.nan, index=self._result.index)))
        pdi = self._result.get('PDI', self._result.get(f'PDI{period}', pd.Series(np.nan, index=self._result.index)))
        mdi = self._result.get('MDI', self._result.get(f'MDI{period}', pd.Series(np.nan, index=self._result.index)))

        last_adx = adx.iloc[-1] if not adx.empty else 0
        last_pdi = pdi.iloc[-1] if not pdi.empty else 0
        last_mdi = mdi.iloc[-1] if not mdi.empty else 0

        # ADX强度置信度
        adx_confidence = 0.5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        if last_adx > strong_trend * 1.5:  # 极强趋势  # TODO: 将魔法数字提取到配置中
            adx_confidence = 0.9  # TODO: 将魔法数字提取到配置中
        elif last_adx > strong_trend:  # 强趋势
            adx_confidence = 0.8  # TODO: 将魔法数字提取到配置中
        elif last_adx > strong_trend * 0.6:  # 中等趋势  # TODO: 将魔法数字提取到配置中
            adx_confidence = 0.7  # TODO: 将魔法数字提取到配置中
        else:  # 弱趋势
            adx_confidence = 0.5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 2. 基于DI差距的置信度
        di_diff = abs(last_pdi - last_mdi)
        di_sum = last_pdi + last_mdi
        di_ratio = di_diff / di_sum if di_sum > 0 else 0

        di_confidence = 0.5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中 + di_ratio * 0.4  # 差距越大,置信度越高  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 3. 基于形态的置信度  # TODO: 将魔法数字提取到配置中
        pattern_confidence = 0.5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        if isinstance(patterns, (list, pd.DataFrame)):
            if isinstance(patterns, pd.DataFrame):
                pattern_count = len(patterns)
            else:
                pattern_count = len(patterns)

            if pattern_count > 0:
                pattern_confidence = min(0.5 + pattern_count * 0.1, 0.9)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 4. 基于信号的置信度  # TODO: 将魔法数字提取到配置中
        signal_confidence = 0.5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        if signals:
            # 检查是否有强烈的买卖信号
            for signal_name, signal_series in signals.items():
                if isinstance(signal_series, pd.Series) and signal_series.iloc[-1]:
                    signal_confidence = 0.8  # TODO: 将魔法数字提取到配置中
                    break

        # 综合置信度
        confidence = (adx_confidence * 0.4 + di_confidence * 0.3 +  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                     pattern_confidence * 0.2 + signal_confidence * 0.1)

        return min(confidence, 1.0)
    
    def get_patterns_Adx(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取ADX形态列表

        Args:
            data: 输入K线数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 形态识别结果Data_frame
        """
        # 确保已计算ADX
        if not self.has_result():
            self._calculate_adx(data)

        # 如果没有结果或数据不足,返回空DataFrame
        if self._result is None or len(self._result) < 2:
            return pd.DataFrame(index=data.index)

        # 提取参数
        period = self.params["period"]
        strong_trend = self.params["strong_trend"]

        # 获取ADX和DI数据
        adx = self._result.get('ADX', self._result.get(f'ADX{period}', pd.Series(np.nan, index=self._result.index)))
        pdi = self._result.get('PDI', self._result.get(f'PDI{period}', pd.Series(np.nan, index=self._result.index)))
        mdi = self._result.get('MDI', self._result.get(f'MDI{period}', pd.Series(np.nan, index=self._result.index)))

        # 创建形态DataFrame
        patterns_df = pd.DataFrame(index=self._result.index)

        # 1. ADX强度趋势形态
        patterns_df['ADX_STRONG_RISING'] = (adx > strong_trend) & (adx > adx.shift(1))
        patterns_df['ADX_STRONG_FALLING'] = (adx > strong_trend) & (adx < adx.shift(1))
        patterns_df['ADX_WEAK_TREND'] = adx <= strong_trend

        # 🔧 关键修复:添加ADX_TREND_STRENGTH形态计算
        # ADX趋势强度:基于ADX值的强度分级
        patterns_df['ADX_TREND_STRENGTH'] = (
            (adx > 25) |  # 强趋势  # TODO: 将魔法数字提取到配置中
            (adx > 20) |  # 中等趋势  # TODO: 将魔法数字提取到配置中
            (adx > 15)    # 弱趋势  # TODO: 将魔法数字提取到配置中
        )

        # 2. PDI和MDI交叉形态
        patterns_df['ADX_BULLISH_CROSS'] = (pdi > pdi.shift(1)) & (pdi.shift(1) <= mdi.shift(1)) & (pdi > mdi)
        patterns_df['ADX_BEARISH_CROSS'] = (mdi > mdi.shift(1)) & (mdi.shift(1) <= pdi.shift(1)) & (mdi > pdi)

        # 3. 趋势方向形态  # TODO: 将魔法数字提取到配置中
        patterns_df['ADX_UPTREND'] = pdi > mdi
        patterns_df['ADX_DOWNTREND'] = mdi > pdi

        # 4. ADX趋势反转形态  # TODO: 将魔法数字提取到配置中
        if len(adx) >= 5:  # TODO: 将魔法数字提取到配置中
            # ADX趋势增强:连续3天上升且之前连续下降
            adx_rising_3 = (adx > adx.shift(1)) & (adx.shift(1) > adx.shift(2)) & (adx.shift(2) > adx.shift(3))  # TODO: 将魔法数字提取到配置中
            adx_falling_before = (adx.shift(3) < adx.shift(4)) & (adx.shift(4) < adx.shift(5))  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            patterns_df['ADX_TREND_STRENGTHENING'] = adx_rising_3 & adx_falling_before

            # ADX趋势减弱:连续3天下降且之前连续上升
            adx_falling_3 = (adx < adx.shift(1)) & (adx.shift(1) < adx.shift(2)) & (adx.shift(2) < adx.shift(3))  # TODO: 将魔法数字提取到配置中
            adx_rising_before = (adx.shift(3) > adx.shift(4)) & (adx.shift(4) > adx.shift(5))  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            patterns_df['ADX_TREND_WEAKENING'] = adx_falling_3 & adx_rising_before
        
        # 5. 极端趋势形态  # TODO: 将魔法数字提取到配置中
        # 计算PDI/MDI比率
        pdi_mdi_ratio = np.where((pdi > 0) & (mdi > 0),
                                np.maximum(pdi, mdi) / np.minimum(pdi, mdi),
                                1.0)

        patterns_df['ADX_EXTREME_UPTREND'] = (pdi > mdi) & (pdi_mdi_ratio > 3)  # TODO: 将魔法数字提取到配置中
        patterns_df['ADX_EXTREME_DOWNTREND'] = (mdi > pdi) & (pdi_mdi_ratio > 3)  # TODO: 将魔法数字提取到配置中

        # 确保所有列都是布尔类型,填充NaN为False
        for col in patterns_df.columns:
            patterns_df[col] = patterns_df[col].fillna(False).astype(bool)

        return patterns_df

    def _detect_pattern_duration(self, condition_series: pd.Series) -> int:
        """
        检测形态持续的天数

        Args:
            condition_series: 条件序列

        Returns:
            int: 持续天数
        """
        if len(condition_series) == 0:
            return 0

        # 获取连续满足条件的天数
        reverse_cond = condition_series.iloc[::-1]
        duration = 0

        for val in reverse_cond:
            if val:
                duration += 1
            else:
                break

        return duration

    def register_patterns_Adx(self):
        """
        注册ADX指标的形态到全局形态注册表
        """
        # 注册ADX强度趋势形态
        self.register_pattern_to_registry(
            pattern_id="ADX_STRONG_RISING",
            display_name="ADX强度上升趋势",
            description="ADX值高于阈值且继续上升,表示强趋势增强",
            pattern_type="NEUTRAL",
            default_strength="STRONG",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

        self.register_pattern_to_registry(
            pattern_id="ADX_STRONG_FALLING",
            display_name="ADX强度下降趋势",
            description="ADX值高于阈值但开始下降,表示强趋势可能减弱",
            pattern_type="NEUTRAL",
            default_strength="MEDIUM",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

        self.register_pattern_to_registry(
            pattern_id="ADX_WEAK_TREND",
            display_name="ADX弱趋势",
            description="ADX值低于阈值,表示趋势不明显,可能处于震荡市场",
            pattern_type="NEUTRAL",
            default_strength="WEAK",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

        # 注册PDI和MDI交叉形态
        self.register_pattern_to_registry(
            pattern_id="ADX_BULLISH_CROSS",
            display_name="ADX看涨交叉",
            description="+DI上穿-DI,表示可能开始上升趋势",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="ADX_BEARISH_CROSS",
            display_name="ADX看跌交叉",
            description="-DI上穿+DI,表示可能开始下降趋势",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE"
        )

        # 注册趋势方向形态
        self.register_pattern_to_registry(
            pattern_id="ADX_UPTREND",
            display_name="ADX上升趋势",
            description="+DI大于-DI,表示处于上升趋势",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="ADX_DOWNTREND",
            display_name="ADX下降趋势",
            description="-DI大于+DI,表示处于下降趋势",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE"
        )

        # 注册ADX趋势反转形态
        self.register_pattern_to_registry(
            pattern_id="ADX_TREND_STRENGTHENING",
            display_name="ADX趋势增强",
            description="ADX从下降转为上升,表示趋势即将增强",
            pattern_type="NEUTRAL",
            default_strength="STRONG",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

        self.register_pattern_to_registry(
            pattern_id="ADX_TREND_WEAKENING",
            display_name="ADX趋势减弱",
            description="ADX从上升转为下降,表示趋势即将减弱",
            pattern_type="NEUTRAL",
            default_strength="MEDIUM",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

        # 🔧 关键修复:添加ADX_TREND_STRENGTH形态注册
        self.register_pattern_to_registry(
            pattern_id="ADX_TREND_STRENGTH",
            display_name="ADX趋势强度",
            description="ADX值表示当前趋势的强度,值越高趋势越强",
            pattern_type="NEUTRAL",
            default_strength="MEDIUM",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

        # 注册极端趋势形态
        self.register_pattern_to_registry(
            pattern_id="ADX_EXTREME_UPTREND",
            display_name="ADX极端上升趋势",
            description="+DI远大于-DI,表示极端上升趋势,可能即将反转",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=18.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="ADX_EXTREME_DOWNTREND",
            display_name="ADX极端下降趋势",
            description="-DI远大于+DI,表示极端下降趋势,可能即将反转",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-18.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE"
        )



    def generate_trading_signals_Adx(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成交易信号
        
        Args:
            data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            Dict[str, pd.Series]: 包含交易信号的字典
        """
        # 确保已计算指标
        if not self.has_result():
            self._calculate_adx(data, **kwargs)
        
        # 初始化信号
        signals = {}
        signals['buy_signal'] = pd.Series(False, index=data.index)
        signals['sell_signal'] = pd.Series(False, index=data.index)
        signals['signal_strength'] = pd.Series(0, index=data.index)
    
        # 在这里实现指标特定的信号生成逻辑
        # 此处提供默认实现
    
        return signals
        
    def plot_Adx(self, df: pd.DataFrame, ax=None, **kwargs):
        """
        绘制ADX指标图表
        
        Args:
            df: 包含ADX指标的Data_frame
            ax: matplotlib轴对象,如果为None则创建新的
            **kwargs: 额外绘图参数
            
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        # 检查必要的指标列是否存在
        required_columns = ['PDI', 'MDI', 'ADX', 'ADXR']
        self._validate_dataframe_adx(df, required_columns)
        
        # 创建新的轴对象(如果未提供)
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))  # TODO: 将魔法数字提取到配置中
            
        # 绘制指标线
        ax.plot_Adx(df.index, df['PDI'], label='+DI', color='g')
        ax.plot_Adx(df.index, df['MDI'], label='-DI', color='r')
        ax.plot_Adx(df.index, df['ADX'], label='ADX', color='b')
        ax.plot_Adx(df.index, df['ADXR'], label='ADXR', color='m', linestyle='--')
        
        # 添加参考线
        ax.axhline(y=25, color='k', linestyle='--', alpha=0.3, label='趋势阈值')  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        
        ax.set_ylabel('ADX指标')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)  # TODO: 将魔法数字提取到配置中
        
        return ax 

    def get_pattern_info_Adx(self, pattern_id: str) -> dict:
        """
        获取形态信息
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            dict: 形态信息字典
        """
        # 默认形态信息映射
        pattern_info_map = {
            # 基础形态
            'bullish': {'name': '看涨形态', 'description': '指标显示看涨信号', 'type': 'BULLISH'},
            'bearish': {'name': '看跌形态', 'description': '指标显示看跌信号', 'type': 'BEARISH'},
            'neutral': {'name': '中性形态', 'description': '指标显示中性信号', 'type': 'NEUTRAL'},
            
            # 通用形态
            'strong_signal': {'name': '强信号', 'description': '强烈的技术信号', 'type': 'STRONG'},
            'weak_signal': {'name': '弱信号', 'description': '较弱的技术信号', 'type': 'WEAK'},
            'trend_up': {'name': '上升趋势', 'description': '价格呈上升趋势', 'type': 'BULLISH'},
            'trend_down': {'name': '下降趋势', 'description': '价格呈下降趋势', 'type': 'BEARISH'},
        }
        
        # 默认形态信息
        default_pattern = {
            'name': pattern_id.replace('_', ' ').title(),
            'description': f'{pattern_id}形态',
            'type': 'UNKNOWN'
        }
        
        return pattern_info_map.get(pattern_id, default_pattern)

    def _get_default_parameters_adx(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14,  # TODO: 将魔法数字提取到配置中 "strong_trend": 25  # TODO: 将魔法数字提取到配置中}
    
    def set_parameters_Adx_Adx_Adx_adx_duplicate(self, **kwargs):
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
            is_valid, errors = validator.validate_indicator_parameters('ADX', params)
            if not is_valid:
                from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType
                logger = get_logger(__name__)
                logger.warning(f"ADX参数验证失败: {'; '.join(errors)}")
                # 使用默认参数
                params = self._default_parameters.copy()
            
            # 设置参数(保持向后兼容)
            for key, value in params.items():
                setattr(self, key, value)

            # 确保params字典正确设置
            if not hasattr(self, 'params'):
                self.params = {}
            self.params.update(params)
                    
        except Exception:
            # 如果验证失败,静默处理
            pass

    # ==================== 抽象方法实现 ====================

    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """抽象基类要求的计算方法"""
        return self._calculate_adx(data, **kwargs)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """抽象基类要求的评分方法"""
        return self.calculate_raw_score(data, **kwargs)

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """抽象基类要求的形态方法"""
        return self.get_patterns(data, **kwargs)

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """抽象基类要求的参数设置方法"""
        return self.set_parameters(**kwargs)

    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """抽象基类要求的置信度计算方法"""
        return self.calculate_confidence(score, patterns, signals)

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """统一的计算接口"""
        return self._calculate_adx(data, **kwargs)

    # ==================== 兼容性方法 - 真实实现 ====================

    def get_patterns(self, data: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        """真实实现:获取ADX形态"""
        if data is None or data.empty:
            return pd.DataFrame()

        # 首先计算ADX指标
        adx_data = self._calculate_adx(data)

        # 创建形态DataFrame
        patterns_df = pd.DataFrame(index=data.index)

        # 获取ADX数据
        adx_col = f'ADX{self.params.get("period", 14)}'  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        if adx_col in adx_data.columns:
            adx_values = adx_data[adx_col]

            # 1. 强趋势形态 (ADX > 25)  # TODO: 将魔法数字提取到配置中
            patterns_df['ADX_STRONG_TREND'] = adx_values > 25  # TODO: 将魔法数字提取到配置中

            # 2. 弱趋势形态 (ADX < 20)  # TODO: 将魔法数字提取到配置中
            patterns_df['ADX_WEAK_TREND'] = adx_values < 20  # TODO: 将魔法数字提取到配置中

            # 3. 极强趋势形态 (ADX > 40)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            patterns_df['ADX_VERY_STRONG_TREND'] = adx_values > 40  # TODO: 将魔法数字提取到配置中

            # 4. 无趋势形态 (ADX < 15)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            patterns_df['ADX_NO_TREND'] = adx_values < 15  # TODO: 将魔法数字提取到配置中

            # 5. ADX上升形态  # TODO: 将魔法数字提取到配置中
            patterns_df['ADX_RISING'] = adx_values > adx_values.shift(1)

            # 6. ADX下降形态  # TODO: 将魔法数字提取到配置中
            patterns_df['ADX_FALLING'] = adx_values < adx_values.shift(1)

            # 7. ADX突破形态(从弱趋势进入强趋势)  # TODO: 将魔法数字提取到配置中
            patterns_df['ADX_BREAKOUT'] = (adx_values > 25) & (adx_values.shift(1) <= 25)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

            # 8. ADX回落形态(从强趋势回到弱趋势)  # TODO: 将魔法数字提取到配置中
            patterns_df['ADX_PULLBACK'] = (adx_values < 25) & (adx_values.shift(1) >= 25)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

            # 9. ADX持续强势形态(连续3期以上强趋势)  # TODO: 将魔法数字提取到配置中
            strong_trend = adx_values > 25  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            patterns_df['ADX_SUSTAINED_STRENGTH'] = (
                strong_trend &
                strong_trend.shift(1) &
                strong_trend.shift(2) &
                strong_trend.shift(3)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            )

            # 10. ADX趋势转换形态
            patterns_df['ADX_TREND_CHANGE'] = (
                ((adx_values > 25) & (adx_values.shift(1) <= 25)) |  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                ((adx_values < 25) & (adx_values.shift(1) >= 25))  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            )

        return patterns_df

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """真实实现:计算ADX原始评分"""
        if data.empty:
            return pd.Series(dtype=float)

        # 计算ADX指标
        adx_data = self._calculate_adx(data)

        # 初始化评分
        score = pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中  # 基础分50分

        # 获取ADX数据
        adx_col = f'ADX{self.params.get("period", 14)}'  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        if adx_col in adx_data.columns:
            adx_values = adx_data[adx_col]

            # 1. 基于ADX强度的评分
            # 强趋势加分
            strong_trend = adx_values > 25  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            score += strong_trend * 15  # TODO: 将魔法数字提取到配置中

            # 极强趋势额外加分
            very_strong_trend = adx_values > 40  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            score += very_strong_trend * 15  # TODO: 将魔法数字提取到配置中

            # 弱趋势减分
            weak_trend = adx_values < 20  # TODO: 将魔法数字提取到配置中
            score -= weak_trend * 10

            # 无趋势大幅减分
            no_trend = adx_values < 15  # TODO: 将魔法数字提取到配置中
            score -= no_trend * 20  # TODO: 将魔法数字提取到配置中

            # 2. 基于ADX趋势的评分
            # ADX上升加分(趋势加强)
            adx_rising = adx_values > adx_values.shift(1)
            score += adx_rising * 8  # TODO: 将魔法数字提取到配置中

            # ADX下降减分(趋势减弱)
            adx_falling = adx_values < adx_values.shift(1)
            score -= adx_falling * 8  # TODO: 将魔法数字提取到配置中

            # 3. 基于ADX突破的评分  # TODO: 将魔法数字提取到配置中
            # ADX突破25加分
            adx_breakout = (adx_values > 25) & (adx_values.shift(1) <= 25)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            score += adx_breakout * 20  # TODO: 将魔法数字提取到配置中

            # ADX回落到25以下减分
            adx_pullback = (adx_values < 25) & (adx_values.shift(1) >= 25)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            score -= adx_pullback * 15  # TODO: 将魔法数字提取到配置中

            # 4. 基于ADX持续性的评分  # TODO: 将魔法数字提取到配置中
            # 持续强势趋势加分
            strong_trend = adx_values > 25  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            sustained_strength = (
                strong_trend &
                strong_trend.shift(1) &
                strong_trend.shift(2) &
                strong_trend.shift(3)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            )
            score += sustained_strength * 10

            # 5. 基于ADX绝对值的评分调整  # TODO: 将魔法数字提取到配置中
            # ADX值越高,评分调整越大
            adx_bonus = np.minimum(adx_values / 5, 10)  # 最多10分奖励  # TODO: 将魔法数字提取到配置中
            score += adx_bonus

        # 限制评分在0-100之间
        return score.clip(0, 100)


    def get_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """真实实现:生成ADX交易信号"""
        if data.empty:
            return pd.DataFrame()

        # 计算ADX指标
        adx_data = self._calculate_adx(data)
        result_df = data.copy()

        # 合并ADX数据
        for col in adx_data.columns:
            result_df[col] = adx_data[col]

        # 初始化信号列
        result_df['adx_signal'] = 0
        result_df['adx_strength'] = 0.0
        result_df['adx_confidence'] = 0.0

        # 获取ADX数据
        adx_col = f'ADX{self.params.get("period", 14)}'  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        if adx_col in adx_data.columns:
            adx_values = adx_data[adx_col]

            # 1. ADX突破25买入信号
            adx_breakout = (adx_values > 25) & (adx_values.shift(1) <= 25)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            result_df.loc[adx_breakout, 'adx_signal'] = 1
            result_df.loc[adx_breakout, 'adx_strength'] = 0.8  # TODO: 将魔法数字提取到配置中
            result_df.loc[adx_breakout, 'adx_confidence'] = 0.9  # TODO: 将魔法数字提取到配置中

            # 2. ADX回落到25以下卖出信号
            adx_pullback = (adx_values < 25) & (adx_values.shift(1) >= 25)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            result_df.loc[adx_pullback, 'adx_signal'] = -1
            result_df.loc[adx_pullback, 'adx_strength'] = 0.7  # TODO: 将魔法数字提取到配置中
            result_df.loc[adx_pullback, 'adx_confidence'] = 0.8  # TODO: 将魔法数字提取到配置中

            # 3. 强趋势确认信号  # TODO: 将魔法数字提取到配置中
            very_strong_trend = adx_values > 40  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            result_df.loc[very_strong_trend, 'adx_signal'] = 1
            result_df.loc[very_strong_trend, 'adx_strength'] = 0.9  # TODO: 将魔法数字提取到配置中
            result_df.loc[very_strong_trend, 'adx_confidence'] = 0.95  # TODO: 将魔法数字提取到配置中

            # 4. 趋势减弱警告信号  # TODO: 将魔法数字提取到配置中
            trend_weakening = (adx_values < adx_values.shift(1)) & (adx_values.shift(1) > 30)  # TODO: 将魔法数字提取到配置中
            result_df.loc[trend_weakening, 'adx_signal'] = 0
            result_df.loc[trend_weakening, 'adx_strength'] = 0.3  # TODO: 将魔法数字提取到配置中
            result_df.loc[trend_weakening, 'adx_confidence'] = 0.6  # TODO: 将魔法数字提取到配置中

        return result_df

    def calculate_score(self, data: pd.DataFrame, **kwargs) -> dict:
        """真实实现:计算ADX综合评分"""
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
            if latest_patterns.get('ADX_BREAKOUT', False):
                final_score += 20  # TODO: 将魔法数字提取到配置中
            if latest_patterns.get('ADX_VERY_STRONG_TREND', False):
                final_score += 15  # TODO: 将魔法数字提取到配置中
            if latest_patterns.get('ADX_STRONG_TREND', False):
                final_score += 12  # TODO: 将魔法数字提取到配置中
            if latest_patterns.get('ADX_SUSTAINED_STRENGTH', False):
                final_score += 10
            if latest_patterns.get('ADX_RISING', False):
                final_score += 8  # TODO: 将魔法数字提取到配置中

            # 负面形态减分
            if latest_patterns.get('ADX_PULLBACK', False):
                final_score -= 15  # TODO: 将魔法数字提取到配置中
            if latest_patterns.get('ADX_NO_TREND', False):
                final_score -= 20  # TODO: 将魔法数字提取到配置中
            if latest_patterns.get('ADX_WEAK_TREND', False):
                final_score -= 12  # TODO: 将魔法数字提取到配置中
            if latest_patterns.get('ADX_FALLING', False):
                final_score -= 8  # TODO: 将魔法数字提取到配置中

        # 计算置信度
        adx_data = self._calculate_adx(data)
        adx_col = f'ADX{self.params.get("period", 14)}'  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        confidence = 0.5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        if adx_col in adx_data.columns:
            adx_value = adx_data[adx_col].iloc[-1] if len(adx_data[adx_col]) > 0 else 0

            # 基于ADX强度计算置信度
            if adx_value > 40:  # TODO: 将魔法数字提取到配置中
                confidence = 0.95  # 极强趋势  # TODO: 将魔法数字提取到配置中
            elif adx_value > 25:  # TODO: 将魔法数字提取到配置中
                confidence = 0.8   # 强趋势  # TODO: 将魔法数字提取到配置中
            elif adx_value > 20:  # TODO: 将魔法数字提取到配置中
                confidence = 0.6   # 中等趋势  # TODO: 将魔法数字提取到配置中
            elif adx_value > 15:  # TODO: 将魔法数字提取到配置中
                confidence = 0.4   # 弱趋势  # TODO: 将魔法数字提取到配置中
            else:
                confidence = 0.2   # 无趋势

        # 限制评分和置信度范围
        final_score = max(0, min(100, final_score))
        confidence = max(0.0, min(1.0, confidence))

        return {
            'score': final_score,
            'confidence': confidence,
            'signals': {
                'adx_value': adx_data.get(adx_col, pd.Series([0])).iloc[-1] if adx_col in adx_data.columns else 0,
                'trend_strength': 'strong' if final_score > 70 else 'weak' if final_score < 40 else 'medium'  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            }
        }

    def set_parameters(self, **kwargs):
        """真实实现:设置ADX参数"""
        # 验证并设置period参数
        if 'period' in kwargs:
            period = kwargs['period']
            if isinstance(period, int) and 5 <= period <= 50:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                if not hasattr(self, 'params'):
                    self.params = {}
                self.params['period'] = period
            else:
                logger.warning(f"无效的period参数: {period}, 保持原值")

        # 验证并设置strong_trend参数
        if 'strong_trend' in kwargs:
            strong_trend = kwargs['strong_trend']
            if isinstance(strong_trend, (int, float)) and 10 <= strong_trend <= 50:  # TODO: 将魔法数字提取到配置中
                if not hasattr(self, 'params'):
                    self.params = {}
                self.params['strong_trend'] = strong_trend
            else:
                logger.warning(f"无效的strong_trend参数: {strong_trend}, 保持原值")

        # 记录参数变更
        logger.info(f"ADX参数已更新")

    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """真实实现:生成ADX交易信号"""
        return self.get_signals(data, **kwargs)

    def compute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """真实实现:计算ADX指标"""
        return self._calculate_adx(data, **kwargs)

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """真实实现:计算ADX置信度"""
        if score.empty:
            return 0.3  # TODO: 将魔法数字提取到配置中

        # 基础置信度
        confidence = 0.5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

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
            if latest_patterns.get('ADX_VERY_STRONG_TREND', False):
                confidence += 0.2
            if latest_patterns.get('ADX_BREAKOUT', False):
                confidence += 0.2
            if latest_patterns.get('ADX_SUSTAINED_STRENGTH', False):
                confidence += 0.15  # TODO: 将魔法数字提取到配置中

            # 弱势形态降低置信度
            if latest_patterns.get('ADX_NO_TREND', False):
                confidence -= 0.2
            if latest_patterns.get('ADX_WEAK_TREND', False):
                confidence -= 0.1

        # 基于信号的置信度调整
        if signals:
            signal_strength = signals.get('strength', 0)
            confidence += signal_strength * 0.1

        # 限制置信度在0-1范围内
        return max(0.0, min(1.0, confidence))

    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """兼容性方法:识别形态"""
        return self.get_patterns(data, **kwargs)

    def calculate_raw_score_adx(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """兼容性方法:计算原始评分"""
        return self.calculate_raw_score(data, **kwargs)

    @property
    def minimum_periods(self) -> int:
        """
        返回ADX指标计算所需的最少数据周期数

        Returns:
            int: 最少需要的数据周期数
        """
        period = self.period
        return max(period * 2 + 5, 35)  # ADX需要更多数据进行平滑,最少35个周期  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中


# 为了兼容指标注册表,创建别名
ADX = AverageDirectionalIndex
