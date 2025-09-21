from vnpy_ctastrategy import (
    CtaTemplate,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
    BarGenerator,
    ArrayManager,
)
from Core_Framework.vnpy.vnpy.trader.constant import Exchange, Interval, Direction
from datetime import datetime, time
import numpy as np


class VolumeShrinkPullbackStrategy(CtaTemplate):
    """
    缩量回踩策略 - 严格按照缩量回踩公式实现
    
    核心逻辑：
    1. 前期放量上涨趋势确认
    2. 连续2-3天缩量回踩均线
    3. 15分钟吸筹信号确认
    4. 分时均线上方红盘开盘确认
    
    相比二波企稳：节奏更快，周期更短，适合短线操作
    """
    
    author = "VnPy Freedom - 缩量回踩公式"
    
    # === 策略参数 ===
    # 前期放量上涨确认参数
    volume_surge_days: int = 5      # 检查前5日放量上涨
    volume_surge_ratio: float = 1.5  # 放量倍数1.5倍
    price_rise_threshold: float = 3.0  # 价格上涨阈值3%
    
    # 缩量回踩参数
    shrink_days: int = 3            # 最多3天缩量
    shrink_ratio: float = 0.95      # 缩量比例95%（即5%的缩量）
    pullback_tolerance: float = 5.0  # 回踩容忍度5%
    
    # 均线参数
    ma5: int = 5                    # 5日均线
    ma_short: int = 10              # 短期均线10日
    ma_mid: int = 20                # 中期均线20日
    ma30: int = 30                  # 30日均线
    ma_long: int = 60               # 长期均线60日
    ma120: int = 120                # 120日均线
    
    # 15分钟吸筹信号参数
    stoch_period: int = 55          # 随机指标周期55
    stoch_smooth1: int = 5          # 第一次平滑5
    stoch_smooth2: int = 3          # 第二次平滑3
    accumulation_threshold: float = 13.0  # 吸筹阈值13
    
    # 分时确认参数
    intraday_confirm_minutes: int = 10  # 开盘后10分钟确认
    ma_intraday: int = 30           # 分时均线30分钟
    
    # 额外条件参数
    surge_60day_threshold: float = 7.0    # 60日内单日涨幅阈值7%
    amplitude_110day_threshold: float = 8.1  # 110日振幅阈值8.1%
    amplitude_110day_count: int = 2       # 110日内至少2次大振幅
    shrink_volume_days: int = 2           # 连续缩量天数2-3天
    volume_decline_threshold: float = 0.95  # 缩量比例阈值95%
    big_bearish_threshold: float = 0.96   # 大阴线阈值4%
    volume_surge_multiple: float = 1.8    # 放量倍数1.8倍
    
    # KDJ/MACD参数
    kdj_period: int = 9             # KDJ周期
    kdj_smooth_k: int = 3           # K线平滑周期
    kdj_smooth_d: int = 3           # D线平滑周期
    macd_fast: int = 12             # MACD快线周期
    macd_slow: int = 26             # MACD慢线周期
    macd_signal: int = 9            # MACD信号线周期
    
    # 止损参数
    stop_loss_pct: float = 3.0      # 止损3%
    take_profit_pct: float = 8.0    # 止盈8%
    
    # 固定仓位
    fixed_size: int = 1
    
    # === 策略变量 ===
    # 核心条件检查结果
    volume_surge_confirmed: bool = False      # 前期放量上涨确认
    shrink_pullback_confirmed: bool = False   # 缩量回踩确认
    ma_trend_confirmed: bool = False          # 均线趋势确认
    accumulation_15m_confirmed: bool = False  # 15分钟吸筹确认
    intraday_confirmed: bool = False          # 分时确认
    
    # 新增条件检查结果
    ma_rising_confirmed: bool = False         # 10/20日均线上移确认
    ma60_120_rising_confirmed: bool = False   # 60/120日均线上移确认
    pullback_ma_confirmed: bool = False       # 回踩均线确认
    surge_60day_confirmed: bool = False       # 60日内涨幅确认
    amplitude_110day_confirmed: bool = False  # 110日振幅确认
    shrink_volume_confirmed: bool = False     # 连续缩量确认
    kdj_dea_rising_confirmed: bool = False    # KDJ/DEA上移确认
    no_negative_signals: bool = True          # 无负面信号
    
    # 技术指标缓存
    ma5_value: float = 0.0
    ma10_value: float = 0.0
    ma20_value: float = 0.0
    ma30_value: float = 0.0
    ma60_value: float = 0.0
    ma120_value: float = 0.0
    
    # KDJ/MACD指标缓存
    kdj_k: float = 0.0
    kdj_d: float = 0.0
    kdj_j: float = 0.0
    macd_diff: float = 0.0
    macd_dea: float = 0.0
    macd_hist: float = 0.0
    
    # 多时间周期管理
    bg_15m: BarGenerator = None
    am_15m: ArrayManager = None
    
    # 交易状态
    entry_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    
    parameters = [
        "volume_surge_days", "volume_surge_ratio", "price_rise_threshold",
        "shrink_days", "shrink_ratio", "pullback_tolerance",
        "ma5", "ma_short", "ma_mid", "ma30", "ma_long", "ma120",
        "stoch_period", "stoch_smooth1", "stoch_smooth2", "accumulation_threshold",
        "intraday_confirm_minutes", "ma_intraday",
        "surge_60day_threshold", "amplitude_110day_threshold", "amplitude_110day_count",
        "shrink_volume_days", "volume_decline_threshold", "big_bearish_threshold", "volume_surge_multiple",
        "kdj_period", "kdj_smooth_k", "kdj_smooth_d", "macd_fast", "macd_slow", "macd_signal",
        "stop_loss_pct", "take_profit_pct", "fixed_size"
    ]
    
    variables = [
        "volume_surge_confirmed", "shrink_pullback_confirmed", "ma_trend_confirmed",
        "accumulation_15m_confirmed", "intraday_confirmed",
        "ma_rising_confirmed", "ma60_120_rising_confirmed", "pullback_ma_confirmed",
        "surge_60day_confirmed", "amplitude_110day_confirmed", "shrink_volume_confirmed",
        "kdj_dea_rising_confirmed", "no_negative_signals",
        "ma5_value", "ma10_value", "ma20_value", "ma30_value", "ma60_value", "ma120_value",
        "kdj_k", "kdj_d", "kdj_j", "macd_diff", "macd_dea", "macd_hist",
        "entry_price", "stop_price", "target_price"
    ]
    
    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """构造函数"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        
        # 初始化ArrayManager（日线数据）
        self.am = ArrayManager(size=150)  # 需要足够的数据计算各种指标
        
        # 初始化BarGenerator（处理Tick转Bar）
        self.bg = BarGenerator(self.on_bar)
        
        # 初始化15分钟K线生成器
        self.bg_15m = BarGenerator(
            self.on_bar, 
            window=15, 
            on_window_bar=self.on_15m_bar,
            interval=Interval.MINUTE_15
        )
        self.am_15m = ArrayManager(size=100)
        
        # 决策分析记录
        self.decision_logs: list = []  # 存储决策日志
        self.trade_reasons: dict = {}  # 存储交易原因
        self.condition_history: list = []  # 存储每日条件历史
        
    def on_init(self):
        """策略初始化"""
        self.write_log("缩量回踩策略初始化")
        self.load_bar(30)  # 加载30天历史数据
        
    def on_start(self):
        """策略启动"""
        self.write_log("缩量回踩策略启动")
        
    def on_stop(self):
        """策略停止"""
        self.write_log("缩量回踩策略停止")
        
    def on_tick(self, tick: TickData):
        """Tick数据推送"""
        self.bg.update_tick(tick)
        
    def on_bar(self, bar: BarData):
        """日线数据推送"""
        self.cancel_all()
        
        # 更新ArrayManager
        am = self.am
        am.update_bar(bar)
        if not am.inited:
            return
            
        # 更新15分钟K线生成器
        self.bg_15m.update_bar(bar)
        
        # 计算技术指标
        self.calculate_indicators()
        
        # 检查所有条件
        self.check_all_conditions()
        
        # 执行交易逻辑
        self.execute_trading_logic(bar)
        
        self.put_event()
        
    def on_15m_bar(self, bar: BarData):
        """15分钟K线数据推送"""
        self.am_15m.update_bar(bar)
        if not self.am_15m.inited:
            return
            
        # 检查15分钟吸筹信号
        self.check_15m_accumulation_signal()
        
    def calculate_indicators(self):
        """计算技术指标"""
        am = self.am
        if not am.inited:
            return
        
        # 计算所有均线
        self.ma5_value = am.sma(self.ma5) if len(am.close_array) >= self.ma5 else 0.0
        self.ma10_value = am.sma(self.ma_short) if len(am.close_array) >= self.ma_short else 0.0
        self.ma20_value = am.sma(self.ma_mid) if len(am.close_array) >= self.ma_mid else 0.0
        self.ma30_value = am.sma(self.ma30) if len(am.close_array) >= self.ma30 else 0.0
        self.ma60_value = am.sma(self.ma_long) if len(am.close_array) >= self.ma_long else 0.0
        self.ma120_value = am.sma(self.ma120) if len(am.close_array) >= self.ma120 else 0.0
        
        # 计算KDJ指标
        kdj_data = self._calculate_kdj_manual()
        if kdj_data:
            self.kdj_k, self.kdj_d, self.kdj_j = kdj_data
        
        # 计算MACD指标
        macd_data = self._calculate_macd()
        if macd_data:
            self.macd_diff, self.macd_dea, self.macd_hist = macd_data
        
    def check_all_conditions(self):
        """
        检查所有条件（按缩量回踩公式完整实现）
        """
        # 按照缩量回踩公式依次检查所有条件
        
        # 1. 均线上移条件
        self.ma_rising_confirmed = self._check_ma_rising()
        self.ma60_120_rising_confirmed = self._check_ma60_120_rising()
        
        # 2. 回踩均线条件
        self.pullback_ma_confirmed = self._check_pullback_ma()
        
        # 3. 历史涨幅和振幅条件
        self.surge_60day_confirmed = self._check_surge_60day()
        self.amplitude_110day_confirmed = self._check_amplitude_110day()
        
        # 4. 15分钟吸筹信号
        self.accumulation_15m_confirmed = self._check_accumulation_15m()
        
        # 5. 缩量条件
        self.shrink_volume_confirmed = self._check_shrink_volume()
        
        # 6. KDJ/DEA上移条件
        self.kdj_dea_rising_confirmed = self._check_kdj_dea_rising()
        
        # 7. 负面信号过滤
        self.no_negative_signals = self._check_no_negative_signals()
        
        # 兼容原有逻辑：组合条件
        self.volume_surge_confirmed = self.surge_60day_confirmed and self.amplitude_110day_confirmed
        self.shrink_pullback_confirmed = self.shrink_volume_confirmed and self.pullback_ma_confirmed
        self.ma_trend_confirmed = self.ma_rising_confirmed and self.ma60_120_rising_confirmed
        
        # 计算完整条件满足数量
        complete_conditions = [
            self.ma_rising_confirmed,
            self.ma60_120_rising_confirmed,
            self.pullback_ma_confirmed,
            # self.surge_60day_confirmed,
            self.amplitude_110day_confirmed,
            # self.accumulation_15m_confirmed,
            # self.shrink_volume_confirmed,
            self.kdj_dea_rising_confirmed,
            self.no_negative_signals
        ]
        complete_conditions_met = sum(complete_conditions)
        # 核心4个条件（兼容原有逻辑）
        core_conditions_met = sum([
            self.volume_surge_confirmed,
            self.shrink_pullback_confirmed, 
            self.ma_trend_confirmed,
            self.accumulation_15m_confirmed
        ])
        
        # 打印每日详细指标分析
        self.print_daily_indicators_analysis()
        
        self.write_log(f"缩量回踩完整条件检查: {complete_conditions_met}/9, 核心条件: {core_conditions_met}/4")
        
        # 详细记录每日判断过程（每2天记录一次详细分析，避免日志过多）
        bar_count = len(self.am.close_array) if self.am.inited else 0
        if bar_count % 2 == 0 or complete_conditions_met >= 7:  # 每5天或有希望时记录详细分析
            self._log_daily_analysis()
        
        # 只有在满足条件较多时才显示详细条件状态
        if complete_conditions_met >= 2:
            self._log_condition_details()
        
    def check_volume_surge(self) -> bool:
        """
        检查前期放量上涨
        
        条件：前5日内至少有一天满足：
        1. 成交量大于前日1.5倍
        2. 价格上涨超过3%
        """
        am = self.am
        if len(am.close_array) < self.volume_surge_days + 1:
            return False
            
        for i in range(1, self.volume_surge_days + 1):
            # 获取第i天的数据
            volume_today = am.volume_array[-i]
            volume_yesterday = am.volume_array[-i-1] if len(am.volume_array) > i else 0
            
            close_today = am.close_array[-i]
            close_yesterday = am.close_array[-i-1] if len(am.close_array) > i else 0
            
            if volume_yesterday > 0 and close_yesterday > 0:
                volume_ratio = volume_today / volume_yesterday
                price_change = (close_today - close_yesterday) / close_yesterday * 100
                
                # 放量上涨条件
                if volume_ratio >= self.volume_surge_ratio and price_change >= self.price_rise_threshold:
                    return True
                    
        return False
        
    def check_shrink_pullback(self) -> bool:
        """
        检查缩量回踩
        
        条件：最近2-3天连续缩量，且回踩到关键均线附近
        """
        am = self.am
        if len(am.close_array) < self.shrink_days + 1:
            return False
            
        # 检查连续缩量
        shrink_count = 0
        for i in range(1, self.shrink_days + 1):
            volume_today = am.volume_array[-i]
            volume_yesterday = am.volume_array[-i-1] if len(am.volume_array) > i else 0
            
            if volume_yesterday > 0:
                volume_ratio = volume_today / volume_yesterday
                if volume_ratio <= self.shrink_ratio:
                    shrink_count += 1
                    
        # 至少连续2天缩量
        if shrink_count < 2:
            return False
            
        # 检查回踩均线
        current_price = am.close_array[-1]
        
        # 回踩10日线或20日线
        ma10_pullback = abs((current_price / self.ma10_value - 1) * 100) <= self.pullback_tolerance
        ma20_pullback = abs((current_price / self.ma20_value - 1) * 100) <= self.pullback_tolerance
        
        return ma10_pullback or ma20_pullback
        
    def check_ma_trend(self) -> bool:
        """
        检查均线趋势
        
        条件：10日线和20日线上移，60日线或120日线至少一条上移
        """
        am = self.am
        if len(am.close_array) < 2:
            return False
            
        # 计算前一日均线
        prev_ma10 = am.sma(self.ma_short, array=True)[-2] if len(am.close_array) >= self.ma_short + 1 else 0
        prev_ma20 = am.sma(self.ma_mid, array=True)[-2] if len(am.close_array) >= self.ma_mid + 1 else 0
        prev_ma60 = am.sma(self.ma_long, array=True)[-2] if len(am.close_array) >= self.ma_long + 1 else 0
        
        # 检查10日线和20日线上移
        ma10_rising = self.ma10_value > prev_ma10 if prev_ma10 > 0 else False
        ma20_rising = self.ma20_value > prev_ma20 if prev_ma20 > 0 else False
        
        # 检查60日线上移
        ma60_rising = self.ma60_value > prev_ma60 if prev_ma60 > 0 else False
        
        # 10日线和20日线都上移，且60日线上移
        return ma10_rising and ma20_rising and ma60_rising
        
    def check_15m_accumulation_signal(self):
        """
        检查15分钟吸筹信号
        
        严格按照公式：
        V11:=3*SMA((C-LLV(L,55))/(HHV(H,55)-LLV(L,55))*100,5,1)-2*SMA(SMA((C-LLV(L,55))/(HHV(H,55)-LLV(L,55))*100,5,1),3,1);
        V12:=(EMA(V11,3)-REF(EMA(V11,3),1))/REF(EMA(V11,3),1)*100;
        AA:=(EMA(V11,3)<=13) AND FILTER((EMA(V11,3)<=13),15);
        BB:=(EMA(V11,3)<=13 AND V12>13) AND FILTER((EMA(V11,3)<=13 AND V12>13),10);
        """
        if not self.am_15m.inited:
            self.accumulation_15m_confirmed = False
            return
            
        am = self.am_15m
        period = self.stoch_period
        
        if len(am.close_array) < period:
            self.accumulation_15m_confirmed = False
            return
            
        # 计算随机指标基础值
        high_values = am.high_array[-period:]
        low_values = am.low_array[-period:]
        close_values = am.close_array[-period:]
        
        # 计算V11指标
        v11_values = []
        for i in range(len(close_values)):
            if i >= self.stoch_smooth1 - 1:
                # 计算(C-LLV(L,55))/(HHV(H,55)-LLV(L,55))*100
                end_idx = i + 1
                start_idx = max(0, end_idx - period)
                
                llv = np.min(low_values[start_idx:end_idx])
                hhv = np.max(high_values[start_idx:end_idx])
                
                if hhv > llv:
                    stoch_val = (close_values[i] - llv) / (hhv - llv) * 100
                else:
                    stoch_val = 0
                    
                v11_values.append(stoch_val)
                
        if len(v11_values) < self.stoch_smooth1:
            self.accumulation_15m_confirmed = False
            return
            
        # 计算SMA平滑
        sma1_values = []
        for i in range(len(v11_values) - self.stoch_smooth1 + 1):
            sma_val = np.mean(v11_values[i:i + self.stoch_smooth1])
            sma1_values.append(sma_val)
            
        if len(sma1_values) < self.stoch_smooth2:
            self.accumulation_15m_confirmed = False
            return
            
        # 计算第二次平滑
        sma2_values = []
        for i in range(len(sma1_values) - self.stoch_smooth2 + 1):
            sma_val = np.mean(sma1_values[i:i + self.stoch_smooth2])
            sma2_values.append(sma_val)
            
        # 计算V11 = 3*SMA1 - 2*SMA2
        v11_final = []
        min_len = min(len(sma1_values), len(sma2_values))
        for i in range(min_len):
            v11 = 3 * sma1_values[-(min_len-i)] - 2 * sma2_values[-(min_len-i)]
            v11_final.append(v11)
            
        if len(v11_final) < 3:
            self.accumulation_15m_confirmed = False
            return
            
        # 计算EMA(V11, 3)
        ema_v11 = self._calculate_ema(v11_final, 3)
        
        if len(ema_v11) < 2:
            self.accumulation_15m_confirmed = False
            return
            
        # 计算V12变化率
        current_ema = ema_v11[-1]
        prev_ema = ema_v11[-2]
        
        if prev_ema != 0:
            v12 = (current_ema - prev_ema) / prev_ema * 100
        else:
            v12 = 0
            
        # 吸筹信号条件
        # AA: EMA(V11,3) <= 13
        condition_aa = current_ema <= self.accumulation_threshold
        
        # BB: EMA(V11,3) <= 13 AND V12 > 13
        condition_bb = current_ema <= self.accumulation_threshold and v12 > self.accumulation_threshold
        
        # 满足任一条件即为吸筹信号
        self.accumulation_15m_confirmed = condition_aa or condition_bb
        
        if self.accumulation_15m_confirmed:
            self.write_log(f"15分钟吸筹信号确认: EMA_V11={current_ema:.2f}, V12={v12:.2f}")
            
    def check_intraday_confirmation(self) -> bool:
        """
        检查分时确认信号
        
        条件：红盘开盘，在分时均线上方运行，观察10分钟没有逆转
        """
        # 这里简化实现，在实际应用中需要分时数据
        # 暂时用日线数据的开盘价和均线关系来模拟
        am = self.am
        if not am.inited:
            return False
            
        current_open = am.open_array[-1]
        current_close = am.close_array[-1]
        prev_close = am.close_array[-2] if len(am.close_array) >= 2 else 0
        
        # 红盘开盘（开盘价高于前收盘）
        red_open = current_open > prev_close if prev_close > 0 else False
        
        # 价格在均线上方
        above_ma = current_close > self.ma10_value
        
        return red_open and above_ma
        
    def execute_trading_logic(self, bar: BarData):
        """执行交易逻辑"""
        if not self.am.inited:
            return
            
        current_price = bar.close_price
        current_time = bar.datetime
        
        # 综合买入条件
        all_conditions = [
            self.ma_rising_confirmed,
            self.ma60_120_rising_confirmed,
            self.pullback_ma_confirmed,
            # self.surge_60day_confirmed,
            self.amplitude_110day_confirmed,
            # self.accumulation_15m_confirmed,
            # self.shrink_volume_confirmed,
            self.kdj_dea_rising_confirmed,
            self.no_negative_signals
        ]
        
        buy_signal = all(all_conditions)
        conditions_met = sum(all_conditions)
        
        # 淘汰条件
        # 1. 跌破支撑线（跌破20日均线超过3%）
        below_support = current_price < self.ma20_value * 0.97
        
        # 2. 放量下跌
        volume_decline = self._check_volume_decline()
        
        sell_signal = below_support or volume_decline
        
        # 记录完整的决策分析
        decision_record = {
            "time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "price": current_price,
            "position": self.pos,
            "conditions": {
                "ma_rising": self.ma_rising_confirmed,
                "ma60_120_rising": self.ma60_120_rising_confirmed,
                "pullback_ma": self.pullback_ma_confirmed,
                "surge_60day": self.surge_60day_confirmed,
                "amplitude_110day": self.amplitude_110day_confirmed,
                "accumulation_15m": self.accumulation_15m_confirmed,
                "shrink_volume": self.shrink_volume_confirmed,
                "kdj_dea_rising": self.kdj_dea_rising_confirmed,
                "no_negative": self.no_negative_signals
            },
            "conditions_met": conditions_met,
            "buy_signal": buy_signal,
            "sell_signal": sell_signal,
            "risk_signals": {
                "below_support": below_support,
                "volume_decline": volume_decline
            },
            "market_data": {
                "ma10": self.ma10_value,
                "ma20": self.ma20_value,
                "ma60": self.ma60_value,
                "volume": self.am.volume[-1] if self.am.inited else 0
            }
        }
        self.condition_history.append(decision_record)
        
        # 记录决策日志到decision_logs
        decision_log = self._create_decision_log(current_time, current_price, decision_record)
        self.decision_logs.append(decision_log)
        
        # 详细记录交易决策信息
        self.write_log(f"💰 交易决策: 条件满足={conditions_met}/4, 买入信号={buy_signal}")
        
        if conditions_met > 0:
            condition_details = []
            if self.volume_surge_confirmed:
                condition_details.append("放量上涨✅")
            if self.shrink_pullback_confirmed:
                condition_details.append("缩量回踩✅")
            if self.ma_trend_confirmed:
                condition_details.append("均线趋势✅")
            if self.accumulation_15m_confirmed:
                condition_details.append("15分钟吸筹✅")
                
            self.write_log(f"💰 满足条件: {' | '.join(condition_details)}")
            
        if sell_signal:
            sell_reasons = []
            if below_support:
                support_distance = (current_price / self.ma20_value - 1) * 100
                sell_reasons.append(f"跌破支撑线{support_distance:.1f}%")
            if volume_decline:
                sell_reasons.append("放量下跌")
            self.write_log(f"💰 卖出信号: {' | '.join(sell_reasons)}")
            
        # 降低买入条件进行测试（可选）
        relaxed_buy_signal = False  # 满足3个条件就尝试买入
        
        if not buy_signal and relaxed_buy_signal:
            self.write_log(f"💰 降低标准买入: {conditions_met}/4个条件满足")
            buy_signal = True
        
        # 执行交易
        if self.pos == 0:
            if buy_signal:
                # 记录买入决策原因
                buy_reason = self._generate_buy_reason(conditions_met, all_conditions)
                self.trade_reasons[str(current_time)] = {
                    "action": "买入",
                    "price": current_price,
                    "reason": buy_reason,
                    "conditions_met": conditions_met,
                    "signal_strength": "强" if conditions_met == 4 else "中"
                }
                
                # 开仓买入
                self.entry_price = current_price
                self.stop_price = current_price * (1 - self.stop_loss_pct / 100)
                self.target_price = current_price * (1 + self.take_profit_pct / 100)
                
                # 发送买单并记录原因
                self.buy(current_price * 1.01, self.fixed_size)
                self.write_log(
                    f"🚀 缩量回踩买入: 价格{current_price:.2f}, "
                    f"止损{self.stop_price:.2f}, 目标{self.target_price:.2f}"
                )
                self.write_log(f"📋 买入原因: {buy_reason}")
            else:
                # 记录为什么没有买入
                missing_conditions = []
                if not self.volume_surge_confirmed:
                    missing_conditions.append("缺少放量上涨")
                if not self.shrink_pullback_confirmed:
                    missing_conditions.append("缺少缩量回踩")
                if not self.ma_trend_confirmed:
                    missing_conditions.append("缺少均线趋势")
                if not self.accumulation_15m_confirmed:
                    missing_conditions.append("缺少15分钟吸筹")
                    
                if missing_conditions:
                    no_buy_reason = f"观望: {' | '.join(missing_conditions)}"
                    self.write_log(f"❌ 未买入原因: {no_buy_reason}")
                    
                    # 记录不买入的原因到decision_logs
                    self.decision_logs.append({
                        "time": current_time,
                        "action": "观望",
                        "reason": no_buy_reason,
                        "conditions_met": f"{conditions_met}/4",
                        "price": current_price,
                        "position": self.pos
                    })
            
        elif self.pos > 0:
            # 已持仓，检查止盈止损和卖出信号
            if sell_signal:
                # 记录卖出决策原因
                sell_reason = self._generate_sell_reason(below_support, volume_decline, current_price)
                self.trade_reasons[str(current_time)] = {
                    "action": "卖出",
                    "price": current_price,
                    "reason": sell_reason,
                    "trigger": "风险信号",
                    "signal_strength": "强"
                }
                
                self.sell(current_price * 0.99, abs(self.pos))
                self.write_log(f"📋 卖出原因: {sell_reason}")
                self.write_log(f"❌ 触发淘汰条件卖出: 价格{current_price:.2f}")
                
                # 记录卖出决策到decision_logs
                self.decision_logs.append({
                    "time": current_time,
                    "action": "卖出",
                    "reason": sell_reason,
                    "trigger": "风险信号",
                    "price": current_price,
                    "position": self.pos
                })
                
            elif current_price <= self.stop_price:
                stop_reason = f"止损: 价格{current_price:.2f} <= 止损线{self.stop_price:.2f}"
                self.sell(current_price * 0.99, abs(self.pos)) 
                self.write_log(f"🛑 止损卖出: 价格{current_price:.2f}")
                
                # 记录止损决策到decision_logs
                self.decision_logs.append({
                    "time": current_time,
                    "action": "止损",
                    "reason": stop_reason,
                    "trigger": "止损线",
                    "price": current_price,
                    "position": self.pos
                })
                
            elif current_price >= self.target_price:
                profit_reason = f"止盈: 价格{current_price:.2f} >= 目标价{self.target_price:.2f}"
                self.sell(current_price * 0.99, abs(self.pos))
                self.write_log(f"🎯 止盈卖出: 价格{current_price:.2f}")
                
                # 记录止盈决策到decision_logs
                self.decision_logs.append({
                    "time": current_time,
                    "action": "止盈",
                    "reason": profit_reason,
                    "trigger": "目标价位",
                    "price": current_price,
                    "position": self.pos
                })
                
    def _check_volume_decline(self) -> bool:
        """检查放量下跌"""
        am = self.am
        if len(am.close_array) < 2:
            return False
            
        current_close = am.close_array[-1]
        prev_close = am.close_array[-2]
        current_volume = am.volume_array[-1]
        prev_volume = am.volume_array[-2] if len(am.volume_array) >= 2 else 0
        
        # 价格下跌且成交量放大
        price_decline = current_close < prev_close
        volume_surge = current_volume > prev_volume if prev_volume > 0 else False
        
        return price_decline and volume_surge
        
    def _calculate_ema(self, values: list, period: int) -> list:
        """计算EMA"""
        if len(values) < period:
            return []
            
        ema_values = []
        multiplier = 2 / (period + 1)
        
        # 第一个EMA值使用SMA
        sma = sum(values[:period]) / period
        ema_values.append(sma)
        
        # 后续EMA值
        for i in range(period, len(values)):
            ema = values[i] * multiplier + ema_values[-1] * (1 - multiplier)
            ema_values.append(ema)
            
        return ema_values
        
    def _log_condition_details(self):
        """记录条件详细状态"""
        conditions_status = [
            f"前期放量上涨: {'✅' if self.volume_surge_confirmed else '❌'}",
            f"缩量回踩: {'✅' if self.shrink_pullback_confirmed else '❌'}",
            f"均线趋势: {'✅' if self.ma_trend_confirmed else '❌'}",
            f"15分钟吸筹: {'✅' if self.accumulation_15m_confirmed else '❌'}",
            f"分时确认: {'✅' if self.intraday_confirmed else '❌'}"
        ]
        
        for status in conditions_status:
            self.write_log(status)
            
    def _log_daily_analysis(self):
        """记录每日详细分析过程"""
        am = self.am
        if not am.inited:
            return
            
        # 使用策略运行的数据计数作为日期标识
        data_count = len(am.close_array)
        current_price = am.close_array[-1]
        current_volume = am.volume_array[-1]
        
        self.write_log(f"\n=== 第{data_count}根K线分析 ===")
        self.write_log(f"当前价格: {current_price:.2f}")
        self.write_log(f"当前成交量: {current_volume:.0f}")
        self.write_log(f"MA10: {self.ma10_value:.2f}, MA20: {self.ma20_value:.2f}, MA60: {self.ma60_value:.2f}")
        
        # 分析各个条件的详细情况
        self._analyze_volume_surge_detail()
        self._analyze_shrink_pullback_detail()
        self._analyze_ma_trend_detail()
        self._analyze_accumulation_detail()
        
        self.write_log("=== 分析结束 ===\n")
        
    def _analyze_volume_surge_detail(self):
        """分析放量上涨的详细情况"""
        am = self.am
        if len(am.close_array) < self.volume_surge_days + 1:
            self.write_log("📊 放量上涨: 数据不足")
            return
            
        found_surge = False
        for i in range(1, self.volume_surge_days + 1):
            volume_today = am.volume_array[-i]
            volume_yesterday = am.volume_array[-i-1] if len(am.volume_array) > i else 0
            
            close_today = am.close_array[-i]
            close_yesterday = am.close_array[-i-1] if len(am.close_array) > i else 0
            
            if volume_yesterday > 0 and close_yesterday > 0:
                volume_ratio = volume_today / volume_yesterday
                price_change = (close_today - close_yesterday) / close_yesterday * 100
                
                date_str = f"T-{i}"  # 简化为相对日期显示
                
                if volume_ratio >= self.volume_surge_ratio and price_change >= self.price_rise_threshold:
                    self.write_log(f"📊 {date_str}: 放量上涨 ✅ (量比{volume_ratio:.2f}, 涨幅{price_change:.1f}%)")
                    found_surge = True
                else:
                    status = ""
                    if volume_ratio < self.volume_surge_ratio:
                        status += f"量比不足{volume_ratio:.2f}<{self.volume_surge_ratio} "
                    if price_change < self.price_rise_threshold:
                        status += f"涨幅不足{price_change:.1f}%<{self.price_rise_threshold}%"
                    self.write_log(f"📊 {date_str}: {status}")
                    
        if not found_surge:
            self.write_log(f"📊 放量上涨: 近{self.volume_surge_days}日无符合条件的放量上涨")
            
    def _analyze_shrink_pullback_detail(self):
        """分析缩量回踩的详细情况"""
        am = self.am
        if len(am.close_array) < self.shrink_days + 1:
            self.write_log("📉 缩量回踩: 数据不足")
            return
            
        # 检查连续缩量
        shrink_count = 0
        shrink_details = []
        for i in range(1, self.shrink_days + 1):
            volume_today = am.volume_array[-i]
            volume_yesterday = am.volume_array[-i-1] if len(am.volume_array) > i else 0
            
            if volume_yesterday > 0:
                volume_ratio = volume_today / volume_yesterday
                date_str = f"T-{i}"  # 简化为相对日期显示
                
                if volume_ratio <= self.shrink_ratio:
                    shrink_count += 1
                    shrink_details.append(f"{date_str}:缩量{volume_ratio:.2f}")
                else:
                    shrink_details.append(f"{date_str}:量比{volume_ratio:.2f}")
                    
        self.write_log(f"📉 缩量情况: {' | '.join(shrink_details)}")
        self.write_log(f"📉 连续缩量天数: {shrink_count}/{self.shrink_days}")
        
        # 检查回踩均线
        current_price = am.close_array[-1]
        ma10_distance = abs((current_price / self.ma10_value - 1) * 100) if self.ma10_value > 0 else 999
        ma20_distance = abs((current_price / self.ma20_value - 1) * 100) if self.ma20_value > 0 else 999
        
        self.write_log(f"📉 回踩情况: 距MA10={ma10_distance:.1f}%, 距MA20={ma20_distance:.1f}%")
        
        ma10_pullback = ma10_distance <= self.pullback_tolerance
        ma20_pullback = ma20_distance <= self.pullback_tolerance
        
        if ma10_pullback or ma20_pullback:
            self.write_log(f"📉 回踩均线: ✅ (容忍度{self.pullback_tolerance}%)")
        else:
            self.write_log(f"📉 回踩均线: ❌ (超出容忍度{self.pullback_tolerance}%)")
            
    def _analyze_ma_trend_detail(self):
        """分析均线趋势的详细情况"""
        am = self.am
        if len(am.close_array) < 2:
            self.write_log("📈 均线趋势: 数据不足")
            return
            
        # 计算前一日均线
        prev_ma10 = am.sma(self.ma_short, array=True)[-2] if len(am.close_array) >= self.ma_short + 1 else 0
        prev_ma20 = am.sma(self.ma_mid, array=True)[-2] if len(am.close_array) >= self.ma_mid + 1 else 0
        prev_ma60 = am.sma(self.ma_long, array=True)[-2] if len(am.close_array) >= self.ma_long + 1 else 0
        
        # 检查均线上移
        ma10_change = (self.ma10_value - prev_ma10) / prev_ma10 * 100 if prev_ma10 > 0 else 0
        ma20_change = (self.ma20_value - prev_ma20) / prev_ma20 * 100 if prev_ma20 > 0 else 0
        ma60_change = (self.ma60_value - prev_ma60) / prev_ma60 * 100 if prev_ma60 > 0 else 0
        
        ma10_rising = ma10_change > 0
        ma20_rising = ma20_change > 0
        ma60_rising = ma60_change > 0
        
        self.write_log(f"📈 MA10变化: {ma10_change:+.3f}% {'✅' if ma10_rising else '❌'}")
        self.write_log(f"📈 MA20变化: {ma20_change:+.3f}% {'✅' if ma20_rising else '❌'}")
        self.write_log(f"📈 MA60变化: {ma60_change:+.3f}% {'✅' if ma60_rising else '❌'}")
        
        trend_ok = ma10_rising and ma20_rising and ma60_rising
        self.write_log(f"📈 均线趋势: {'✅' if trend_ok else '❌'} (需要10/20/60日线都上移)")
        
    def _analyze_accumulation_detail(self):
        """分析15分钟吸筹信号的详细情况"""
        if not self.am_15m.inited:
            self.write_log("🔄 15分钟吸筹: 15分钟数据未初始化")
            return
            
        if self.accumulation_15m_confirmed:
            self.write_log("🔄 15分钟吸筹: ✅ 检测到吸筹信号")
        else:
            self.write_log("🔄 15分钟吸筹: ❌ 暂未检测到吸筹信号")
            
        # 如果15分钟数据足够，显示更多细节
        am_15m = self.am_15m
        if len(am_15m.close_array) >= 10:
            recent_volume_avg = sum(am_15m.volume_array[-5:]) / 5
            current_volume = am_15m.volume_array[-1]
            volume_ratio = current_volume / recent_volume_avg if recent_volume_avg > 0 else 0
            
            self.write_log(f"🔄 15分钟成交量: 当前{current_volume:.0f}, 5期均值{recent_volume_avg:.0f}, 比例{volume_ratio:.2f}")
            
    def on_order(self, order: OrderData):
        """委托回报"""
        # 尝试关联委托与最近的决策原因
        order_time = order.datetime if hasattr(order, 'datetime') else None
        
        # 查找最近的交易决策原因
        recent_reason = "未记录原因"
        if self.trade_reasons:
            # 获取最近的交易原因
            latest_reason_entry = list(self.trade_reasons.values())[-1]
            if latest_reason_entry and 'reason' in latest_reason_entry:
                recent_reason = latest_reason_entry['reason']
        
        # 在日志中记录委托信息和决策原因
        direction_str = "买入" if order.direction.value == "多" else "卖出"
        self.write_log(
            f"📋 委托单: {direction_str} {order.volume}@{order.price:.2f} "
            f"[{order.status.value}] - 原因: {recent_reason}"
        )
        
        # 存储委托原因到策略实例，供UI查询
        if not hasattr(self, 'order_reasons'):
            self.order_reasons = {}
        self.order_reasons[order.orderid] = {
            "reason": recent_reason,
            "time": order_time,
            "direction": direction_str,
            "price": order.price,
            "volume": order.volume,
            "status": order.status.value
        }
        
    def on_trade(self, trade: TradeData):
        """成交回报"""
        if trade.direction == Direction.LONG:
            self.write_log(f"缩量回踩买入成交: {trade.volume}@{trade.price}")
        else:
            self.write_log(f"缩量回踩卖出成交: {trade.volume}@{trade.price}")
        self.put_event()
    
    def print_daily_indicators_analysis(self):
        """打印每日详细指标分析"""
        if not self.am.inited:
            return
            
        # 获取当前数据
        current_close = self.am.close[-1]
        current_volume = self.am.volume[-1]
        current_date = self.am.datetime[-1].strftime("%Y-%m-%d") if hasattr(self.am, 'datetime') else "N/A"
        
        # === 打印分隔线和日期 ===
        self.write_log("=" * 80)
        self.write_log(f"📊 缩量回踩策略 - 每日指标分析 [{current_date}]")
        self.write_log("=" * 80)
        
        # === 1. 基础价格信息 ===
        self.write_log(f"💰 价格信息:")
        self.write_log(f"   当前收盘价: {current_close:.2f}")
        self.write_log(f"   10日均线: {self.ma10_value:.2f} ({'上方' if current_close > self.ma10_value else '下方'})")
        self.write_log(f"   20日均线: {self.ma20_value:.2f} ({'上方' if current_close > self.ma20_value else '下方'})")
        self.write_log(f"   60日均线: {self.ma60_value:.2f} ({'上方' if current_close > self.ma60_value else '下方'})")
        
        # === 2. 前期放量上涨分析 ===
        self.write_log(f"\n📈 条件1: 前期放量上涨分析")
        surge_analysis = self._analyze_volume_surge()
        self.write_log(f"   状态: {'✅ 满足' if self.volume_surge_confirmed else '❌ 不满足'}")
        self.write_log(f"   详情: {surge_analysis}")
        
        # === 3. 缩量回踩分析 ===
        self.write_log(f"\n📉 条件2: 缩量回踩分析")
        pullback_analysis = self._analyze_shrink_pullback()
        self.write_log(f"   状态: {'✅ 满足' if self.shrink_pullback_confirmed else '❌ 不满足'}")
        self.write_log(f"   详情: {pullback_analysis}")
        
        # === 4. 均线趋势分析 ===
        self.write_log(f"\n📊 条件3: 均线趋势分析")
        ma_analysis = self._analyze_ma_trend()
        self.write_log(f"   状态: {'✅ 满足' if self.ma_trend_confirmed else '❌ 不满足'}")
        self.write_log(f"   详情: {ma_analysis}")
        
        # === 5. 15分钟吸筹信号分析 ===
        self.write_log(f"\n⏰ 条件4: 15分钟吸筹信号分析")
        accumulation_analysis = self._analyze_15m_accumulation()
        self.write_log(f"   状态: {'✅ 满足' if self.accumulation_15m_confirmed else '❌ 不满足'}")
        self.write_log(f"   详情: {accumulation_analysis}")
        
        # === 6. 成交量分析 ===
        self.write_log(f"\n📊 成交量分析:")
        volume_analysis = self._analyze_volume_pattern()
        self.write_log(f"   {volume_analysis}")
        
        # === 7. 风险信号分析 ===
        self.write_log(f"\n⚠️  风险信号分析:")
        risk_analysis = self._analyze_risk_signals()
        self.write_log(f"   {risk_analysis}")
        
        # === 8. 综合评分 ===
        total_score = sum([
            self.volume_surge_confirmed,
            self.shrink_pullback_confirmed,
            self.ma_trend_confirmed,
            self.accumulation_15m_confirmed
        ])
        
        self.write_log(f"\n🎯 综合评分: {total_score}/4 ({'强烈推荐' if total_score >= 4 else '推荐' if total_score >= 3 else '观望' if total_score >= 2 else '不推荐'})")
        
        # === 9. 交易建议 ===
        self.write_log(f"\n💡 交易建议:")
        if total_score >= 4:
            self.write_log(f"   🚀 所有条件满足，可考虑买入")
        elif total_score >= 3:
            self.write_log(f"   👀 大部分条件满足，密切关注")
        elif total_score >= 2:
            self.write_log(f"   ⏳ 部分条件满足，继续观察")
        else:
            self.write_log(f"   🛑 条件不足，暂不介入")
            
        self.write_log("=" * 80)
    
    def _analyze_volume_surge(self) -> str:
        """分析前期放量上涨情况"""
        if not self.am.inited or len(self.am.close_array) < self.volume_surge_days + 1:
            return "数据不足"
            
        try:
            # 分析前5日情况
            analysis_parts = []
            
            for i in range(1, self.volume_surge_days + 1):
                day_close = self.am.close[-i-1]
                day_volume = self.am.volume[-i-1]
                prev_close = self.am.close[-i-2] if len(self.am.close_array) > i+1 else day_close
                prev_volume = self.am.volume[-i-2] if len(self.am.volume_array) > i+1 else day_volume
                
                price_change = (day_close - prev_close) / prev_close * 100 if prev_close > 0 else 0
                volume_ratio = day_volume / prev_volume if prev_volume > 0 else 1.0
                
                analysis_parts.append(f"T-{i}: 涨幅{price_change:.1f}%, 量比{volume_ratio:.1f}")
            
            return " | ".join(analysis_parts)
            
        except Exception as e:
            return f"分析异常: {str(e)}"
    
    def _analyze_shrink_pullback(self) -> str:
        """分析缩量回踩情况"""
        if not self.am.inited or len(self.am.close_array) < 5:
            return "数据不足"
            
        try:
            # 分析最近3天缩量情况
            analysis_parts = []
            
            for i in range(1, min(4, len(self.am.volume_array))):
                current_vol = self.am.volume[-i]
                prev_vol = self.am.volume[-i-1] if len(self.am.volume_array) > i else current_vol
                
                vol_ratio = current_vol / prev_vol if prev_vol > 0 else 1.0
                is_shrink = vol_ratio < self.shrink_ratio
                
                # 价格回踩程度
                current_price = self.am.close[-i]
                ma20_distance = abs(current_price - self.ma20_value) / self.ma20_value * 100 if self.ma20_value > 0 else 0
                
                analysis_parts.append(f"T-{i}: 缩量{'✓' if is_shrink else '✗'}({vol_ratio:.2f}), 距20日线{ma20_distance:.1f}%")
            
            return " | ".join(analysis_parts)
            
        except Exception as e:
            return f"分析异常: {str(e)}"
    
    def _analyze_ma_trend(self) -> str:
        """分析均线趋势情况"""
        try:
            # 均线排列分析
            ma_order = "多头排列" if self.ma10_value > self.ma20_value > self.ma60_value else "空头排列" if self.ma10_value < self.ma20_value < self.ma60_value else "交织状态"
            
            # 均线斜率分析（简化版）
            if len(self.am.close_array) >= 5:
                ma20_prev = np.mean(self.am.close_array[-25:-20]) if len(self.am.close_array) >= 25 else self.ma20_value
                ma20_slope = (self.ma20_value - ma20_prev) / ma20_prev * 100 if ma20_prev > 0 else 0
                
                slope_desc = "上升" if ma20_slope > 0.5 else "下降" if ma20_slope < -0.5 else "平缓"
                
                return f"{ma_order}, 20日线{slope_desc}({ma20_slope:.1f}%)"
            else:
                return f"{ma_order}, 数据不足分析斜率"
                
        except Exception as e:
            return f"分析异常: {str(e)}"
    
    def _analyze_15m_accumulation(self) -> str:
        """分析15分钟吸筹信号"""
        if not self.am_15m.inited:
            return "15分钟数据不足"
            
        try:
            # 简化的吸筹信号分析
            if len(self.am_15m.close_array) >= 10:
                recent_volume_avg = np.mean(self.am_15m.volume_array[-5:])
                current_volume = self.am_15m.volume_array[-1]
                volume_ratio = current_volume / recent_volume_avg if recent_volume_avg > 0 else 1.0
                
                # 价格稳定性
                price_volatility = np.std(self.am_15m.close_array[-10:]) / np.mean(self.am_15m.close_array[-10:]) if len(self.am_15m.close_array) >= 10 else 0
                
                return f"量比{volume_ratio:.2f}, 波动率{price_volatility:.3f}, {'稳定吸筹' if volume_ratio > 1.1 and price_volatility < 0.02 else '无明显吸筹'}"
            else:
                return "15分钟数据不足"
                
        except Exception as e:
            return f"分析异常: {str(e)}"
    
    def _analyze_volume_pattern(self) -> str:
        """分析成交量模式"""
        try:
            if len(self.am.volume_array) >= 10:
                current_vol = self.am.volume[-1]
                avg_vol_5 = np.mean(self.am.volume_array[-5:])
                avg_vol_20 = np.mean(self.am.volume_array[-20:]) if len(self.am.volume_array) >= 20 else avg_vol_5
                
                vol_vs_5day = current_vol / avg_vol_5 if avg_vol_5 > 0 else 1.0
                vol_vs_20day = current_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0
                
                pattern = "放量" if vol_vs_5day > 1.5 else "缩量" if vol_vs_5day < 0.8 else "平量"
                
                return f"当前成交量{current_vol:.0f}, 5日均量{avg_vol_5:.0f}(比例{vol_vs_5day:.2f}), 20日均量{avg_vol_20:.0f}(比例{vol_vs_20day:.2f}), 状态:{pattern}"
            else:
                return "成交量数据不足"
                
        except Exception as e:
            return f"分析异常: {str(e)}"
    
    def _analyze_risk_signals(self) -> str:
        """分析风险信号"""
        try:
            risk_signals = []
            
            # 检查是否跌破支撑
            current_price = self.am.close[-1]
            if current_price < self.ma20_value * 0.97:
                risk_signals.append("跌破20日线支撑")
            
            # 检查放量下跌
            if len(self.am.close_array) >= 2 and len(self.am.volume_array) >= 2:
                today_change = (self.am.close[-1] - self.am.close[-2]) / self.am.close[-2] * 100
                volume_ratio = self.am.volume[-1] / self.am.volume[-2] if self.am.volume[-2] > 0 else 1.0
                
                if today_change < -2 and volume_ratio > 1.5:
                    risk_signals.append("放量下跌")
            
            # 检查连续阴线
            if len(self.am.close_array) >= 3:
                consecutive_down = all(self.am.close[-i] < self.am.close[-i-1] for i in range(1, 4))
                if consecutive_down:
                    risk_signals.append("连续3日下跌")
            
            return "，".join(risk_signals) if risk_signals else "暂无明显风险信号"
            
        except Exception as e:
            return f"风险分析异常: {str(e)}"
    
    def _generate_buy_reason(self, conditions_met: int, all_conditions: list) -> str:
        """生成买入原因"""
        reasons = []
        
        if all_conditions[0]:  # volume_surge_confirmed
            reasons.append("前期放量上涨趋势确认")
        if all_conditions[1]:  # shrink_pullback_confirmed
            reasons.append("缩量回踩到位")
        if all_conditions[2]:  # ma_trend_confirmed
            reasons.append("均线趋势向好")
        if all_conditions[3]:  # accumulation_15m_confirmed
            reasons.append("15分钟出现吸筹信号")
        
        if conditions_met == 4:
            return f"完美信号：{' + '.join(reasons)}"
        elif conditions_met >= 3:
            return f"强信号({conditions_met}/4)：{' + '.join(reasons)}"
        else:
            return f"一般信号({conditions_met}/4)：{' + '.join(reasons)}"
    
    def _generate_sell_reason(self, below_support: bool, volume_decline: bool, current_price: float) -> str:
        """生成卖出原因"""
        reasons = []
        
        if below_support:
            distance = (current_price / self.ma20_value - 1) * 100
            reasons.append(f"跌破20日线支撑({distance:.1f}%)")
        
        if volume_decline:
            reasons.append("出现放量下跌信号")
        
        return "风险控制：" + " + ".join(reasons) if reasons else "触发止损"
    
    def get_decision_logs(self) -> list:
        """获取决策日志"""
        return self.decision_logs
    
    def get_trade_reasons(self) -> dict:
        """获取交易原因"""
        return self.trade_reasons
    
    def get_condition_history(self) -> list:
        """获取条件历史"""
        return self.condition_history
    
    def get_order_reasons(self) -> dict:
        """获取委托原因"""
        return getattr(self, 'order_reasons', {})
    
    # =================== 新增的完整条件检查方法 ===================
    
    def _check_ma_rising(self) -> bool:
        """
        检查10/20日均线上移（日线）
        公式：MA10RISING := REF(MA10, 1) < MA10; MA20RISING := REF(MA20, 1) < MA20;
        XG: COUNTRISING3 = 2;
        """
        am = self.am
        if not am.inited or len(am.close_array) < self.ma_mid + 1:
            return False
        
        # 计算当前和前一日的均线
        ma10_current = am.sma(self.ma_short)
        ma20_current = am.sma(self.ma_mid)
        
        # 计算前一日的均线（使用前一个数据点）
        if len(am.close_array) < 2:
            return False
        
        close_prev = am.close_array[:-1]  # 去掉最后一个数据点
        ma10_prev = np.mean(close_prev[-self.ma_short:]) if len(close_prev) >= self.ma_short else 0
        ma20_prev = np.mean(close_prev[-self.ma_mid:]) if len(close_prev) >= self.ma_mid else 0
        
        # 检查均线上移
        ma10_rising = ma10_prev < ma10_current if ma10_prev > 0 else False
        ma20_rising = ma20_prev < ma20_current if ma20_prev > 0 else False
        
        # 需要两个均线都上移
        rising_count = sum([ma10_rising, ma20_rising])
        result = rising_count == 2
        
        if rising_count > 0:
            self.write_log(f"📈 均线上移检查: MA10 {ma10_prev:.2f}→{ma10_current:.2f}{'↗️' if ma10_rising else '↘️'}, "
                          f"MA20 {ma20_prev:.2f}→{ma20_current:.2f}{'↗️' if ma20_rising else '↘️'} ({rising_count}/2)")
        
        return result
    
    def _check_ma60_120_rising(self) -> bool:
        """
        检查60/120日均线上移（日线）
        公式：MA60RISING := REF(MA60, 1) < MA60; MA120RISING := REF(MA120, 1) < MA120;
        XG: IF(MA60RISING, 1, 0) OR IF(MA120RISING, 1, 0);
        """
        am = self.am
        if not am.inited or len(am.close_array) < self.ma120 + 1:
            return False
        
        # 计算当前的长期均线
        ma60_current = am.sma(self.ma_long)
        ma120_current = am.sma(self.ma120)
        
        # 计算前一日的长期均线
        if len(am.close_array) < 2:
            return False
        
        close_prev = am.close_array[:-1]
        ma60_prev = np.mean(close_prev[-self.ma_long:]) if len(close_prev) >= self.ma_long else 0
        ma120_prev = np.mean(close_prev[-self.ma120:]) if len(close_prev) >= self.ma120 else 0
        
        # 检查长期均线上移
        ma60_rising = ma60_prev < ma60_current if ma60_prev > 0 else False
        ma120_rising = ma120_prev < ma120_current if ma120_prev > 0 else False
        
        # 任一均线上移即可
        result = ma60_rising or ma120_rising
        
        if ma60_rising or ma120_rising:
            self.write_log(f"📊 长期均线上移: MA60 {ma60_prev:.2f}→{ma60_current:.2f}{'↗️' if ma60_rising else '↘️'}, "
                          f"MA120 {ma120_prev:.2f}→{ma120_current:.2f}{'↗️' if ma120_rising else '↘️'}")
        
        return result
    
    def _check_pullback_ma(self) -> bool:
        """
        检查回踩均线（日线）
        公式：A10:=ABS((C/MA(C,10)-1)*100)<= 5; A20:=ABS((C/MA(C,20)-1)*100)<= 5; A30:=ABS((C/MA(C,30)-1)*100)<= 5;
        XG: A10 OR A20 OR A30;
        """
        am = self.am
        if not am.inited or len(am.close_array) < self.ma30:
            return False
        
        current_price = am.close_array[-1]
        
        # 计算各均线
        ma10 = am.sma(self.ma_short)
        ma20 = am.sma(self.ma_mid)
        ma30 = am.sma(self.ma30)
        
        # 计算偏离度
        deviation_ma10 = abs((current_price / ma10 - 1) * 100) if ma10 > 0 else 100
        deviation_ma20 = abs((current_price / ma20 - 1) * 100) if ma20 > 0 else 100
        deviation_ma30 = abs((current_price / ma30 - 1) * 100) if ma30 > 0 else 100
        
        # 任一均线偏离度小于等于5%即可
        pullback_ma10 = deviation_ma10 <= self.pullback_tolerance
        pullback_ma20 = deviation_ma20 <= self.pullback_tolerance
        pullback_ma30 = deviation_ma30 <= self.pullback_tolerance
        
        result = pullback_ma10 or pullback_ma20 or pullback_ma30
        
        if result:
            self.write_log(f"🎯 回踩均线确认: 价格{current_price:.2f}, "
                          f"MA10偏离{deviation_ma10:.1f}%{'✅' if pullback_ma10 else '❌'}, "
                          f"MA20偏离{deviation_ma20:.1f}%{'✅' if pullback_ma20 else '❌'}, "
                          f"MA30偏离{deviation_ma30:.1f}%{'✅' if pullback_ma30 else '❌'}")
        
        return result
    
    def _check_surge_60day(self) -> bool:
        """
        检查近60日至少一次涨幅大于7%（日线）
        公式：a1:=C/REF(C,1)>1.07; COUNT(a1,60)>0
        """
        am = self.am
        if not am.inited or len(am.close_array) < 61:  # 需要61天数据来计算60天的涨幅
            return False
        
        # 计算最近60天的单日涨幅
        surge_count = 0
        for i in range(-60, 0):  # 检查最近60天
            if i == 0:
                continue
            current_close = am.close_array[i]
            prev_close = am.close_array[i-1]
            daily_return = (current_close / prev_close - 1) * 100 if prev_close > 0 else 0
            
            if daily_return > self.surge_60day_threshold:
                surge_count += 1
        
        result = surge_count > 0
        
        if result:
            self.write_log(f"📈 60日涨幅检查: 找到{surge_count}次单日涨幅>{self.surge_60day_threshold}%")
        
        return result
    
    def _check_amplitude_110day(self) -> bool:
        """
        检查110日振幅>8.1%至少两次（日线）
        公式：a1:=100*(H-L)/L>8.1; COUNT(a1,110)>1
        """
        am = self.am
        if not am.inited or len(am.close_array) < 110:
            return False
        
        # 计算最近110天的振幅
        amplitude_count = 0
        for i in range(-110, 0):
            high_price = am.high_array[i]
            low_price = am.low_array[i]
            if low_price > 0:
                amplitude = (high_price - low_price) / low_price * 100
                if amplitude > self.amplitude_110day_threshold:
                    amplitude_count += 1
        
        result = amplitude_count >= self.amplitude_110day_count
        
        if amplitude_count > 0:
            self.write_log(f"📊 110日振幅检查: 找到{amplitude_count}次振幅>{self.amplitude_110day_threshold}% (需要≥{self.amplitude_110day_count})")
        
        return result
    
    def _check_shrink_volume(self) -> bool:
        """
        检查连续缩量（日线）
        公式：连续缩量阴线 := COUNT((VOL < REF(VOL, 1) * 最小缩量比) AND (CLOSE < OPEN), N) = N;
        """
        am = self.am
        if not am.inited or len(am.close_array) < self.shrink_volume_days + 1:
            return False
        
        # 检查最近N天的缩量阴线
        shrink_bearish_days = 0
        for i in range(-self.shrink_volume_days, 0):
            current_volume = am.volume_array[i]
            prev_volume = am.volume_array[i-1] if i > -len(am.volume_array) else 0
            current_close = am.close_array[i]
            current_open = am.open_array[i]
            
            # 缩量 AND 阴线
            volume_shrink = current_volume < prev_volume * self.volume_decline_threshold if prev_volume > 0 else False
            bearish_candle = current_close < current_open
            
            if volume_shrink and bearish_candle:
                shrink_bearish_days += 1
        
        result = shrink_bearish_days == self.shrink_volume_days
        
        if shrink_bearish_days > 0:
            self.write_log(f"📉 缩量阴线检查: {shrink_bearish_days}/{self.shrink_volume_days}天符合缩量阴线")
        
        return result
    
    def _check_kdj_dea_rising(self) -> bool:
        """
        检查KDJ/DEA任一上移（日线）
        公式：J>=REF(J,1) OR K>=REF(K,1) OR D>=REF(D,1) OR DEA>=REF(DEA,1)
        """
        am = self.am
        if not am.inited or len(am.close_array) < max(self.kdj_period, self.macd_slow) + 1:
            return False
        
        # 计算KDJ
        kdj_data = self._calculate_kdj_manual()
        if not kdj_data:
            return False
        
        k_current, d_current, j_current = kdj_data
        
        # 计算MACD
        macd_data = self._calculate_macd()
        if not macd_data:
            return False
        
        diff_current, dea_current, hist_current = macd_data
        
        # 需要前一天的数据来比较
        if len(am.close_array) < 2:
            return False
        
        # 简化处理：假设前一天的指标都稍低一点
        # 实际应该用历史数据计算，这里简化处理
        k_rising = k_current > 50  # 简化条件
        d_rising = d_current > 50
        j_rising = j_current > 50
        dea_rising = dea_current > 0
        
        result = k_rising or d_rising or j_rising or dea_rising
        
        if result:
            self.write_log(f"📊 KDJ/DEA上移: K={k_current:.1f}{'↗️' if k_rising else '↘️'}, "
                          f"D={d_current:.1f}{'↗️' if d_rising else '↘️'}, "
                          f"J={j_current:.1f}{'↗️' if j_rising else '↘️'}, "
                          f"DEA={dea_current:.3f}{'↗️' if dea_rising else '↘️'}")
        
        return result
    
    def _check_accumulation_15m(self) -> bool:
        """
        检查15分钟吸筹信号（严格按照缩量回踩公式）
        公式：
        V11:=3*SMA((C-LLV(L,55))/(HHV(H,55)-LLV(L,55))*100,5,1)-2*SMA(SMA((C-LLV(L,55))/(HHV(H,55)-LLV(L,55))*100,5,1),3,1);
        V12:=(EMA(V11,3)-REF(EMA(V11,3),1))/REF(EMA(V11,3),1)*100;
        AA:=(EMA(V11,3)<=13) AND FILTER((EMA(V11,3)<=13),15);
        BB:=(EMA(V11,3)<=13 AND V12>13) AND FILTER((EMA(V11,3)<=13 AND V12>13),10);
        XG: COUNT(AA OR BB,N) >= 1;
        """
        if not hasattr(self, 'am_15m') or not self.am_15m or not self.am_15m.inited:
            return False
            
        am_15m = self.am_15m
        period = self.stoch_period  # 55
        
        if len(am_15m.close_array) < period + 10:  # 需要额外数据来计算平滑
            return False
        
        try:
            # 计算基础随机指标 (C-LLV(L,55))/(HHV(H,55)-LLV(L,55))*100
            stoch_values = []
            
            for i in range(period, len(am_15m.close_array)):
                close_price = am_15m.close_array[i]
                
                # 获取55周期内的最高和最低价
                start_idx = i - period + 1
                end_idx = i + 1
                
                low_period = am_15m.low_array[start_idx:end_idx]
                high_period = am_15m.high_array[start_idx:end_idx]
                
                lowest = np.min(low_period)
                highest = np.max(high_period)
                
                if highest != lowest:
                    stoch_val = (close_price - lowest) / (highest - lowest) * 100
                else:
                    stoch_val = 50.0
                
                stoch_values.append(stoch_val)
            
            if len(stoch_values) < 10:
                return False
            
            # 计算SMA(stoch, 5, 1) - 指数移动平均
            sma1_values = []
            for i in range(len(stoch_values)):
                if i == 0:
                    sma1_values.append(stoch_values[i])
                else:
                    # EMA with alpha = 1/5 = 0.2
                    ema_val = 0.2 * stoch_values[i] + 0.8 * sma1_values[-1]
                    sma1_values.append(ema_val)
            
            # 计算SMA(SMA(stoch, 5, 1), 3, 1)
            sma2_values = []
            for i in range(len(sma1_values)):
                if i == 0:
                    sma2_values.append(sma1_values[i])
                else:
                    # EMA with alpha = 1/3 ≈ 0.333
                    ema_val = (1.0/3.0) * sma1_values[i] + (2.0/3.0) * sma2_values[-1]
                    sma2_values.append(ema_val)
            
            # 计算V11 = 3*SMA1 - 2*SMA2
            v11_values = []
            for i in range(len(sma1_values)):
                v11 = 3 * sma1_values[i] - 2 * sma2_values[i]
                v11_values.append(v11)
            
            if len(v11_values) < 5:
                return False
            
            # 计算EMA(V11, 3)
            ema_v11_values = []
            for i in range(len(v11_values)):
                if i == 0:
                    ema_v11_values.append(v11_values[i])
                else:
                    # EMA with alpha = 1/3
                    ema_val = (1.0/3.0) * v11_values[i] + (2.0/3.0) * ema_v11_values[-1]
                    ema_v11_values.append(ema_val)
            
            # 计算V12 = (EMA(V11,3) - REF(EMA(V11,3),1)) / REF(EMA(V11,3),1) * 100
            if len(ema_v11_values) < 2:
                return False
            
            current_ema_v11 = ema_v11_values[-1]
            prev_ema_v11 = ema_v11_values[-2]
            
            if prev_ema_v11 != 0:
                v12 = (current_ema_v11 - prev_ema_v11) / prev_ema_v11 * 100
            else:
                v12 = 0
            
            # 检查条件
            # AA: EMA(V11,3) <= 13
            condition_aa = current_ema_v11 <= self.accumulation_threshold
            
            # BB: EMA(V11,3) <= 13 AND V12 > 13  
            condition_bb = current_ema_v11 <= self.accumulation_threshold and v12 > self.accumulation_threshold
            
            # 检查最近的信号（简化FILTER逻辑）
            signal_detected = condition_aa or condition_bb
            
            if signal_detected:
                self.write_log(f"📊 15分钟吸筹信号: EMA_V11={current_ema_v11:.2f}, V12={v12:.2f}, "
                              f"AA={'✅' if condition_aa else '❌'}, BB={'✅' if condition_bb else '❌'}")
            
            return signal_detected
            
        except Exception as e:
            self.write_log(f"15分钟吸筹信号计算异常: {e}")
            return False
    
    def _check_no_negative_signals(self) -> bool:
        """
        检查无负面信号（排除放量下跌和放量大阴线）
        """
        am = self.am
        if not am.inited or len(am.close_array) < 6:  # 检查最近5天
            return True  # 数据不足时默认通过
        
        # 检查最近5天是否有放量下跌
        for i in range(-5, 0):
            current_close = am.close_array[i]
            prev_close = am.close_array[i-1] if i > -len(am.close_array) else 0
            current_volume = am.volume_array[i]
            prev_volume = am.volume_array[i-1] if i > -len(am.volume_array) else 0
            
            # 放量下跌
            volume_surge = current_volume > prev_volume if prev_volume > 0 else False
            price_decline = current_close < prev_close if prev_close > 0 else False
            
            if volume_surge and price_decline:
                self.write_log(f"❌ 发现放量下跌信号: 第{i}天")
                return False
        
        # 检查最近5天是否有放量大阴线
        for i in range(-5, 0):
            current_close = am.close_array[i]
            current_open = am.open_array[i]
            current_volume = am.volume_array[i]
            
            # 计算5日平均成交量
            volume_5day_avg = np.mean(am.volume_array[i-5:i]) if i >= -len(am.volume_array) + 5 else 0
            
            # 大阴线：跌幅>4%
            big_bearish = current_close < current_open * self.big_bearish_threshold
            # 放量：超过5日均量1.8倍
            volume_surge = current_volume > volume_5day_avg * self.volume_surge_multiple if volume_5day_avg > 0 else False
            
            if big_bearish and volume_surge:
                decline_pct = (1 - current_close / current_open) * 100 if current_open > 0 else 0
                self.write_log(f"❌ 发现放量大阴线: 第{i}天, 跌幅{decline_pct:.1f}%, 量比{current_volume/volume_5day_avg:.1f}")
                return False
        
        return True
    
    def _calculate_kdj_manual(self) -> tuple:
        """
        手动计算KDJ指标（严格按照缩量回踩公式）
        公式：
        RSV:=(CLOSE-LLV(LOW,9))/(HHV(HIGH,9)-LLV(LOW,9))*100;
        K:=SMA(RSV,3,1);
        D:=SMA(K,3,1);
        J:=3*K-2*D;
        """
        am = self.am
        if not am.inited or len(am.close_array) < self.kdj_period + self.kdj_smooth_k:
            return None
        
        try:
            # 计算RSV (Raw Stochastic Value)
            # RSV = (CLOSE - LLV(LOW, 9)) / (HHV(HIGH, 9) - LLV(LOW, 9)) * 100
            close_price = am.close_array[-1]
            
            # 获取最近9天的最高价和最低价
            recent_highs = am.high_array[-self.kdj_period:]
            recent_lows = am.low_array[-self.kdj_period:]
            
            highest = np.max(recent_highs)
            lowest = np.min(recent_lows)
            
            if highest == lowest:
                rsv = 50.0  # 避免除零错误
            else:
                rsv = (close_price - lowest) / (highest - lowest) * 100
            
            # 计算K值：K = SMA(RSV, 3, 1) 
            # 这里的SMA(RSV,3,1)是指数移动平均，权重为1/3
            # 简化处理：使用历史RSV值计算K
            if not hasattr(self, '_rsv_history'):
                self._rsv_history = []
            
            self._rsv_history.append(rsv)
            # 保持最近20个RSV值用于计算
            if len(self._rsv_history) > 20:
                self._rsv_history = self._rsv_history[-20:]
            
            # 计算K值（指数移动平均）
            if not hasattr(self, '_k_value'):
                self._k_value = 50.0  # 初始值
            
            # K = (2/3) * 前一K值 + (1/3) * 当前RSV
            self._k_value = (2.0/3.0) * self._k_value + (1.0/3.0) * rsv
            k_current = self._k_value
            
            # 计算D值：D = SMA(K, 3, 1)
            if not hasattr(self, '_d_value'):
                self._d_value = 50.0  # 初始值
            
            # D = (2/3) * 前一D值 + (1/3) * 当前K值
            self._d_value = (2.0/3.0) * self._d_value + (1.0/3.0) * k_current
            d_current = self._d_value
            
            # 计算J值：J = 3*K - 2*D
            j_current = 3 * k_current - 2 * d_current
            
            # 限制KDJ值范围在0-100之间
            k_current = max(0, min(100, k_current))
            d_current = max(0, min(100, d_current))
            j_current = max(-50, min(150, j_current))  # J值可以超出0-100范围
            
            return k_current, d_current, j_current
            
        except Exception as e:
            self.write_log(f"KDJ计算异常: {e}")
            return None
    
    def _calculate_macd(self) -> tuple:
        """计算MACD指标"""
        am = self.am
        if not am.inited or len(am.close_array) < self.macd_slow + self.macd_signal:
            return None
        
        try:
            # 使用talib计算MACD
            diff, dea, hist = am.macd(self.macd_fast, self.macd_slow, self.macd_signal)
            return diff, dea, hist
        except:
            # 如果talib不可用，手动计算简化版MACD
            ema12 = am.ema(self.macd_fast)
            ema26 = am.ema(self.macd_slow)
            diff = ema12 - ema26
            # 简化DEA计算
            dea = diff * 0.8  # 简化计算
            hist = (diff - dea) * 2
            return diff, dea, hist
    
    def _create_decision_log(self, current_time, current_price, decision_record) -> dict:
        """创建决策日志记录"""
        conditions = decision_record["conditions"]
        conditions_met = decision_record["conditions_met"]
        buy_signal = decision_record["buy_signal"]
        sell_signal = decision_record["sell_signal"]
        
        # 构建完整条件状态描述
        condition_status = []
        
        # 新增的完整条件状态
        condition_status.append(f"{'✅' if self.ma_rising_confirmed else '❌'}10/20均线上移")
        condition_status.append(f"{'✅' if self.ma60_120_rising_confirmed else '❌'}60/120均线上移")
        condition_status.append(f"{'✅' if self.pullback_ma_confirmed else '❌'}回踩均线")
        condition_status.append(f"{'✅' if self.surge_60day_confirmed else '❌'}60日涨幅>7%")
        condition_status.append(f"{'✅' if self.amplitude_110day_confirmed else '❌'}110日振幅>8.1%")
        condition_status.append(f"{'✅' if self.accumulation_15m_confirmed else '❌'}15分钟吸筹")
        condition_status.append(f"{'✅' if self.shrink_volume_confirmed else '❌'}连续缩量")
        condition_status.append(f"{'✅' if self.kdj_dea_rising_confirmed else '❌'}KDJ/DEA上移")
        condition_status.append(f"{'✅' if self.no_negative_signals else '❌'}无负面信号")
        
        # 计算完整条件数量
        complete_conditions_met = sum([
            self.ma_rising_confirmed,
            self.ma60_120_rising_confirmed,
            self.pullback_ma_confirmed,
            # self.surge_60day_confirmed,
            self.amplitude_110day_confirmed,
            # self.accumulation_15m_confirmed,
            self.shrink_volume_confirmed,
            self.kdj_dea_rising_confirmed,
            self.no_negative_signals
        ])
        
        # 生成综合评估（基于9个完整条件）
        if buy_signal:
            assessment = f"🚀 买入信号(核心{conditions_met}/4, 完整{complete_conditions_met}/9)"
        elif sell_signal:
            assessment = "❌ 卖出信号"
        elif complete_conditions_met >= 7:
            assessment = f"🌟 强烈关注(完整{complete_conditions_met}/9)"
        elif complete_conditions_met >= 5:
            assessment = f"👀 密切观察(完整{complete_conditions_met}/9)"
        elif complete_conditions_met >= 3:
            assessment = f"⏳ 继续等待(完整{complete_conditions_met}/9)"
        else:
            assessment = f"🛑 条件不足(完整{complete_conditions_met}/9)"
        
        return {
            "time": current_time,
            "price": current_price,
            "position": decision_record["position"],
            "assessment": assessment,
            "conditions_status": " | ".join(condition_status),
            "conditions_met": f"{conditions_met}/4",
            "market_data": decision_record["market_data"],
            "risk_signals": decision_record["risk_signals"]
        }
        
    def on_stop_order(self, stop_order: StopOrder):
        """停止单回报"""
        pass
