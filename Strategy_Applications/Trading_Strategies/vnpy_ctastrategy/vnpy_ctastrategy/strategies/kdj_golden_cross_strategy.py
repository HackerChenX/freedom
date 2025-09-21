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


class KdjGoldenCrossStrategy(CtaTemplate):
    """
    KDJ金叉选股策略
    
    基于KDJ指标的金叉信号进行选股交易：
    - 当K线上穿D线时产生买入信号
    - 当K线下穿D线时产生卖出信号
    """

    author = "VnPy Freedom"

    # 策略参数
    kdj_period: int = 9        # KDJ计算周期
    k_smooth: int = 3          # K值平滑周期
    d_smooth: int = 3          # D值平滑周期
    overbought: float = 80.0   # 超买线
    oversold: float = 20.0     # 超卖线
    
    # 策略变量
    k_value: float = 0.0       # 当前K值
    d_value: float = 0.0       # 当前D值
    j_value: float = 0.0       # 当前J值
    k_value_last: float = 0.0  # 上一个K值
    d_value_last: float = 0.0  # 上一个D值
    
    signal_type: str = ""      # 信号类型
    
    parameters = ["kdj_period", "k_smooth", "d_smooth", "overbought", "oversold"]
    variables = ["k_value", "d_value", "j_value", "signal_type"]

    def on_init(self) -> None:
        """
        策略初始化回调
        """
        self.write_log("KDJ金叉策略初始化")
        
        self.bg: BarGenerator = BarGenerator(self.on_bar)
        self.am: ArrayManager = ArrayManager()
        
        # 加载历史数据
        self.load_bar(30)

    def on_start(self) -> None:
        """
        策略启动回调
        """
        self.write_log("KDJ金叉策略启动")
        self.put_event()

    def on_stop(self) -> None:
        """
        策略停止回调
        """
        self.write_log("KDJ金叉策略停止")
        self.put_event()

    def on_tick(self, tick: TickData) -> None:
        """
        Tick数据更新回调
        """
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData) -> None:
        """
        K线数据更新回调
        """
        # 取消所有未成交订单
        self.cancel_all()
        
        # 更新数组管理器
        am = self.am
        am.update_bar(bar)
        if not am.inited:
            return
        
        # 计算KDJ指标
        self.calculate_kdj()
        
        # 判断交易信号
        self.check_trading_signal(bar)
        
        # 更新界面
        self.put_event()

    def calculate_kdj(self) -> None:
        """
        计算KDJ指标
        """
        am = self.am
        
        # 获取最高价、最低价、收盘价数组
        high_array = am.high_array[-self.kdj_period:]
        low_array = am.low_array[-self.kdj_period:]
        close_array = am.close_array[-self.kdj_period:]
        
        if len(high_array) < self.kdj_period:
            return
        
        # 计算RSV (Raw Stochastic Value)
        highest = max(high_array)
        lowest = min(low_array)
        current_close = close_array[-1]
        
        if highest == lowest:
            rsv = 50.0
        else:
            rsv = (current_close - lowest) / (highest - lowest) * 100
        
        # 保存上一个K、D值
        self.k_value_last = self.k_value
        self.d_value_last = self.d_value
        
        # 计算K值 (K = (2/3) * K前值 + (1/3) * RSV)
        if self.k_value == 0.0:  # 首次计算
            self.k_value = rsv
        else:
            self.k_value = (2 * self.k_value + rsv) / 3
        
        # 计算D值 (D = (2/3) * D前值 + (1/3) * K)
        if self.d_value == 0.0:  # 首次计算
            self.d_value = self.k_value
        else:
            self.d_value = (2 * self.d_value + self.k_value) / 3
        
        # 计算J值 (J = 3K - 2D)
        self.j_value = 3 * self.k_value - 2 * self.d_value

    def check_trading_signal(self, bar: BarData) -> None:
        """
        检查交易信号
        """
        # 需要有历史K、D值才能判断金叉死叉
        if self.k_value_last == 0.0 or self.d_value_last == 0.0:
            return
        
        # 金叉信号：K线上穿D线
        golden_cross = (
            self.k_value > self.d_value and 
            self.k_value_last <= self.d_value_last
        )
        
        # 死叉信号：K线下穿D线
        death_cross = (
            self.k_value < self.d_value and 
            self.k_value_last >= self.d_value_last
        )
        
        # 超卖区金叉（强买入信号）
        oversold_golden = golden_cross and self.d_value < self.oversold
        
        # 超买区死叉（强卖出信号）
        overbought_death = death_cross and self.d_value > self.overbought
        
        # 执行交易逻辑
        if oversold_golden:
            self.signal_type = "超卖金叉"
            if self.pos == 0:
                self.buy(bar.close_price, 1)
                self.write_log(f"超卖区金叉买入: K={self.k_value:.2f}, D={self.d_value:.2f}")
            elif self.pos < 0:
                self.cover(bar.close_price, abs(self.pos))
                self.buy(bar.close_price, 1)
                self.write_log(f"超卖区金叉平空买入: K={self.k_value:.2f}, D={self.d_value:.2f}")
                
        elif golden_cross and not oversold_golden:
            self.signal_type = "普通金叉"
            if self.pos == 0:
                self.buy(bar.close_price, 1)
                self.write_log(f"金叉买入: K={self.k_value:.2f}, D={self.d_value:.2f}")
            elif self.pos < 0:
                self.cover(bar.close_price, abs(self.pos))
                self.write_log(f"金叉平空: K={self.k_value:.2f}, D={self.d_value:.2f}")
                
        elif overbought_death:
            self.signal_type = "超买死叉"
            if self.pos > 0:
                self.sell(bar.close_price, self.pos)
                self.write_log(f"超买区死叉卖出: K={self.k_value:.2f}, D={self.d_value:.2f}")
                
        elif death_cross and not overbought_death:
            self.signal_type = "普通死叉"
            if self.pos > 0:
                self.sell(bar.close_price, self.pos)
                self.write_log(f"死叉卖出: K={self.k_value:.2f}, D={self.d_value:.2f}")
        else:
            self.signal_type = "无信号"

    def on_order(self, order: OrderData) -> None:
        """
        订单状态更新回调
        """
        pass

    def on_trade(self, trade: TradeData) -> None:
        """
        成交数据更新回调
        """
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder) -> None:
        """
        停止单状态更新回调
        """
        pass
