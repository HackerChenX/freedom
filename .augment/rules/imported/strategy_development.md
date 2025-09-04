---
type: "agent_requested"
description: "Example description"
---

# 策略开发规范（策略系统）

## 🎯 策略系统架构概览

策略系统基于六层架构的L5业务应用层，主要组件：

- **策略管理器** ([strategy_manager.py](mdc:strategy/strategy_manager.py)): 策略生命周期管理
- **策略执行器** ([strategy_executor.py](mdc:strategy/strategy_executor.py)): 策略运行引擎
- **策略解析器** ([strategy_parser.py](mdc:strategy/strategy_parser.py)): 策略配置解析
- **回测框架** ([backtester.py](mdc:strategy/backtester.py)): 策略回测系统
- **多周期策略** ([multi_period_strategy.py](mdc:strategy/multi_period_strategy.py)): 跨周期策略支持

## 📋 策略开发标准模板

### 基础策略类结构
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from datetime import datetime
from utils.logger import get_logger
from utils.performance_monitor import performance_monitor
from utils.exception_handler import exception_handler
from enums.signal_types import SignalType
from enums.position_types import PositionType

logger = get_logger(__name__)

class BaseStrategy(ABC):
    """
    策略基类 - 所有策略必须继承此类
    
    Attributes:
        name (str): 策略名称
        version (str): 策略版本
        params (Dict): 策略参数
        indicators (List): 使用的指标列表
    """
    
    def __init__(self, name: str, version: str = "1.0.0", **params):
        """
        初始化策略
        
        Args:
            name: 策略名称
            version: 策略版本
            **params: 策略参数
        """
        self.name = name
        self.version = version
        self.params = params
        self.indicators = []
        self.position = PositionType.NONE
        self.last_signal_time = None
        self._validate_params()
        self._initialize_indicators()
    
    def _validate_params(self):
        """验证策略参数"""
        required_params = self.get_required_params()
        for param in required_params:
            if param not in self.params:
                raise ValueError(f"策略 {self.name} 缺少必需参数: {param}")
    
    @abstractmethod
    def get_required_params(self) -> List[str]:
        """返回策略必需参数列表"""
        pass
    
    @abstractmethod
    def _initialize_indicators(self):
        """初始化策略使用的指标"""
        pass
    
    @abstractmethod
    @performance_monitor(threshold_seconds=3.0)
    @exception_handler(reraise=True)
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        生成交易信号
        
        Args:
            data: 股票历史数据
            
        Returns:
            Dict[str, Any]: 交易信号
            {
                'signal': SignalType,  # BUY/SELL/HOLD
                'strength': float,     # 信号强度 0-1
                'confidence': float,   # 置信度 0-1
                'entry_price': float,  # 建议入场价格
                'stop_loss': float,    # 止损价格
                'take_profit': float,  # 止盈价格
                'position_size': float, # 仓位大小
                'reason': str,         # 信号原因
                'indicators': Dict     # 指标值
            }
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, data: pd.DataFrame, signal_strength: float) -> float:
        """
        计算仓位大小
        
        Args:
            data: 股票数据
            signal_strength: 信号强度
            
        Returns:
            float: 仓位比例 (0-1)
        """
        pass
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """获取策略信息"""
        return {
            'name': self.name,
            'version': self.version,
            'type': self.__class__.__name__,
            'params': self.params,
            'indicators': [ind.name for ind in self.indicators],
            'last_signal_time': self.last_signal_time
        }
    
    def update_position(self, new_position: PositionType):
        """更新持仓状态"""
        self.position = new_position
        logger.info(f"策略 {self.name} 仓位更新为: {new_position}")
```

### 具体策略实现示例
```python
from indicators.complete_indicator_registry import get_indicator
from enums.signal_types import SignalType
from enums.position_types import PositionType

class MACDCrossStrategy(BaseStrategy):
    """
    MACD金叉死叉策略
    
    策略逻辑：
    - MACD金叉时买入
    - MACD死叉时卖出
    - 结合RSI过滤信号
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, 
                 rsi_period: int = 14, rsi_oversold: float = 30, rsi_overbought: float = 70):
        params = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period,
            'rsi_period': rsi_period,
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought
        }
        super().__init__("MACD_Cross_Strategy", "1.0.0", **params)
    
    def get_required_params(self) -> List[str]:
        return ['fast_period', 'slow_period', 'signal_period', 'rsi_period']
    
    def _initialize_indicators(self):
        """初始化MACD和RSI指标"""
        try:
            self.macd = get_indicator('MACD')(
                fast_period=self.params['fast_period'],
                slow_period=self.params['slow_period'],
                signal_period=self.params['signal_period']
            )
            self.rsi = get_indicator('RSI')(period=self.params['rsi_period'])
            self.indicators = [self.macd, self.rsi]
            logger.info(f"策略 {self.name} 指标初始化成功")
        except Exception as e:
            logger.error(f"策略 {self.name} 指标初始化失败: {e}")
            raise
    
    @performance_monitor(threshold_seconds=2.0)
    @exception_handler(reraise=True)
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成MACD交叉信号"""
        if len(data) < max(self.params['slow_period'], self.params['rsi_period']) + 10:
            return self._create_hold_signal("数据不足")
        
        # 计算指标
        macd_data = self.macd.calculate(data)
        rsi_data = self.rsi.calculate(macd_data)
        
        # 获取最新值
        current_macd = macd_data['MACD'].iloc[-1]
        current_signal = macd_data['MACD_Signal'].iloc[-1]
        prev_macd = macd_data['MACD'].iloc[-2]
        prev_signal = macd_data['MACD_Signal'].iloc[-2]
        current_rsi = rsi_data['RSI'].iloc[-1]
        current_price = data['close'].iloc[-1]
        
        # 检查金叉/死叉
        golden_cross = (prev_macd <= prev_signal) and (current_macd > current_signal)
        death_cross = (prev_macd >= prev_signal) and (current_macd < current_signal)
        
        # 生成信号
        if golden_cross and current_rsi < self.params['rsi_overbought']:
            return self._create_buy_signal(current_price, current_rsi, current_macd)
        elif death_cross or current_rsi > self.params['rsi_overbought']:
            return self._create_sell_signal(current_price, current_rsi, current_macd)
        else:
            return self._create_hold_signal("无明确信号")
    
    def _create_buy_signal(self, price: float, rsi: float, macd: float) -> Dict[str, Any]:
        """创建买入信号"""
        strength = min(abs(macd) / 0.1, 1.0)  # 基于MACD强度
        confidence = 0.8 if rsi < self.params['rsi_oversold'] else 0.6
        
        return {
            'signal': SignalType.BUY,
            'strength': strength,
            'confidence': confidence,
            'entry_price': price,
            'stop_loss': price * 0.95,  # 5%止损
            'take_profit': price * 1.10,  # 10%止盈
            'position_size': self.calculate_position_size(None, strength),
            'reason': 'MACD金叉买入信号',
            'indicators': {
                'MACD': macd,
                'RSI': rsi,
                'price': price
            }
        }
    
    def _create_sell_signal(self, price: float, rsi: float, macd: float) -> Dict[str, Any]:
        """创建卖出信号"""
        return {
            'signal': SignalType.SELL,
            'strength': 0.8,
            'confidence': 0.9,
            'entry_price': price,
            'stop_loss': 0,
            'take_profit': 0,
            'position_size': 1.0,  # 全部卖出
            'reason': 'MACD死叉或RSI超买',
            'indicators': {
                'MACD': macd,
                'RSI': rsi,
                'price': price
            }
        }
    
    def _create_hold_signal(self, reason: str) -> Dict[str, Any]:
        """创建持有信号"""
        return {
            'signal': SignalType.HOLD,
            'strength': 0.0,
            'confidence': 0.5,
            'entry_price': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'position_size': 0,
            'reason': reason,
            'indicators': {}
        }
    
    def calculate_position_size(self, data: pd.DataFrame, signal_strength: float) -> float:
        """基于信号强度计算仓位"""
        base_position = 0.3  # 基础仓位30%
        max_position = 0.8   # 最大仓位80%
        
        return min(base_position + (signal_strength * 0.5), max_position)
```

## 🔄 策略管理系统

### 策略注册和管理
参考 [strategy_manager.py](mdc:strategy/strategy_manager.py) 的实现：

```python
from typing import Dict, List, Type
from utils.container import container

class StrategyManager:
    """策略管理器 - 管理所有策略的生命周期"""
    
    def __init__(self):
        self.strategies: Dict[str, Type[BaseStrategy]] = {}
        self.active_strategies: Dict[str, BaseStrategy] = {}
        self.data_access = container.resolve("DataAccessInterface")
        self._register_built_in_strategies()
    
    def register_strategy(self, strategy_class: Type[BaseStrategy]) -> bool:
        """注册策略类"""
        try:
            strategy_name = strategy_class.__name__
            self.strategies[strategy_name] = strategy_class
            logger.info(f"策略 {strategy_name} 注册成功")
            return True
        except Exception as e:
            logger.error(f"策略注册失败: {e}")
            return False
    
    def create_strategy(self, strategy_name: str, **params) -> Optional[BaseStrategy]:
        """创建策略实例"""
        if strategy_name not in self.strategies:
            logger.error(f"未找到策略: {strategy_name}")
            return None
        
        try:
            strategy_class = self.strategies[strategy_name]
            strategy = strategy_class(**params)
            self.active_strategies[strategy.name] = strategy
            logger.info(f"策略实例 {strategy.name} 创建成功")
            return strategy
        except Exception as e:
            logger.error(f"创建策略实例失败: {e}")
            return None
    
    def run_strategy(self, strategy_name: str, stock_code: str, 
                    start_date: str, end_date: str) -> Optional[Dict[str, Any]]:
        """运行策略"""
        if strategy_name not in self.active_strategies:
            logger.error(f"策略实例 {strategy_name} 不存在")
            return None
        
        try:
            strategy = self.active_strategies[strategy_name]
            
            # 获取股票数据
            data = self.data_access.get_stock_data(stock_code, start_date, end_date)
            if data.empty:
                logger.warning(f"股票 {stock_code} 无数据")
                return None
            
            # 生成信号
            signal = strategy.generate_signal(data)
            signal['strategy'] = strategy_name
            signal['stock_code'] = stock_code
            signal['timestamp'] = datetime.now().isoformat()
            
            return signal
            
        except Exception as e:
            logger.error(f"运行策略 {strategy_name} 失败: {e}")
            return None
```

### 策略执行器
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

class StrategyExecutor:
    """策略执行器 - 并行执行多个策略"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.strategy_manager = StrategyManager()
    
    @performance_monitor(threshold_seconds=10.0)
    def execute_batch_strategies(self, stock_codes: List[str], 
                                strategy_configs: List[Dict]) -> Dict[str, List[Dict]]:
        """批量执行策略"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            futures = {}
            for stock_code in stock_codes:
                for config in strategy_configs:
                    future = executor.submit(
                        self._execute_single_strategy,
                        stock_code, config
                    )
                    futures[future] = (stock_code, config['name'])
            
            # 收集结果
            for future in as_completed(futures):
                stock_code, strategy_name = futures[future]
                try:
                    signal = future.result()
                    if signal:
                        if stock_code not in results:
                            results[stock_code] = []
                        results[stock_code].append(signal)
                except Exception as e:
                    logger.error(f"策略执行异常 {stock_code}-{strategy_name}: {e}")
        
        return results
    
    def _execute_single_strategy(self, stock_code: str, config: Dict) -> Optional[Dict]:
        """执行单个策略"""
        try:
            # 创建策略实例
            strategy = self.strategy_manager.create_strategy(
                config['name'], **config.get('params', {})
            )
            if not strategy:
                return None
            
            # 运行策略
            return self.strategy_manager.run_strategy(
                strategy.name, stock_code, 
                config['start_date'], config['end_date']
            )
            
        except Exception as e:
            logger.error(f"单策略执行失败: {e}")
            return None
```

## 📊 回测框架

### 回测引擎
```python
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class BacktestResult:
    """回测结果数据类"""
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float

class Backtester:
    """策略回测框架"""
    
    def __init__(self, initial_capital: float = 100000.0, commission: float = 0.003):
        """
        初始化回测器
        
        Args:
            initial_capital: 初始资金
            commission: 交易手续费率
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.trades = []
        self.positions = {}
        self.cash = initial_capital
        self.portfolio_value = initial_capital
        self.data_access = container.resolve("DataAccessInterface")
    
    @performance_monitor(threshold_seconds=30.0)
    def run_backtest(self, strategy: BaseStrategy, stock_code: str,
                    start_date: str, end_date: str) -> BacktestResult:
        """
        运行策略回测
        
        Args:
            strategy: 策略实例
            stock_code: 股票代码
            start_date: 回测开始日期
            end_date: 回测结束日期
            
        Returns:
            BacktestResult: 回测结果
        """
        # 获取历史数据
        data = self.data_access.get_stock_data(stock_code, start_date, end_date)
        if data.empty:
            raise ValueError(f"无法获取股票 {stock_code} 的历史数据")
        
        # 重置回测状态
        self._reset_backtest_state()
        
        # 逐日回测
        for i in range(len(data)):
            if i < 50:  # 需要足够的历史数据
                continue
            
            current_data = data.iloc[:i+1]
            current_date = data.iloc[i]['date']
            current_price = data.iloc[i]['close']
            
            # 生成策略信号
            try:
                signal = strategy.generate_signal(current_data)
                if signal['signal'] != SignalType.HOLD:
                    self._process_signal(signal, stock_code, current_price, current_date)
            except Exception as e:
                logger.warning(f"策略信号生成失败 {current_date}: {e}")
                continue
            
            # 更新组合价值
            self._update_portfolio_value(current_price)
        
        # 计算回测结果
        return self._calculate_backtest_result(start_date, end_date)
    
    def _reset_backtest_state(self):
        """重置回测状态"""
        self.trades = []
        self.positions = {}
        self.cash = self.initial_capital
        self.portfolio_value = self.initial_capital
    
    def _process_signal(self, signal: Dict, stock_code: str, price: float, date: str):
        """处理交易信号"""
        if signal['signal'] == SignalType.BUY:
            self._execute_buy(stock_code, price, signal['position_size'], date)
        elif signal['signal'] == SignalType.SELL:
            self._execute_sell(stock_code, price, signal['position_size'], date)
    
    def _execute_buy(self, stock_code: str, price: float, position_size: float, date: str):
        """执行买入操作"""
        available_cash = self.cash * position_size
        shares = int(available_cash / price)
        cost = shares * price * (1 + self.commission)
        
        if cost <= self.cash and shares > 0:
            self.cash -= cost
            if stock_code in self.positions:
                self.positions[stock_code] += shares
            else:
                self.positions[stock_code] = shares
            
            self.trades.append({
                'date': date,
                'type': 'BUY',
                'stock_code': stock_code,
                'price': price,
                'shares': shares,
                'amount': cost
            })
            logger.debug(f"买入 {stock_code}: {shares}股 @ {price}")
    
    def _execute_sell(self, stock_code: str, price: float, position_size: float, date: str):
        """执行卖出操作"""
        if stock_code not in self.positions:
            return
        
        shares_to_sell = int(self.positions[stock_code] * position_size)
        if shares_to_sell <= 0:
            return
        
        proceeds = shares_to_sell * price * (1 - self.commission)
        self.cash += proceeds
        self.positions[stock_code] -= shares_to_sell
        
        if self.positions[stock_code] <= 0:
            del self.positions[stock_code]
        
        self.trades.append({
            'date': date,
            'type': 'SELL',
            'stock_code': stock_code,
            'price': price,
            'shares': shares_to_sell,
            'amount': proceeds
        })
        logger.debug(f"卖出 {stock_code}: {shares_to_sell}股 @ {price}")
    
    def _update_portfolio_value(self, current_price: float):
        """更新组合价值"""
        stock_value = sum(shares * current_price for shares in self.positions.values())
        self.portfolio_value = self.cash + stock_value
    
    def _calculate_backtest_result(self, start_date: str, end_date: str) -> BacktestResult:
        """计算回测结果"""
        if not self.trades:
            return BacktestResult(
                total_return=0, annual_return=0, max_drawdown=0, sharpe_ratio=0,
                win_rate=0, profit_factor=0, total_trades=0, winning_trades=0,
                losing_trades=0, avg_win=0, avg_loss=0, start_date=start_date,
                end_date=end_date, initial_capital=self.initial_capital,
                final_capital=self.portfolio_value
            )
        
        # 计算基本指标
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        # 计算年化收益率
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        days = (end_dt - start_dt).days
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # 分析交易
        buy_trades = [t for t in self.trades if t['type'] == 'BUY']
        sell_trades = [t for t in self.trades if t['type'] == 'SELL']
        
        # 配对交易计算盈亏
        wins = []
        losses = []
        for sell_trade in sell_trades:
            # 简化匹配：找最近的买入交易
            matching_buys = [b for b in buy_trades 
                           if b['stock_code'] == sell_trade['stock_code'] 
                           and b['date'] <= sell_trade['date']]
            if matching_buys:
                buy_trade = matching_buys[-1]
                profit = (sell_trade['price'] - buy_trade['price']) * sell_trade['shares']
                if profit > 0:
                    wins.append(profit)
                else:
                    losses.append(abs(profit))
        
        # 计算交易统计
        total_trades = len(wins) + len(losses)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        profit_factor = sum(wins) / sum(losses) if losses else float('inf') if wins else 0
        
        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            max_drawdown=0,  # 简化，实际需要计算最大回撤
            sharpe_ratio=0,  # 简化，实际需要计算夏普比率
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=self.portfolio_value
        )
```

## 🔧 策略配置系统

### 策略配置文件标准
```yaml
# config/strategies/macd_cross_strategy.yaml
strategy:
  name: "MACD_Cross_Strategy"
  version: "1.0.0"
  description: "MACD金叉死叉策略"
  
parameters:
  fast_period: 12
  slow_period: 26
  signal_period: 9
  rsi_period: 14
  rsi_oversold: 30
  rsi_overbought: 70
  
risk_management:
  max_position_size: 0.8
  stop_loss_pct: 0.05
  take_profit_pct: 0.10
  max_drawdown: 0.15
  
backtest:
  initial_capital: 100000
  commission: 0.003
  benchmark: "000001"
  
universe:
  - "000001"
  - "000002"
  - "600036"
```

### 策略解析器
参考 [strategy_parser.py](mdc:strategy/strategy_parser.py) 的实现：

```python
import yaml
from pathlib import Path

class StrategyParser:
    """策略配置解析器"""
    
    def __init__(self, config_dir: str = "config/strategies"):
        self.config_dir = Path(config_dir)
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def parse_strategy_config(self, config_file: str) -> Dict[str, Any]:
        """解析策略配置文件"""
        config_path = self.config_dir / config_file
        if not config_path.exists():
            raise FileNotFoundError(f"策略配置文件不存在: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 验证配置结构
            self._validate_config(config)
            return config
            
        except Exception as e:
            logger.error(f"解析策略配置失败: {e}")
            raise
    
    def _validate_config(self, config: Dict[str, Any]):
        """验证配置文件结构"""
        required_sections = ['strategy', 'parameters']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"配置文件缺少必需部分: {section}")
        
        strategy_info = config['strategy']
        required_fields = ['name', 'version']
        for field in required_fields:
            if field not in strategy_info:
                raise ValueError(f"策略信息缺少必需字段: {field}")
    
    def get_strategy_dir(self) -> str:
        """获取策略配置目录"""
        return str(self.config_dir)
```

## ✅ 策略开发检查清单

新策略开发必须满足：

- [ ] 继承BaseStrategy基类
- [ ] 实现所有抽象方法
- [ ] 包含完整的参数验证
- [ ] 添加性能监控装饰器
- [ ] 包含异常处理机制
- [ ] 创建对应的配置文件
- [ ] 编写单元测试
- [ ] 通过回测验证
- [ ] 注册到策略管理器
- [ ] 文档完整（策略说明、参数说明、风险提示）

### 策略性能要求
- 信号生成时间 < 3秒
- 支持并行执行
- 内存使用合理
- 错误恢复能力
- 完整的日志记录

### 风险管理要求
- 强制止损机制
- 仓位管理
- 最大回撤控制
- 相关性检查
- 压力测试通过

这些规范确保我们的策略系统安全、高效、可扩展。
description:
globs:
alwaysApply: false
---
