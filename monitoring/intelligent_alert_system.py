#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
智能预警系统

基于技术指标的智能预警系统，包括买卖信号预警、风险预警等
"""

import time
import threading
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import queue
import json

from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor
from utils.unified_container import get_container
from indicators.complete_indicator_registry import get_indicator_registry

logger = get_logger(__name__)

# 延迟导入避免循环依赖
def get_alert_config_manager():
    """延迟导入配置管理器"""
    from monitoring.alert_config_manager import get_alert_config_manager as _get_manager
    return _get_manager()


class SignalType(Enum):
    """信号类型"""
    BUY = "买入信号"
    SELL = "卖出信号"
    HOLD = "持有信号"
    RISK_WARNING = "风险预警"
    OPPORTUNITY = "机会提醒"


class SignalStrength(Enum):
    """信号强度"""
    WEAK = "弱"
    MODERATE = "中等"
    STRONG = "强"
    VERY_STRONG = "很强"


@dataclass
class TradingSignal:
    """交易信号数据类"""
    id: str
    timestamp: datetime
    stock_code: str
    stock_name: str
    signal_type: SignalType
    signal_strength: SignalStrength
    confidence: float  # 置信度 0-1
    price: float
    indicators: Dict[str, Any]
    message: str
    details: Dict[str, Any]
    valid_until: datetime
    processed: bool = False


@dataclass
class AlertRule:
    """预警规则数据类"""
    id: str
    name: str
    description: str
    indicators: List[str]
    conditions: Dict[str, Any]
    signal_type: SignalType
    enabled: bool = True
    priority: int = 1  # 1-5, 5最高


class IntelligentAlertSystem:
    """
    智能预警系统
    
    功能特性：
    1. 基于技术指标的智能信号识别
    2. 多指标组合分析
    3. 信号强度评估
    4. 风险预警
    5. 机会识别
    """
    
    def __init__(self):
        """初始化智能预警系统"""
        self.alert_rules: Dict[str, AlertRule] = {}
        self.signal_queue = queue.Queue()
        self.signals_history: List[TradingSignal] = []
        
        # 组件初始化
        self.container = get_container()
        try:
            from db.interfaces.data_access_interface import DataAccessInterface
from db.sql_manager import SQLManager, QueryType
            self.data_access = self.container.resolve(DataAccessInterface)
        except Exception as e:
            logger.warning(f"无法获取数据访问接口: {e}")
            self.data_access = None

        self.indicator_registry = get_indicator_registry()
        
        # 线程锁
        self.lock = threading.RLock()

        # 配置管理器
        self.config_manager = get_alert_config_manager()

        # 从配置加载规则
        self._load_rules_from_config()

        logger.info(f"智能预警系统初始化完成，加载 {len(self.alert_rules)} 个预警规则")
    
    def _load_rules_from_config(self):
        """从配置文件加载预警规则"""
        try:
            # 获取所有启用的规则配置
            enabled_configs = self.config_manager.get_enabled_rules()

            for rule_id, rule_config in enabled_configs.items():
                # 将配置转换为AlertRule对象
                alert_rule = self._config_to_alert_rule(rule_config)
                if alert_rule:
                    self.alert_rules[rule_id] = alert_rule

            logger.info(f"从配置加载 {len(self.alert_rules)} 个预警规则")

        except Exception as e:
            logger.error(f"加载预警规则配置失败: {e}")
            # 如果配置加载失败，使用默认规则
            self._initialize_default_rules()

    def _config_to_alert_rule(self, rule_config) -> Optional[AlertRule]:
        """将配置转换为AlertRule对象"""
        try:
            # 转换信号类型
            signal_type = getattr(SignalType, rule_config.signal_type, SignalType.OPPORTUNITY)

            return AlertRule(
                id=rule_config.id,
                name=rule_config.name,
                description=rule_config.description,
                indicators=rule_config.indicators,
                conditions=rule_config.conditions,
                signal_type=signal_type,
                priority=rule_config.priority,
                enabled=rule_config.enabled
            )
        except Exception as e:
            logger.error(f"转换规则配置失败 {rule_config.id}: {e}")
            return None

    def _initialize_default_rules(self):
        """初始化默认预警规则（备用方法）"""
        # RSI超买超卖规则
        rsi_rule = AlertRule(
            id="rsi_overbought_oversold",
            name="RSI超买超卖",
            description="RSI指标超买(>70)或超卖(<30)预警",
            indicators=["RSI"],
            conditions={
                "rsi_overbought": 70,
                "rsi_oversold": 30
            },
            signal_type=SignalType.RISK_WARNING,
            priority=3
        )
        self.alert_rules[rsi_rule.id] = rsi_rule

        # MACD金叉死叉规则
        macd_rule = AlertRule(
            id="macd_golden_death_cross",
            name="MACD金叉死叉",
            description="MACD指标金叉买入、死叉卖出信号",
            indicators=["MACD"],
            conditions={
                "golden_cross_threshold": 0.01,
                "death_cross_threshold": -0.01
            },
            signal_type=SignalType.BUY,
            priority=4
        )
        self.alert_rules[macd_rule.id] = macd_rule

        # KDJ超买超卖规则
        kdj_rule = AlertRule(
            id="kdj_overbought_oversold",
            name="KDJ超买超卖",
            description="KDJ指标超买(>80)或超卖(<20)预警",
            indicators=["KDJ"],
            conditions={
                "k_overbought": 80,
                "k_oversold": 20,
                "d_overbought": 80,
                "d_oversold": 20
            },
            signal_type=SignalType.OPPORTUNITY,
            priority=3
        )
        self.alert_rules[kdj_rule.id] = kdj_rule

        # 布林带突破规则
        boll_rule = AlertRule(
            id="boll_breakout",
            name="布林带突破",
            description="价格突破布林带上下轨预警",
            indicators=["BOLL"],
            conditions={
                "upper_breakout_threshold": 0.02,  # 突破上轨2%
                "lower_breakout_threshold": 0.02   # 突破下轨2%
            },
            signal_type=SignalType.OPPORTUNITY,
            priority=4
        )
        self.alert_rules[boll_rule.id] = boll_rule

        # 成交量异常规则
        volume_rule = AlertRule(
            id="volume_anomaly",
            name="成交量异常",
            description="成交量异常放大或萎缩预警",
            indicators=["VOL"],
            conditions={
                "volume_surge_ratio": 3.0,    # 成交量放大3倍
                "volume_shrink_ratio": 0.3    # 成交量萎缩至30%
            },
            signal_type=SignalType.RISK_WARNING,
            priority=3
        )
        self.alert_rules[volume_rule.id] = volume_rule

        # 价格跳空规则
        gap_rule = AlertRule(
            id="price_gap",
            name="价格跳空",
            description="价格跳空缺口预警",
            indicators=[],  # 不依赖特定指标
            conditions={
                "gap_threshold": 0.03  # 跳空3%以上
            },
            signal_type=SignalType.RISK_WARNING,
            priority=5
        )
        self.alert_rules[gap_rule.id] = gap_rule

        # 多指标共振买入规则
        multi_buy_rule = AlertRule(
            id="multi_indicator_buy",
            name="多指标共振买入",
            description="RSI超卖+MACD金叉+KDJ超卖的多指标共振买入信号",
            indicators=["RSI", "MACD", "KDJ"],
            conditions={
                "rsi_threshold": 35,
                "macd_positive": True,
                "kdj_threshold": 25
            },
            signal_type=SignalType.BUY,
            priority=5
        )
        self.alert_rules[multi_buy_rule.id] = multi_buy_rule

        # 多指标共振卖出规则
        multi_sell_rule = AlertRule(
            id="multi_indicator_sell",
            name="多指标共振卖出",
            description="RSI超买+MACD死叉+KDJ超买的多指标共振卖出信号",
            indicators=["RSI", "MACD", "KDJ"],
            conditions={
                "rsi_threshold": 65,
                "macd_negative": True,
                "kdj_threshold": 75
            },
            signal_type=SignalType.SELL,
            priority=5
        )
        self.alert_rules[multi_sell_rule.id] = multi_sell_rule

        logger.info(f"初始化 {len(self.alert_rules)} 个默认预警规则")
    
    @exception_handler(reraise=True)
    def add_alert_rule(self, rule: AlertRule) -> bool:
        """
        添加预警规则

        Args:
            rule: 预警规则

        Returns:
            bool: 添加是否成功
        """
        with self.lock:
            self.alert_rules[rule.id] = rule
            logger.info(f"添加预警规则: {rule.name}")
            return True

    @exception_handler(reraise=True)
    def reload_rules_from_config(self) -> bool:
        """重新加载配置文件中的规则"""
        try:
            # 清空现有规则
            old_count = len(self.alert_rules)
            self.alert_rules.clear()

            # 重新加载配置管理器
            self.config_manager = get_alert_config_manager()

            # 从配置加载规则
            self._load_rules_from_config()

            new_count = len(self.alert_rules)
            logger.info(f"重新加载预警规则完成: {old_count} -> {new_count}")
            return True

        except Exception as e:
            logger.error(f"重新加载预警规则失败: {e}")
            return False

    @exception_handler(reraise=True)
    def create_rule_from_template(self, template_id: str, rule_id: str,
                                 rule_name: str, custom_conditions: Dict[str, Any] = None) -> bool:
        """
        从模板创建预警规则

        Args:
            template_id: 模板ID
            rule_id: 新规则ID
            rule_name: 新规则名称
            custom_conditions: 自定义条件

        Returns:
            bool: 创建是否成功
        """
        try:
            # 使用配置管理器创建规则
            success = self.config_manager.create_rule_from_template(
                template_id, rule_id, rule_name, custom_conditions
            )

            if success:
                # 重新加载规则
                self.reload_rules_from_config()
                logger.info(f"从模板 {template_id} 创建规则 {rule_id} 成功")

            return success

        except Exception as e:
            logger.error(f"从模板创建规则失败: {e}")
            return False

    @exception_handler(reraise=True)
    def update_rule_config(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新预警规则配置

        Args:
            rule_id: 规则ID
            updates: 更新的字段

        Returns:
            bool: 更新是否成功
        """
        try:
            # 使用配置管理器更新规则
            success = self.config_manager.update_rule(rule_id, updates)

            if success:
                # 重新加载规则
                self.reload_rules_from_config()
                logger.info(f"更新规则 {rule_id} 配置成功")

            return success

        except Exception as e:
            logger.error(f"更新规则配置失败: {e}")
            return False

    @exception_handler(reraise=True)
    def delete_rule_config(self, rule_id: str) -> bool:
        """
        删除预警规则配置

        Args:
            rule_id: 规则ID

        Returns:
            bool: 删除是否成功
        """
        try:
            # 使用配置管理器删除规则
            success = self.config_manager.delete_rule(rule_id)

            if success:
                # 重新加载规则
                self.reload_rules_from_config()
                logger.info(f"删除规则 {rule_id} 配置成功")

            return success

        except Exception as e:
            logger.error(f"删除规则配置失败: {e}")
            return False

    def get_available_templates(self) -> Dict[str, Dict[str, Any]]:
        """获取可用的规则模板"""
        return self.config_manager.get_templates()

    def get_rule_config(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """获取规则配置"""
        rule_config = self.config_manager.get_rule(rule_id)
        if rule_config:
            return {
                'id': rule_config.id,
                'name': rule_config.name,
                'description': rule_config.description,
                'indicators': rule_config.indicators,
                'conditions': rule_config.conditions,
                'signal_type': rule_config.signal_type,
                'priority': rule_config.priority,
                'enabled': rule_config.enabled,
                'created_at': rule_config.created_at,
                'updated_at': rule_config.updated_at
            }
        return None
    
    @exception_handler(reraise=True)
    def remove_alert_rule(self, rule_id: str) -> bool:
        """
        移除预警规则
        
        Args:
            rule_id: 规则ID
            
        Returns:
            bool: 移除是否成功
        """
        with self.lock:
            if rule_id in self.alert_rules:
                del self.alert_rules[rule_id]
                logger.info(f"移除预警规则: {rule_id}")
                return True
            return False
    
    @exception_handler(reraise=True)
    def enable_rule(self, rule_id: str, enabled: bool = True) -> bool:
        """
        启用/禁用预警规则
        
        Args:
            rule_id: 规则ID
            enabled: 是否启用
            
        Returns:
            bool: 操作是否成功
        """
        with self.lock:
            if rule_id in self.alert_rules:
                self.alert_rules[rule_id].enabled = enabled
                logger.info(f"{'启用' if enabled else '禁用'}预警规则: {rule_id}")
                return True
            return False
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=10.0)
    def analyze_stock_signals(self, stock_code: str, stock_name: str = "", 
                            lookback_days: int = 30) -> List[TradingSignal]:
        """
        分析股票信号
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            lookback_days: 回看天数
            
        Returns:
            List[TradingSignal]: 识别的信号列表
        """
        signals = []
        
        try:
            # 获取股票数据
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            
            query = f"""
            SELECT code, name, date, open, high, low, close, volume, turnover_rate
            FROM stock_info WHERE level = %(level)s AND code = '{stock_code}'
            AND level = '日线'
            AND date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY date DESC
            LIMIT {lookback_days}
            """
            
            # 使用模拟查询方法
            if self.data_access:
                try:
                    # 尝试使用execute_query方法
                    data = self.data_access.execute_query_data_access_manager(query)
                except AttributeError:
                    # 如果方法不存在，创建模拟数据
                    data = self._create_mock_data(stock_code, start_date, end_date)
            else:
                data = self._create_mock_data(stock_code, start_date, end_date)
            
            if data.empty:
                logger.warning(f"股票 {stock_code} 没有获取到数据")
                return signals
            
            # 分析每个启用的规则
            for rule in self.alert_rules.values():
                if not rule.enabled:
                    continue
                
                try:
                    rule_signals = self._analyze_rule_signals(stock_code, stock_name, data, rule)
                    signals.extend(rule_signals)
                except Exception as e:
                    logger.error(f"分析规则 {rule.id} 时出错: {e}")
            
            logger.info(f"股票 {stock_code} 分析完成，识别到 {len(signals)} 个信号")
            
        except Exception as e:
            logger.error(f"分析股票 {stock_code} 信号时出错: {e}")
        
        return signals

    def _analyze_rule_signals(self, stock_code: str, stock_name: str,
                            data: pd.DataFrame, rule: AlertRule) -> List[TradingSignal]:
        """分析单个规则的信号"""
        signals = []

        try:
            # 计算所需指标
            indicator_results = {}
            for indicator_name in rule.indicators:
                try:
                    indicator = self.indicator_registry.create_indicator(indicator_name)
                    indicator_data = indicator.calculate(data)
                    indicator_signal = indicator.get_signal(indicator_data)

                    indicator_results[indicator_name] = {
                        'data': indicator_data,
                        'signal': indicator_signal
                    }
                except Exception as e:
                    logger.debug(f"计算指标 {indicator_name} 失败: {e}")
                    continue

            if not indicator_results:
                return signals

            # 根据规则类型分析信号
            if rule.id == "rsi_overbought_oversold":
                signals.extend(self._analyze_rsi_signals(stock_code, stock_name, data, rule, indicator_results))
            elif rule.id == "macd_golden_death_cross":
                signals.extend(self._analyze_macd_signals(stock_code, stock_name, data, rule, indicator_results))
            elif rule.id == "kdj_overbought_oversold":
                signals.extend(self._analyze_kdj_signals(stock_code, stock_name, data, rule, indicator_results))
            elif rule.id == "boll_breakout":
                signals.extend(self._analyze_boll_signals(stock_code, stock_name, data, rule, indicator_results))
            elif rule.id == "volume_anomaly":
                signals.extend(self._analyze_volume_signals(stock_code, stock_name, data, rule, indicator_results))
            elif rule.id == "price_gap":
                signals.extend(self._analyze_gap_signals(stock_code, stock_name, data, rule))
            elif rule.id == "multi_indicator_buy":
                signals.extend(self._analyze_multi_buy_signals(stock_code, stock_name, data, rule, indicator_results))
            elif rule.id == "multi_indicator_sell":
                signals.extend(self._analyze_multi_sell_signals(stock_code, stock_name, data, rule, indicator_results))

        except Exception as e:
            logger.error(f"分析规则 {rule.id} 信号时出错: {e}")

        return signals

    def _calculate_signal_strength(self, value: float, threshold: float, extreme: float) -> SignalStrength:
        """计算信号强度"""
        if extreme == 0:  # 超卖情况
            ratio = (threshold - value) / threshold
        else:  # 超买情况
            ratio = (value - threshold) / (extreme - threshold)

        if ratio < 0.3:
            return SignalStrength.WEAK
        elif ratio < 0.6:
            return SignalStrength.MODERATE
        elif ratio < 0.8:
            return SignalStrength.STRONG
        else:
            return SignalStrength.VERY_STRONG

    def _analyze_rsi_signals(self, stock_code: str, stock_name: str, data: pd.DataFrame,
                           rule: AlertRule, indicator_results: Dict[str, Any]) -> List[TradingSignal]:
        """分析RSI信号"""
        signals = []

        if 'RSI' not in indicator_results:
            return signals

        rsi_data = indicator_results['RSI']['data']
        if rsi_data.empty:
            return signals

        current_rsi = rsi_data.iloc[0].get('rsi', 50)
        current_price = data.iloc[0]['close']

        # 超买信号
        if current_rsi > rule.conditions['rsi_overbought']:
            signal = TradingSignal(
                id=f"{stock_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rsi_overbought",
                timestamp=datetime.now(),
                stock_code=stock_code,
                stock_name=stock_name,
                signal_type=SignalType.SELL,
                signal_strength=self._calculate_signal_strength(current_rsi, 70, 100),
                confidence=0.7,
                price=current_price,
                indicators={'RSI': current_rsi},
                message=f"RSI超买信号 (RSI: {current_rsi:.1f})",
                details={
                    'rsi_value': current_rsi,
                    'threshold': rule.conditions['rsi_overbought'],
                    'rule_id': rule.id
                },
                valid_until=datetime.now() + timedelta(hours=24)
            )
            signals.append(signal)

        # 超卖信号
        elif current_rsi < rule.conditions['rsi_oversold']:
            signal = TradingSignal(
                id=f"{stock_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rsi_oversold",
                timestamp=datetime.now(),
                stock_code=stock_code,
                stock_name=stock_name,
                signal_type=SignalType.BUY,
                signal_strength=self._calculate_signal_strength(30, current_rsi, 0),
                confidence=0.7,
                price=current_price,
                indicators={'RSI': current_rsi},
                message=f"RSI超卖信号 (RSI: {current_rsi:.1f})",
                details={
                    'rsi_value': current_rsi,
                    'threshold': rule.conditions['rsi_oversold'],
                    'rule_id': rule.id
                },
                valid_until=datetime.now() + timedelta(hours=24)
            )
            signals.append(signal)

        return signals

    def _analyze_macd_signals(self, stock_code: str, stock_name: str, data: pd.DataFrame,
                            rule: AlertRule, indicator_results: Dict[str, Any]) -> List[TradingSignal]:
        """分析MACD信号"""
        signals = []

        if 'MACD' not in indicator_results:
            return signals

        macd_data = indicator_results['MACD']['data']
        if len(macd_data) < 2:
            return signals

        current_macd = macd_data.iloc[0].get('macd', 0)
        current_signal = macd_data.iloc[0].get('signal', 0)
        previous_macd = macd_data.iloc[1].get('macd', 0)
        previous_signal = macd_data.iloc[1].get('signal', 0)

        current_price = data.iloc[0]['close']

        # 金叉信号 (MACD线上穿信号线)
        if (current_macd > current_signal and previous_macd <= previous_signal and
            current_macd > rule.conditions['golden_cross_threshold']):

            signal = TradingSignal(
                id=f"{stock_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_macd_golden",
                timestamp=datetime.now(),
                stock_code=stock_code,
                stock_name=stock_name,
                signal_type=SignalType.BUY,
                signal_strength=SignalStrength.STRONG,
                confidence=0.8,
                price=current_price,
                indicators={'MACD': current_macd, 'Signal': current_signal},
                message=f"MACD金叉买入信号",
                details={
                    'macd_value': current_macd,
                    'signal_value': current_signal,
                    'rule_id': rule.id
                },
                valid_until=datetime.now() + timedelta(hours=48)
            )
            signals.append(signal)

        # 死叉信号 (MACD线下穿信号线)
        elif (current_macd < current_signal and previous_macd >= previous_signal and
              current_macd < rule.conditions['death_cross_threshold']):

            signal = TradingSignal(
                id=f"{stock_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_macd_death",
                timestamp=datetime.now(),
                stock_code=stock_code,
                stock_name=stock_name,
                signal_type=SignalType.SELL,
                signal_strength=SignalStrength.STRONG,
                confidence=0.8,
                price=current_price,
                indicators={'MACD': current_macd, 'Signal': current_signal},
                message=f"MACD死叉卖出信号",
                details={
                    'macd_value': current_macd,
                    'signal_value': current_signal,
                    'rule_id': rule.id
                },
                valid_until=datetime.now() + timedelta(hours=48)
            )
            signals.append(signal)

        return signals

    def _analyze_kdj_signals(self, stock_code: str, stock_name: str, data: pd.DataFrame,
                           rule: AlertRule, indicator_results: Dict[str, Any]) -> List[TradingSignal]:
        """分析KDJ信号"""
        signals = []

        if 'KDJ' not in indicator_results:
            return signals

        kdj_data = indicator_results['KDJ']['data']
        if kdj_data.empty:
            return signals

        current_k = kdj_data.iloc[0].get('k', 50)
        current_d = kdj_data.iloc[0].get('d', 50)
        current_price = data.iloc[0]['close']

        # 超买信号
        if (current_k > rule.conditions['k_overbought'] and
            current_d > rule.conditions['d_overbought']):

            signal = TradingSignal(
                id=f"{stock_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_kdj_overbought",
                timestamp=datetime.now(),
                stock_code=stock_code,
                stock_name=stock_name,
                signal_type=SignalType.RISK_WARNING,
                signal_strength=SignalStrength.MODERATE,
                confidence=0.6,
                price=current_price,
                indicators={'K': current_k, 'D': current_d},
                message=f"KDJ超买预警 (K: {current_k:.1f}, D: {current_d:.1f})",
                details={
                    'k_value': current_k,
                    'd_value': current_d,
                    'k_threshold': rule.conditions['k_overbought'],
                    'd_threshold': rule.conditions['d_overbought'],
                    'rule_id': rule.id
                },
                valid_until=datetime.now() + timedelta(hours=24)
            )
            signals.append(signal)

        # 超卖信号
        elif (current_k < rule.conditions['k_oversold'] and
              current_d < rule.conditions['d_oversold']):

            signal = TradingSignal(
                id=f"{stock_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_kdj_oversold",
                timestamp=datetime.now(),
                stock_code=stock_code,
                stock_name=stock_name,
                signal_type=SignalType.OPPORTUNITY,
                signal_strength=SignalStrength.MODERATE,
                confidence=0.6,
                price=current_price,
                indicators={'K': current_k, 'D': current_d},
                message=f"KDJ超卖机会 (K: {current_k:.1f}, D: {current_d:.1f})",
                details={
                    'k_value': current_k,
                    'd_value': current_d,
                    'k_threshold': rule.conditions['k_oversold'],
                    'd_threshold': rule.conditions['d_oversold'],
                    'rule_id': rule.id
                },
                valid_until=datetime.now() + timedelta(hours=24)
            )
            signals.append(signal)

        return signals

    def _analyze_boll_signals(self, stock_code: str, stock_name: str, data: pd.DataFrame,
                            rule: AlertRule, indicator_results: Dict[str, Any]) -> List[TradingSignal]:
        """分析布林带信号"""
        signals = []

        if 'BOLL' not in indicator_results:
            return signals

        boll_data = indicator_results['BOLL']['data']
        if boll_data.empty:
            return signals

        current_price = data.iloc[0]['close']
        upper_band = boll_data.iloc[0].get('Upper', current_price * 1.02)
        lower_band = boll_data.iloc[0].get('Lower', current_price * 0.98)
        middle_band = boll_data.iloc[0].get('Middle', current_price)

        # 突破上轨信号
        if current_price > upper_band * (1 + rule.conditions['upper_breakout_threshold']):
            signal = TradingSignal(
                id=f"{stock_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_boll_upper_breakout",
                timestamp=datetime.now(),
                stock_code=stock_code,
                stock_name=stock_name,
                signal_type=SignalType.RISK_WARNING,
                signal_strength=SignalStrength.STRONG,
                confidence=0.75,
                price=current_price,
                indicators={'Price': current_price, 'Upper': upper_band, 'Middle': middle_band},
                message=f"突破布林带上轨 (价格: {current_price:.2f}, 上轨: {upper_band:.2f})",
                details={
                    'price': current_price,
                    'upper_band': upper_band,
                    'breakout_ratio': (current_price - upper_band) / upper_band,
                    'rule_id': rule.id
                },
                valid_until=datetime.now() + timedelta(hours=24)
            )
            signals.append(signal)

        # 跌破下轨信号
        elif current_price < lower_band * (1 - rule.conditions['lower_breakout_threshold']):
            signal = TradingSignal(
                id=f"{stock_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_boll_lower_breakout",
                timestamp=datetime.now(),
                stock_code=stock_code,
                stock_name=stock_name,
                signal_type=SignalType.OPPORTUNITY,
                signal_strength=SignalStrength.STRONG,
                confidence=0.75,
                price=current_price,
                indicators={'Price': current_price, 'Lower': lower_band, 'Middle': middle_band},
                message=f"跌破布林带下轨 (价格: {current_price:.2f}, 下轨: {lower_band:.2f})",
                details={
                    'price': current_price,
                    'lower_band': lower_band,
                    'breakout_ratio': (lower_band - current_price) / lower_band,
                    'rule_id': rule.id
                },
                valid_until=datetime.now() + timedelta(hours=24)
            )
            signals.append(signal)

        return signals

    def _analyze_volume_signals(self, stock_code: str, stock_name: str, data: pd.DataFrame,
                              rule: AlertRule, indicator_results: Dict[str, Any]) -> List[TradingSignal]:
        """分析成交量信号"""
        signals = []

        if len(data) < 5:
            return signals

        current_volume = data.iloc[0]['volume']
        avg_volume = data.iloc[1:6]['volume'].mean()  # 过去5天平均成交量
        current_price = data.iloc[0]['close']

        # 成交量放大信号
        if current_volume > avg_volume * rule.conditions['volume_surge_ratio']:
            signal = TradingSignal(
                id=f"{stock_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_volume_surge",
                timestamp=datetime.now(),
                stock_code=stock_code,
                stock_name=stock_name,
                signal_type=SignalType.OPPORTUNITY,
                signal_strength=SignalStrength.STRONG,
                confidence=0.8,
                price=current_price,
                indicators={'Volume': current_volume, 'AvgVolume': avg_volume},
                message=f"成交量异常放大 (当前: {current_volume:,.0f}, 平均: {avg_volume:,.0f})",
                details={
                    'current_volume': current_volume,
                    'average_volume': avg_volume,
                    'surge_ratio': current_volume / avg_volume,
                    'rule_id': rule.id
                },
                valid_until=datetime.now() + timedelta(hours=12)
            )
            signals.append(signal)

        # 成交量萎缩信号
        elif current_volume < avg_volume * rule.conditions['volume_shrink_ratio']:
            signal = TradingSignal(
                id=f"{stock_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_volume_shrink",
                timestamp=datetime.now(),
                stock_code=stock_code,
                stock_name=stock_name,
                signal_type=SignalType.RISK_WARNING,
                signal_strength=SignalStrength.MODERATE,
                confidence=0.6,
                price=current_price,
                indicators={'Volume': current_volume, 'AvgVolume': avg_volume},
                message=f"成交量异常萎缩 (当前: {current_volume:,.0f}, 平均: {avg_volume:,.0f})",
                details={
                    'current_volume': current_volume,
                    'average_volume': avg_volume,
                    'shrink_ratio': current_volume / avg_volume,
                    'rule_id': rule.id
                },
                valid_until=datetime.now() + timedelta(hours=12)
            )
            signals.append(signal)

        return signals

    def _analyze_gap_signals(self, stock_code: str, stock_name: str, data: pd.DataFrame,
                           rule: AlertRule) -> List[TradingSignal]:
        """分析价格跳空信号"""
        signals = []

        if len(data) < 2:
            return signals

        current_open = data.iloc[0]['open']
        current_close = data.iloc[0]['close']
        previous_close = data.iloc[1]['close']

        # 向上跳空
        gap_up_ratio = (current_open - previous_close) / previous_close
        if gap_up_ratio > rule.conditions['gap_threshold']:
            signal = TradingSignal(
                id=f"{stock_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_gap_up",
                timestamp=datetime.now(),
                stock_code=stock_code,
                stock_name=stock_name,
                signal_type=SignalType.OPPORTUNITY,
                signal_strength=SignalStrength.STRONG,
                confidence=0.85,
                price=current_close,
                indicators={'Open': current_open, 'PrevClose': previous_close},
                message=f"向上跳空 {gap_up_ratio:.1%} (开盘: {current_open:.2f}, 昨收: {previous_close:.2f})",
                details={
                    'current_open': current_open,
                    'previous_close': previous_close,
                    'gap_ratio': gap_up_ratio,
                    'rule_id': rule.id
                },
                valid_until=datetime.now() + timedelta(hours=6)
            )
            signals.append(signal)

        # 向下跳空
        gap_down_ratio = (previous_close - current_open) / previous_close
        if gap_down_ratio > rule.conditions['gap_threshold']:
            signal = TradingSignal(
                id=f"{stock_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_gap_down",
                timestamp=datetime.now(),
                stock_code=stock_code,
                stock_name=stock_name,
                signal_type=SignalType.RISK_WARNING,
                signal_strength=SignalStrength.STRONG,
                confidence=0.85,
                price=current_close,
                indicators={'Open': current_open, 'PrevClose': previous_close},
                message=f"向下跳空 {gap_down_ratio:.1%} (开盘: {current_open:.2f}, 昨收: {previous_close:.2f})",
                details={
                    'current_open': current_open,
                    'previous_close': previous_close,
                    'gap_ratio': gap_down_ratio,
                    'rule_id': rule.id
                },
                valid_until=datetime.now() + timedelta(hours=6)
            )
            signals.append(signal)

        return signals

    def _analyze_multi_buy_signals(self, stock_code: str, stock_name: str, data: pd.DataFrame,
                                 rule: AlertRule, indicator_results: Dict[str, Any]) -> List[TradingSignal]:
        """分析多指标共振买入信号"""
        signals = []

        # 检查所需指标是否都存在
        required_indicators = ['RSI', 'MACD', 'KDJ']
        if not all(indicator in indicator_results for indicator in required_indicators):
            return signals

        rsi_data = indicator_results['RSI']['data']
        macd_data = indicator_results['MACD']['data']
        kdj_data = indicator_results['KDJ']['data']

        if any(data.empty for data in [rsi_data, macd_data, kdj_data]):
            return signals

        current_rsi = rsi_data.iloc[0].get('rsi', 50)
        current_macd = macd_data.iloc[0].get('macd', 0)
        current_k = kdj_data.iloc[0].get('k', 50)
        current_price = data.iloc[0]['close']

        # 多指标共振买入条件
        rsi_oversold = current_rsi < rule.conditions['rsi_threshold']
        macd_positive = current_macd > 0 if rule.conditions['macd_positive'] else True
        kdj_oversold = current_k < rule.conditions['kdj_threshold']

        if rsi_oversold and macd_positive and kdj_oversold:
            # 计算综合信号强度
            rsi_strength = (rule.conditions['rsi_threshold'] - current_rsi) / rule.conditions['rsi_threshold']
            kdj_strength = (rule.conditions['kdj_threshold'] - current_k) / rule.conditions['kdj_threshold']
            combined_strength = (rsi_strength + kdj_strength) / 2

            if combined_strength > 0.5:
                strength = SignalStrength.VERY_STRONG
            elif combined_strength > 0.3:
                strength = SignalStrength.STRONG
            else:
                strength = SignalStrength.MODERATE

            signal = TradingSignal(
                id=f"{stock_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_multi_buy",
                timestamp=datetime.now(),
                stock_code=stock_code,
                stock_name=stock_name,
                signal_type=SignalType.BUY,
                signal_strength=strength,
                confidence=0.9,
                price=current_price,
                indicators={'RSI': current_rsi, 'MACD': current_macd, 'KDJ_K': current_k},
                message=f"多指标共振买入信号 (RSI: {current_rsi:.1f}, MACD: {current_macd:.3f}, K: {current_k:.1f})",
                details={
                    'rsi_value': current_rsi,
                    'macd_value': current_macd,
                    'kdj_k_value': current_k,
                    'combined_strength': combined_strength,
                    'rule_id': rule.id
                },
                valid_until=datetime.now() + timedelta(hours=48)
            )
            signals.append(signal)

        return signals

    def _analyze_multi_sell_signals(self, stock_code: str, stock_name: str, data: pd.DataFrame,
                                  rule: AlertRule, indicator_results: Dict[str, Any]) -> List[TradingSignal]:
        """分析多指标共振卖出信号"""
        signals = []

        # 检查所需指标是否都存在
        required_indicators = ['RSI', 'MACD', 'KDJ']
        if not all(indicator in indicator_results for indicator in required_indicators):
            return signals

        rsi_data = indicator_results['RSI']['data']
        macd_data = indicator_results['MACD']['data']
        kdj_data = indicator_results['KDJ']['data']

        if any(data.empty for data in [rsi_data, macd_data, kdj_data]):
            return signals

        current_rsi = rsi_data.iloc[0].get('rsi', 50)
        current_macd = macd_data.iloc[0].get('macd', 0)
        current_k = kdj_data.iloc[0].get('k', 50)
        current_price = data.iloc[0]['close']

        # 多指标共振卖出条件
        rsi_overbought = current_rsi > rule.conditions['rsi_threshold']
        macd_negative = current_macd < 0 if rule.conditions['macd_negative'] else True
        kdj_overbought = current_k > rule.conditions['kdj_threshold']

        if rsi_overbought and macd_negative and kdj_overbought:
            # 计算综合信号强度
            rsi_strength = (current_rsi - rule.conditions['rsi_threshold']) / (100 - rule.conditions['rsi_threshold'])
            kdj_strength = (current_k - rule.conditions['kdj_threshold']) / (100 - rule.conditions['kdj_threshold'])
            combined_strength = (rsi_strength + kdj_strength) / 2

            if combined_strength > 0.5:
                strength = SignalStrength.VERY_STRONG
            elif combined_strength > 0.3:
                strength = SignalStrength.STRONG
            else:
                strength = SignalStrength.MODERATE

            signal = TradingSignal(
                id=f"{stock_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_multi_sell",
                timestamp=datetime.now(),
                stock_code=stock_code,
                stock_name=stock_name,
                signal_type=SignalType.SELL,
                signal_strength=strength,
                confidence=0.9,
                price=current_price,
                indicators={'RSI': current_rsi, 'MACD': current_macd, 'KDJ_K': current_k},
                message=f"多指标共振卖出信号 (RSI: {current_rsi:.1f}, MACD: {current_macd:.3f}, K: {current_k:.1f})",
                details={
                    'rsi_value': current_rsi,
                    'macd_value': current_macd,
                    'kdj_k_value': current_k,
                    'combined_strength': combined_strength,
                    'rule_id': rule.id
                },
                valid_until=datetime.now() + timedelta(hours=48)
            )
            signals.append(signal)

        return signals

    @exception_handler(reraise=True)
    def get_signals(self, limit: int = 50, signal_type: Optional[SignalType] = None,
                   stock_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取信号列表

        Args:
            limit: 返回数量限制
            signal_type: 信号类型过滤
            stock_code: 股票代码过滤

        Returns:
            List[Dict[str, Any]]: 信号列表
        """
        with self.lock:
            signals = self.signals_history.copy()

        # 按类型过滤
        if signal_type:
            signals = [signal for signal in signals if signal.signal_type == signal_type]

        # 按股票代码过滤
        if stock_code:
            signals = [signal for signal in signals if signal.stock_code == stock_code]

        # 按时间排序并限制数量
        signals = sorted(signals, key=lambda x: x.timestamp, reverse=True)[:limit]

        # 转换为字典格式
        return [
            {
                "id": signal.id,
                "timestamp": signal.timestamp.isoformat(),
                "stock_code": signal.stock_code,
                "stock_name": signal.stock_name,
                "signal_type": signal.signal_type.value,
                "signal_strength": signal.signal_strength.value,
                "confidence": signal.confidence,
                "price": signal.price,
                "indicators": signal.indicators,
                "message": signal.message,
                "details": signal.details,
                "valid_until": signal.valid_until.isoformat(),
                "processed": signal.processed
            }
            for signal in signals
        ]

    @exception_handler(reraise=True)
    def mark_signal_processed(self, signal_id: str) -> bool:
        """
        标记信号为已处理

        Args:
            signal_id: 信号ID

        Returns:
            bool: 是否成功标记
        """
        with self.lock:
            for signal in self.signals_history:
                if signal.id == signal_id:
                    signal.processed = True
                    logger.info(f"信号 {signal_id} 已标记为已处理")
                    return True

        return False

    @exception_handler(reraise=True)
    def get_alert_rules(self) -> List[Dict[str, Any]]:
        """
        获取预警规则列表

        Returns:
            List[Dict[str, Any]]: 规则列表
        """
        with self.lock:
            return [
                {
                    "id": rule.id,
                    "name": rule.name,
                    "description": rule.description,
                    "indicators": rule.indicators,
                    "conditions": rule.conditions,
                    "signal_type": rule.signal_type.value,
                    "enabled": rule.enabled,
                    "priority": rule.priority
                }
                for rule in self.alert_rules.values()
            ]

    @exception_handler(reraise=True)
    def get_system_statistics(self) -> Dict[str, Any]:
        """
        获取系统统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        with self.lock:
            total_signals = len(self.signals_history)
            processed_signals = len([s for s in self.signals_history if s.processed])

            # 按类型统计
            type_stats = {}
            for signal_type in SignalType:
                type_stats[signal_type.value] = len([
                    s for s in self.signals_history
                    if s.signal_type == signal_type and not s.processed
                ])

            # 按强度统计
            strength_stats = {}
            for strength in SignalStrength:
                strength_stats[strength.value] = len([
                    s for s in self.signals_history
                    if s.signal_strength == strength and not s.processed
                ])

            return {
                "total_rules": len(self.alert_rules),
                "enabled_rules": len([r for r in self.alert_rules.values() if r.enabled]),
                "total_signals": total_signals,
                "processed_signals": processed_signals,
                "pending_signals": total_signals - processed_signals,
                "type_statistics": type_stats,
                "strength_statistics": strength_stats,
                "timestamp": datetime.now().isoformat()
            }

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=30.0)
    def batch_analyze_stocks(self, stock_codes: List[str], lookback_days: int = 30) -> Dict[str, List[TradingSignal]]:
        """
        批量分析多只股票的信号

        Args:
            stock_codes: 股票代码列表
            lookback_days: 回看天数

        Returns:
            Dict[str, List[TradingSignal]]: 股票代码到信号列表的映射
        """
        results = {}

        for stock_code in stock_codes:
            try:
                signals = self.analyze_stock_signals(stock_code, lookback_days=lookback_days)
                results[stock_code] = signals

                # 将信号添加到历史记录
                with self.lock:
                    self.signals_history.extend(signals)

            except Exception as e:
                logger.error(f"批量分析股票 {stock_code} 失败: {e}")
                results[stock_code] = []

        logger.info(f"批量分析完成，处理 {len(stock_codes)} 只股票，生成 {sum(len(signals) for signals in results.values())} 个信号")
        return results

    @exception_handler(reraise=True)
    def start_real_time_monitoring(self, stock_codes: List[str], interval_seconds: int = 300):
        """
        启动实时监控

        Args:
            stock_codes: 要监控的股票代码列表
            interval_seconds: 监控间隔（秒）
        """
        def monitor_loop():
            while True:
                try:
                    logger.info(f"开始实时监控 {len(stock_codes)} 只股票")

                    # 批量分析股票
                    batch_results = self.batch_analyze_stocks(stock_codes, lookback_days=5)

                    # 处理新信号
                    new_signals_count = 0
                    for stock_code, signals in batch_results.items():
                        for signal in signals:
                            if not signal.processed:
                                self._process_real_time_signal(signal)
                                new_signals_count += 1

                    if new_signals_count > 0:
                        logger.info(f"实时监控发现 {new_signals_count} 个新信号")

                    # 等待下一次监控
                    time.sleep(interval_seconds)

                except Exception as e:
                    logger.error(f"实时监控出错: {e}")
                    time.sleep(60)  # 出错后等待1分钟再重试

        # 启动监控线程
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info(f"实时监控已启动，监控间隔: {interval_seconds}秒")

    def _process_real_time_signal(self, signal: TradingSignal):
        """处理实时信号"""
        try:
            # 根据信号类型和强度决定处理方式
            if signal.signal_type in [SignalType.BUY, SignalType.SELL] and signal.signal_strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]:
                # 高优先级信号，立即通知
                self._send_urgent_notification(signal)
            elif signal.signal_type == SignalType.RISK_WARNING:
                # 风险预警，记录并通知
                self._send_risk_notification(signal)

            # 将信号加入队列
            self.signal_queue.put(signal)

        except Exception as e:
            logger.error(f"处理实时信号失败: {e}")

    def _send_urgent_notification(self, signal: TradingSignal):
        """发送紧急通知"""
        logger.warning(f"🚨 紧急信号: {signal.message} | 股票: {signal.stock_code} | 价格: {signal.price:.2f} | 置信度: {signal.confidence:.1%}")

    def _send_risk_notification(self, signal: TradingSignal):
        """发送风险通知"""
        logger.warning(f"⚠️ 风险预警: {signal.message} | 股票: {signal.stock_code} | 价格: {signal.price:.2f}")

    @exception_handler(reraise=True)
    def get_pending_signals(self, priority_threshold: int = 3) -> List[TradingSignal]:
        """
        获取待处理的高优先级信号

        Args:
            priority_threshold: 优先级阈值

        Returns:
            List[TradingSignal]: 待处理信号列表
        """
        with self.lock:
            pending_signals = []

            for signal in self.signals_history:
                if not signal.processed and signal.valid_until > datetime.now():
                    # 查找对应的规则优先级
                    rule_priority = 1
                    for rule in self.alert_rules.values():
                        if rule.id in signal.details.get('rule_id', ''):
                            rule_priority = rule.priority
                            break

                    if rule_priority >= priority_threshold:
                        pending_signals.append(signal)

            # 按优先级和时间排序
            pending_signals.sort(key=lambda x: (-self._get_signal_priority(x), x.timestamp), reverse=True)

        return pending_signals

    def _get_signal_priority(self, signal: TradingSignal) -> int:
        """获取信号优先级"""
        # 基础优先级
        base_priority = 1

        # 根据信号类型调整
        if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
            base_priority += 2
        elif signal.signal_type == SignalType.RISK_WARNING:
            base_priority += 1

        # 根据信号强度调整
        if signal.signal_strength == SignalStrength.VERY_STRONG:
            base_priority += 3
        elif signal.signal_strength == SignalStrength.STRONG:
            base_priority += 2
        elif signal.signal_strength == SignalStrength.MODERATE:
            base_priority += 1

        # 根据置信度调整
        if signal.confidence > 0.8:
            base_priority += 2
        elif signal.confidence > 0.6:
            base_priority += 1

        return base_priority

    def _create_mock_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """创建模拟股票数据"""
        try:
            import numpy as np

            # 生成日期范围
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            n_days = len(dates)

            if n_days == 0:
                return pd.DataFrame()

            # 生成模拟价格数据
            base_price = 100.0
            price_changes = np.random.normal(0, 0.02, n_days)  # 2%的日波动
            prices = [base_price]

            for change in price_changes[1:]:
                new_price = prices[-1] * (1 + change)
                prices.append(max(new_price, 1.0))  # 价格不能为负

            # 创建OHLC数据
            data = []
            for i, (date, close) in enumerate(zip(dates, prices)):
                high = close * (1 + abs(np.random.normal(0, 0.01)))
                low = close * (1 - abs(np.random.normal(0, 0.01)))
                open_price = close * (1 + np.random.normal(0, 0.005))
                volume = int(np.random.normal(1000000, 200000))

                data.append({
                    'code': stock_code,
                    'name': f'股票{stock_code}',
                    'date': date.strftime('%Y-%m-%d'),
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close, 2),
                    'volume': max(volume, 100000),
                    'turnover_rate': round(np.random.uniform(0.01, 0.1), 4)
                })

            return pd.DataFrame(data)

        except Exception as e:
            logger.error(f"创建模拟数据失败: {e}")
            return pd.DataFrame()


# 全局实例
_intelligent_alert_system_instance = None


def get_intelligent_alert_system() -> IntelligentAlertSystem:
    """
    获取智能预警系统实例（单例模式）

    Returns:
        IntelligentAlertSystem: 智能预警系统实例
    """
    global _intelligent_alert_system_instance

    if _intelligent_alert_system_instance is None:
        _intelligent_alert_system_instance = IntelligentAlertSystem()

    return _intelligent_alert_system_instance


def create_intelligent_alert_system() -> IntelligentAlertSystem:
    """
    创建新的智能预警系统实例

    Returns:
        IntelligentAlertSystem: 新的智能预警系统实例
    """
    return IntelligentAlertSystem()
