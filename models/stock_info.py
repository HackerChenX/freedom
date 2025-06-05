#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
股票信息数据模型

提供标准化的股票数据结构和相关操作方法
作为数据库层和业务逻辑层之间的桥梁
"""

import datetime
from typing import Dict, Any, Optional, Union, List
import pandas as pd
import numpy as np
from enums.period import Period


class StockInfo:
    """
    股票信息数据模型类
    
    用于封装股票相关数据，提供统一的数据结构和访问接口
    支持单条数据和批量数据的处理
    """
    
    def __init__(self, data: Union[Dict[str, Any], pd.DataFrame] = None):
        """
        初始化股票信息对象
        
        Args:
            data: 股票数据，可以是字典或DataFrame
                  如果是DataFrame，则构造一个包含多条数据的StockInfo集合
        """
        # 定义基本属性
        self._code: str = ""  # 股票代码
        self._name: str = ""  # 股票名称
        self._date: Optional[datetime.date] = None  # 日期
        self._level: str = Period.DAILY.value  # K线周期
        self._open: float = 0.0  # 开盘价
        self._high: float = 0.0  # 最高价
        self._low: float = 0.0  # 最低价
        self._close: float = 0.0  # 收盘价
        self._volume: float = 0.0  # 成交量
        self._turnover_rate: float = 0.0  # 换手率
        self._price_change: float = 0.0  # 价格变动
        self._price_range: float = 0.0  # 价格区间
        self._industry: str = ""  # 行业
        self._datetime: Optional[datetime.datetime] = None  # 日期时间
        self._seq: int = 0  # 序号
        
        # 多条数据支持
        self._is_collection: bool = False  # 是否为数据集合
        self._collection: List['StockInfo'] = []  # 数据集合
        self._data_frame: Optional[pd.DataFrame] = None  # 原始DataFrame
        
        # 初始化数据
        if data is not None:
            self._load_data(data)
    
    def _load_data(self, data: Union[Dict[str, Any], pd.DataFrame]) -> None:
        """
        加载数据到对象
        
        Args:
            data: 股票数据，可以是字典或DataFrame
        """
        if isinstance(data, dict):
            # 加载单条数据
            self._load_from_dict(data)
        elif isinstance(data, pd.DataFrame):
            # 加载多条数据
            self._is_collection = True
            self._data_frame = data
            
            # 将DataFrame转换为StockInfo对象列表
            for _, row in data.iterrows():
                stock_info = StockInfo(row.to_dict())
                self._collection.append(stock_info)
        else:
            raise TypeError(f"不支持的数据类型: {type(data)}")
    
    def _load_from_dict(self, data: Dict[str, Any]) -> None:
        """
        从字典加载数据
        
        Args:
            data: 包含股票数据的字典
        """
        # 映射字段名，处理可能的列名差异
        field_mapping = {
            'code': ['code', 'stock_code', 'symbol'],
            'name': ['name', 'stock_name', 'code_name'],
            'date': ['date', 'trading_date', 'trade_date'],
            'level': ['level', 'period', 'kline_period'],
            'open': ['open', 'open_price'],
            'high': ['high', 'high_price'],
            'low': ['low', 'low_price'],
            'close': ['close', 'close_price'],
            'volume': ['volume', 'vol'],
            'turnover_rate': ['turnover_rate', 'turnover'],
            'price_change': ['price_change', 'change', 'pct_chg'],
            'price_range': ['price_range', 'range'],
            'industry': ['industry', 'sector'],
            'datetime': ['datetime', 'date_time'],
            'seq': ['seq', 'sequence']
        }
        
        # 尝试从不同的可能字段名中提取数据
        for attr, possible_keys in field_mapping.items():
            for key in possible_keys:
                if key in data:
                    value = data[key]
                    
                    # 类型转换
                    if attr == 'date' and value is not None:
                        if isinstance(value, str):
                            try:
                                if '-' in value:
                                    value = datetime.datetime.strptime(value, '%Y-%m-%d').date()
                                else:
                                    value = datetime.datetime.strptime(value, '%Y%m%d').date()
                            except ValueError:
                                pass
                        elif isinstance(value, datetime.datetime):
                            value = value.date()
                    
                    if attr == 'datetime' and value is not None:
                        if isinstance(value, str):
                            try:
                                value = datetime.datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
                            except ValueError:
                                pass
                    
                    # 数值类型转换
                    if attr in ['open', 'high', 'low', 'close', 'volume', 'turnover_rate', 'price_change', 'price_range']:
                        try:
                            value = float(value) if value is not None else 0.0
                        except (ValueError, TypeError):
                            value = 0.0
                    
                    if attr == 'seq':
                        try:
                            value = int(value) if value is not None else 0
                        except (ValueError, TypeError):
                            value = 0
                    
                    # 设置属性
                    setattr(self, f"_{attr}", value)
                    break
    
    @staticmethod
    def get_fields() -> List[str]:
        """获取所有字段"""
        return [
            "code", "name", "date", "level", "open", "high", "low", "close",
            "volume", "turnover_rate", "price_change", "price_range", "industry", "datetime", "seq"
        ]

    @property
    def code(self) -> str:
        """获取股票代码"""
        return self._code
    
    @code.setter
    def code(self, value: str) -> None:
        """设置股票代码"""
        self._code = value
    
    @property
    def name(self) -> str:
        """获取股票名称"""
        return self._name
    
    @name.setter
    def name(self, value: str) -> None:
        """设置股票名称"""
        self._name = value
    
    @property
    def date(self) -> Optional[datetime.date]:
        """获取日期"""
        return self._date
    
    @date.setter
    def date(self, value: Union[str, datetime.date, datetime.datetime]) -> None:
        """设置日期"""
        if isinstance(value, str):
            try:
                if '-' in value:
                    self._date = datetime.datetime.strptime(value, '%Y-%m-%d').date()
                else:
                    self._date = datetime.datetime.strptime(value, '%Y%m%d').date()
            except ValueError:
                self._date = None
        elif isinstance(value, datetime.datetime):
            self._date = value.date()
        elif isinstance(value, datetime.date):
            self._date = value
        else:
            self._date = None
    
    @property
    def level(self) -> str:
        """获取K线周期"""
        return self._level
    
    @level.setter
    def level(self, value: Union[str, Period]) -> None:
        """设置K线周期"""
        if isinstance(value, Period):
            self._level = value.value
        else:
            self._level = value
    
    @property
    def open(self) -> float:
        """获取开盘价"""
        return self._open
    
    @open.setter
    def open(self, value: float) -> None:
        """设置开盘价"""
        try:
            self._open = float(value)
        except (ValueError, TypeError):
            self._open = 0.0
    
    @property
    def high(self) -> float:
        """获取最高价"""
        return self._high
    
    @high.setter
    def high(self, value: float) -> None:
        """设置最高价"""
        try:
            self._high = float(value)
        except (ValueError, TypeError):
            self._high = 0.0
    
    @property
    def low(self) -> float:
        """获取最低价"""
        return self._low
    
    @low.setter
    def low(self, value: float) -> None:
        """设置最低价"""
        try:
            self._low = float(value)
        except (ValueError, TypeError):
            self._low = 0.0
    
    @property
    def close(self) -> float:
        """获取收盘价"""
        return self._close
    
    @close.setter
    def close(self, value: float) -> None:
        """设置收盘价"""
        try:
            self._close = float(value)
        except (ValueError, TypeError):
            self._close = 0.0
    
    @property
    def volume(self) -> float:
        """获取成交量"""
        return self._volume
    
    @volume.setter
    def volume(self, value: float) -> None:
        """设置成交量"""
        try:
            self._volume = float(value)
        except (ValueError, TypeError):
            self._volume = 0.0
    
    @property
    def turnover_rate(self) -> float:
        """获取换手率"""
        return self._turnover_rate
    
    @turnover_rate.setter
    def turnover_rate(self, value: float) -> None:
        """设置换手率"""
        try:
            self._turnover_rate = float(value)
        except (ValueError, TypeError):
            self._turnover_rate = 0.0
    
    @property
    def price_change(self) -> float:
        """获取价格变动"""
        return self._price_change
    
    @price_change.setter
    def price_change(self, value: float) -> None:
        """设置价格变动"""
        try:
            self._price_change = float(value)
        except (ValueError, TypeError):
            self._price_change = 0.0
    
    @property
    def price_range(self) -> float:
        """获取价格区间"""
        return self._price_range
    
    @price_range.setter
    def price_range(self, value: float) -> None:
        """设置价格区间"""
        try:
            self._price_range = float(value)
        except (ValueError, TypeError):
            self._price_range = 0.0
    
    @property
    def industry(self) -> str:
        """获取行业"""
        return self._industry
    
    @industry.setter
    def industry(self, value: str) -> None:
        """设置行业"""
        self._industry = value
    
    @property
    def datetime_value(self) -> Optional[datetime.datetime]:
        """获取日期时间"""
        return self._datetime
    
    @datetime_value.setter
    def datetime_value(self, value: Union[str, datetime.datetime]) -> None:
        """设置日期时间"""
        if isinstance(value, str):
            try:
                self._datetime = datetime.datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                self._datetime = None
        elif isinstance(value, datetime.datetime):
            self._datetime = value
        else:
            self._datetime = None
    
    @property
    def seq(self) -> int:
        """获取序号"""
        return self._seq
    
    @seq.setter
    def seq(self, value: int) -> None:
        """设置序号"""
        try:
            self._seq = int(value)
        except (ValueError, TypeError):
            self._seq = 0
    
    @property
    def is_collection(self) -> bool:
        """是否为数据集合"""
        return self._is_collection
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            Dict[str, Any]: 包含所有属性的字典
        """
        if self._is_collection:
            raise ValueError("集合对象不能直接转换为字典，请使用to_dicts()方法")
        
        return {
            'code': self._code,
            'name': self._name,
            'date': self._date,
            'level': self._level,
            'open': self._open,
            'high': self._high,
            'low': self._low,
            'close': self._close,
            'volume': self._volume,
            'turnover_rate': self._turnover_rate,
            'price_change': self._price_change,
            'price_range': self._price_range,
            'industry': self._industry,
            'datetime': self._datetime,
            'seq': self._seq
        }
    
    def to_dicts(self) -> List[Dict[str, Any]]:
        """
        将集合转换为字典列表
        
        Returns:
            List[Dict[str, Any]]: 字典列表
            
        Raises:
            ValueError: 如果不是集合对象
        """
        if not self._is_collection:
            return [self.to_dict()]
        
        return [item.to_dict() for item in self._collection]
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        转换为DataFrame
        
        Returns:
            pd.DataFrame: 包含所有数据的DataFrame
        """
        if not self._is_collection:
            return pd.DataFrame([self.to_dict()])
        
        if self._data_frame is not None:
            return self._data_frame
        
        return pd.DataFrame(self.to_dicts())
    
    def __getitem__(self, index: int) -> 'StockInfo':
        """
        获取集合中的某一项
        
        Args:
            index: 索引
            
        Returns:
            StockInfo: 指定索引的股票信息对象
            
        Raises:
            ValueError: 如果不是集合对象
            IndexError: 如果索引超出范围
        """
        if not self._is_collection:
            raise ValueError("非集合对象不支持索引访问")
        
        if index < 0 or index >= len(self._collection):
            raise IndexError(f"索引 {index} 超出范围 [0, {len(self._collection) - 1}]")
        
        return self._collection[index]
    
    def __len__(self) -> int:
        """
        获取集合长度
        
        Returns:
            int: 集合长度，如果不是集合则返回1
        """
        if not self._is_collection:
            return 1
        
        return len(self._collection)
    
    def __iter__(self):
        """
        返回迭代器
        
        Returns:
            iterator: 迭代器
        """
        if not self._is_collection:
            return iter([self])
        
        return iter(self._collection)
    
    def __str__(self) -> str:
        """
        返回字符串表示
        
        Returns:
            str: 字符串表示
        """
        if self._is_collection:
            return f"StockInfoCollection(size={len(self._collection)})"
        
        return f"StockInfo(code={self._code}, name={self._name}, date={self._date}, close={self._close})"
    
    def __repr__(self) -> str:
        """
        返回表示
        
        Returns:
            str: 表示
        """
        return self.__str__() 