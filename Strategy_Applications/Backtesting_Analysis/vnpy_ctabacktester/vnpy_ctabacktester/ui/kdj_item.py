"""
KDJ指标图表项
"""

import pyqtgraph as pg
import numpy as np
from typing import List, Tuple
from vnpy.trader.ui import QtCore, QtGui


class KdjItem:
    """KDJ指标图表项"""

    def __init__(self):
        """构造函数"""

        # KDJ线条
        self.k_line: pg.PlotCurveItem = pg.PlotCurveItem(
            pen=pg.mkPen(color=(255, 255, 0), width=1.5)  # 黄色K线
        )
        self.d_line: pg.PlotCurveItem = pg.PlotCurveItem(
            pen=pg.mkPen(color=(0, 255, 255), width=1.5)  # 青色D线
        )
        self.j_line: pg.PlotCurveItem = pg.PlotCurveItem(
            pen=pg.mkPen(color=(255, 0, 255), width=1.5)  # 紫色J线
        )

        # 超买超卖线
        self.overbought_line: pg.InfiniteLine = pg.InfiniteLine(
            pos=80, angle=0,
            pen=pg.mkPen(color=(255, 0, 0), width=1, style=QtCore.Qt.PenStyle.DashLine)
        )
        self.oversold_line: pg.InfiniteLine = pg.InfiniteLine(
            pos=20, angle=0,
            pen=pg.mkPen(color=(0, 255, 0), width=1, style=QtCore.Qt.PenStyle.DashLine)
        )

        # 中线
        self.middle_line: pg.InfiniteLine = pg.InfiniteLine(
            pos=50, angle=0,
            pen=pg.mkPen(color=(128, 128, 128), width=1, style=QtCore.Qt.PenStyle.DotLine)
        )

        # KDJ数据
        self.k_data: List[float] = []
        self.d_data: List[float] = []
        self.j_data: List[float] = []

    def clear_all(self) -> None:
        """清除所有数据"""
        self.k_data.clear()
        self.d_data.clear()
        self.j_data.clear()

    def update_kdj_data(self, k_data: List[float], d_data: List[float], j_data: List[float]) -> None:
        """更新KDJ数据"""
        self.k_data = k_data.copy()
        self.d_data = d_data.copy()
        self.j_data = j_data.copy()

        # 更新图表线条
        if self.k_data:
            x_data = list(range(len(self.k_data)))
            self.k_line.setData(x_data, self.k_data)
            self.d_line.setData(x_data, self.d_data)
            self.j_line.setData(x_data, self.j_data)

    def add_to_plot(self, plot: pg.PlotItem) -> None:
        """添加到图表"""
        # 添加KDJ线条
        plot.addItem(self.k_line)
        plot.addItem(self.d_line)
        plot.addItem(self.j_line)

        # 添加水平线
        plot.addItem(self.overbought_line)
        plot.addItem(self.oversold_line)
        plot.addItem(self.middle_line)

        # 设置Y轴范围
        plot.setYRange(-10, 110)  # 扩大范围以容纳J值可能超出0-100的情况

        # 设置Y轴标签
        plot.setLabel('left', 'KDJ', color='white', size='8pt')

        # 添加图例
        self._add_legend(plot)

    def _add_legend(self, plot: pg.PlotItem) -> None:
        """添加图例"""
        try:
            legend = plot.addLegend(offset=(10, 10))
            legend.addItem(self.k_line, "K")
            legend.addItem(self.d_line, "D") 
            legend.addItem(self.j_line, "J")
        except Exception:
            # 如果图例添加失败，忽略错误
            pass
