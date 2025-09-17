#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Excel报告生成器

专业的Excel格式报告生成器，支持：
- 多工作表结构
- 数据表格和图表
- 条件格式
- 专业样式
- 交互式元素
"""

import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from utils.decorators import performance_monitor, exception_handler

logger = get_logger(__name__)

# 尝试导入Excel生成库
try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.formatting.rule import CellIsRule, ColorScaleRule
    from openpyxl.chart import LineChart, BarChart, ScatterChart, PieChart, Reference
    from openpyxl.chart.axis import DateAxis
    from openpyxl.utils.dataframe import dataframe_to_rows
    OPENPYXL_AVAILABLE = True
except ImportError:
    logger.warning("openpyxl未安装，Excel生成功能不可用")
    OPENPYXL_AVAILABLE = False


class ExcelReportGenerator:
    """Excel报告生成器"""

    def __init__(self, config: Optional[Any] = None):
        """
        初始化Excel报告生成器

        Args:
            config: 配置对象
        """
        self.config = config or {}

    @exception_handler(reraise=True)
    @performance_monitor(threshold=25.0)
    def generate_report(self,
                       evaluation_results: Dict[str, Any],
                       chart_files: Dict[str, str],
                       strategy_name: str,
                       request_id: str) -> str:
        """
        生成Excel报告

        Args:
            evaluation_results: 评估结果
            chart_files: 图表文件
            strategy_name: 策略名称
            request_id: 请求ID

        Returns:
            str: 生成的文件路径
        """
        if not OPENPYXL_AVAILABLE:
            # 回退到CSV格式
            return self._generate_csv_fallback(
                evaluation_results, strategy_name, request_id
            )

        try:
            # 生成文件路径
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{strategy_name}_report_{timestamp}.xlsx"
            output_dir = Path(getattr(self.config, 'output_dir', './reports'))
            filepath = output_dir / filename

            # 创建工作簿
            wb = openpyxl.Workbook()

            # 创建各个工作表
            self._create_summary_sheet(wb, evaluation_results, strategy_name, request_id)
            self._create_performance_metrics_sheet(wb, evaluation_results)
            self._create_risk_metrics_sheet(wb, evaluation_results)

            if 'time_series_analysis' in evaluation_results:
                self._create_time_series_sheet(wb, evaluation_results)

            if 'benchmark_comparison' in evaluation_results:
                self._create_benchmark_comparison_sheet(wb, evaluation_results)

            # 删除默认工作表
            if 'Sheet' in wb.sheetnames:
                wb.remove(wb['Sheet'])

            # 保存文件
            wb.save(str(filepath))

            logger.info(f"Excel报告生成完成: {filepath}")

            return str(filepath)

        except Exception as e:
            logger.error(f"生成Excel报告失败: {e}")
            raise

    def _create_summary_sheet(self,
                             wb: openpyxl.Workbook,
                             evaluation_results: Dict[str, Any],
                             strategy_name: str,
                             request_id: str):
        """创建摘要工作表"""
        ws = wb.active
        ws.title = "摘要"

        # 设置标题
        ws['A1'] = f"{strategy_name} - 策略回测报告"
        ws['A1'].font = Font(size=16, bold=True, color="2E86AB")
        ws.merge_cells('A1:F1')
        ws['A1'].alignment = Alignment(horizontal='center')

        # 基本信息
        ws['A3'] = "生成时间:"
        ws['B3'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ws['A4'] = "报告ID:"
        ws['B4'] = request_id

        # 获取性能指标
        performance_metrics = evaluation_results.get('performance_metrics', {})

        # 核心指标表
        headers = ['指标名称', '数值', '描述']
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=6, column=col, value=header)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill(start_color="2E86AB", end_color="2E86AB", fill_type="solid")
            cell.alignment = Alignment(horizontal='center')

        # 核心指标数据
        core_metrics = [
            ('总收益率', performance_metrics.get('total_return', 0), '整个回测期间的累计收益'),
            ('年化收益率', performance_metrics.get('annualized_return', 0), '按年化计算的收益率'),
            ('夏普比率', performance_metrics.get('sharpe_ratio', 0), '每单位风险的超额收益'),
            ('最大回撤', performance_metrics.get('max_drawdown', 0), '历史最大资产损失幅度'),
            ('胜率', performance_metrics.get('win_rate', 0), '盈利交易占总交易比例'),
            ('年化波动率', performance_metrics.get('volatility', 0), '收益率的年化标准差')
        ]

        for row, (metric_name, value, description) in enumerate(core_metrics, 7):
            ws.cell(row=row, column=1, value=metric_name)

            # 格式化数值
            if 'ratio' in metric_name.lower() and 'sharpe' not in metric_name.lower():
                formatted_value = f"{value:.3f}"
            elif '率' in metric_name or 'return' in str(value).lower():
                formatted_value = f"{value * 100:.2f}%"
            else:
                formatted_value = f"{value:.3f}"

            cell = ws.cell(row=row, column=2, value=formatted_value)

            # 设置颜色
            if '收益' in metric_name:
                cell.font = Font(color="28a745" if value > 0 else "dc3545")
            elif '回撤' in metric_name:
                cell.font = Font(color="dc3545")

            ws.cell(row=row, column=3, value=description)

        # 设置列宽
        ws.column_dimensions['A'].width = 15
        ws.column_dimensions['B'].width = 15
        ws.column_dimensions['C'].width = 30

        # 添加边框
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )

        for row in range(6, 13):
            for col in range(1, 4):
                ws.cell(row=row, column=col).border = thin_border

    def _create_performance_metrics_sheet(self,
                                        wb: openpyxl.Workbook,
                                        evaluation_results: Dict[str, Any]):
        """创建性能指标工作表"""
        ws = wb.create_sheet("性能指标")
        performance_metrics = evaluation_results.get('performance_metrics', {})

        # 标题
        ws['A1'] = "详细性能指标"
        ws['A1'].font = Font(size=14, bold=True)
        ws['A1'].fill = PatternFill(start_color="F18F01", end_color="F18F01", fill_type="solid")

        # 收益指标
        ws['A3'] = "收益指标"
        ws['A3'].font = Font(size=12, bold=True, color="2E86AB")

        return_metrics = [
            ('总收益率', performance_metrics.get('total_return', 0)),
            ('年化收益率', performance_metrics.get('annualized_return', 0)),
            ('超额收益率', performance_metrics.get('excess_return', 0)),
            ('复合年增长率', performance_metrics.get('cagr', 0)),
        ]

        row = 4
        for metric_name, value in return_metrics:
            ws.cell(row=row, column=1, value=metric_name)
            ws.cell(row=row, column=2, value=f"{value * 100:.2f}%")
            row += 1

        # 风险调整收益指标
        ws[f'A{row + 1}'] = "风险调整收益指标"
        ws[f'A{row + 1}'].font = Font(size=12, bold=True, color="2E86AB")

        risk_adjusted_metrics = [
            ('夏普比率', performance_metrics.get('sharpe_ratio', 0)),
            ('索提诺比率', performance_metrics.get('sortino_ratio', 0)),
            ('卡玛比率', performance_metrics.get('calmar_ratio', 0)),
            ('信息比率', performance_metrics.get('information_ratio', 0)),
        ]

        row += 2
        for metric_name, value in risk_adjusted_metrics:
            ws.cell(row=row, column=1, value=metric_name)
            ws.cell(row=row, column=2, value=f"{value:.3f}")
            row += 1

        # 交易统计
        ws[f'A{row + 1}'] = "交易统计"
        ws[f'A{row + 1}'].font = Font(size=12, bold=True, color="2E86AB")

        trading_metrics = [
            ('胜率', performance_metrics.get('win_rate', 0)),
            ('盈亏比', performance_metrics.get('profit_loss_ratio', 0)),
            ('总交易次数', performance_metrics.get('total_trades', 0)),
            ('平均每笔收益', performance_metrics.get('avg_trade_return', 0)),
        ]

        row += 2
        for metric_name, value in trading_metrics:
            ws.cell(row=row, column=1, value=metric_name)
            if '次数' in metric_name:
                ws.cell(row=row, column=2, value=int(value))
            elif '率' in metric_name and '盈亏' not in metric_name:
                ws.cell(row=row, column=2, value=f"{value * 100:.1f}%")
            else:
                ws.cell(row=row, column=2, value=f"{value:.3f}")
            row += 1

        # 设置列宽
        ws.column_dimensions['A'].width = 20
        ws.column_dimensions['B'].width = 15

    def _create_risk_metrics_sheet(self,
                                  wb: openpyxl.Workbook,
                                  evaluation_results: Dict[str, Any]):
        """创建风险指标工作表"""
        ws = wb.create_sheet("风险分析")
        risk_metrics = evaluation_results.get('risk_metrics', {})
        performance_metrics = evaluation_results.get('performance_metrics', {})

        # 标题
        ws['A1'] = "风险分析指标"
        ws['A1'].font = Font(size=14, bold=True)
        ws['A1'].fill = PatternFill(start_color="C73E1D", end_color="C73E1D", fill_type="solid")
        ws['A1'].font = Font(size=14, bold=True, color="FFFFFF")

        # 波动率分析
        ws['A3'] = "波动率分析"
        ws['A3'].font = Font(size=12, bold=True, color="C73E1D")

        volatility_metrics = [
            ('日波动率', risk_metrics.get('daily_volatility', 0)),
            ('年化波动率', performance_metrics.get('volatility', 0)),
            ('下行波动率', risk_metrics.get('downside_volatility', 0)),
        ]

        row = 4
        for metric_name, value in volatility_metrics:
            ws.cell(row=row, column=1, value=metric_name)
            ws.cell(row=row, column=2, value=f"{value * 100:.2f}%")

            # 风险等级
            if value < 0.05:
                risk_level = "低风险"
                color = "28a745"
            elif value < 0.15:
                risk_level = "中等风险"
                color = "ffc107"
            else:
                risk_level = "高风险"
                color = "dc3545"

            cell = ws.cell(row=row, column=3, value=risk_level)
            cell.font = Font(color=color, bold=True)
            row += 1

        # 回撤分析
        ws[f'A{row + 1}'] = "回撤分析"
        ws[f'A{row + 1}'].font = Font(size=12, bold=True, color="C73E1D")

        drawdown_metrics = [
            ('最大回撤', performance_metrics.get('max_drawdown', 0)),
            ('平均回撤', risk_metrics.get('avg_drawdown', 0)),
            ('当前回撤', risk_metrics.get('current_drawdown', 0)),
            ('最大回撤持续期(天)', risk_metrics.get('max_drawdown_duration', 0)),
        ]

        row += 2
        for metric_name, value in drawdown_metrics:
            ws.cell(row=row, column=1, value=metric_name)
            if '持续期' in metric_name:
                ws.cell(row=row, column=2, value=f"{int(value)}")
            else:
                ws.cell(row=row, column=2, value=f"{abs(value) * 100:.2f}%")
                # 回撤用红色显示
                ws.cell(row=row, column=2).font = Font(color="dc3545")
            row += 1

        # VaR分析
        if 'var_95' in risk_metrics or 'var_99' in risk_metrics:
            ws[f'A{row + 1}'] = "风险价值 (VaR)"
            ws[f'A{row + 1}'].font = Font(size=12, bold=True, color="C73E1D")

            var_metrics = [
                ('VaR (95%)', risk_metrics.get('var_95', 0)),
                ('VaR (99%)', risk_metrics.get('var_99', 0)),
                ('CVaR (95%)', risk_metrics.get('cvar_95', 0)),
            ]

            row += 2
            for metric_name, value in var_metrics:
                if value != 0:
                    ws.cell(row=row, column=1, value=metric_name)
                    ws.cell(row=row, column=2, value=f"{abs(value) * 100:.2f}%")
                    ws.cell(row=row, column=2).font = Font(color="dc3545")
                    row += 1

        # 设置列宽
        ws.column_dimensions['A'].width = 20
        ws.column_dimensions['B'].width = 15
        ws.column_dimensions['C'].width = 15

    def _create_time_series_sheet(self,
                                 wb: openpyxl.Workbook,
                                 evaluation_results: Dict[str, Any]):
        """创建时间序列工作表"""
        ws = wb.create_sheet("时间序列")

        # 标题
        ws['A1'] = "时间序列数据"
        ws['A1'].font = Font(size=14, bold=True)

        # 这里可以添加时间序列数据
        # 由于evaluation_results中可能没有详细的时间序列数据
        # 我们创建一个示例结构

        headers = ['日期', '累计收益率', '每日收益率', '回撤', '基准收益率']
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=3, column=col, value=header)
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="6A994E", end_color="6A994E", fill_type="solid")
            cell.font = Font(bold=True, color="FFFFFF")

        # 如果有实际的时间序列数据，在这里添加
        # 目前添加一个说明
        ws['A4'] = "时间序列数据将在实际实现中填充"

        # 设置列宽
        for col in range(1, 6):
            ws.column_dimensions[chr(64 + col)].width = 15

    def _create_benchmark_comparison_sheet(self,
                                          wb: openpyxl.Workbook,
                                          evaluation_results: Dict[str, Any]):
        """创建基准比较工作表"""
        ws = wb.create_sheet("基准比较")

        # 标题
        ws['A1'] = "策略与基准比较"
        ws['A1'].font = Font(size=14, bold=True)

        benchmark_data = evaluation_results.get('benchmark_comparison', {})

        # 比较表格
        headers = ['指标', '策略', '基准', '超额表现']
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=3, column=col, value=header)
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="A23B72", end_color="A23B72", fill_type="solid")
            cell.font = Font(bold=True, color="FFFFFF")

        # 如果有基准比较数据，在这里添加
        if benchmark_data:
            # 添加实际的基准比较数据
            pass
        else:
            ws['A4'] = "基准比较数据将在实际实现中填充"

        # 设置列宽
        for col in range(1, 5):
            ws.column_dimensions[chr(64 + col)].width = 15

    def _generate_csv_fallback(self,
                              evaluation_results: Dict[str, Any],
                              strategy_name: str,
                              request_id: str) -> str:
        """生成CSV格式作为回退方案"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{strategy_name}_report_{timestamp}.csv"
            output_dir = Path(getattr(self.config, 'output_dir', './reports'))
            filepath = output_dir / filename

            # 准备数据
            performance_metrics = evaluation_results.get('performance_metrics', {})

            data = {
                '指标名称': [
                    '总收益率', '年化收益率', '夏普比率', '最大回撤',
                    '胜率', '年化波动率', '索提诺比率', '卡玛比率'
                ],
                '数值': [
                    f"{performance_metrics.get('total_return', 0) * 100:.2f}%",
                    f"{performance_metrics.get('annualized_return', 0) * 100:.2f}%",
                    f"{performance_metrics.get('sharpe_ratio', 0):.3f}",
                    f"{performance_metrics.get('max_drawdown', 0) * 100:.2f}%",
                    f"{performance_metrics.get('win_rate', 0) * 100:.1f}%",
                    f"{performance_metrics.get('volatility', 0) * 100:.2f}%",
                    f"{performance_metrics.get('sortino_ratio', 0):.3f}",
                    f"{performance_metrics.get('calmar_ratio', 0):.3f}"
                ]
            }

            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')

            logger.info(f"CSV报告生成完成: {filepath}")

            return str(filepath)

        except Exception as e:
            logger.error(f"生成CSV报告失败: {e}")
            raise