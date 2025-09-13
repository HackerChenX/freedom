#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDF报告生成器

专业的PDF格式报告生成器，支持：
- 高质量PDF输出
- 矢量图表嵌入
- 多页面布局
- 专业排版
- 中文字体支持
"""

import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, root_dir)

from utils.dependency_injection import get_logger
from utils.decorators import performance_monitor, exception_handler

logger = get_logger(__name__)

# 尝试导入PDF生成库
try:
    from weasyprint import HTML, CSS
    from weasyprint.text.fonts import FontConfiguration
    WEASYPRINT_AVAILABLE = True
except ImportError:
    logger.warning("weasyprint未安装，PDF生成功能受限")
    WEASYPRINT_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    logger.warning("reportlab未安装，PDF生成功能受限")
    REPORTLAB_AVAILABLE = False


class PDFReportGenerator:
    """PDF报告生成器"""

    def __init__(self,
                 template_manager: Any,
                 config: Optional[Any] = None):
        """
        初始化PDF报告生成器

        Args:
            template_manager: 模板管理器
            config: 配置对象
        """
        self.template_manager = template_manager
        self.config = config or {}

    @exception_handler(reraise=True)
    @performance_monitor(threshold=30.0)
    def generate_report(self,
                       evaluation_results: Dict[str, Any],
                       chart_files: Dict[str, str],
                       strategy_name: str,
                       request_id: str) -> str:
        """
        生成PDF报告

        Args:
            evaluation_results: 评估结果
            chart_files: 图表文件
            strategy_name: 策略名称
            request_id: 请求ID

        Returns:
            str: 生成的文件路径
        """
        try:
            # 生成文件路径
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{strategy_name}_report_{timestamp}.pdf"
            output_dir = Path(getattr(self.config, 'output_dir', './reports'))
            filepath = output_dir / filename

            # 选择PDF生成方法
            if WEASYPRINT_AVAILABLE:
                self._generate_with_weasyprint(
                    evaluation_results, chart_files, strategy_name, request_id, filepath
                )
            elif REPORTLAB_AVAILABLE:
                self._generate_with_reportlab(
                    evaluation_results, chart_files, strategy_name, request_id, filepath
                )
            else:
                # 回退到简单文本PDF
                self._generate_simple_pdf(
                    evaluation_results, chart_files, strategy_name, request_id, filepath
                )

            logger.info(f"PDF报告生成完成: {filepath}")

            return str(filepath)

        except Exception as e:
            logger.error(f"生成PDF报告失败: {e}")
            raise

    def _generate_with_weasyprint(self,
                                 evaluation_results: Dict[str, Any],
                                 chart_files: Dict[str, str],
                                 strategy_name: str,
                                 request_id: str,
                                 filepath: Path):
        """使用WeasyPrint生成PDF"""
        # 准备模板上下文
        context = self._prepare_context(
            evaluation_results, chart_files, strategy_name, request_id
        )

        # 渲染HTML模板
        html_content = self.template_manager.render_template(
            'standard',
            context,
            'html'
        )

        # 添加PDF特定样式
        pdf_css = CSS(string="""
            @page {
                size: A4;
                margin: 2cm;
                @bottom-center {
                    content: "第 " counter(page) " 页，共 " counter(pages) " 页";
                    font-size: 10pt;
                    color: #666;
                }
            }

            body {
                font-family: 'SimHei', 'Microsoft YaHei', Arial, sans-serif;
                line-height: 1.4;
                color: #333;
            }

            .page-break {
                page-break-before: always;
            }

            .no-page-break {
                page-break-inside: avoid;
            }

            img {
                max-width: 100%;
                height: auto;
                page-break-inside: avoid;
            }

            table {
                page-break-inside: avoid;
            }
        """)

        # 生成PDF
        font_config = FontConfiguration()
        HTML(string=html_content).write_pdf(
            str(filepath),
            stylesheets=[pdf_css],
            font_config=font_config
        )

    def _generate_with_reportlab(self,
                                evaluation_results: Dict[str, Any],
                                chart_files: Dict[str, str],
                                strategy_name: str,
                                request_id: str,
                                filepath: Path):
        """使用ReportLab生成PDF"""
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )

        # 获取样式
        styles = getSampleStyleSheet()

        # 创建自定义样式
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2E86AB'),
            alignment=1  # 居中
        )

        # 构建PDF内容
        story = []

        # 标题页
        story.append(Paragraph(f"{strategy_name}", title_style))
        story.append(Paragraph("量化策略回测报告", styles['Heading2']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"生成时间: {datetime.now().strftime('%Y年%m月%d日')}", styles['Normal']))
        story.append(Paragraph(f"报告ID: {request_id}", styles['Normal']))
        story.append(Spacer(1, 30))

        # 执行摘要
        story.append(Paragraph("执行摘要", styles['Heading2']))

        performance_metrics = evaluation_results.get('performance_metrics', {})
        summary_data = [
            ['指标', '数值'],
            ['总收益率', f"{performance_metrics.get('total_return', 0) * 100:.2f}%"],
            ['年化收益率', f"{performance_metrics.get('annualized_return', 0) * 100:.2f}%"],
            ['夏普比率', f"{performance_metrics.get('sharpe_ratio', 0):.3f}"],
            ['最大回撤', f"{performance_metrics.get('max_drawdown', 0) * 100:.2f}%"],
            ['胜率', f"{performance_metrics.get('win_rate', 0) * 100:.1f}%"],
        ]

        summary_table = Table(summary_data, colWidths=[2*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(summary_table)
        story.append(Spacer(1, 20))

        # 添加图表
        for chart_type, chart_path in chart_files.items():
            if chart_path and Path(chart_path).exists():
                try:
                    story.append(Paragraph(f"{chart_type.replace('_', ' ').title()}", styles['Heading3']))

                    # 调整图片大小
                    img = Image(chart_path)
                    img._restrictSize(6*inch, 4*inch)
                    story.append(img)
                    story.append(Spacer(1, 12))

                except Exception as e:
                    logger.warning(f"添加图表失败 {chart_type}: {e}")

        # 生成PDF
        doc.build(story)

    def _generate_simple_pdf(self,
                            evaluation_results: Dict[str, Any],
                            chart_files: Dict[str, str],
                            strategy_name: str,
                            request_id: str,
                            filepath: Path):
        """生成简单PDF（回退方案）"""
        # 如果没有专业PDF库，生成一个简单的文本文件
        content = self._generate_text_report(evaluation_results, strategy_name, request_id)

        # 将扩展名改为txt
        txt_filepath = filepath.with_suffix('.txt')
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.warning(f"使用简单文本格式生成报告: {txt_filepath}")
        return str(txt_filepath)

    def _generate_text_report(self,
                             evaluation_results: Dict[str, Any],
                             strategy_name: str,
                             request_id: str) -> str:
        """生成文本格式报告"""
        performance_metrics = evaluation_results.get('performance_metrics', {})

        return f"""
{strategy_name} - 量化策略回测报告
{'=' * 50}

生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
报告ID: {request_id}

执行摘要
{'-' * 20}
总收益率: {performance_metrics.get('total_return', 0) * 100:.2f}%
年化收益率: {performance_metrics.get('annualized_return', 0) * 100:.2f}%
夏普比率: {performance_metrics.get('sharpe_ratio', 0):.3f}
最大回撤: {performance_metrics.get('max_drawdown', 0) * 100:.2f}%
胜率: {performance_metrics.get('win_rate', 0) * 100:.1f}%
年化波动率: {performance_metrics.get('volatility', 0) * 100:.2f}%

风险分析
{'-' * 20}
最大回撤: {performance_metrics.get('max_drawdown', 0) * 100:.2f}%
年化波动率: {performance_metrics.get('volatility', 0) * 100:.2f}%

注：本报告由于缺少PDF生成库，以文本格式提供。
建议安装 weasyprint 或 reportlab 以获得完整PDF报告功能。
"""

    def _prepare_context(self,
                        evaluation_results: Dict[str, Any],
                        chart_files: Dict[str, str],
                        strategy_name: str,
                        request_id: str) -> Dict[str, Any]:
        """准备模板上下文"""
        return {
            'strategy_name': strategy_name,
            'request_id': request_id,
            'performance_metrics': evaluation_results.get('performance_metrics', {}),
            'risk_metrics': evaluation_results.get('risk_metrics', {}),
            'benchmark_comparison': evaluation_results.get('benchmark_comparison', {}),
            'time_series_analysis': evaluation_results.get('time_series_analysis', {}),
            'stress_test_results': evaluation_results.get('stress_test_results', {}),
            'chart_files': chart_files,
            'generation_time': datetime.now(),
            'colors': getattr(self.config, 'brand_colors', {
                'primary': '#2E86AB',
                'secondary': '#A23B72',
                'success': '#F18F01',
                'warning': '#C73E1D',
                'accent': '#6A994E',
                'neutral': '#6C757D'
            })
        }