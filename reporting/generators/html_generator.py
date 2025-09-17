#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HTML报告生成器

专业的HTML格式报告生成器，支持：
- 响应式设计
- 高质量图表嵌入
- 交互式元素
- 专业样式主题
- 完整的金融报告结构
"""

import os
import sys
import base64
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from utils.decorators import performance_monitor, exception_handler

logger = get_logger(__name__)


class HTMLReportGenerator:
    """HTML报告生成器"""

    def __init__(self,
                 template_manager: Any,
                 config: Optional[Any] = None):
        """
        初始化HTML报告生成器

        Args:
            template_manager: 模板管理器
            config: 配置对象
        """
        self.template_manager = template_manager
        self.config = config or {}

    @exception_handler(reraise=True)
    @performance_monitor(threshold=20.0)
    def generate_report(self,
                       evaluation_results: Dict[str, Any],
                       chart_files: Dict[str, str],
                       strategy_name: str,
                       request_id: str) -> str:
        """
        生成HTML报告

        Args:
            evaluation_results: 评估结果
            chart_files: 图表文件
            strategy_name: 策略名称
            request_id: 请求ID

        Returns:
            str: 生成的文件路径
        """
        try:
            # 准备模板上下文
            context = self._prepare_context(
                evaluation_results, chart_files, strategy_name, request_id
            )

            # 渲染模板
            html_content = self.template_manager.render_template(
                'standard',
                context,
                'html'
            )

            # 嵌入图表
            html_content = self._embed_charts(html_content, chart_files)

            # 生成文件路径
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{strategy_name}_report_{timestamp}.html"
            output_dir = Path(getattr(self.config, 'output_dir', './reports'))
            filepath = output_dir / filename

            # 写入文件
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"HTML报告生成完成: {filepath}")

            return str(filepath)

        except Exception as e:
            logger.error(f"生成HTML报告失败: {e}")
            raise

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

    def _embed_charts(self, html_content: str, chart_files: Dict[str, str]) -> str:
        """嵌入图表到HTML中"""
        for chart_type, chart_path in chart_files.items():
            if chart_path and Path(chart_path).exists():
                try:
                    # 将图片转换为base64
                    with open(chart_path, 'rb') as f:
                        img_data = f.read()
                        img_base64 = base64.b64encode(img_data).decode()

                    # 替换占位符
                    placeholder = f'src="{chart_path}"'
                    replacement = f'src="data:image/png;base64,{img_base64}"'
                    html_content = html_content.replace(placeholder, replacement)

                except Exception as e:
                    logger.warning(f"嵌入图表失败 {chart_type}: {e}")

        return html_content