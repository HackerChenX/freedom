#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
专业报告模板管理系统

核心功能：
- 多种报告模板（HTML、PDF、Excel、Markdown）
- 模板缓存和版本管理
- 自定义品牌设置
- 动态内容渲染
- 模板继承和组合

支持的模板类型：
- 标准财务报告模板
- 风险管理报告模板
- 监管合规报告模板
- 投资者报告模板
- 自定义企业品牌模板
"""

import os
import sys
import json
import hashlib
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
import jinja2

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, root_dir)

from utils.dependency_injection import get_logger
from utils.decorators import performance_monitor, exception_handler

logger = get_logger(__name__)


@dataclass
class TemplateMetadata:
    """模板元数据"""
    name: str
    version: str
    description: str
    author: str
    created_at: datetime
    updated_at: datetime
    supported_formats: List[str]
    template_type: str  # 'standard', 'risk', 'regulatory', 'investor', 'custom'
    brand_settings: Dict[str, Any]
    variables: List[str]  # 模板所需变量列表


class TemplateManager:
    """
    专业报告模板管理系统

    功能：
    1. 模板注册和管理
    2. 动态模板渲染
    3. 模板缓存优化
    4. 品牌定制支持
    5. 模板版本控制
    """

    def __init__(self,
                 template_dir: str = "./templates/reports",
                 cache_dir: str = "./cache/templates"):
        """
        初始化模板管理器

        Args:
            template_dir: 模板目录
            cache_dir: 缓存目录
        """
        self.template_dir = Path(template_dir)
        self.cache_dir = Path(cache_dir)

        # 创建目录
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 初始化Jinja2环境
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader([str(self.template_dir)]),
            autoescape=jinja2.select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )

        # 注册自定义过滤器
        self._register_custom_filters()

        # 模板缓存
        self.template_cache = {}
        self.metadata_cache = {}

        # 创建默认模板
        self._create_default_templates()

        logger.info(f"模板管理器初始化完成")
        logger.info(f"模板目录: {self.template_dir}")
        logger.info(f"缓存目录: {self.cache_dir}")

    def _register_custom_filters(self):
        """注册自定义Jinja2过滤器"""

        def format_percentage(value, decimals=2):
            """格式化为百分比"""
            try:
                return f"{float(value) * 100:.{decimals}f}%"
            except (ValueError, TypeError):
                return "N/A"

        def format_number(value, decimals=2, thousands_sep=True):
            """格式化数字"""
            try:
                num = float(value)
                if thousands_sep:
                    return f"{num:,.{decimals}f}"
                else:
                    return f"{num:.{decimals}f}"
            except (ValueError, TypeError):
                return "N/A"

        def format_currency(value, currency='CNY', decimals=2):
            """格式化为货币"""
            try:
                num = float(value)
                if currency == 'CNY':
                    return f"¥{num:,.{decimals}f}"
                elif currency == 'USD':
                    return f"${num:,.{decimals}f}"
                else:
                    return f"{num:,.{decimals}f} {currency}"
            except (ValueError, TypeError):
                return "N/A"

        def format_date(value, format_str='%Y-%m-%d'):
            """格式化日期"""
            try:
                if isinstance(value, str):
                    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                elif isinstance(value, datetime):
                    dt = value
                else:
                    return str(value)
                return dt.strftime(format_str)
            except Exception:
                return str(value)

        def color_by_value(value, positive_color='#28a745', negative_color='#dc3545', neutral_color='#6c757d'):
            """根据数值设置颜色"""
            try:
                num = float(value)
                if num > 0:
                    return positive_color
                elif num < 0:
                    return negative_color
                else:
                    return neutral_color
            except (ValueError, TypeError):
                return neutral_color

        def risk_level(value):
            """根据数值判断风险等级"""
            try:
                num = float(value)
                if abs(num) < 0.05:
                    return '低风险'
                elif abs(num) < 0.15:
                    return '中等风险'
                elif abs(num) < 0.3:
                    return '高风险'
                else:
                    return '极高风险'
            except (ValueError, TypeError):
                return '未知风险'

        # 注册过滤器
        self.jinja_env.filters['percentage'] = format_percentage
        self.jinja_env.filters['number'] = format_number
        self.jinja_env.filters['currency'] = format_currency
        self.jinja_env.filters['date'] = format_date
        self.jinja_env.filters['color_by_value'] = color_by_value
        self.jinja_env.filters['risk_level'] = risk_level

    @exception_handler(reraise=True)
    @performance_monitor(threshold=5.0)
    def render_template(self,
                       template_name: str,
                       context: Dict[str, Any],
                       output_format: str = 'html') -> str:
        """
        渲染模板

        Args:
            template_name: 模板名称
            context: 模板上下文变量
            output_format: 输出格式

        Returns:
            str: 渲染后的内容
        """
        try:
            # 检查缓存
            cache_key = self._generate_template_cache_key(template_name, context, output_format)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                logger.debug(f"从缓存获取模板渲染结果: {template_name}")
                return cached_result

            # 获取模板文件
            template_file = self._get_template_file(template_name, output_format)
            if not template_file.exists():
                raise FileNotFoundError(f"模板文件不存在: {template_file}")

            # 加载模板
            template = self.jinja_env.get_template(template_file.name)

            # 准备上下文
            render_context = self._prepare_context(context, template_name, output_format)

            # 渲染模板
            rendered_content = template.render(**render_context)

            # 缓存结果
            self._save_to_cache(cache_key, rendered_content)

            logger.debug(f"模板渲染完成: {template_name} ({output_format})")

            return rendered_content

        except Exception as e:
            logger.error(f"渲染模板失败: {template_name} - {e}")
            raise

    def _get_template_file(self, template_name: str, output_format: str) -> Path:
        """获取模板文件路径"""
        # 优先查找特定格式的模板
        specific_template = self.template_dir / f"{template_name}_{output_format}.jinja2"
        if specific_template.exists():
            return specific_template

        # 查找通用模板
        general_template = self.template_dir / f"{template_name}.jinja2"
        if general_template.exists():
            return general_template

        # 查找默认模板
        default_template = self.template_dir / f"default_{output_format}.jinja2"
        if default_template.exists():
            return default_template

        raise FileNotFoundError(f"未找到合适的模板: {template_name}")

    def _prepare_context(self, context: Dict[str, Any], template_name: str, output_format: str) -> Dict[str, Any]:
        """准备模板上下文"""
        render_context = context.copy()

        # 添加系统变量
        render_context.update({
            'template_name': template_name,
            'output_format': output_format,
            'generation_time': datetime.now(),
            'system_info': {
                'version': '1.0.0',
                'generator': '回测报告自动生成系统'
            }
        })

        # 添加默认样式变量
        render_context.setdefault('colors', {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#F18F01',
            'warning': '#C73E1D',
            'accent': '#6A994E',
            'neutral': '#6C757D'
        })

        # 添加格式化辅助函数
        render_context['helpers'] = {
            'format_large_number': self._format_large_number,
            'generate_summary_text': self._generate_summary_text,
            'calculate_performance_grade': self._calculate_performance_grade
        }

        return render_context

    def _format_large_number(self, value: float) -> str:
        """格式化大数字"""
        try:
            if value >= 1e8:
                return f"{value/1e8:.2f}亿"
            elif value >= 1e4:
                return f"{value/1e4:.2f}万"
            else:
                return f"{value:.2f}"
        except (ValueError, TypeError):
            return "N/A"

    def _generate_summary_text(self, performance_data: Dict[str, Any]) -> str:
        """生成性能摘要文本"""
        try:
            total_return = performance_data.get('total_return', 0)
            sharpe_ratio = performance_data.get('sharpe_ratio', 0)
            max_drawdown = performance_data.get('max_drawdown', 0)

            summary_parts = []

            if total_return > 0.2:
                summary_parts.append("策略表现优异")
            elif total_return > 0.05:
                summary_parts.append("策略表现良好")
            elif total_return > 0:
                summary_parts.append("策略获得正收益")
            else:
                summary_parts.append("策略出现亏损")

            if sharpe_ratio > 2:
                summary_parts.append("风险调整收益优秀")
            elif sharpe_ratio > 1:
                summary_parts.append("风险调整收益良好")
            else:
                summary_parts.append("风险调整收益一般")

            if abs(max_drawdown) < 0.05:
                summary_parts.append("回撤控制良好")
            elif abs(max_drawdown) < 0.15:
                summary_parts.append("回撤控制一般")
            else:
                summary_parts.append("存在较大回撤风险")

            return "；".join(summary_parts) + "。"

        except Exception as e:
            logger.warning(f"生成摘要文本失败: {e}")
            return "策略性能摘要生成中..."

    def _calculate_performance_grade(self, performance_data: Dict[str, Any]) -> str:
        """计算性能等级"""
        try:
            total_return = performance_data.get('total_return', 0)
            sharpe_ratio = performance_data.get('sharpe_ratio', 0)
            max_drawdown = abs(performance_data.get('max_drawdown', 0))

            # 综合评分
            score = 0

            # 收益率评分 (40%)
            if total_return > 0.3:
                score += 40
            elif total_return > 0.15:
                score += 32
            elif total_return > 0.05:
                score += 24
            elif total_return > 0:
                score += 16
            else:
                score += 0

            # 夏普比率评分 (35%)
            if sharpe_ratio > 2:
                score += 35
            elif sharpe_ratio > 1.5:
                score += 28
            elif sharpe_ratio > 1:
                score += 21
            elif sharpe_ratio > 0.5:
                score += 14
            else:
                score += 0

            # 回撤控制评分 (25%)
            if max_drawdown < 0.05:
                score += 25
            elif max_drawdown < 0.1:
                score += 20
            elif max_drawdown < 0.2:
                score += 15
            elif max_drawdown < 0.3:
                score += 10
            else:
                score += 0

            # 等级划分
            if score >= 85:
                return "A+"
            elif score >= 75:
                return "A"
            elif score >= 65:
                return "B+"
            elif score >= 55:
                return "B"
            elif score >= 45:
                return "C+"
            elif score >= 35:
                return "C"
            else:
                return "D"

        except Exception as e:
            logger.warning(f"计算性能等级失败: {e}")
            return "N/A"

    def _create_default_templates(self):
        """创建默认模板"""
        templates = {
            'standard_html.jinja2': self._get_standard_html_template(),
            'standard_pdf.jinja2': self._get_standard_pdf_template(),
            'risk_report_html.jinja2': self._get_risk_html_template(),
            'executive_summary.jinja2': self._get_executive_summary_template()
        }

        for template_name, template_content in templates.items():
            template_file = self.template_dir / template_name
            if not template_file.exists():
                try:
                    with open(template_file, 'w', encoding='utf-8') as f:
                        f.write(template_content)
                    logger.debug(f"创建默认模板: {template_name}")
                except Exception as e:
                    logger.warning(f"创建默认模板失败 {template_name}: {e}")

    def _get_standard_html_template(self) -> str:
        """获取标准HTML模板"""
        return '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ strategy_name }} - 量化策略回测报告</title>
    <style>
        body {
            font-family: 'Microsoft YaHei', 'SimHei', Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, {{ colors.primary }} 0%, {{ colors.secondary }} 100%);
            color: white;
            padding: 2rem;
            text-align: center;
        }
        .header h1 {
            margin: 0 0 1rem 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }
        .section {
            padding: 2rem;
            border-bottom: 1px solid #eee;
        }
        .section h2 {
            color: {{ colors.primary }};
            border-bottom: 3px solid {{ colors.primary }};
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
            font-size: 1.8em;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: linear-gradient(145deg, #f8f9fa, #e9ecef);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 5px solid {{ colors.primary }};
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .metric-card h3 {
            margin: 0 0 0.5rem 0;
            color: #495057;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }
        .metric-value {
            font-size: 2.2em;
            font-weight: 700;
            margin: 0;
            color: {{ colors.primary }};
        }
        .metric-description {
            font-size: 0.9em;
            color: #666;
            margin-top: 0.5rem;
        }
        .positive { color: {{ colors.success }}; }
        .negative { color: {{ colors.warning }}; }
        .neutral { color: {{ colors.neutral }}; }
        .chart-container {
            text-align: center;
            margin: 2rem 0;
            padding: 1rem;
            background-color: #fafafa;
            border-radius: 8px;
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .performance-grade {
            font-size: 3em;
            font-weight: bold;
            padding: 1rem;
            border-radius: 50%;
            background: linear-gradient(45deg, {{ colors.success }}, {{ colors.accent }});
            color: white;
            width: 100px;
            height: 100px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin: 1rem;
        }
        .footer {
            background: #343a40;
            color: white;
            text-align: center;
            padding: 2rem;
            margin-top: 2rem;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        table th, table td {
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        table th {
            background: linear-gradient(135deg, {{ colors.primary }}, {{ colors.secondary }});
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        table tr:hover {
            background-color: rgba({{ colors.primary | replace('#', '') }}, 0.05);
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- 报告头部 -->
        <div class="header">
            <h1>{{ strategy_name }}</h1>
            <div class="subtitle">量化策略回测报告</div>
            <div style="margin-top: 1rem; font-size: 1em;">
                报告生成时间: {{ generation_time | date('%Y-%m-%d %H:%M:%S') }}
            </div>
        </div>

        <!-- 执行摘要 -->
        <div class="section">
            <h2>📊 执行摘要</h2>
            <div style="display: flex; align-items: center; gap: 2rem; margin-bottom: 2rem;">
                <div class="performance-grade">
                    {{ helpers.calculate_performance_grade(performance_metrics) }}
                </div>
                <div>
                    <h3>策略评级</h3>
                    <p style="font-size: 1.1em;">{{ helpers.generate_summary_text(performance_metrics) }}</p>
                </div>
            </div>

            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>总收益率</h3>
                    <p class="metric-value {{ 'positive' if performance_metrics.total_return > 0 else 'negative' if performance_metrics.total_return < 0 else 'neutral' }}">
                        {{ performance_metrics.total_return | percentage }}
                    </p>
                    <p class="metric-description">整个回测期间的累计收益</p>
                </div>

                <div class="metric-card">
                    <h3>年化收益率</h3>
                    <p class="metric-value {{ 'positive' if performance_metrics.annualized_return > 0 else 'negative' if performance_metrics.annualized_return < 0 else 'neutral' }}">
                        {{ performance_metrics.annualized_return | percentage }}
                    </p>
                    <p class="metric-description">按年化计算的收益率</p>
                </div>

                <div class="metric-card">
                    <h3>夏普比率</h3>
                    <p class="metric-value {{ 'positive' if performance_metrics.sharpe_ratio > 1 else 'neutral' if performance_metrics.sharpe_ratio > 0 else 'negative' }}">
                        {{ performance_metrics.sharpe_ratio | number(3) }}
                    </p>
                    <p class="metric-description">每单位风险的超额收益</p>
                </div>

                <div class="metric-card">
                    <h3>最大回撤</h3>
                    <p class="metric-value negative">
                        {{ performance_metrics.max_drawdown | percentage }}
                    </p>
                    <p class="metric-description">历史最大资产损失幅度</p>
                </div>

                <div class="metric-card">
                    <h3>胜率</h3>
                    <p class="metric-value {{ 'positive' if performance_metrics.win_rate > 0.5 else 'neutral' }}">
                        {{ performance_metrics.win_rate | percentage }}
                    </p>
                    <p class="metric-description">盈利交易占总交易比例</p>
                </div>

                <div class="metric-card">
                    <h3>年化波动率</h3>
                    <p class="metric-value neutral">
                        {{ performance_metrics.volatility | percentage }}
                    </p>
                    <p class="metric-description">收益率的年化标准差</p>
                </div>
            </div>
        </div>

        <!-- 图表展示区域 -->
        {% if chart_files %}
        <div class="section">
            <h2>📈 图表分析</h2>
            {% for chart_type, chart_path in chart_files.items() %}
            {% if chart_path %}
            <div class="chart-container">
                <h3>{{ chart_type | replace('_', ' ') | title }}</h3>
                <img src="{{ chart_path }}" alt="{{ chart_type }}">
            </div>
            {% endif %}
            {% endfor %}
        </div>
        {% endif %}

        <!-- 详细指标 -->
        <div class="section">
            <h2>📋 详细性能指标</h2>

            <h3>收益指标</h3>
            <table>
                <thead>
                    <tr>
                        <th>指标名称</th>
                        <th>数值</th>
                        <th>说明</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>总收益率</td>
                        <td class="{{ 'positive' if performance_metrics.total_return > 0 else 'negative' }}">
                            {{ performance_metrics.total_return | percentage }}
                        </td>
                        <td>整个评估期间的累计收益率</td>
                    </tr>
                    <tr>
                        <td>年化收益率</td>
                        <td class="{{ 'positive' if performance_metrics.annualized_return > 0 else 'negative' }}">
                            {{ performance_metrics.annualized_return | percentage }}
                        </td>
                        <td>按年化计算的收益率</td>
                    </tr>
                    <tr>
                        <td>累计超额收益</td>
                        <td class="{{ 'positive' if performance_metrics.get('excess_return', 0) > 0 else 'negative' }}">
                            {{ performance_metrics.get('excess_return', 0) | percentage }}
                        </td>
                        <td>相对于基准的超额收益</td>
                    </tr>
                </tbody>
            </table>

            <h3>风险调整收益指标</h3>
            <table>
                <thead>
                    <tr>
                        <th>指标名称</th>
                        <th>数值</th>
                        <th>说明</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>夏普比率</td>
                        <td class="{{ 'positive' if performance_metrics.sharpe_ratio > 1 else 'neutral' }}">
                            {{ performance_metrics.sharpe_ratio | number(3) }}
                        </td>
                        <td>每单位风险的超额收益</td>
                    </tr>
                    <tr>
                        <td>索提诺比率</td>
                        <td class="{{ 'positive' if performance_metrics.get('sortino_ratio', 0) > 1 else 'neutral' }}">
                            {{ performance_metrics.get('sortino_ratio', 0) | number(3) }}
                        </td>
                        <td>每单位下行风险的超额收益</td>
                    </tr>
                    <tr>
                        <td>卡玛比率</td>
                        <td class="{{ 'positive' if performance_metrics.get('calmar_ratio', 0) > 1 else 'neutral' }}">
                            {{ performance_metrics.get('calmar_ratio', 0) | number(3) }}
                        </td>
                        <td>年化收益率与最大回撤的比值</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <!-- 风险分析 -->
        {% if risk_metrics %}
        <div class="section">
            <h2>⚠️ 风险分析</h2>

            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>最大回撤</h3>
                    <p class="metric-value negative">{{ risk_metrics.max_drawdown | percentage }}</p>
                    <p class="metric-description">风险级别: {{ risk_metrics.max_drawdown | risk_level }}</p>
                </div>

                <div class="metric-card">
                    <h3>年化波动率</h3>
                    <p class="metric-value neutral">{{ risk_metrics.get('annual_volatility', 0) | percentage }}</p>
                    <p class="metric-description">收益率的年化标准差</p>
                </div>

                <div class="metric-card">
                    <h3>VaR (95%)</h3>
                    <p class="metric-value negative">{{ risk_metrics.get('var_95', 0) | percentage }}</p>
                    <p class="metric-description">95%置信度下的风险价值</p>
                </div>

                <div class="metric-card">
                    <h3>回撤持续期</h3>
                    <p class="metric-value neutral">{{ risk_metrics.get('max_drawdown_duration', 0) | number(0) }} 天</p>
                    <p class="metric-description">最大回撤的持续时间</p>
                </div>
            </div>
        </div>
        {% endif %}

        <!-- 报告尾部 -->
        <div class="footer">
            <p>本报告由量化策略回测系统自动生成</p>
            <p>生成时间: {{ generation_time | date('%Y年%m月%d日 %H:%M:%S') }}</p>
            <p>版本: {{ system_info.version }}</p>
        </div>
    </div>
</body>
</html>'''

    def _get_standard_pdf_template(self) -> str:
        """获取标准PDF模板 - 简化版"""
        return '''# {{ strategy_name }} - 量化策略回测报告

生成时间: {{ generation_time | date('%Y-%m-%d %H:%M:%S') }}

## 执行摘要

**策略评级**: {{ helpers.calculate_performance_grade(performance_metrics) }}

{{ helpers.generate_summary_text(performance_metrics) }}

### 核心指标

- **总收益率**: {{ performance_metrics.total_return | percentage }}
- **年化收益率**: {{ performance_metrics.annualized_return | percentage }}
- **夏普比率**: {{ performance_metrics.sharpe_ratio | number(3) }}
- **最大回撤**: {{ performance_metrics.max_drawdown | percentage }}
- **胜率**: {{ performance_metrics.win_rate | percentage }}

## 详细性能指标

### 收益指标
- 总收益率: {{ performance_metrics.total_return | percentage }}
- 年化收益率: {{ performance_metrics.annualized_return | percentage }}
- 超额收益: {{ performance_metrics.get('excess_return', 0) | percentage }}

### 风险指标
- 年化波动率: {{ performance_metrics.volatility | percentage }}
- 最大回撤: {{ performance_metrics.max_drawdown | percentage }}
- 夏普比率: {{ performance_metrics.sharpe_ratio | number(3) }}

---
报告生成系统: {{ system_info.generator }}
版本: {{ system_info.version }}'''

    def _get_risk_html_template(self) -> str:
        """获取风险报告HTML模板"""
        return '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>{{ strategy_name }} - 风险分析报告</title>
    <style>
        body { font-family: 'Microsoft YaHei', sans-serif; }
        .risk-high { color: #dc3545; font-weight: bold; }
        .risk-medium { color: #ffc107; font-weight: bold; }
        .risk-low { color: #28a745; font-weight: bold; }
    </style>
</head>
<body>
    <h1>{{ strategy_name }} - 风险分析报告</h1>

    <h2>风险概览</h2>
    <p>最大回撤: <span class="risk-high">{{ risk_metrics.max_drawdown | percentage }}</span></p>
    <p>波动率: <span class="risk-medium">{{ risk_metrics.get('annual_volatility', 0) | percentage }}</span></p>

    {% if stress_test_results %}
    <h2>压力测试结果</h2>
    <ul>
    {% for scenario, result in stress_test_results.items() %}
        <li>{{ scenario }}: {{ result | percentage }}</li>
    {% endfor %}
    </ul>
    {% endif %}
</body>
</html>'''

    def _get_executive_summary_template(self) -> str:
        """获取执行摘要模板"""
        return '''## 执行摘要

**策略名称**: {{ strategy_name }}
**评估期间**: {{ evaluation_period | default('未指定') }}
**策略评级**: {{ helpers.calculate_performance_grade(performance_metrics) }}

### 核心表现
{{ helpers.generate_summary_text(performance_metrics) }}

### 关键指标
- 总收益率: {{ performance_metrics.total_return | percentage }}
- 年化收益率: {{ performance_metrics.annualized_return | percentage }}
- 夏普比率: {{ performance_metrics.sharpe_ratio | number(3) }}
- 最大回撤: {{ performance_metrics.max_drawdown | percentage }}

### 风险评估
- 波动率水平: {{ performance_metrics.volatility | risk_level }}
- 回撤控制: {{ 'excellent' if performance_metrics.max_drawdown|abs < 0.1 else 'good' if performance_metrics.max_drawdown|abs < 0.2 else 'needs_improvement' }}

---
*此摘要由系统自动生成于 {{ generation_time | date('%Y-%m-%d %H:%M') }}*'''

    def _generate_template_cache_key(self, template_name: str, context: Dict[str, Any], output_format: str) -> str:
        """生成模板缓存键"""
        # 只使用关键参数生成缓存键，避免因时间戳等变化导致缓存失效
        key_data = {
            'template': template_name,
            'format': output_format,
            'context_hash': hashlib.md5(str(sorted(context.keys())).encode()).hexdigest()[:8]
        }
        return hashlib.md5(str(key_data).encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """从缓存获取模板"""
        cache_file = self.cache_dir / f"{cache_key}.html"

        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"读取模板缓存失败: {e}")

        return None

    def _save_to_cache(self, cache_key: str, content: str):
        """保存模板到缓存"""
        cache_file = self.cache_dir / f"{cache_key}.html"

        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            logger.warning(f"保存模板缓存失败: {e}")

    def list_templates(self) -> List[str]:
        """列出可用模板"""
        templates = []
        for template_file in self.template_dir.glob("*.jinja2"):
            templates.append(template_file.stem)
        return sorted(templates)

    def clear_cache(self):
        """清理模板缓存"""
        try:
            for file in self.cache_dir.glob("*"):
                if file.is_file():
                    file.unlink()
            logger.info("模板缓存已清理")
        except Exception as e:
            logger.error(f"清理模板缓存失败: {e}")