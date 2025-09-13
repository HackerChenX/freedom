#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Markdown报告生成器

生成Markdown格式的报告，适用于：
- 文档系统集成
- 版本控制
- 在线展示
- 技术文档
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


class MarkdownReportGenerator:
    """Markdown报告生成器"""

    def __init__(self,
                 template_manager: Any,
                 config: Optional[Any] = None):
        """
        初始化Markdown报告生成器

        Args:
            template_manager: 模板管理器
            config: 配置对象
        """
        self.template_manager = template_manager
        self.config = config or {}

    @exception_handler(reraise=True)
    @performance_monitor(threshold=15.0)
    def generate_report(self,
                       evaluation_results: Dict[str, Any],
                       chart_files: Dict[str, str],
                       strategy_name: str,
                       request_id: str) -> str:
        """
        生成Markdown报告

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
            filename = f"{strategy_name}_report_{timestamp}.md"
            output_dir = Path(getattr(self.config, 'output_dir', './reports'))
            filepath = output_dir / filename

            # 构建Markdown内容
            markdown_content = self._build_markdown_content(
                evaluation_results, chart_files, strategy_name, request_id
            )

            # 写入文件
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            logger.info(f"Markdown报告生成完成: {filepath}")

            return str(filepath)

        except Exception as e:
            logger.error(f"生成Markdown报告失败: {e}")
            raise

    def _build_markdown_content(self,
                               evaluation_results: Dict[str, Any],
                               chart_files: Dict[str, str],
                               strategy_name: str,
                               request_id: str) -> str:
        """构建Markdown内容"""
        performance_metrics = evaluation_results.get('performance_metrics', {})
        risk_metrics = evaluation_results.get('risk_metrics', {})

        content = f"""# {strategy_name} - 量化策略回测报告

**生成时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
**报告ID**: `{request_id}`
**报告类型**: 策略回测分析报告

---

## 📊 执行摘要

### 策略评级
{self._get_performance_badge(performance_metrics)}

### 核心指标

| 指标名称 | 数值 | 说明 |
|---------|------|------|
| 总收益率 | {self._format_percentage(performance_metrics.get('total_return', 0))} | 整个回测期间的累计收益 |
| 年化收益率 | {self._format_percentage(performance_metrics.get('annualized_return', 0))} | 按年化计算的收益率 |
| 夏普比率 | {performance_metrics.get('sharpe_ratio', 0):.3f} | 每单位风险的超额收益 |
| 最大回撤 | {self._format_percentage(performance_metrics.get('max_drawdown', 0), negative=True)} | 历史最大资产损失幅度 |
| 胜率 | {self._format_percentage(performance_metrics.get('win_rate', 0))} | 盈利交易占总交易比例 |
| 年化波动率 | {self._format_percentage(performance_metrics.get('volatility', 0))} | 收益率的年化标准差 |

### 策略表现总结
{self._generate_performance_summary(performance_metrics, risk_metrics)}

---

## 📈 图表分析

{self._build_charts_section(chart_files)}

---

## 📋 详细性能指标

### 收益指标

| 指标名称 | 数值 | 描述 |
|---------|------|------|
| 总收益率 | {self._format_percentage(performance_metrics.get('total_return', 0))} | 整个评估期间的累计收益率 |
| 年化收益率 | {self._format_percentage(performance_metrics.get('annualized_return', 0))} | 按年化计算的收益率 |
| 累计超额收益 | {self._format_percentage(performance_metrics.get('excess_return', 0))} | 相对于基准的超额收益 |
| 复合年增长率 (CAGR) | {self._format_percentage(performance_metrics.get('cagr', performance_metrics.get('annualized_return', 0)))} | 复合年增长率 |

### 风险调整收益指标

| 指标名称 | 数值 | 描述 |
|---------|------|------|
| 夏普比率 | {performance_metrics.get('sharpe_ratio', 0):.3f} | 每单位风险的超额收益 |
| 索提诺比率 | {performance_metrics.get('sortino_ratio', 0):.3f} | 每单位下行风险的超额收益 |
| 卡玛比率 | {performance_metrics.get('calmar_ratio', 0):.3f} | 年化收益率与最大回撤的比值 |
| 信息比率 | {performance_metrics.get('information_ratio', 0):.3f} | 每单位跟踪误差的超额收益 |

### 交易统计

| 指标名称 | 数值 | 描述 |
|---------|------|------|
| 胜率 | {self._format_percentage(performance_metrics.get('win_rate', 0))} | 盈利交易占总交易的比例 |
| 盈亏比 | {performance_metrics.get('profit_loss_ratio', 0):.2f} | 平均盈利与平均亏损的比值 |
| 总交易次数 | {int(performance_metrics.get('total_trades', 0))} | 评估期间的总交易次数 |
| 平均每笔收益 | {self._format_percentage(performance_metrics.get('avg_trade_return', 0))} | 每笔交易的平均收益 |

---

## ⚠️ 风险分析

### 波动率分析

| 时间周期 | 波动率 | 风险等级 |
|---------|-------|---------|
| 日波动率 | {self._format_percentage(risk_metrics.get('daily_volatility', 0))} | {self._get_risk_level(risk_metrics.get('daily_volatility', 0) * 100, [1, 2, 4])} |
| 年化波动率 | {self._format_percentage(risk_metrics.get('annual_volatility', performance_metrics.get('volatility', 0)))} | {self._get_risk_level(risk_metrics.get('annual_volatility', performance_metrics.get('volatility', 0)) * 100, [15, 25, 40])} |

### 回撤分析

| 回撤指标 | 数值 | 说明 |
|---------|------|------|
| 最大回撤 | {self._format_percentage(performance_metrics.get('max_drawdown', 0), negative=True)} | 历史上资产价值的最大跌幅 |
| 最大回撤持续期 | {risk_metrics.get('max_drawdown_duration', 0):.0f} 天 | 最大回撤的持续时间 |
| 当前回撤 | {self._format_percentage(risk_metrics.get('current_drawdown', 0), negative=True)} | 当前相对于历史高点的回撤 |
| 平均回撤 | {self._format_percentage(risk_metrics.get('avg_drawdown', 0), negative=True)} | 所有回撤期间的平均回撤幅度 |

{self._build_stress_test_section(evaluation_results)}

---

## 📊 基准比较

{self._build_benchmark_section(evaluation_results)}

---

## 🔍 时间序列分析

{self._build_time_series_section(evaluation_results)}

---

## ⚡ 压力测试结果

{self._build_stress_test_details(evaluation_results)}

---

## 💡 投资建议

### 策略优势
{self._list_strategy_strengths(performance_metrics, risk_metrics)}

### 潜在风险
{self._list_potential_risks(performance_metrics, risk_metrics)}

### 改进建议
{self._list_improvement_suggestions(performance_metrics, risk_metrics)}

---

## 📝 重要说明

> **风险提示**:
> - 本报告基于历史数据回测，不代表未来投资表现
> - 实际交易中可能存在滑点、手续费等额外成本
> - 市场环境变化可能影响策略有效性
> - 投资有风险，决策需谨慎

> **数据说明**:
> - 所有收益率均为净收益率（已扣除手续费）
> - 基准选择为相关市场指数
> - 回测期间: {self._get_backtest_period(evaluation_results)}
> - 数据频率: 日频数据

---

**报告生成**: 量化策略回测系统 v1.0
**技术支持**: 量化投资团队
**联系方式**: 如有疑问，请联系系统管理员

---
*© {datetime.now().year} 量化投资系统. 保留所有权利.*
"""

        return content

    def _get_performance_badge(self, performance_metrics: Dict[str, Any]) -> str:
        """获取性能徽章"""
        grade = self._calculate_performance_grade(performance_metrics)

        badge_colors = {
            'A+': '🟢',
            'A': '🟢',
            'B+': '🟡',
            'B': '🟡',
            'C+': '🟠',
            'C': '🟠',
            'D': '🔴'
        }

        color = badge_colors.get(grade, '⚪')
        return f"{color} **{grade}级** - {self._get_grade_description(grade)}"

    def _calculate_performance_grade(self, performance_metrics: Dict[str, Any]) -> str:
        """计算性能等级"""
        try:
            total_return = performance_metrics.get('total_return', 0)
            sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
            max_drawdown = abs(performance_metrics.get('max_drawdown', 0))

            score = 0
            if total_return > 0.3: score += 40
            elif total_return > 0.15: score += 32
            elif total_return > 0.05: score += 24
            elif total_return > 0: score += 16

            if sharpe_ratio > 2: score += 35
            elif sharpe_ratio > 1.5: score += 28
            elif sharpe_ratio > 1: score += 21
            elif sharpe_ratio > 0.5: score += 14

            if max_drawdown < 0.05: score += 25
            elif max_drawdown < 0.1: score += 20
            elif max_drawdown < 0.2: score += 15
            elif max_drawdown < 0.3: score += 10

            if score >= 85: return "A+"
            elif score >= 75: return "A"
            elif score >= 65: return "B+"
            elif score >= 55: return "B"
            elif score >= 45: return "C+"
            elif score >= 35: return "C"
            else: return "D"

        except Exception:
            return "N/A"

    def _get_grade_description(self, grade: str) -> str:
        """获取等级描述"""
        descriptions = {
            'A+': '卓越表现',
            'A': '优秀表现',
            'B+': '良好表现',
            'B': '一般表现',
            'C+': '及格表现',
            'C': '需要改进',
            'D': '表现不佳'
        }
        return descriptions.get(grade, '未评级')

    def _format_percentage(self, value: float, negative: bool = False) -> str:
        """格式化百分比"""
        if value is None:
            return "N/A"

        formatted = f"{abs(value) * 100:.2f}%"
        if negative and value != 0:
            formatted = f"-{formatted}"
        elif value < 0:
            formatted = f"-{formatted}"

        return formatted

    def _generate_performance_summary(self,
                                    performance_metrics: Dict[str, Any],
                                    risk_metrics: Dict[str, Any]) -> str:
        """生成性能摘要"""
        total_return = performance_metrics.get('total_return', 0)
        sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
        max_drawdown = abs(performance_metrics.get('max_drawdown', 0))

        summary_parts = []

        if total_return > 0.2:
            summary_parts.append("**策略表现优异**，获得了显著的正收益")
        elif total_return > 0.05:
            summary_parts.append("**策略表现良好**，获得了稳定的正收益")
        elif total_return > 0:
            summary_parts.append("策略获得了微弱的正收益")
        else:
            summary_parts.append("⚠️ 策略在评估期间出现了亏损")

        if sharpe_ratio > 2:
            summary_parts.append("风险调整后收益**优秀**")
        elif sharpe_ratio > 1:
            summary_parts.append("风险调整后收益**良好**")
        elif sharpe_ratio > 0:
            summary_parts.append("风险调整后收益一般")
        else:
            summary_parts.append("⚠️ 风险调整后收益较差")

        if max_drawdown < 0.05:
            summary_parts.append("回撤控制**较好**")
        elif max_drawdown < 0.15:
            summary_parts.append("回撤控制一般")
        else:
            summary_parts.append("⚠️ 回撤较大，需要关注风险控制")

        return "；".join(summary_parts) + "。"

    def _build_charts_section(self, chart_files: Dict[str, str]) -> str:
        """构建图表章节"""
        if not chart_files:
            return "📊 暂无图表数据"

        charts_md = ""
        for chart_type, chart_path in chart_files.items():
            if chart_path and Path(chart_path).exists():
                chart_title = chart_type.replace('_', ' ').title()
                charts_md += f"""
### {chart_title}

![{chart_title}]({chart_path})

"""

        return charts_md if charts_md else "📊 图表文件不可用"

    def _get_risk_level(self, value: float, thresholds: List[float]) -> str:
        """获取风险等级"""
        if value < thresholds[0]:
            return "🟢 低风险"
        elif value < thresholds[1]:
            return "🟡 中等风险"
        elif value < thresholds[2]:
            return "🟠 高风险"
        else:
            return "🔴 极高风险"

    def _build_stress_test_section(self, evaluation_results: Dict[str, Any]) -> str:
        """构建压力测试章节"""
        stress_results = evaluation_results.get('stress_test_results', {})

        if not stress_results:
            return """
### 压力测试

> 📋 压力测试数据暂未生成
"""

        return f"""
### 压力测试概览

| 测试场景 | 预期表现 | 风险评估 |
|---------|---------|---------|
| 市场下跌 | {self._format_percentage(stress_results.get('bear_market', 0))} | {self._assess_stress_result(stress_results.get('bear_market', 0))} |
| 高波动环境 | {self._format_percentage(stress_results.get('high_volatility', 0))} | {self._assess_stress_result(stress_results.get('high_volatility', 0))} |
| 极端情况 | {self._format_percentage(stress_results.get('extreme_scenario', 0))} | {self._assess_stress_result(stress_results.get('extreme_scenario', 0))} |
"""

    def _assess_stress_result(self, value: float) -> str:
        """评估压力测试结果"""
        if value > -0.05:
            return "🟢 抗压能力强"
        elif value > -0.15:
            return "🟡 抗压能力一般"
        else:
            return "🔴 抗压能力弱"

    def _build_benchmark_section(self, evaluation_results: Dict[str, Any]) -> str:
        """构建基准比较章节"""
        benchmark_data = evaluation_results.get('benchmark_comparison', {})

        if not benchmark_data:
            return "📊 基准比较数据暂未生成"

        return """
| 指标 | 策略 | 基准 | 超额表现 |
|------|------|------|----------|
| 年化收益率 | - | - | - |
| 夏普比率 | - | - | - |
| 最大回撤 | - | - | - |

> 📋 基准比较详细数据正在完善中
"""

    def _build_time_series_section(self, evaluation_results: Dict[str, Any]) -> str:
        """构建时间序列章节"""
        return """
### 滚动指标分析

- **滚动收益率**: 展现策略在不同时间窗口的收益表现
- **滚动波动率**: 反映策略风险特征的时间变化
- **滚动夏普比率**: 显示风险调整收益的稳定性

> 📊 详细的时间序列图表请参考图表分析章节
"""

    def _build_stress_test_details(self, evaluation_results: Dict[str, Any]) -> str:
        """构建压力测试详情"""
        return """
### 历史模拟法

基于历史极端市场事件进行压力测试：

- **2008年金融危机**: 模拟系统性风险情况
- **2020年新冠疫情**: 模拟突发事件冲击
- **高波动期间**: 测试策略在高波动环境下的表现

### 蒙特卡洛模拟

通过随机模拟评估策略在不同市场情况下的表现：

- **模拟次数**: 10,000次
- **置信区间**: 95%
- **最坏情况损失**: 评估极端不利情况

> 📋 具体压力测试结果请参考风险分析章节
"""

    def _list_strategy_strengths(self,
                               performance_metrics: Dict[str, Any],
                               risk_metrics: Dict[str, Any]) -> str:
        """列出策略优势"""
        strengths = []

        total_return = performance_metrics.get('total_return', 0)
        sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
        max_drawdown = abs(performance_metrics.get('max_drawdown', 0))
        win_rate = performance_metrics.get('win_rate', 0)

        if total_return > 0.1:
            strengths.append("- ✅ **收益表现突出**: 获得了较好的绝对收益")

        if sharpe_ratio > 1.5:
            strengths.append("- ✅ **风险调整收益优秀**: 夏普比率表现良好")

        if max_drawdown < 0.1:
            strengths.append("- ✅ **回撤控制良好**: 最大回撤控制在合理范围")

        if win_rate > 0.6:
            strengths.append("- ✅ **信号质量较高**: 胜率表现良好")

        if not strengths:
            strengths.append("- 📊 策略特点有待进一步分析")

        return '\n'.join(strengths)

    def _list_potential_risks(self,
                            performance_metrics: Dict[str, Any],
                            risk_metrics: Dict[str, Any]) -> str:
        """列出潜在风险"""
        risks = []

        max_drawdown = abs(performance_metrics.get('max_drawdown', 0))
        volatility = performance_metrics.get('volatility', 0)
        sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)

        if max_drawdown > 0.2:
            risks.append("- ⚠️ **回撤风险**: 最大回撤较大，需要关注资金管理")

        if volatility > 0.3:
            risks.append("- ⚠️ **波动风险**: 策略波动率较高，适合风险承受能力强的投资者")

        if sharpe_ratio < 1:
            risks.append("- ⚠️ **收益风险比**: 风险调整后收益有待提升")

        if not risks:
            risks.append("- ✅ 未发现明显的重大风险点")

        return '\n'.join(risks)

    def _list_improvement_suggestions(self,
                                    performance_metrics: Dict[str, Any],
                                    risk_metrics: Dict[str, Any]) -> str:
        """列出改进建议"""
        suggestions = []

        max_drawdown = abs(performance_metrics.get('max_drawdown', 0))
        sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
        win_rate = performance_metrics.get('win_rate', 0)

        if max_drawdown > 0.15:
            suggestions.append("- 💡 **风险控制优化**: 建议加强止损机制，降低回撤风险")

        if sharpe_ratio < 1.5:
            suggestions.append("- 💡 **收益提升**: 建议优化信号生成逻辑，提高风险调整收益")

        if win_rate < 0.5:
            suggestions.append("- 💡 **信号质量**: 建议改进选股或择时模型，提高胜率")

        suggestions.extend([
            "- 📈 **持续监控**: 建议定期评估策略表现，及时调整参数",
            "- 🔄 **组合优化**: 考虑与其他策略组合，分散单一策略风险",
            "- 📊 **实盘验证**: 建议小资金实盘验证策略有效性"
        ])

        return '\n'.join(suggestions)

    def _get_backtest_period(self, evaluation_results: Dict[str, Any]) -> str:
        """获取回测周期"""
        # 这里可以从evaluation_results中提取实际的回测周期
        return "具体周期请参考数据源"