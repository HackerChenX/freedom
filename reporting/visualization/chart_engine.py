#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
专业级图表生成引擎

核心功能：
- 高质量金融图表生成（matplotlib + plotly）
- 多种图表类型支持
- 智能缓存系统
- 高DPI输出（>300DPI）
- 中文字体支持
- 自定义主题和样式

支持的图表类型：
- 收益曲线图
- 回撤曲线图
- 风险收益散点图
- 滚动性能指标
- 基准比较图
- 因子分析图表
- 持仓分析图表
"""

import os
import sys
import json
import hashlib
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from utils.decorators import performance_monitor, exception_handler

logger = get_logger(__name__)

# 尝试导入可视化库
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib import font_manager
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logger.warning("matplotlib未安装，部分图表功能不可用")
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
from db.sql_manager import SQLManager, QueryType
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    logger.warning("plotly未安装，交互式图表功能不可用")
    PLOTLY_AVAILABLE = False


class ChartTheme:
    """图表主题管理"""

    PROFESSIONAL_THEME = {
        'figure_size': (12, 8),
        'dpi': 300,
        'background_color': '#ffffff',
        'grid_alpha': 0.3,
        'line_width': 2,
        'marker_size': 6,
        'font_size': 12,
        'title_size': 16,
        'label_size': 10,
        'colors': {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#F18F01',
            'warning': '#C73E1D',
            'accent': '#6A994E',
            'neutral': '#6C757D'
        }
    }

    DARK_THEME = {
        'figure_size': (12, 8),
        'dpi': 300,
        'background_color': '#1e1e1e',
        'grid_alpha': 0.2,
        'line_width': 2,
        'marker_size': 6,
        'font_size': 12,
        'title_size': 16,
        'label_size': 10,
        'colors': {
            'primary': '#4FC3F7',
            'secondary': '#E91E63',
            'success': '#4CAF50',
            'warning': '#FF9800',
            'accent': '#9C27B0',
            'neutral': '#9E9E9E'
        }
    }

    @classmethod
    def get_theme(cls, theme_name: str) -> Dict[str, Any]:
        """获取主题配置"""
        themes = {
            'professional': cls.PROFESSIONAL_THEME,
            'dark': cls.DARK_THEME,
            'light': cls.PROFESSIONAL_THEME  # 默认使用专业主题
        }
        return themes.get(theme_name, cls.PROFESSIONAL_THEME)


class ChartEngine:
    """
    专业级图表生成引擎

    功能：
    1. 多种金融图表类型
    2. 高质量输出（300+ DPI）
    3. 智能缓存机制
    4. 中文字体支持
    5. 自定义主题
    """

    def __init__(self,
                 cache_dir: str = "./cache/charts",
                 config: Optional[Any] = None):
        """
        初始化图表引擎

        Args:
            cache_dir: 缓存目录
            config: 配置对象
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.config = config
        self.cache = {}

        # 设置中文字体
        self._setup_chinese_fonts()

        # 设置图表样式
        self._setup_chart_styles()

        logger.info(f"图表引擎初始化完成，缓存目录: {self.cache_dir}")

    def _setup_chinese_fonts(self):
        """设置中文字体支持"""
        if not MATPLOTLIB_AVAILABLE:
            return

        try:
            # 尝试使用系统中文字体
            chinese_fonts = [
                'SimHei',  # 黑体
                'Microsoft YaHei',  # 微软雅黑
                'Arial Unicode MS',  # Arial Unicode
                'DejaVu Sans',  # 默认字体
            ]

            for font_name in chinese_fonts:
                try:
                    plt.rcParams['font.sans-serif'] = [font_name]
                    plt.rcParams['axes.unicode_minus'] = False
                    # 测试字体
                    fig, ax = plt.subplots(figsize=(1, 1))
                    ax.text(0.5, 0.5, '测试中文', fontsize=12)
                    plt.close(fig)
                    logger.info(f"成功设置中文字体: {font_name}")
                    break
                except Exception:
                    continue
            else:
                logger.warning("未找到合适的中文字体，可能影响中文显示")

        except Exception as e:
            logger.warning(f"设置中文字体失败: {e}")

    def _setup_chart_styles(self):
        """设置图表样式"""
        if not MATPLOTLIB_AVAILABLE:
            return

        try:
            # 设置默认样式
            plt.style.use('default')

            # 自定义样式
            plt.rcParams.update({
                'figure.figsize': [12, 8],
                'figure.dpi': 100,
                'savefig.dpi': 300,
                'figure.facecolor': 'white',
                'axes.facecolor': 'white',
                'axes.edgecolor': '#CCCCCC',
                'axes.linewidth': 1,
                'axes.grid': True,
                'grid.alpha': 0.3,
                'grid.linewidth': 0.5,
                'lines.linewidth': 2,
                'lines.markersize': 6,
                'font.size': 12,
                'axes.titlesize': 16,
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 11,
                'legend.frameon': True,
                'legend.fancybox': True,
                'legend.shadow': True
            })

            logger.info("图表样式设置完成")

        except Exception as e:
            logger.warning(f"设置图表样式失败: {e}")

    @exception_handler(reraise=True)
    @performance_monitor(threshold=10.0)
    def generate_chart(self,
                      chart_spec: Dict[str, Any],
                      data: Dict[str, Any],
                      config: Optional[Any] = None) -> Dict[str, Any]:
        """
        生成图表

        Args:
            chart_spec: 图表规格
            data: 数据
            config: 配置

        Returns:
            Dict[str, Any]: 生成结果
        """
        start_time = time.time()

        chart_type = chart_spec.get('type')
        title = chart_spec.get('title', '图表')

        logger.debug(f"开始生成图表: {title} ({chart_type})")

        try:
            # 检查缓存
            cache_key = self._generate_cache_key(chart_spec, data)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                logger.debug(f"从缓存获取图表: {title}")
                return cached_result

            # 生成图表
            chart_result = self._create_chart(chart_spec, data, config)
            chart_result['generation_time'] = time.time() - start_time
            chart_result['from_cache'] = False

            # 缓存结果
            self._save_to_cache(cache_key, chart_result)

            logger.debug(f"图表生成完成: {title}, 耗时: {chart_result['generation_time']:.2f}秒")

            return chart_result

        except Exception as e:
            logger.error(f"生成图表失败: {title} - {e}")
            raise

    def _create_chart(self,
                     chart_spec: Dict[str, Any],
                     data: Dict[str, Any],
                     config: Optional[Any] = None) -> Dict[str, Any]:
        """创建图表"""
        chart_type = chart_spec.get('type')

        # 根据图表类型调用相应的生成方法
        chart_generators = {
            'returns_curve': self._generate_returns_curve,
            'drawdown_curve': self._generate_drawdown_curve,
            'risk_return_scatter': self._generate_risk_return_scatter,
            'rolling_performance': self._generate_rolling_performance,
            'benchmark_comparison': self._generate_benchmark_comparison,
            'factor_exposure': self._generate_factor_exposure,
            'factor_attribution': self._generate_factor_attribution,
            'position_concentration': self._generate_position_concentration,
            'turnover_rate_analysis': self._generate_turnover_rate_analysis,
            'monthly_returns_heatmap': self._generate_monthly_returns_heatmap,
            'performance_attribution': self._generate_performance_attribution
        }

        generator = chart_generators.get(chart_type)
        if not generator:
            raise ValueError(f"不支持的图表类型: {chart_type}")

        return generator(chart_spec, data, config)

    def _generate_returns_curve(self,
                               chart_spec: Dict[str, Any],
                               data: Dict[str, Any],
                               config: Optional[Any] = None) -> Dict[str, Any]:
        """生成收益率曲线图"""
        if not MATPLOTLIB_AVAILABLE:
            return self._create_placeholder_chart(chart_spec)

        # 获取主题
        theme = ChartTheme.get_theme(getattr(config, 'chart_theme', 'professional'))

        # 创建图表
        fig, ax = plt.subplots(figsize=theme['figure_size'], dpi=theme['dpi'])

        # 模拟数据（实际应从data中提取）
        dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
        strategy_returns = np.cumsum(np.random.normal(0.0008, 0.02, 252))
        benchmark_returns = np.cumsum(np.random.normal(0.0005, 0.015, 252))

        # 绘制曲线
        ax.plot(dates, strategy_returns,
                color=theme['colors']['primary'],
                linewidth=theme['line_width'],
                label='策略收益')
        ax.plot(dates, benchmark_returns,
                color=theme['colors']['secondary'],
                linewidth=theme['line_width'],
                label='基准收益')

        # 设置标题和标签
        ax.set_title(chart_spec.get('title', '累计收益率曲线'),
                    fontsize=theme['title_size'], pad=20)
        ax.set_xlabel('日期', fontsize=theme['label_size'])
        ax.set_ylabel('累计收益率', fontsize=theme['label_size'])

        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)

        # 设置网格和图例
        ax.grid(True, alpha=theme['grid_alpha'])
        ax.legend(loc='upper left')

        # 调整布局
        plt.tight_layout()

        # 保存图表
        filename = f"returns_curve_{int(time.time())}.png"
        filepath = self.cache_dir / filename
        plt.savefig(filepath,
                   dpi=theme['dpi'],
                   bbox_inches='tight',
                   facecolor=theme['background_color'])
        plt.close()

        return {
            'file_path': str(filepath),
            'chart_type': 'returns_curve',
            'title': chart_spec.get('title', '累计收益率曲线'),
            'format': 'png'
        }

    def _generate_drawdown_curve(self,
                                chart_spec: Dict[str, Any],
                                data: Dict[str, Any],
                                config: Optional[Any] = None) -> Dict[str, Any]:
        """生成回撤曲线图"""
        if not MATPLOTLIB_AVAILABLE:
            return self._create_placeholder_chart(chart_spec)

        theme = ChartTheme.get_theme(getattr(config, 'chart_theme', 'professional'))

        fig, ax = plt.subplots(figsize=theme['figure_size'], dpi=theme['dpi'])

        # 模拟回撤数据
        dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
        returns = np.random.normal(0.0008, 0.02, 252)
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max

        # 绘制回撤曲线
        ax.fill_between(dates, drawdown, 0,
                       color=theme['colors']['warning'],
                       alpha=0.7, label='回撤')
        ax.plot(dates, drawdown,
               color=theme['colors']['warning'],
               linewidth=theme['line_width'])

        # 标注最大回撤点
        max_dd_idx = np.argmin(drawdown)
        ax.annotate(f'最大回撤: {drawdown[max_dd_idx]:.2%}',
                   xy=(dates[max_dd_idx], drawdown[max_dd_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        ax.set_title(chart_spec.get('title', '回撤曲线'),
                    fontsize=theme['title_size'], pad=20)
        ax.set_xlabel('日期', fontsize=theme['label_size'])
        ax.set_ylabel('回撤幅度', fontsize=theme['label_size'])

        # 格式化y轴为百分比
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)

        ax.grid(True, alpha=theme['grid_alpha'])
        ax.legend()

        plt.tight_layout()

        filename = f"drawdown_curve_{int(time.time())}.png"
        filepath = self.cache_dir / filename
        plt.savefig(filepath,
                   dpi=theme['dpi'],
                   bbox_inches='tight',
                   facecolor=theme['background_color'])
        plt.close()

        return {
            'file_path': str(filepath),
            'chart_type': 'drawdown_curve',
            'title': chart_spec.get('title', '回撤曲线'),
            'format': 'png'
        }

    def _generate_risk_return_scatter(self,
                                    chart_spec: Dict[str, Any],
                                    data: Dict[str, Any],
                                    config: Optional[Any] = None) -> Dict[str, Any]:
        """生成风险收益散点图"""
        if not MATPLOTLIB_AVAILABLE:
            return self._create_placeholder_chart(chart_spec)

        theme = ChartTheme.get_theme(getattr(config, 'chart_theme', 'professional'))

        fig, ax = plt.subplots(figsize=theme['figure_size'], dpi=theme['dpi'])

        # 模拟多个策略的风险收益数据
        np.random.seed(42)
        n_strategies = 20
        returns = np.random.normal(0.12, 0.08, n_strategies)
        volatilities = np.random.normal(0.15, 0.05, n_strategies)
        sharpe_ratios = returns / volatilities

        # 根据夏普比率设置颜色
        scatter = ax.scatter(volatilities, returns,
                           c=sharpe_ratios,
                           cmap='viridis',
                           s=100, alpha=0.7, edgecolors='black', linewidth=1)

        # 添加颜色条
        cbar = plt.colorbar(scatter)
        cbar.set_label('夏普比率', fontsize=theme['label_size'])

        # 标注当前策略
        current_return = 0.15
        current_vol = 0.12
        ax.scatter(current_vol, current_return,
                  color=theme['colors']['primary'],
                  s=200, marker='*',
                  edgecolors='black', linewidth=2,
                  label='当前策略', zorder=5)

        ax.set_title(chart_spec.get('title', '风险收益散点图'),
                    fontsize=theme['title_size'], pad=20)
        ax.set_xlabel('年化波动率', fontsize=theme['label_size'])
        ax.set_ylabel('年化收益率', fontsize=theme['label_size'])

        # 格式化坐标轴为百分比
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

        ax.grid(True, alpha=theme['grid_alpha'])
        ax.legend()

        plt.tight_layout()

        filename = f"risk_return_scatter_{int(time.time())}.png"
        filepath = self.cache_dir / filename
        plt.savefig(filepath,
                   dpi=theme['dpi'],
                   bbox_inches='tight',
                   facecolor=theme['background_color'])
        plt.close()

        return {
            'file_path': str(filepath),
            'chart_type': 'risk_return_scatter',
            'title': chart_spec.get('title', '风险收益散点图'),
            'format': 'png'
        }

    def _generate_rolling_performance(self,
                                    chart_spec: Dict[str, Any],
                                    data: Dict[str, Any],
                                    config: Optional[Any] = None) -> Dict[str, Any]:
        """生成滚动性能指标图"""
        if not MATPLOTLIB_AVAILABLE:
            return self._create_placeholder_chart(chart_spec)

        theme = ChartTheme.get_theme(getattr(config, 'chart_theme', 'professional'))

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=theme['dpi'])

        # 生成模拟数据
        dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
        returns = np.random.normal(0.0008, 0.02, 252)

        # 计算滚动指标
        window = 60
        rolling_return = pd.Series(returns).rolling(window).mean() * 252
        rolling_vol = pd.Series(returns).rolling(window).std() * np.sqrt(252)
        rolling_sharpe = rolling_return / rolling_vol
        rolling_max_dd = pd.Series(np.cumprod(1 + returns)).rolling(window).apply(
            lambda x: (x.iloc[-1] - x.max()) / x.max()
        )

        # 滚动年化收益率
        ax1.plot(dates[window-1:], rolling_return[window-1:],
                color=theme['colors']['primary'], linewidth=theme['line_width'])
        ax1.set_title('滚动年化收益率', fontsize=theme['title_size'])
        ax1.set_ylabel('年化收益率', fontsize=theme['label_size'])
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        ax1.grid(True, alpha=theme['grid_alpha'])

        # 滚动波动率
        ax2.plot(dates[window-1:], rolling_vol[window-1:],
                color=theme['colors']['secondary'], linewidth=theme['line_width'])
        ax2.set_title('滚动年化波动率', fontsize=theme['title_size'])
        ax2.set_ylabel('年化波动率', fontsize=theme['label_size'])
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        ax2.grid(True, alpha=theme['grid_alpha'])

        # 滚动夏普比率
        ax3.plot(dates[window-1:], rolling_sharpe[window-1:],
                color=theme['colors']['success'], linewidth=theme['line_width'])
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='优秀线(1.0)')
        ax3.set_title('滚动夏普比率', fontsize=theme['title_size'])
        ax3.set_ylabel('夏普比率', fontsize=theme['label_size'])
        ax3.grid(True, alpha=theme['grid_alpha'])
        ax3.legend()

        # 滚动最大回撤
        ax4.fill_between(dates[window-1:], rolling_max_dd[window-1:], 0,
                        color=theme['colors']['warning'], alpha=0.7)
        ax4.plot(dates[window-1:], rolling_max_dd[window-1:],
                color=theme['colors']['warning'], linewidth=theme['line_width'])
        ax4.set_title('滚动最大回撤', fontsize=theme['title_size'])
        ax4.set_ylabel('回撤幅度', fontsize=theme['label_size'])
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        ax4.grid(True, alpha=theme['grid_alpha'])

        # 设置x轴格式
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.suptitle(chart_spec.get('title', '滚动性能指标'),
                    fontsize=theme['title_size'] + 2, y=0.98)
        plt.tight_layout()

        filename = f"rolling_performance_{int(time.time())}.png"
        filepath = self.cache_dir / filename
        plt.savefig(filepath,
                   dpi=theme['dpi'],
                   bbox_inches='tight',
                   facecolor=theme['background_color'])
        plt.close()

        return {
            'file_path': str(filepath),
            'chart_type': 'rolling_performance',
            'title': chart_spec.get('title', '滚动性能指标'),
            'format': 'png'
        }

    def _generate_monthly_returns_heatmap(self,
                                        chart_spec: Dict[str, Any],
                                        data: Dict[str, Any],
                                        config: Optional[Any] = None) -> Dict[str, Any]:
        """生成月度收益热力图"""
        if not MATPLOTLIB_AVAILABLE:
            return self._create_placeholder_chart(chart_spec)

        theme = ChartTheme.get_theme(getattr(config, 'chart_theme', 'professional'))

        fig, ax = plt.subplots(figsize=(14, 8), dpi=theme['dpi'])

        # 生成模拟月度收益数据
        years = [2021, 2022, 2023]
        months = ['1月', '2月', '3月', '4月', '5月', '6月',
                 '7月', '8月', '9月', '10月', '11月', '12月']

        np.random.seed(42)
        monthly_returns = np.random.normal(0.02, 0.08, (len(years), len(months)))

        # 创建DataFrame
        df = pd.DataFrame(monthly_returns, index=years, columns=months)

        # 创建热力图
        im = ax.imshow(df.values, cmap='RdYlGn', aspect='auto', vmin=-0.2, vmax=0.2)

        # 设置坐标轴
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months)
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels(years)

        # 添加数值标签
        for i in range(len(years)):
            for j in range(len(months)):
                value = df.iloc[i, j]
                color = 'white' if abs(value) > 0.1 else 'black'
                ax.text(j, i, f'{value:.1%}',
                       ha='center', va='center',
                       color=color, fontweight='bold')

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('月度收益率', fontsize=theme['label_size'])
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

        ax.set_title(chart_spec.get('title', '月度收益热力图'),
                    fontsize=theme['title_size'], pad=20)

        plt.tight_layout()

        filename = f"monthly_returns_heatmap_{int(time.time())}.png"
        filepath = self.cache_dir / filename
        plt.savefig(filepath,
                   dpi=theme['dpi'],
                   bbox_inches='tight',
                   facecolor=theme['background_color'])
        plt.close()

        return {
            'file_path': str(filepath),
            'chart_type': 'monthly_returns_heatmap',
            'title': chart_spec.get('title', '月度收益热力图'),
            'format': 'png'
        }

    # 占位符实现（将在后续完善）
    def _generate_benchmark_comparison(self, chart_spec, data, config):
        """生成基准比较图 - 占位符"""
        return self._create_placeholder_chart(chart_spec)

    def _generate_factor_exposure(self, chart_spec, data, config):
        """生成因子暴露图 - 占位符"""
        return self._create_placeholder_chart(chart_spec)

    def _generate_factor_attribution(self, chart_spec, data, config):
        """生成因子归因图 - 占位符"""
        return self._create_placeholder_chart(chart_spec)

    def _generate_position_concentration(self, chart_spec, data, config):
        """生成持仓集中度图 - 占位符"""
        return self._create_placeholder_chart(chart_spec)

    def _generate_turnover_rate_analysis(self, chart_spec, data, config):
        """生成换手率分析图 - 占位符"""
        return self._create_placeholder_chart(chart_spec)

    def _generate_performance_attribution(self, chart_spec, data, config):
        """生成业绩归因图 - 占位符"""
        return self._create_placeholder_chart(chart_spec)

    def _create_placeholder_chart(self, chart_spec: Dict[str, Any]) -> Dict[str, Any]:
        """创建占位符图表"""
        filename = f"placeholder_{chart_spec.get('type', 'unknown')}_{int(time.time())}.png"
        filepath = self.cache_dir / filename

        if MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"图表生成中...\n{chart_spec.get('title', '未知图表')}",
                   ha='center', va='center', fontsize=16,
                   bbox=dict(boxstyle='round', facecolor='lightgray'))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            # 创建一个简单的文本文件作为占位符
            with open(filepath.with_suffix('.txt'), 'w', encoding='utf-8') as f:
                f.write(f"图表占位符: {chart_spec.get('title', '未知图表')}")
            filepath = filepath.with_suffix('.txt')

        return {
            'file_path': str(filepath),
            'chart_type': chart_spec.get('type', 'placeholder'),
            'title': chart_spec.get('title', '占位符图表'),
            'format': 'png' if MATPLOTLIB_AVAILABLE else 'txt'
        }

    def _generate_cache_key(self, chart_spec: Dict[str, Any], data: Dict[str, Any]) -> str:
        """生成缓存键"""
        key_data = {
            'chart_type': chart_spec.get('type'),
            'title': chart_spec.get('title'),
            'data_hash': hashlib.md5(str(data).encode()).hexdigest()[:8]
        }
        return hashlib.md5(str(key_data).encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """从缓存获取结果"""
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)

                # 检查图表文件是否存在
                chart_file = Path(cached_data.get('file_path', ''))
                if chart_file.exists():
                    cached_data['from_cache'] = True
                    return cached_data

            except Exception as e:
                logger.warning(f"读取缓存失败: {e}")

        return None

    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]):
        """保存到缓存"""
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            cache_data = {
                **result,
                'cached_at': datetime.now().isoformat()
            }

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")

    def clear_cache(self):
        """清理缓存"""
        try:
            for file in self.cache_dir.glob("*"):
                if file.is_file():
                    file.unlink()
            logger.info("图表缓存已清理")
        except Exception as e:
            logger.error(f"清理缓存失败: {e}")