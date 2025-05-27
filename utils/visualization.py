"""
可视化工具模块

提供选股结果和技术指标的可视化功能
"""

import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from typing import Dict, List, Optional, Tuple, Any, Union
import io
from datetime import datetime
import base64

from utils.logger import get_logger
from utils.decorators import safe_run, performance_monitor

logger = get_logger(__name__)

# 设置中文字体支持
try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Bitstream Vera Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception as e:
    logger.warning(f"设置matplotlib中文字体失败: {e}")


@performance_monitor(threshold=1.0)
def plot_selection_result_distribution(
    result_df: pd.DataFrame, 
    output_file: Optional[str] = None,
    title: str = "选股结果分布",
    figsize: Tuple[int, int] = (10, 12)
) -> Optional[str]:
    """
    生成选股结果的分布可视化图表
    
    Args:
        result_df: 选股结果DataFrame
        output_file: 输出文件路径，None表示返回base64编码的图像
        title: 图表标题
        figsize: 图表尺寸
        
    Returns:
        如果output_file为None，返回base64编码的图像字符串；否则返回None
        
    Raises:
        ValueError: 输入数据无效
    """
    if result_df is None or len(result_df) == 0:
        raise ValueError("选股结果为空，无法生成可视化")
    
    # 创建图表
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # 子图数量
    subplot_count = 0
    
    # 行业分布
    if 'industry' in result_df.columns:
        subplot_count += 1
        ax1 = fig.add_subplot(3, 1, subplot_count)
        industry_counts = result_df['industry'].value_counts()
        industry_counts.plot(kind='pie', ax=ax1, autopct='%1.1f%%')
        ax1.set_title('行业分布')
        ax1.set_ylabel('')
    
    # 市值分布
    if 'market_cap' in result_df.columns:
        subplot_count += 1
        ax2 = fig.add_subplot(3, 1, subplot_count)
        
        # 创建市值区间
        bins = [0, 50, 100, 200, 500, 1000, 2000, float('inf')]
        labels = ['0-50亿', '50-100亿', '100-200亿', '200-500亿', 
                  '500-1000亿', '1000-2000亿', '2000亿以上']
        
        # 计算市值分布
        result_df['market_cap_range'] = pd.cut(result_df['market_cap'], bins=bins, labels=labels)
        market_cap_counts = result_df['market_cap_range'].value_counts().sort_index()
        
        market_cap_counts.plot(kind='bar', ax=ax2)
        ax2.set_title('市值分布')
        ax2.set_xlabel('市值范围')
        ax2.set_ylabel('股票数量')
    
    # 信号强度分布
    if 'signal_strength' in result_df.columns:
        subplot_count += 1
        ax3 = fig.add_subplot(3, 1, subplot_count)
        
        # 创建信号强度区间
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
        
        # 计算信号强度分布
        result_df['signal_strength_range'] = pd.cut(result_df['signal_strength'], bins=bins, labels=labels)
        signal_strength_counts = result_df['signal_strength_range'].value_counts().sort_index()
        
        signal_strength_counts.plot(kind='bar', ax=ax3)
        ax3.set_title('信号强度分布')
        ax3.set_xlabel('信号强度范围')
        ax3.set_ylabel('股票数量')
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存或返回图像
    if output_file:
        plt.savefig(output_file, dpi=300)
        logger.info(f"图表已保存至: {output_file}")
        plt.close(fig)
        return None
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return img_base64


@performance_monitor(threshold=1.0)
def plot_technical_indicators(
    kline_data: pd.DataFrame, 
    indicators: Dict[str, pd.DataFrame],
    output_file: Optional[str] = None,
    title: str = "技术指标可视化",
    figsize: Tuple[int, int] = (12, 10)
) -> Optional[str]:
    """
    生成技术指标可视化图表
    
    Args:
        kline_data: K线数据DataFrame
        indicators: 指标数据字典，键为指标名称，值为指标DataFrame
        output_file: 输出文件路径，None表示返回base64编码的图像
        title: 图表标题
        figsize: 图表尺寸
        
    Returns:
        如果output_file为None，返回base64编码的图像字符串；否则返回None
        
    Raises:
        ValueError: 输入数据无效
    """
    if kline_data is None or len(kline_data) == 0:
        raise ValueError("K线数据为空，无法生成可视化")
    
    if not indicators:
        raise ValueError("指标数据为空，无法生成可视化")
    
    # 确保日期列为索引
    if 'date' in kline_data.columns:
        kline_data = kline_data.set_index('date')
    
    # 创建图表
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # 创建子图
    n_indicators = len(indicators)
    grid_size = n_indicators + 1  # K线图 + 每个指标一个子图
    
    # K线图
    ax_candle = plt.subplot2grid((grid_size, 1), (0, 0), rowspan=2)
    
    # 绘制K线图
    candle_colors = {
        'up': 'red',
        'down': 'green'
    }
    
    for i in range(len(kline_data)):
        date = kline_data.index[i]
        open_price = kline_data['open'].iloc[i]
        close = kline_data['close'].iloc[i]
        high = kline_data['high'].iloc[i]
        low = kline_data['low'].iloc[i]
        
        color = candle_colors['up'] if close >= open_price else candle_colors['down']
        
        # 绘制实体
        ax_candle.plot([i, i], [open_price, close], color=color, linewidth=6)
        # 绘制影线
        ax_candle.plot([i, i], [low, high], color=color, linewidth=1)
    
    # 设置K线图属性
    ax_candle.set_title('K线图')
    ax_candle.set_ylabel('价格')
    ax_candle.grid(True)
    
    # 绘制指标图
    row = 2  # 从第3行开始（K线图占用前2行）
    
    for name, data in indicators.items():
        ax = plt.subplot2grid((grid_size, 1), (row, 0))
        
        # 获取指标数据中的列
        for column in data.columns:
            if column not in ['date']:
                ax.plot(data[column], label=column)
        
        ax.set_title(name)
        ax.legend(loc='best')
        ax.grid(True)
        
        row += 1
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存或返回图像
    if output_file:
        plt.savefig(output_file, dpi=300)
        logger.info(f"图表已保存至: {output_file}")
        plt.close(fig)
        return None
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return img_base64


@performance_monitor(threshold=1.0)
def plot_strategy_performance(
    performance_data: pd.DataFrame,
    output_file: Optional[str] = None,
    title: str = "策略性能分析",
    figsize: Tuple[int, int] = (12, 10)
) -> Optional[str]:
    """
    生成策略性能分析图表
    
    Args:
        performance_data: 策略性能数据DataFrame，包含日期、选股数量、胜率等
        output_file: 输出文件路径，None表示返回base64编码的图像
        title: 图表标题
        figsize: 图表尺寸
        
    Returns:
        如果output_file为None，返回base64编码的图像字符串；否则返回None
        
    Raises:
        ValueError: 输入数据无效
    """
    if performance_data is None or len(performance_data) == 0:
        raise ValueError("策略性能数据为空，无法生成可视化")
    
    # 确保日期列为索引
    if 'date' in performance_data.columns:
        performance_data = performance_data.set_index('date')
    
    # 创建图表
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # 绘制子图
    # 1. 选股数量趋势
    ax1 = fig.add_subplot(2, 2, 1)
    if 'stock_count' in performance_data.columns:
        performance_data['stock_count'].plot(ax=ax1)
        ax1.set_title('选股数量趋势')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('股票数量')
        ax1.grid(True)
    
    # 2. 胜率趋势
    ax2 = fig.add_subplot(2, 2, 2)
    if 'win_rate' in performance_data.columns:
        performance_data['win_rate'].plot(ax=ax2)
        ax2.set_title('胜率趋势')
        ax2.set_xlabel('日期')
        ax2.set_ylabel('胜率')
        ax2.grid(True)
    
    # 3. 平均收益趋势
    ax3 = fig.add_subplot(2, 2, 3)
    if 'avg_return' in performance_data.columns:
        performance_data['avg_return'].plot(ax=ax3)
        ax3.set_title('平均收益趋势')
        ax3.set_xlabel('日期')
        ax3.set_ylabel('平均收益(%)')
        ax3.grid(True)
    
    # 4. 最大收益和最大亏损
    ax4 = fig.add_subplot(2, 2, 4)
    if 'max_gain' in performance_data.columns and 'max_loss' in performance_data.columns:
        performance_data[['max_gain', 'max_loss']].plot(ax=ax4)
        ax4.set_title('最大收益和最大亏损')
        ax4.set_xlabel('日期')
        ax4.set_ylabel('收益率(%)')
        ax4.grid(True)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存或返回图像
    if output_file:
        plt.savefig(output_file, dpi=300)
        logger.info(f"图表已保存至: {output_file}")
        plt.close(fig)
        return None
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return img_base64


@performance_monitor(threshold=1.0)
def plot_condition_heatmap(
    result_df: pd.DataFrame,
    output_file: Optional[str] = None,
    title: str = "条件满足热力图",
    figsize: Tuple[int, int] = (12, 10)
) -> Optional[str]:
    """
    生成条件满足热力图
    
    Args:
        result_df: 选股结果DataFrame，必须包含satisfied_conditions列
        output_file: 输出文件路径，None表示返回base64编码的图像
        title: 图表标题
        figsize: 图表尺寸
        
    Returns:
        如果output_file为None，返回base64编码的图像字符串；否则返回None
        
    Raises:
        ValueError: 输入数据无效
    """
    if result_df is None or len(result_df) == 0:
        raise ValueError("选股结果为空，无法生成可视化")
    
    if 'satisfied_conditions' not in result_df.columns:
        raise ValueError("选股结果缺少satisfied_conditions列，无法生成热力图")
    
    # 提取所有条件
    all_conditions = set()
    for conditions in result_df['satisfied_conditions']:
        if isinstance(conditions, list):
            all_conditions.update(conditions)
        elif isinstance(conditions, str):
            # 尝试解析JSON字符串
            try:
                import json
                parsed = json.loads(conditions)
                if isinstance(parsed, list):
                    all_conditions.update(parsed)
            except:
                pass
    
    all_conditions = sorted(list(all_conditions))
    
    if not all_conditions:
        raise ValueError("未找到有效的条件数据")
    
    # 创建热力图数据
    heatmap_data = pd.DataFrame(0, index=result_df['stock_code'], columns=all_conditions)
    
    for i, row in result_df.iterrows():
        conditions = row['satisfied_conditions']
        if isinstance(conditions, list):
            for condition in conditions:
                heatmap_data.loc[row['stock_code'], condition] = 1
        elif isinstance(conditions, str):
            # 尝试解析JSON字符串
            try:
                import json
                parsed = json.loads(conditions)
                if isinstance(parsed, list):
                    for condition in parsed:
                        heatmap_data.loc[row['stock_code'], condition] = 1
            except:
                pass
    
    # 创建图表
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # 绘制热力图
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(heatmap_data, cmap='YlGnBu', aspect='auto')
    
    # 设置坐标轴
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns, rotation=45, ha='right')
    
    # 添加颜色条
    plt.colorbar(label='满足条件 (1=是, 0=否)')
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存或返回图像
    if output_file:
        plt.savefig(output_file, dpi=300)
        logger.info(f"图表已保存至: {output_file}")
        plt.close(fig)
        return None
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return img_base64


def create_html_report(
    title: str,
    charts: List[str],
    descriptions: List[str],
    output_file: str
) -> str:
    """
    创建HTML报告
    
    Args:
        title: 报告标题
        charts: 图表的base64编码列表
        descriptions: 图表描述列表
        output_file: 输出文件路径
        
    Returns:
        HTML报告文件路径
    """
    if len(charts) != len(descriptions):
        raise ValueError("图表数量与描述数量不匹配")
    
    # 创建HTML内容
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            h1 {{
                color: #333;
                text-align: center;
            }}
            .chart-container {{
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                margin: 20px 0;
                padding: 20px;
            }}
            .chart-title {{
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 10px;
            }}
            .chart-description {{
                margin-top: 10px;
                color: #666;
            }}
            img {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 0 auto;
            }}
            .footer {{
                text-align: center;
                margin-top: 30px;
                color: #999;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
    """
    
    # 添加图表和描述
    for i, (chart, description) in enumerate(zip(charts, descriptions)):
        html_content += f"""
        <div class="chart-container">
            <div class="chart-title">图表 {i+1}</div>
            <img src="data:image/png;base64,{chart}" alt="图表 {i+1}">
            <div class="chart-description">{description}</div>
        </div>
        """
    
    # 添加页脚
    html_content += f"""
        <div class="footer">
            生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </body>
    </html>
    """
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"HTML报告已保存至: {output_file}")
    return output_file 