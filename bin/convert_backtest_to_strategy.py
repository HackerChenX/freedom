#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
回测结果转换选股策略工具

将回测结果中的有效信号和模式转换为可配置的选股策略
确保保留所有重要指标，特别是ZXM体系指标
"""

import os
import sys
import yaml
import json
import re
import pandas as pd
from datetime import datetime
import argparse

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.logger import get_logger, init_logging
from utils.path_utils import get_result_dir
from indicators.indicator_registry import indicator_registry, IndicatorEnum, STANDARD_PARAMETER_MAPPING

logger = get_logger(__name__)

# 指标映射，将回测中的指标名映射到系统中的指标ID
INDICATOR_MAPPING = {
    # ZXM体系指标
    "ZXM吸筹": IndicatorEnum.ZXM_ABSORB,
    "ZXM换手买点": IndicatorEnum.ZXM_TURNOVER,
    "ZXM日MACD买点": IndicatorEnum.ZXM_DAILY_MACD,
    "ZXM回踩均线买点": IndicatorEnum.ZXM_MA_CALLBACK,
    "ZXM涨幅弹性": IndicatorEnum.ZXM_RISE_ELASTICITY,
    "ZXM振幅弹性": IndicatorEnum.ZXM_AMPLITUDE_ELASTICITY,
    "ZXM弹性满足": IndicatorEnum.ZXM_ELASTICITY_SCORE,
    "ZXM买点满足": IndicatorEnum.ZXM_BUYPOINT_SCORE,
    "ZXM趋势满足": IndicatorEnum.ZXM_DAILY_TREND_UP,
    
    # 常规指标
    "MACD底背离": IndicatorEnum.DIVERGENCE,  # 特殊处理
    "KDJ金叉": IndicatorEnum.KDJ,  # 特殊处理
    "均线多头": IndicatorEnum.MA,  # 特殊处理
    "BOLL中轨突破": IndicatorEnum.BOLL,  # 特殊处理
    "VR上穿均线": IndicatorEnum.VR,  # 特殊处理
    "OBV上穿均线": IndicatorEnum.OBV,  # 特殊处理
}

# 默认参数，使用标准参数映射替代
DEFAULT_PARAMETERS = {
    IndicatorEnum.ZXM_ABSORB: {
        "absorb_threshold": 3
    },
    IndicatorEnum.ZXM_TURNOVER: {
        "threshold": 1.0
    },
    IndicatorEnum.ZXM_DAILY_MACD: {
        "threshold": 0.0
    },
    IndicatorEnum.ZXM_MA_CALLBACK: {
        "periods": [20, 30, 60, 120]
    },
    IndicatorEnum.ZXM_RISE_ELASTICITY: {
        "rise_threshold": 1.02
    },
    IndicatorEnum.ZXM_AMPLITUDE_ELASTICITY: {
        "amplitude_threshold": 10.0
    },
    IndicatorEnum.ZXM_ELASTICITY_SCORE: {
        "threshold": 75
    },
    IndicatorEnum.ZXM_BUYPOINT_SCORE: {
        "threshold": 75
    },
    IndicatorEnum.ZXM_DAILY_TREND_UP: {
        "periods": [60, 120]
    },
    IndicatorEnum.DIVERGENCE: {
        "type": "MACD",
        "divergence_type": "BULLISH",
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
    },
    IndicatorEnum.KDJ: {
        "n": 9,
        "m1": 3,
        "m2": 3,
        "cross_type": "GOLDEN_CROSS"
    },
    IndicatorEnum.MA: {
        "periods": [5, 10, 20],
        "is_bullish": True
    },
    IndicatorEnum.BOLL: {
        "period": 20,
        "std_dev": 2,
        "breakout_type": "MIDDLE"
    },
    IndicatorEnum.VR: {
        "period": 26,
        "ma_period": 6,
        "cross_type": "GOLDEN_CROSS"
    },
    IndicatorEnum.OBV: {
        "ma_period": 30,
        "cross_type": "GOLDEN_CROSS"
    }
}

# 周期映射
PERIOD_MAPPING = {
    "日线": "DAILY",
    "周线": "WEEKLY",
    "月线": "MONTHLY",
    "15分钟": "MIN_15",
    "30分钟": "MIN_30",
    "60分钟": "MIN_60"
}

def extract_indicators_from_backtest(backtest_file):
    """从回测文件中提取指标信息"""
    logger.info(f"从回测文件中提取指标信息: {backtest_file}")
    
    with open(backtest_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取最常见的技术形态
    common_patterns = []
    common_patterns_match = re.search(r'最常见的技术形态:(.*?)(?=\n\n|\Z)', content, re.DOTALL)
    
    if common_patterns_match:
        patterns_text = common_patterns_match.group(1)
        # 提取每个形态的信息
        pattern_matches = re.findall(r'(\d+\.\s+([^\(]+)\s+\(频率:\s+(\d+(?:\.\d+)?)%,\s+成功率:\s+(\d+(?:\.\d+)?)%\))', patterns_text)
        
        for _, name, frequency, success_rate in pattern_matches:
            common_patterns.append({
                'name': name.strip(),
                'frequency': float(frequency),
                'success_rate': float(success_rate)
            })
            
    logger.info(f"提取到 {len(common_patterns)} 个常见技术形态")
    
    # 尝试提取指标唯一ID列表
    indicator_ids = []
    indicator_ids_match = re.search(r'命中指标ID列表:(.*?)(?=\n\n|\Z)', content, re.DOTALL)
    
    if indicator_ids_match:
        ids_text = indicator_ids_match.group(1)
        # 提取每个指标ID
        id_matches = re.findall(r'- ([A-Z0-9_]+)(?:\s+\(([^)]+)\))?', ids_text)
        
        for indicator_id, desc in id_matches:
            indicator_ids.append({
                'id': indicator_id.strip(),
                'description': desc.strip() if desc else None
            })
        
        logger.info(f"直接提取到 {len(indicator_ids)} 个指标ID")
    
    # 提取跨周期共性特征
    cross_period_patterns = []
    cross_period_match = re.search(r'跨周期共性特征:(.*?)(?=\n\n|\Z)', content, re.DOTALL)
    
    if cross_period_match:
        cross_text = cross_period_match.group(1)
        # 提取跨周期形态
        cross_matches = re.findall(r'- ([^:]+):\s+([^\n]+)', cross_text)
        
        for period_desc, patterns in cross_matches:
            patterns_list = [p.strip() for p in patterns.split(',')]
            cross_period_patterns.append({
                'description': period_desc.strip(),
                'patterns': patterns_list
            })
    
    logger.info(f"提取到 {len(cross_period_patterns)} 个跨周期共性特征")
    
    # 提取各周期详细分析
    period_analysis = {}
    period_match = re.search(r'各周期详细分析:(.*?)(?=\n\n|\Z)', content, re.DOTALL)
    
    if period_match:
        period_text = period_match.group(1)
        
        # 提取每个周期的分析
        for period_name in PERIOD_MAPPING.keys():
            period_pattern = re.search(f'{period_name}[^\n]*\n(.*?)(?=\n\n|\Z)', period_text, re.DOTALL)
            if period_pattern:
                period_content = period_pattern.group(1)
                
                # 提取指标列表和评分
                indicators = []
                indicator_matches = re.findall(r'- ([^:]+):\s+评分 (\d+)', period_content)
                
                for indicator_name, score in indicator_matches:
                    indicators.append({
                        'name': indicator_name.strip(),
                        'score': int(score)
                    })
                
                period_analysis[period_name] = indicators
    
    logger.info(f"提取到 {len(period_analysis)} 个周期的详细分析")
    
    return {
        'common_patterns': common_patterns,
        'indicator_ids': indicator_ids,  # 新增直接提取的指标ID列表
        'cross_period_patterns': cross_period_patterns,
        'period_analysis': period_analysis
    }

def extract_parameters_from_source(source_file):
    """从源策略文件中提取指标参数"""
    logger.info(f"从源策略文件中提取指标参数: {source_file}")
    
    parameters = {}
    
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取各种指标的参数
        # ZXM吸筹
        zxm_absorb_match = re.search(r'ZXM吸筹\([^\)]+\)', content)
        if zxm_absorb_match:
            param_match = re.search(r'threshold=(\d+(?:\.\d+)?)', zxm_absorb_match.group(0))
            if param_match:
                parameters[IndicatorEnum.ZXM_ABSORB] = {
                    'threshold': float(param_match.group(1))
                }
        
        # KDJ参数
        kdj_match = re.search(r'KDJ\((\d+),\s*(\d+),\s*(\d+)\)', content)
        if kdj_match:
            parameters[IndicatorEnum.KDJ] = {
                'n': int(kdj_match.group(1)),
                'm1': int(kdj_match.group(2)),
                'm2': int(kdj_match.group(3)),
                'signal_type': 'GOLDEN_CROSS'
            }
        
        # MACD参数
        macd_match = re.search(r'MACD\((\d+),\s*(\d+),\s*(\d+)\)', content)
        if macd_match:
            parameters[IndicatorEnum.DIVERGENCE] = {
                'indicator_type': 'MACD',
                'divergence_type': 'BULLISH',
                'fast_period': int(macd_match.group(1)),
                'slow_period': int(macd_match.group(2)),
                'signal_period': int(macd_match.group(3))
            }
        
        # MA参数
        ma_match = re.search(r'MA\((\d+),\s*(\d+),\s*(\d+)\)', content)
        if ma_match:
            parameters[IndicatorEnum.MA] = {
                'periods': [int(ma_match.group(1)), int(ma_match.group(2)), int(ma_match.group(3))],
                'bullish_alignment': True
            }
        
        # 其他参数的提取...
        
        logger.info(f"从源文件提取到 {len(parameters)} 个指标的参数")
        
    except Exception as e:
        logger.error(f"提取参数时出错: {e}")
    
    return parameters

def create_strategy_from_backtest(backtest_data, parameters, output_file):
    """根据回测数据创建策略配置"""
    logger.info("开始创建策略配置")
    
    # 创建策略ID和名称
    strategy_id = f"BACKTEST_STRATEGY_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    strategy_name = "回测策略"
    
    # 创建策略条件列表
    conditions = []
    
    # 优先使用直接提取的指标ID列表
    if 'indicator_ids' in backtest_data and backtest_data['indicator_ids']:
        logger.info("使用直接提取的指标ID列表创建策略条件")
        
        for idx, indicator in enumerate(backtest_data['indicator_ids']):
            indicator_id = indicator['id']
            
            # 查找参数
            params = parameters.get(indicator_id, DEFAULT_PARAMETERS.get(indicator_id, {}))
            
            # 应用参数映射
            mapped_params = {}
            parameter_mapping = STANDARD_PARAMETER_MAPPING.get(indicator_id, {})
            
            for param_name, param_value in params.items():
                if param_name in parameter_mapping:
                    mapped_params[parameter_mapping[param_name]] = param_value
                else:
                    mapped_params[param_name] = param_value
            
            # 添加指标条件
            conditions.append({
                'indicator_id': indicator_id,
                'period': 'DAILY',  # 默认使用日线
                'parameters': mapped_params,
                'signal_type': 'BUY'
            })
            
            # 除了最后一个，每个指标后面都添加OR
            if idx < len(backtest_data['indicator_ids']) - 1:
                conditions.append({
                    'logic': 'OR'
                })
    else:
        logger.info("使用常见技术形态创建策略条件")
        
        # 添加ZXM体系指标（优先级最高）
        zxm_indicators = [p for p in backtest_data['common_patterns'] 
                         if any(zxm_key in p['name'] for zxm_key in ['ZXM', 'zxm'])]
        
        for idx, indicator in enumerate(zxm_indicators):
            indicator_name = indicator['name'].strip()
            indicator_id = INDICATOR_MAPPING.get(indicator_name)
            
            if not indicator_id and any(key in indicator_name for key in INDICATOR_MAPPING.keys()):
                # 尝试模糊匹配
                for key in INDICATOR_MAPPING.keys():
                    if key in indicator_name:
                        indicator_id = INDICATOR_MAPPING[key]
                        break
            
            if indicator_id:
                # 查找参数
                params = parameters.get(indicator_id, DEFAULT_PARAMETERS.get(indicator_id, {}))
                
                # 应用参数映射
                mapped_params = {}
                parameter_mapping = STANDARD_PARAMETER_MAPPING.get(indicator_id, {})
                
                for param_name, param_value in params.items():
                    if param_name in parameter_mapping:
                        mapped_params[parameter_mapping[param_name]] = param_value
                    else:
                        mapped_params[param_name] = param_value
                
                # 添加指标条件
                conditions.append({
                    'indicator_id': indicator_id,
                    'period': 'DAILY',  # 默认使用日线
                    'parameters': mapped_params,
                    'signal_type': 'BUY'
                })
                
                # 除了最后一个，每个指标后面都添加OR
                if idx < len(zxm_indicators) - 1:
                    conditions.append({
                        'logic': 'OR'
                    })
        
        # 添加其他高成功率指标
        other_indicators = [p for p in backtest_data['common_patterns'] 
                           if not any(zxm_key in p['name'] for zxm_key in ['ZXM', 'zxm'])
                           and p['success_rate'] >= 70]
        
        for idx, indicator in enumerate(other_indicators):
            indicator_name = indicator['name'].strip()
            indicator_id = INDICATOR_MAPPING.get(indicator_name)
            
            if not indicator_id and any(key in indicator_name for key in INDICATOR_MAPPING.keys()):
                # 尝试模糊匹配
                for key in INDICATOR_MAPPING.keys():
                    if key in indicator_name:
                        indicator_id = INDICATOR_MAPPING[key]
                        break
            
            if indicator_id:
                # 如果已经有ZXM指标，添加OR
                if len(conditions) > 0:
                    conditions.append({
                        'logic': 'OR'
                    })
                
                # 查找参数
                params = parameters.get(indicator_id, DEFAULT_PARAMETERS.get(indicator_id, {}))
                
                # 应用参数映射
                mapped_params = {}
                parameter_mapping = STANDARD_PARAMETER_MAPPING.get(indicator_id, {})
                
                for param_name, param_value in params.items():
                    if param_name in parameter_mapping:
                        mapped_params[parameter_mapping[param_name]] = param_value
                    else:
                        mapped_params[param_name] = param_value
                
                # 添加指标条件
                conditions.append({
                    'indicator_id': indicator_id,
                    'period': 'DAILY',  # 默认使用日线
                    'parameters': mapped_params,
                    'signal_type': 'BUY'
                })
                
                # 除了最后一个，每个指标后面都添加OR
                if idx < len(other_indicators) - 1:
                    conditions.append({
                        'logic': 'OR'
                    })
        
        # 添加跨周期指标
        cross_period_indicators = []
        for cross_group in backtest_data['cross_period_patterns']:
            for pattern in cross_group['patterns']:
                pattern = pattern.strip()
                if pattern not in [i['name'] for i in zxm_indicators + other_indicators]:
                    for key in INDICATOR_MAPPING.keys():
                        if key in pattern:
                            cross_period_indicators.append({
                                'name': pattern,
                                'indicator_id': INDICATOR_MAPPING[key],
                                'success_rate': 75  # 默认成功率
                            })
                            break
        
        for idx, indicator in enumerate(cross_period_indicators):
            # 如果已经有其他指标，添加OR
            if len(conditions) > 0:
                conditions.append({
                    'logic': 'OR'
                })
                
            # 查找参数
            params = parameters.get(indicator['indicator_id'], DEFAULT_PARAMETERS.get(indicator['indicator_id'], {}))
            
            # 应用参数映射
            mapped_params = {}
            parameter_mapping = STANDARD_PARAMETER_MAPPING.get(indicator['indicator_id'], {})
            
            for param_name, param_value in params.items():
                if param_name in parameter_mapping:
                    mapped_params[parameter_mapping[param_name]] = param_value
                else:
                    mapped_params[param_name] = param_value
            
            # 添加指标条件
            conditions.append({
                'indicator_id': indicator['indicator_id'],
                'period': 'DAILY',  # 默认使用日线
                'parameters': mapped_params,
                'signal_type': 'BUY'
            })
            
            # 除了最后一个，每个指标后面都添加OR
            if idx < len(cross_period_indicators) - 1:
                conditions.append({
                    'logic': 'OR'
                })
    
    # 创建策略配置
    strategy_config = {
        'strategy': {
            'id': strategy_id,
            'name': strategy_name,
            'description': f"基于回测结果生成的策略 - 生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            'version': "1.0",
            'author': "system",
            'create_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'conditions': conditions,
            'filters': {
                'market': [],  # 不限制市场
                'industry': [],  # 不限制行业
                'market_cap': {
                    'min': 0,
                    'max': 10000
                },
                'price': {
                    'min': 0,
                    'max': 500
                }
            },
            'sort': [
                {
                    'field': 'signal_strength',
                    'direction': 'DESC'
                },
                {
                    'field': 'market_cap',
                    'direction': 'ASC'
                }
            ]
        }
    }
    
    # 写入配置文件
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(strategy_config, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"策略配置已保存到: {output_file}")
    return strategy_config

def main():
    parser = argparse.ArgumentParser(description="将回测结果转换为选股策略")
    parser.add_argument("-b", "--backtest", required=True, help="回测结果文件路径")
    parser.add_argument("-s", "--source", help="源策略文件路径，用于提取参数")
    parser.add_argument("-o", "--output", help="输出策略配置文件路径")
    parser.add_argument("-i", "--indicator_ids", help="指标ID列表，用逗号分隔，直接指定要使用的指标")
    
    args = parser.parse_args()
    
    # 初始化日志
    init_logging(level="INFO")
    
    # 默认输出文件路径
    if not args.output:
        output_file = os.path.join(
            get_result_dir(), 
            f"backtest_strategy_{datetime.now().strftime('%Y%m%d%H%M%S')}.yaml"
        )
    else:
        output_file = args.output
    
    # 如果直接指定了指标ID列表，使用这些ID创建策略
    if args.indicator_ids:
        indicator_ids = [id.strip() for id in args.indicator_ids.split(',')]
        logger.info(f"使用指定的指标ID列表: {indicator_ids}")
        
        # 创建一个模拟的回测数据结构
        backtest_data = {
            'indicator_ids': [{'id': id, 'description': None} for id in indicator_ids],
            'common_patterns': [],
            'cross_period_patterns': [],
            'period_analysis': {}
        }
    else:
        # 提取回测数据
        backtest_data = extract_indicators_from_backtest(args.backtest)
    
    # 提取参数（如果有源文件）
    parameters = {}
    if args.source:
        parameters = extract_parameters_from_source(args.source)
    
    # 创建策略配置
    strategy_config = create_strategy_from_backtest(backtest_data, parameters, output_file)
    
    logger.info(f"策略生成完成，包含 {len(strategy_config['strategy']['conditions'])} 个条件")
    
    # 输出指标统计
    indicator_count = sum(1 for c in strategy_config['strategy']['conditions'] 
                         if isinstance(c, dict) and 'indicator_id' in c)
    zxm_count = sum(1 for c in strategy_config['strategy']['conditions'] 
                   if isinstance(c, dict) and 'indicator_id' in c and 'ZXM' in c.get('indicator_id', ''))
    
    logger.info(f"策略中包含 {indicator_count} 个指标，其中 {zxm_count} 个ZXM系列指标")
    
    # 验证是否包含必要的ZXM指标
    required_zxm = [IndicatorEnum.ZXM_ABSORB, IndicatorEnum.ZXM_TURNOVER, 
                    IndicatorEnum.ZXM_DAILY_MACD, IndicatorEnum.ZXM_BUYPOINT_SCORE]
    missing_zxm = [zxm for zxm in required_zxm if not any(
        isinstance(c, dict) and c.get('indicator_id') == zxm 
        for c in strategy_config['strategy']['conditions']
    )]
    
    if missing_zxm:
        logger.warning(f"策略缺少以下重要的ZXM指标: {', '.join(missing_zxm)}")
    else:
        logger.info("策略包含所有必要的ZXM指标")

if __name__ == "__main__":
    main() 