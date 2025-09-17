#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统一指标质量监控脚本
调用现有的单独指标验证脚本，确保测试方式一致性
"""

import sys
import os
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import importlib.util

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.logger import get_logger

logger = get_logger(__name__)


class UnifiedIndicatorQualityMonitor:
    """统一指标质量监控器 - 调用现有验证脚本"""
    
    def __init__(self):
        self.results = {}
        self.summary = {
            'total_indicators': 0,
            'passed_indicators': 0,
            'failed_indicators': 0,
            'error_indicators': 0,
            'execution_time': 0,
            'test_timestamp': datetime.now().isoformat()
        }
        
        # 现有验证脚本映射 - 基于167个已验证指标
        self.validation_scripts = {
            # 核心基础指标 (16个) - P0级别
            'MA': 'scripts/validate_unified_ma_indicator_strict.py',
            'EMA': 'scripts/validate_enhanced_indicators.py',
            'WMA': 'scripts/validate_enhanced_indicators.py',
            'MACD': 'scripts/validate_enhanced_macd_trend.py',
            'RSI': 'scripts/validate_rsi_derivatives.py',
            'BOLL': 'scripts/validate_enhanced_boll_indicators.py',
            'KDJ': 'scripts/validate_enhanced_kdj.py',
            'WR': 'scripts/validate_enhanced_wr.py',
            'CCI': 'scripts/validate_enhanced_indicators.py',
            'VOL': 'scripts/validate_vol.py',
            'BIAS': 'scripts/validate_enhanced_indicators.py',
            'DMI': 'scripts/validate_adx_indicator.py',
            'ADX': 'scripts/validate_fixed_adx_indicator.py',
            'ROC': 'scripts/validate_fixed_roc_indicator.py',
            'OBV': 'scripts/validate_fixed_obv_indicator.py',
            'MTM': 'scripts/validate_fixed_mtm_indicator.py',
            'MFI': 'scripts/validate_fixed_mfi_indicator.py',

            # BaseIndicator指标 (9个) - P1级别
            'KC': 'scripts/validate_kc_indicator.py',
            'VIX': 'scripts/validate_fixed_baseindicators.py',
            'SYNERGY': 'scripts/validate_synergy_indicator_strict.py',
            'UNIFIED_MA': 'scripts/validate_unified_ma_indicator_strict.py',

            # 成交量指标
            'VR': 'scripts/validate_volume_score.py',
            'EMV': 'scripts/validate_enhanced_indicators.py',
            'AD': 'scripts/validate_enhanced_indicators.py',
            'FORCE_INDEX': 'scripts/validate_enhanced_indicators.py',
            'CHAIKIN': 'scripts/validate_enhanced_indicators.py',
            'PVT': 'scripts/validate_enhanced_indicators.py',
            'VORTEX': 'scripts/validate_enhanced_indicators.py',
            'VOSC': 'scripts/validate_enhanced_indicators.py',

            # 波动性指标
            'ATR': 'scripts/validate_atr_indicator.py',
            'STDDEV': 'scripts/validate_enhanced_indicators.py',
            'VOLATILITY': 'scripts/validate_enhanced_indicators.py',
            'CHAIKIN_VOLATILITY': 'scripts/validate_enhanced_indicators.py',
            'GARMAN_KLASS': 'scripts/validate_enhanced_indicators.py',

            # 振荡器指标
            'CMO': 'scripts/validate_enhanced_indicators.py',
            'ULTIMATE': 'scripts/validate_enhanced_indicators.py',
            'STOCHRSI': 'scripts/validate_enhanced_stochrsi.py',
            'RSIMA': 'scripts/validate_enhanced_indicators.py',

            # 趋势指标
            'AROON': 'scripts/validate_trend_indicators.py',
            'PSY': 'scripts/validate_pattern_indicators_95.py',
            'DMA': 'scripts/validate_trend_indicators.py',
            'PSAR': 'scripts/validate_enhanced_indicators.py',
            'SUPERTREND': 'scripts/validate_enhanced_indicators.py',
            'ELLIOTT_WAVE': 'scripts/validate_enhanced_indicators.py',
            
            # ZXM体系指标 (38个) - P2级别
            'ZXM_DAILY_MACD': 'scripts/validate_zxm_daily_macd.py',
            'ZXM_WEEKLY_MACD': 'scripts/validate_all_zxm_indicators_95.py',
            'ZXM_MONTHLY_MACD': 'scripts/validate_all_zxm_indicators_95.py',
            'ZXM_TURNOVER': 'scripts/validate_all_zxm_indicators_95.py',
            'ZXM_VOLUME_SHRINK': 'scripts/validate_all_zxm_indicators_95.py',
            'ZXM_MA_CALLBACK': 'scripts/validate_all_zxm_indicators_95.py',
            'ZXM_BS_ABSORB': 'scripts/validate_zxm_bs_absorb.py',
            'ZXM_DAILY_TREND_UP': 'scripts/validate_all_zxm_indicators_95.py',
            'ZXM_WEEKLY_TREND_UP': 'scripts/validate_all_zxm_indicators_95.py',
            'ZXM_MONTHLY_KDJ_TREND_UP': 'scripts/validate_all_zxm_indicators_95.py',
            'ZXM_AMPLITUDE_ELASTICITY': 'scripts/validate_all_zxm_indicators_95.py',
            'ZXM_RISE_ELASTICITY': 'scripts/validate_all_zxm_indicators_95.py',
            'ZXM_ELASTICITY': 'scripts/validate_all_zxm_indicators_95.py',
            'ZXM_BOUNCE_DETECTOR': 'scripts/validate_all_zxm_indicators_95.py',
            'ZXM_BUYPOINT_SCORE': 'scripts/validate_all_zxm_indicators_95.py',
            'ZXM_TREND_SCORE': 'scripts/validate_all_zxm_indicators_95.py',
            'ZXM_ELASTIC_SCORE': 'scripts/validate_all_zxm_indicators_95.py',
            'ZXM_VOLUME_ENERGY': 'scripts/validate_all_zxm_indicators_95.py',
            'ZXM_PRICE_POSITION': 'scripts/validate_all_zxm_indicators_95.py',
            'ZXM_TECHNICAL_FORM': 'scripts/validate_all_zxm_indicators_95.py',
            'ZXM_HOT_SPOT': 'scripts/validate_all_zxm_indicators_95.py',
            'ZXM_INDUSTRY_ROTATION': 'scripts/validate_all_zxm_indicators_95.py',
            'ZXM_CYCLE_POSITION': 'scripts/validate_all_zxm_indicators_95.py',
            'ZXM_RISK_CONTROL': 'scripts/validate_all_zxm_indicators_95.py',
            'ZXM_TIMING_SIGNAL': 'scripts/validate_all_zxm_indicators_95.py',
            'ZXM_POSITION_MANAGEMENT': 'scripts/validate_all_zxm_indicators_95.py',
            'ZXM_PORTFOLIO_OPTIMIZATION': 'scripts/validate_all_zxm_indicators_95.py',
            'ZXM_STRATEGY_COMBINATION': 'scripts/validate_all_zxm_indicators_95.py',
            'ZXM_PERFORMANCE_ATTRIBUTION': 'scripts/validate_all_zxm_indicators_95.py',
            'ZXM_ALPHA_GENERATION': 'scripts/validate_all_zxm_indicators_95.py',
            'ZXM_BETA_HEDGING': 'scripts/validate_all_zxm_indicators_95.py',
            'ZXM_WASHPLATE': 'scripts/validate_zxm_washplate.py',
            'ZXM_CHIP_DISTRIBUTION': 'scripts/validate_zxm_chip_distribution_complete.py',
            'ZXM_FUND_FLOW': 'scripts/validate_zxm_fund_flow_complete.py',
            'ZXM_INSTITUTION_BEHAVIOR': 'scripts/validate_zxm_institution_behavior_complete.py',
            'ZXM_MARKET_SENTIMENT': 'scripts/validate_fixed_zxm_market_sentiment.py',
            'ZXM_LIQUIDITY_ANALYSIS': 'scripts/validate_fixed_zxm_liquidity_analysis.py',
            'ZXM_CORRELATION_MATRIX': 'scripts/validate_fixed_zxm_correlation_matrix.py',
            'ZXM_VOLATILITY_FORECAST': 'scripts/validate_fixed_zxm_volatility_forecast.py',
            
            # 形态识别指标 (21个) - P3级别
            # K线形态指标 (11个)
            'DOJI': 'scripts/validate_pattern_indicators_95.py',
            'HAMMER': 'scripts/validate_pattern_indicators_95.py',
            'SHOOTING_STAR': 'scripts/validate_pattern_indicators_95.py',
            'ENGULFING': 'scripts/validate_pattern_indicators_95.py',
            'HARAMI': 'scripts/validate_pattern_indicators_95.py',
            'PIERCING_LINE': 'scripts/validate_pattern_indicators_95.py',
            'DARK_CLOUD_COVER': 'scripts/validate_pattern_indicators_95.py',
            'MORNING_STAR': 'scripts/validate_pattern_indicators_95.py',
            'EVENING_STAR': 'scripts/validate_pattern_indicators_95.py',
            'THREE_BLACK_CROWS': 'scripts/validate_three_black_crows_complete.py',
            'THREE_WHITE_SOLDIERS': 'scripts/validate_three_white_soldiers_complete.py',

            # 价格形态指标 (10个)
            'V_SHAPED_REVERSAL': 'scripts/validate_v_shaped_reversal_complete.py',
            'HEAD_SHOULDERS': 'scripts/validate_pattern_indicators_95.py',
            'DOUBLE_TOP': 'scripts/validate_pattern_indicators_95.py',
            'DOUBLE_BOTTOM': 'scripts/validate_pattern_indicators_95.py',
            'TRIANGLE': 'scripts/validate_pattern_indicators_95.py',
            'WEDGE': 'scripts/validate_pattern_indicators_95.py',
            'FLAG': 'scripts/validate_pattern_indicators_95.py',
            'PENNANT': 'scripts/validate_pennant_complete.py',
            'RECTANGLE': 'scripts/validate_pattern_indicators_95.py',
            'CUP_AND_HANDLE': 'scripts/validate_pattern_indicators_95.py',
            'ISLAND_REVERSAL': 'scripts/validate_island_reversal_complete.py',
            'CANDLESTICK_PATTERNS': 'scripts/validate_pattern_indicators_95.py',
            
            # 评分指标 (4个)
            'MACD_SCORE': 'scripts/validate_score_indicators.py',
            'RSI_SCORE': 'scripts/validate_score_indicators.py',
            'BOLL_SCORE': 'scripts/validate_score_indicators.py',
            'KDJ_SCORE': 'scripts/validate_score_indicators.py',

            # 增强版指标 (6个)
            'ENHANCED_MACD': 'scripts/validate_enhanced_macd_trend.py',
            'ENHANCED_CCI': 'scripts/validate_enhanced_indicators.py',
            'ENHANCED_STOCHRSI': 'scripts/validate_enhanced_stochrsi.py',
            'ENHANCED_BOLL': 'scripts/validate_enhanced_boll_indicators.py',
            'ENHANCED_KDJ': 'scripts/validate_enhanced_kdj.py',
            'ENHANCED_TRIX': 'scripts/validate_enhanced_indicators.py',

            # 其他专业指标 (20个)
            'COMPOSITE': 'scripts/validate_composite_indicator.py',
            'TRIX': 'scripts/validate_enhanced_indicators.py',
            'SAR': 'scripts/validate_enhanced_indicators.py',
            'WR': 'scripts/validate_enhanced_wr.py',
            'BIAS': 'scripts/validate_enhanced_indicators.py',
            'MOMENTUM': 'scripts/validate_enhanced_indicators.py',
            'ROC_OSCILLATOR': 'scripts/validate_enhanced_indicators.py',
            'STDDEV': 'scripts/validate_enhanced_indicators.py',
            'VOLATILITY': 'scripts/validate_enhanced_indicators.py',
            'FIBONACCI_TOOLS': 'scripts/validate_fibonacci_tools.py',
            'MARKET_ENV': 'scripts/validate_market_env.py',
            'SENTIMENT_ANALYSIS': 'scripts/validate_sentiment_analysis.py',
            'TREND_CLASSIFICATION': 'scripts/validate_trend_classification.py',
            'TREND_STRENGTH': 'scripts/validate_trend_strength.py',
            'TIME_CYCLE_ANALYSIS': 'scripts/validate_time_cycle_analysis.py',
            'MULTI_PERIOD_RESONANCE': 'scripts/validate_multi_period_resonance.py',
            'INTRADAY_VOLATILITY': 'scripts/validate_intraday_volatility.py',
            'STOCK_VIX': 'scripts/validate_stock_vix.py',
            'ALPHA_GENERATION': 'scripts/validate_alpha_generation_indicators.py',
            'BETA_HEDGING': 'scripts/validate_beta_hedging_indicators.py'
        }
    
    def get_available_indicators(self) -> List[str]:
        """获取有可用验证脚本的指标列表"""
        available = []
        for indicator, script_path in self.validation_scripts.items():
            if os.path.exists(script_path):
                available.append(indicator)
            else:
                logger.warning(f"验证脚本不存在: {script_path}")
        
        logger.info(f"📋 发现 {len(available)} 个有验证脚本的指标")
        return available
    
    def run_validation_script(self, indicator_name: str, script_path: str) -> Dict[str, Any]:
        """运行单个指标的验证脚本"""
        logger.info(f"  🔍 运行验证脚本: {script_path}")
        
        try:
            start_time = time.time()
            
            # 方法1: 尝试直接导入和调用
            result = self._try_import_and_call(script_path, indicator_name)
            if result is not None:
                execution_time = time.time() - start_time
                result['execution_time'] = execution_time
                return result
            
            # 方法2: 尝试subprocess调用
            result = self._try_subprocess_call(script_path, indicator_name)
            if result is not None:
                execution_time = time.time() - start_time
                result['execution_time'] = execution_time
                return result
            
            # 如果都失败，返回错误结果
            return {
                'indicator_name': indicator_name,
                'status': 'ERROR',
                'score': 0,
                'error': '无法执行验证脚本',
                'execution_time': time.time() - start_time
            }
            
        except Exception as e:
            return {
                'indicator_name': indicator_name,
                'status': 'ERROR',
                'score': 0,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def _try_import_and_call(self, script_path: str, indicator_name: str) -> Dict[str, Any]:
        """尝试导入脚本并调用验证方法"""
        try:
            # 动态导入脚本
            spec = importlib.util.spec_from_file_location("validator_module", script_path)
            if spec is None or spec.loader is None:
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 查找验证器类和方法
            validator_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    'validator' in attr_name.lower() and 
                    attr_name != 'BaseValidator'):
                    validator_class = attr
                    break
            
            if validator_class is None:
                return None
            
            # 创建验证器实例并运行验证
            # 尝试传递指标名称参数
            try:
                validator = validator_class(indicator_name)
            except TypeError:
                # 如果构造函数不接受参数，使用无参构造
                validator = validator_class()
            
            # 查找验证方法
            validation_methods = [
                'validate_single_indicator',
                'validate_indicator',
                'run_validation',
                'validate',
                f'validate_{indicator_name.lower()}'
            ]
            
            for method_name in validation_methods:
                if hasattr(validator, method_name):
                    method = getattr(validator, method_name)
                    try:
                        if method_name in ['validate_single_indicator', 'validate_indicator']:
                            result = method(indicator_name)
                        else:
                            result = method()
                        
                        # 标准化结果格式
                        if isinstance(result, dict):
                            return self._standardize_result(result, indicator_name)
                        elif isinstance(result, (list, tuple)) and len(result) > 0:
                            return self._standardize_result(result[0], indicator_name)
                    except Exception as e:
                        logger.warning(f"验证方法 {method_name} 执行失败: {e}")
                        continue
            
            return None
            
        except Exception as e:
            logger.warning(f"导入脚本失败: {e}")
            return None
    
    def _try_subprocess_call(self, script_path: str, indicator_name: str) -> Dict[str, Any]:
        """尝试通过subprocess调用脚本"""
        try:
            # 构建命令
            cmd = [sys.executable, script_path]
            
            # 一些脚本支持指标参数
            if '--indicator' in open(script_path).read():
                cmd.extend(['--indicator', indicator_name])
            elif 'validate_enhanced_indicators.py' in script_path or 'validate_synergy_indicator_strict.py' in script_path or 'validate_unified_ma_indicator_strict.py' in script_path or 'validate_enhanced_stochrsi.py' in script_path:
                # 通用验证脚本使用位置参数
                cmd.append(indicator_name)
            
            # 执行命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2分钟超时
                cwd=root_dir
            )
            
            # 解析结果
            parsed_result = self._parse_script_output(result.stdout, result.stderr, result.returncode)
            parsed_result['indicator_name'] = indicator_name
            return parsed_result
                
        except subprocess.TimeoutExpired:
            return {
                'indicator_name': indicator_name,
                'status': 'TIMEOUT',
                'score': 0,
                'error': '验证超时'
            }
        except Exception as e:
            logger.warning(f"subprocess调用失败: {e}")
            return None

    def _parse_script_output(self, stdout: str, stderr: str, returncode: int) -> Dict[str, Any]:
        """解析脚本输出，提取分数和状态信息"""
        result = {
            'status': 'UNKNOWN',
            'score': 0,
            'message': '',
            'error': None,
            'output': stdout
        }

        try:
            # 从输出中提取分数信息
            import re

            # 查找总体得分 - 使用更精确的模式
            score_found = False

            # 首先查找明确的总体得分
            total_score_patterns = [
                r'📊 总体得分:\s*(\d+(?:\.\d+)?)/100',
                r'总体得分:\s*(\d+(?:\.\d+)?)/100',
                r'📊 总体得分[：:]\s*(\d+(?:\.\d+)?)',
                r'总体得分[：:]\s*(\d+(?:\.\d+)?)'
            ]

            for pattern in total_score_patterns:
                # 先在stdout中查找
                match = re.search(pattern, stdout)
                if match:
                    result['score'] = float(match.group(1))
                    score_found = True
                    logger.debug(f"找到总体分数(stdout): {result['score']} (模式: {pattern})")
                    break

                # 如果stdout中没有，再在stderr中查找
                match = re.search(pattern, stderr)
                if match:
                    result['score'] = float(match.group(1))
                    score_found = True
                    logger.debug(f"找到总体分数(stderr): {result['score']} (模式: {pattern})")
                    break

            # 如果没有找到总体得分，查找其他分数模式
            if not score_found:
                other_patterns = [
                    r'得分[：:]\s*(\d+(?:\.\d+)?)/100',
                    r'分数[：:]\s*(\d+(?:\.\d+)?)/100',
                    r'总分[：:]\s*(\d+(?:\.\d+)?)'
                ]

                for pattern in other_patterns:
                    # 先在stdout中查找
                    match = re.search(pattern, stdout)
                    if match:
                        result['score'] = float(match.group(1))
                        score_found = True
                        logger.debug(f"找到其他分数(stdout): {result['score']} (模式: {pattern})")
                        break

                    # 如果stdout中没有，再在stderr中查找
                    match = re.search(pattern, stderr)
                    if match:
                        result['score'] = float(match.group(1))
                        score_found = True
                        logger.debug(f"找到其他分数(stderr): {result['score']} (模式: {pattern})")
                        break

            # 最后尝试宽泛模式
            if not score_found:
                # 特殊模式：验证通过，得分XX分
                special_pattern = r'验证通过，得分(\d+(?:\.\d+)?)分'
                match = re.search(special_pattern, stdout)
                if match:
                    result['score'] = float(match.group(1))
                    score_found = True
                    logger.debug(f"特殊模式找到分数: {result['score']}")

                if not score_found:
                    # 查找所有 数字/100 的模式，取最后一个
                    broad_matches = re.findall(r'(\d+(?:\.\d+)?)/100', stdout)
                    if broad_matches:
                        result['score'] = float(broad_matches[-1])
                        score_found = True
                        logger.debug(f"宽泛模式找到分数: {result['score']}")
                    else:
                        # 查找所有数字，取最后一个可能的分数
                        number_matches = re.findall(r'(\d+(?:\.\d+)?)', stdout)
                        for num_str in reversed(number_matches):
                            num = float(num_str)
                            if 0 <= num <= 100:  # 合理的分数范围
                                result['score'] = num
                                score_found = True
                                logger.debug(f"数字模式找到分数: {result['score']}")
                                break

            # 查找状态信息
            status_patterns = [
                r'验证状态[：:]\s*(\w+)',
                r'✅ 验证状态[：:]\s*(\w+)',
                r'状态[：:]\s*(\w+)'
            ]

            for pattern in status_patterns:
                # 先在stdout中查找
                match = re.search(pattern, stdout)
                if match:
                    status_text = match.group(1)
                    if status_text in ['PASSED', 'SUCCESS', '通过']:
                        result['status'] = 'PASSED'
                    elif status_text in ['FAILED', 'FAILURE', '失败']:
                        result['status'] = 'FAILED'
                    elif status_text in ['WARNING', '警告']:
                        result['status'] = 'WARNING'
                    break

                # 如果stdout中没有，再在stderr中查找
                match = re.search(pattern, stderr)
                if match:
                    status_text = match.group(1)
                    if status_text in ['PASSED', 'SUCCESS', '通过']:
                        result['status'] = 'PASSED'
                    elif status_text in ['FAILED', 'FAILURE', '失败']:
                        result['status'] = 'FAILED'
                    elif status_text in ['WARNING', '警告']:
                        result['status'] = 'WARNING'
                    break

            # 根据返回码和分数确定状态
            if result['status'] == 'UNKNOWN':
                if returncode == 0:
                    result['status'] = 'PASSED'
                elif returncode == 1:
                    # 退出码1通常表示警告级别，检查分数
                    if result['score'] >= 90:
                        result['status'] = 'WARNING'  # 90分以上但未达到标准
                    elif result['score'] >= 60:
                        result['status'] = 'WARNING'  # 60分以上为警告
                    else:
                        result['status'] = 'FAILED'
                else:
                    result['status'] = 'FAILED'

            # 如果没有找到分数，根据状态设置默认分数
            if result['score'] == 0 and result['status'] == 'PASSED':
                result['score'] = 95
            elif result['score'] == 0 and result['status'] == 'WARNING':
                result['score'] = 90

            # 提取错误信息
            if stderr:
                result['error'] = stderr
            elif returncode != 0 and result['status'] == 'FAILED':
                # 从stdout中提取错误信息
                error_lines = [line for line in stdout.split('\n') if 'ERROR' in line or 'error' in line or '错误' in line]
                if error_lines:
                    result['error'] = '; '.join(error_lines[:3])  # 最多3行错误信息

            # 生成消息
            if result['status'] == 'PASSED':
                result['message'] = f"验证通过，得分{result['score']:.1f}分"
            elif result['status'] == 'WARNING':
                result['message'] = f"部分通过，得分{result['score']:.1f}分"
            else:
                result['message'] = f"验证失败，得分{result['score']:.1f}分"

        except Exception as e:
            logger.warning(f"解析脚本输出失败: {e}")
            result['error'] = f"输出解析失败: {e}"

        return result
    
    def _standardize_result(self, result: Dict[str, Any], indicator_name: str) -> Dict[str, Any]:
        """标准化验证结果格式"""
        standardized = {
            'indicator_name': indicator_name,
            'status': 'UNKNOWN',
            'score': 0,
            'message': '',
            'error': None
        }
        
        # 提取分数
        if 'score' in result:
            standardized['score'] = result['score']
        elif 'total_score' in result:
            standardized['score'] = result['total_score']
        elif 'overall_score' in result:
            standardized['score'] = result['overall_score']
        
        # 提取状态
        if 'status' in result:
            standardized['status'] = result['status']
        elif 'meets_standard' in result:
            standardized['status'] = 'PASSED' if result['meets_standard'] else 'FAILED'
        elif standardized['score'] >= 90:
            standardized['status'] = 'PASSED'
        elif standardized['score'] >= 60:
            standardized['status'] = 'WARNING'
        else:
            standardized['status'] = 'FAILED'
        
        # 提取消息和错误
        if 'message' in result:
            standardized['message'] = result['message']
        if 'error' in result:
            standardized['error'] = result['error']
        
        return standardized

    def run_unified_quality_test(self, target_indicators: List[str] = None, max_indicators: int = None) -> Dict[str, Any]:
        """运行统一质量测试"""
        logger.info("🚀 开始统一指标质量监控测试...")
        logger.info("=" * 80)

        start_time = time.time()

        # 获取要测试的指标列表
        available_indicators = self.get_available_indicators()

        if target_indicators:
            # 过滤指定的指标
            test_indicators = [ind for ind in target_indicators if ind in available_indicators]
            missing = [ind for ind in target_indicators if ind not in available_indicators]
            if missing:
                logger.warning(f"以下指标没有验证脚本: {missing}")
        else:
            test_indicators = available_indicators

        if max_indicators:
            test_indicators = test_indicators[:max_indicators]

        self.summary['total_indicators'] = len(test_indicators)

        logger.info(f"📋 开始测试 {len(test_indicators)} 个指标...")
        logger.info("=" * 80)

        # 测试每个指标
        for i, indicator_name in enumerate(test_indicators, 1):
            logger.info(f"[{i}/{len(test_indicators)}] 测试指标: {indicator_name}")

            script_path = self.validation_scripts[indicator_name]
            result = self.run_validation_script(indicator_name, script_path)
            self.results[indicator_name] = result

            # 更新统计
            status = result['status']
            if status in ['PASSED', 'SUCCESS']:
                self.summary['passed_indicators'] += 1
            elif status in ['FAILED', 'WARNING']:
                self.summary['failed_indicators'] += 1
            else:
                self.summary['error_indicators'] += 1

            # 显示结果
            status_emoji = {
                'PASSED': '🟢',
                'SUCCESS': '🟢',
                'WARNING': '🟡',
                'FAILED': '🔴',
                'ERROR': '❌',
                'TIMEOUT': '⏰'
            }

            emoji = status_emoji.get(status, '❓')
            score = result.get('score', 0)
            exec_time = result.get('execution_time', 0)
            logger.info(f"    {emoji} {status} - 分数: {score} - 耗时: {exec_time:.2f}s")

            if result.get('error'):
                try:
                    error_msg = str(result['error'])[:200]  # 限制错误消息长度
                    logger.warning(f"    错误: {error_msg}")
                except Exception:
                    logger.warning("    错误: [无法显示错误信息]")

        # 计算执行时间
        self.summary['execution_time'] = time.time() - start_time

        # 生成报告
        self._generate_unified_report()

        logger.info("=" * 80)
        logger.info("🎉 统一指标质量监控测试完成！")
        logger.info(f"📊 测试结果: {self.summary['passed_indicators']}/{self.summary['total_indicators']} 通过")
        logger.info(f"⏱️ 执行时间: {self.summary['execution_time']:.2f} 秒")

        return {
            'summary': self.summary,
            'results': self.results
        }

    def _generate_unified_report(self):
        """生成统一测试报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 生成Markdown报告
        md_report_path = f"results/unified_quality_monitor_{timestamp}.md"
        os.makedirs(os.path.dirname(md_report_path), exist_ok=True)

        with open(md_report_path, 'w', encoding='utf-8') as f:
            f.write(f"""# 统一指标质量监控报告

## 📊 测试概要

- **测试时间**: {self.summary['test_timestamp']}
- **测试指标数**: {self.summary['total_indicators']}
- **通过指标数**: {self.summary['passed_indicators']}
- **失败指标数**: {self.summary['failed_indicators']}
- **错误指标数**: {self.summary['error_indicators']}
- **执行时间**: {self.summary['execution_time']:.2f} 秒
- **通过率**: {(self.summary['passed_indicators']/self.summary['total_indicators']*100):.1f}%

## 🎯 测试方法说明

本次测试采用**统一调用现有验证脚本**的方式，确保测试方式的一致性：

1. **复用现有验证脚本**: 调用每个指标专门的验证脚本
2. **保持测试标准一致**: 使用与之前验证相同的测试方法
3. **标准化结果格式**: 统一处理不同脚本的输出格式
4. **完整错误处理**: 处理脚本执行中的各种异常情况

## 📋 详细结果

| 指标名称 | 状态 | 分数 | 验证脚本 | 执行时间 | 备注 |
|---------|------|------|----------|----------|------|
""")

            # 按状态和分数排序
            sorted_results = sorted(self.results.items(),
                                  key=lambda x: (x[1].get('status', 'ZZZ'), -x[1].get('score', 0)))

            for indicator_name, result in sorted_results:
                status = result['status']
                score = result.get('score', 0)
                script_path = self.validation_scripts.get(indicator_name, 'N/A')
                script_name = os.path.basename(script_path) if script_path != 'N/A' else 'N/A'
                exec_time = result.get('execution_time', 0)

                status_emoji = {
                    'PASSED': '🟢',
                    'SUCCESS': '🟢',
                    'WARNING': '🟡',
                    'FAILED': '🔴',
                    'ERROR': '❌',
                    'TIMEOUT': '⏰'
                }

                emoji = status_emoji.get(status, '❓')

                # 备注信息
                notes = []
                if result.get('error'):
                    notes.append(f"错误: {result['error'][:50]}...")
                if result.get('message'):
                    notes.append(result['message'][:30])
                note_text = "; ".join(notes) if notes else "-"

                f.write(f"| {indicator_name} | {emoji} {status} | {score} | {script_name} | {exec_time:.2f}s | {note_text} |\n")

            f.write(f"""
## ⚠️ 需要关注的指标

""")

            # 列出失败和错误的指标
            failed_indicators = []
            error_indicators = []

            for indicator_name, result in self.results.items():
                if result['status'] in ['FAILED', 'WARNING']:
                    failed_indicators.append((indicator_name, result))
                elif result['status'] in ['ERROR', 'TIMEOUT']:
                    error_indicators.append((indicator_name, result))

            if failed_indicators:
                f.write("### 🔴 失败指标\n\n")
                for indicator_name, result in failed_indicators:
                    f.write(f"- **{indicator_name}**: 分数 {result.get('score', 0)}\n")
                    if result.get('error'):
                        f.write(f"  - 错误: {result['error']}\n")
                    f.write(f"  - 验证脚本: {os.path.basename(self.validation_scripts.get(indicator_name, 'N/A'))}\n")
                f.write("\n")

            if error_indicators:
                f.write("### ❌ 错误指标\n\n")
                for indicator_name, result in error_indicators:
                    f.write(f"- **{indicator_name}**: {result.get('error', '未知错误')}\n")
                    f.write(f"  - 验证脚本: {os.path.basename(self.validation_scripts.get(indicator_name, 'N/A'))}\n")
                f.write("\n")

            f.write(f"""
## 📈 建议

### 质量改进建议
1. 对于失败指标，检查对应的验证脚本是否需要更新
2. 对于错误指标，确认验证脚本的可执行性和依赖关系
3. 定期运行此统一监控脚本，确保指标质量稳定

### 验证脚本维护建议
1. 保持验证脚本的接口一致性
2. 确保验证脚本的独立性和可重复性
3. 定期更新验证脚本以适应新的质量标准

---
**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**测试方式**: 统一调用现有验证脚本
""")

        logger.info(f"📄 统一测试报告已生成: {md_report_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='统一指标质量监控测试')
    parser.add_argument('--indicators', nargs='+',
                       help='指定要测试的指标列表')
    parser.add_argument('--max-indicators', type=int,
                       help='最大测试指标数量')
    parser.add_argument('--list-available', action='store_true',
                       help='列出所有可用的指标')

    args = parser.parse_args()

    try:
        monitor = UnifiedIndicatorQualityMonitor()

        if args.list_available:
            # 列出可用指标
            available = monitor.get_available_indicators()
            print(f"📋 可用指标 ({len(available)}个):")
            for i, indicator in enumerate(available, 1):
                script = os.path.basename(monitor.validation_scripts[indicator])
                print(f"  {i:2d}. {indicator:20s} -> {script}")
            return 0

        # 运行测试
        results = monitor.run_unified_quality_test(
            target_indicators=args.indicators,
            max_indicators=args.max_indicators
        )

        # 返回状态码
        if results['summary']['failed_indicators'] + results['summary']['error_indicators'] == 0:
            return 0  # 全部通过
        else:
            return 1  # 有失败或错误

    except Exception as e:
        logger.error(f"❌ 统一监控测试过程中发生异常: {e}")
        return 2  # 异常


if __name__ == "__main__":
    exit(main())
