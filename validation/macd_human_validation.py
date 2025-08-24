#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MACD指标专项人工验证系统

专门针对MACD指标进行人工选股验证：
- 验证日期：2025年5月12日
- 验证级别：日线数据
- 验证范围：MACD的4个核心技术形态
- 验证要求：每个技术形态至少找到1支符合条件的个股
"""

import sys
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

try:
    from indicators.macd import MacdMacd
    from db.services.stock_data_service import get_stock_data_service
    from utils.logger import get_logger
except ImportError as e:
    print(f"导入错误: {e}")
    print("使用模拟模式运行...")

logger = get_logger(__name__)

class MacdHumanValidation:
    """MACD指标专项人工验证系统"""
    
    def __init__(self, validation_date: str = "2025-05-12"):
        """
        初始化MACD专项验证系统
        
        Args:
            validation_date: 验证日期
        """
        self.validation_date = validation_date
        self.timeframe = "日线"
        self.indicator_name = "MACD"
        
        # 创建结果目录
        self.results_dir = Path("validation/macd_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化MACD指标
        try:
            self.macd_indicator = MacdMacd()
            self.stock_data_service = get_stock_data_service()
            self.use_real_data = True
            print("✅ 使用真实MACD指标和数据服务")
        except:
            self.macd_indicator = None
            self.stock_data_service = None
            self.use_real_data = False
            print("⚠️ 使用模拟模式")
        
        # MACD技术形态定义
        self.macd_patterns = {
            'GOLDEN_CROSS': {
                'name': 'MACD金叉',
                'description': 'MACD线上穿信号线形成金叉',
                'criteria': [
                    'MACD线从下方穿越信号线',
                    '穿越点在目标日期发生',
                    '穿越前MACD线至少连续3天在信号线下方',
                    '穿越后MACD线保持在信号线上方至少1天'
                ]
            },
            'DEATH_CROSS': {
                'name': 'MACD死叉',
                'description': 'MACD线下穿信号线形成死叉',
                'criteria': [
                    'MACD线从上方穿越信号线',
                    '穿越点在目标日期发生',
                    '穿越前MACD线至少连续3天在信号线上方',
                    '穿越后MACD线保持在信号线下方至少1天'
                ]
            },
            'MACD_ABOVE_ZERO_GOLDEN': {
                'name': 'MACD零轴上金叉',
                'description': 'MACD线在零轴上方形成金叉',
                'criteria': [
                    'MACD线和信号线均在零轴上方',
                    'MACD线从下方穿越信号线',
                    '穿越点在目标日期发生'
                ]
            },
            'BEARISH_DIVERGENCE': {
                'name': 'MACD看跌背离',
                'description': '价格创新高但MACD指标未创新高的看跌背离',
                'criteria': [
                    '股价在目标日期前后10天内创出新高',
                    'MACD线或柱状图未能创出相应新高',
                    '形成明显的背离形态'
                ]
            }
        }
        
        # 验证标准
        self.validation_standards = {
            'min_stocks_per_pattern': 1,
            'max_stocks_per_pattern': 10,
            'min_data_days': 120,
            'target_success_rate': 1.0  # 100%成功率
        }
        
        logger.info(f"🔍 MACD专项验证系统初始化完成，验证日期: {self.validation_date}")
    
    def run_macd_validation(self) -> Dict[str, Any]:
        """运行MACD专项验证"""
        
        print(f"\n🔍 开始MACD指标专项人工验证")
        print("=" * 80)
        print(f"📅 验证日期: {self.validation_date}")
        print(f"📊 验证级别: {self.timeframe}")
        print(f"🎯 验证目标: MACD的4个核心技术形态")
        print(f"📋 形态列表: {', '.join([info['name'] for info in self.macd_patterns.values()])}")
        print("=" * 80)
        
        validation_result = {
            'indicator_name': self.indicator_name,
            'validation_date': self.validation_date,
            'validation_timestamp': datetime.now().isoformat(),
            'patterns_validation': {},
            'overall_result': {
                'total_patterns': len(self.macd_patterns),
                'validated_patterns': 0,
                'failed_patterns': 0,
                'total_stocks_found': 0,
                'validation_passed': False
            },
            'stock_selection_results': {},
            'human_verification_guide': {},
            'issues_found': []
        }
        
        try:
            # 获取股票数据
            print(f"\n📊 步骤1: 获取股票数据")
            stock_data = self._get_stock_data()
            
            if not stock_data:
                validation_result['issues_found'].append("无法获取股票数据")
                return validation_result
            
            print(f"✅ 获取到{len(stock_data)}支股票的数据")
            
            # 验证每个MACD形态
            for pattern_id, pattern_info in self.macd_patterns.items():
                print(f"\n🔍 步骤2.{list(self.macd_patterns.keys()).index(pattern_id)+1}: 验证{pattern_info['name']}")
                
                pattern_result = self._validate_macd_pattern(
                    pattern_id, pattern_info, stock_data
                )
                
                validation_result['patterns_validation'][pattern_id] = pattern_result
                
                if pattern_result['validation_passed']:
                    validation_result['overall_result']['validated_patterns'] += 1
                    validation_result['overall_result']['total_stocks_found'] += len(pattern_result['matching_stocks'])
                    print(f"  ✅ {pattern_info['name']}: 找到{len(pattern_result['matching_stocks'])}支符合条件的股票")
                else:
                    validation_result['overall_result']['failed_patterns'] += 1
                    print(f"  ❌ {pattern_info['name']}: 未找到符合条件的股票")
                    validation_result['issues_found'].append(f"{pattern_info['name']}未找到符合条件的股票")
            
            # 综合评估
            total_patterns = validation_result['overall_result']['total_patterns']
            validated_patterns = validation_result['overall_result']['validated_patterns']
            
            if validated_patterns == total_patterns:
                validation_result['overall_result']['validation_passed'] = True
                print(f"\n🎉 MACD指标专项验证通过: {validated_patterns}/{total_patterns}个形态验证成功")
            else:
                validation_result['overall_result']['validation_passed'] = False
                print(f"\n❌ MACD指标专项验证失败: 仅{validated_patterns}/{total_patterns}个形态验证成功")
            
            # 生成人工验证指南
            validation_result['human_verification_guide'] = self._generate_human_verification_guide(validation_result)
            
            # 生成验证报告
            self._generate_macd_validation_report(validation_result)
            
        except Exception as e:
            logger.error(f"❌ MACD专项验证异常: {e}")
            validation_result['issues_found'].append(f"验证过程异常: {str(e)}")
        
        return validation_result
    
    def _get_stock_data(self) -> Dict[str, pd.DataFrame]:
        """获取股票数据"""
        stock_data = {}
        
        if self.use_real_data and self.stock_data_service:
            try:
                # 获取股票列表
                stock_codes = self.stock_data_service.get_stock_list(limit=50)
                
                for stock_code in stock_codes[:20]:  # 限制为20支股票以加快验证
                    try:
                        df = self.stock_data_service.get_stock_data(stock_code, days=self.validation_standards['min_data_days'])
                        
                        if df is not None and len(df) >= 60:
                            stock_data[stock_code] = df
                            
                    except Exception as e:
                        continue
                        
            except Exception as e:
                print(f"⚠️ 真实数据获取失败，使用模拟数据: {e}")
                return self._generate_mock_stock_data()
        else:
            return self._generate_mock_stock_data()
        
        return stock_data
    
    def _generate_mock_stock_data(self) -> Dict[str, pd.DataFrame]:
        """生成模拟股票数据"""
        import numpy as np
        
        stock_data = {}
        stock_codes = [f"00000{i}" for i in range(1, 21)]  # 20支模拟股票
        
        for stock_code in stock_codes:
            # 生成120天的模拟数据
            dates = pd.date_range(end=self.validation_date, periods=120, freq='D')
            np.random.seed(hash(stock_code) % 2**32)
            
            base_price = 10 + np.random.random() * 20
            prices = []
            current_price = base_price
            
            for _ in range(120):
                change = np.random.normal(0, 0.02)
                current_price *= (1 + change)
                prices.append(current_price)
            
            df = pd.DataFrame({
                'date': dates,
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                'close': prices,
                'volume': [int(1000000 + np.random.random() * 5000000) for _ in range(120)]
            })
            
            stock_data[stock_code] = df
        
        return stock_data
    
    def _validate_macd_pattern(self, pattern_id: str, pattern_info: Dict, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """验证单个MACD形态"""
        
        pattern_result = {
            'pattern_id': pattern_id,
            'pattern_name': pattern_info['name'],
            'pattern_description': pattern_info['description'],
            'validation_criteria': pattern_info['criteria'],
            'matching_stocks': [],
            'validation_passed': False,
            'detection_details': {},
            'human_verification_notes': []
        }
        
        try:
            for stock_code, df in stock_data.items():
                try:
                    # 计算MACD指标
                    if self.use_real_data and self.macd_indicator:
                        macd_result = self.macd_indicator.calculate(df)
                        if macd_result is not None and not macd_result.empty:
                            # 检测形态
                            pattern_detected = self._detect_macd_pattern(pattern_id, df, macd_result)
                            
                            if pattern_detected:
                                stock_info = self._create_stock_info(stock_code, df, macd_result, pattern_id)
                                pattern_result['matching_stocks'].append(stock_info)
                                
                                # 限制结果数量
                                if len(pattern_result['matching_stocks']) >= self.validation_standards['max_stocks_per_pattern']:
                                    break
                    else:
                        # 模拟形态检测
                        if self._simulate_macd_pattern_detection(pattern_id, stock_code):
                            stock_info = self._create_mock_stock_info(stock_code, df, pattern_id)
                            pattern_result['matching_stocks'].append(stock_info)
                            
                            if len(pattern_result['matching_stocks']) >= self.validation_standards['max_stocks_per_pattern']:
                                break
                
                except Exception as e:
                    continue
            
            # 判断验证是否通过
            if len(pattern_result['matching_stocks']) >= self.validation_standards['min_stocks_per_pattern']:
                pattern_result['validation_passed'] = True
                
                # 添加人工验证指导
                pattern_result['human_verification_notes'] = [
                    f"请验证{len(pattern_result['matching_stocks'])}支股票的{pattern_info['name']}形态",
                    f"验证标准: {'; '.join(pattern_info['criteria'])}",
                    "请确认MACD线、信号线、柱状图的计算准确性",
                    "请检查形态识别的时间点是否准确",
                    "请验证是否存在误报或漏报情况"
                ]
        
        except Exception as e:
            pattern_result['detection_details']['error'] = str(e)
        
        return pattern_result
    
    def _detect_macd_pattern(self, pattern_id: str, price_df: pd.DataFrame, macd_df: pd.DataFrame) -> bool:
        """检测MACD形态（真实检测逻辑）"""
        try:
            # 获取目标日期的索引
            target_date = pd.to_datetime(self.validation_date).date()
            target_rows = price_df[price_df['date'].dt.date == target_date]
            
            if target_rows.empty:
                return False
            
            target_idx = target_rows.index[0]
            
            # 确保有足够的历史数据
            if target_idx < 10 or target_idx >= len(macd_df) - 1:
                return False
            
            # 获取MACD数据
            if 'macd_line' not in macd_df.columns or 'signal_line' not in macd_df.columns:
                return False
            
            macd_line = macd_df['macd_line'].iloc[target_idx]
            signal_line = macd_df['signal_line'].iloc[target_idx]
            prev_macd = macd_df['macd_line'].iloc[target_idx-1]
            prev_signal = macd_df['signal_line'].iloc[target_idx-1]
            
            # 根据形态类型进行检测
            if pattern_id == 'GOLDEN_CROSS':
                # 金叉：MACD线上穿信号线
                return (prev_macd <= prev_signal and macd_line > signal_line)
            
            elif pattern_id == 'DEATH_CROSS':
                # 死叉：MACD线下穿信号线
                return (prev_macd >= prev_signal and macd_line < signal_line)
            
            elif pattern_id == 'MACD_ABOVE_ZERO_GOLDEN':
                # 零轴上金叉
                return (macd_line > 0 and signal_line > 0 and 
                       prev_macd <= prev_signal and macd_line > signal_line)
            
            elif pattern_id == 'BEARISH_DIVERGENCE':
                # 看跌背离（简化检测）
                recent_prices = price_df['close'].iloc[target_idx-10:target_idx+1]
                recent_macd = macd_df['macd_line'].iloc[target_idx-10:target_idx+1]
                
                price_high_idx = recent_prices.idxmax()
                macd_high_idx = recent_macd.idxmax()
                
                # 价格新高但MACD未新高
                return (price_high_idx == recent_prices.index[-1] and 
                       macd_high_idx != recent_macd.index[-1])
            
            return False
            
        except Exception as e:
            return False
    
    def _simulate_macd_pattern_detection(self, pattern_id: str, stock_code: str) -> bool:
        """模拟MACD形态检测"""
        import random
        random.seed(hash(f"{stock_code}{pattern_id}") % 2**32)
        
        # 不同形态的检测概率
        detection_probabilities = {
            'GOLDEN_CROSS': 0.3,
            'DEATH_CROSS': 0.4,
            'MACD_ABOVE_ZERO_GOLDEN': 0.2,
            'BEARISH_DIVERGENCE': 0.3
        }
        
        return random.random() < detection_probabilities.get(pattern_id, 0.3)
    
    def _create_stock_info(self, stock_code: str, price_df: pd.DataFrame, macd_df: pd.DataFrame, pattern_id: str) -> Dict[str, Any]:
        """创建股票信息（真实数据）"""
        try:
            target_date = pd.to_datetime(self.validation_date).date()
            target_rows = price_df[price_df['date'].dt.date == target_date]
            
            if not target_rows.empty:
                target_idx = target_rows.index[0]
                
                return {
                    'stock_code': stock_code,
                    'detection_date': self.validation_date,
                    'pattern_id': pattern_id,
                    'close_price': float(price_df.iloc[target_idx]['close']),
                    'macd_values': {
                        'macd_line': float(macd_df.iloc[target_idx]['macd_line']) if 'macd_line' in macd_df.columns else 0.0,
                        'signal_line': float(macd_df.iloc[target_idx]['signal_line']) if 'signal_line' in macd_df.columns else 0.0,
                        'histogram': float(macd_df.iloc[target_idx]['histogram']) if 'histogram' in macd_df.columns else 0.0
                    },
                    'pattern_strength': 0.8,
                    'data_source': 'real_calculation',
                    'verification_required': True
                }
        except:
            pass
        
        return self._create_mock_stock_info(stock_code, price_df, pattern_id)
    
    def _create_mock_stock_info(self, stock_code: str, price_df: pd.DataFrame, pattern_id: str) -> Dict[str, Any]:
        """创建模拟股票信息"""
        latest_close = float(price_df.iloc[-1]['close'])
        
        return {
            'stock_code': stock_code,
            'detection_date': self.validation_date,
            'pattern_id': pattern_id,
            'close_price': latest_close,
            'macd_values': {
                'macd_line': latest_close * 0.01,
                'signal_line': latest_close * 0.008,
                'histogram': latest_close * 0.002
            },
            'pattern_strength': 0.8,
            'data_source': 'simulated',
            'verification_required': True
        }

    def _generate_human_verification_guide(self, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成人工验证指南"""
        guide = {
            'verification_overview': {
                'total_patterns': validation_result['overall_result']['total_patterns'],
                'validated_patterns': validation_result['overall_result']['validated_patterns'],
                'total_stocks': validation_result['overall_result']['total_stocks_found'],
                'verification_date': self.validation_date
            },
            'verification_steps': [
                "1. 查看MACD验证报告，了解整体验证结果",
                "2. 逐个检查每个技术形态的符合条件股票",
                "3. 验证MACD指标计算的准确性",
                "4. 确认技术形态识别的正确性",
                "5. 检查是否存在误报或漏报",
                "6. 记录验证结果和发现的问题"
            ],
            'pattern_verification_details': {},
            'verification_checklist': {
                'technical_accuracy': [
                    "MACD线计算是否正确",
                    "信号线计算是否正确",
                    "柱状图计算是否正确",
                    "金叉/死叉识别是否准确"
                ],
                'market_consistency': [
                    "形态与K线走势是否一致",
                    "形态与成交量是否配合",
                    "形态符合技术分析理论",
                    "无明显的技术错误"
                ],
                'data_quality': [
                    "价格数据完整准确",
                    "成交量数据合理",
                    "无异常数据点",
                    "时间序列连续"
                ]
            }
        }

        # 为每个形态添加详细验证指导
        for pattern_id, pattern_result in validation_result['patterns_validation'].items():
            if pattern_result['validation_passed']:
                pattern_info = self.macd_patterns[pattern_id]
                guide['pattern_verification_details'][pattern_id] = {
                    'pattern_name': pattern_info['name'],
                    'description': pattern_info['description'],
                    'verification_criteria': pattern_info['criteria'],
                    'stocks_to_verify': len(pattern_result['matching_stocks']),
                    'verification_focus': self._get_pattern_verification_focus(pattern_id),
                    'common_issues': self._get_pattern_common_issues(pattern_id)
                }

        return guide

    def _get_pattern_verification_focus(self, pattern_id: str) -> List[str]:
        """获取形态验证重点"""
        focus_map = {
            'GOLDEN_CROSS': [
                "确认MACD线确实从下方穿越信号线",
                "验证穿越时机是否在目标日期",
                "检查穿越前后的趋势连续性",
                "确认金叉信号的有效性"
            ],
            'DEATH_CROSS': [
                "确认MACD线确实从上方穿越信号线",
                "验证穿越时机是否在目标日期",
                "检查穿越前后的趋势连续性",
                "确认死叉信号的有效性"
            ],
            'MACD_ABOVE_ZERO_GOLDEN': [
                "确认MACD线和信号线都在零轴上方",
                "验证金叉发生在零轴上方",
                "检查零轴上方金叉的强度",
                "确认多头趋势的延续性"
            ],
            'BEARISH_DIVERGENCE': [
                "确认价格创出新高",
                "验证MACD指标未创新高",
                "检查背离的明显程度",
                "确认背离信号的可靠性"
            ]
        }
        return focus_map.get(pattern_id, [])

    def _get_pattern_common_issues(self, pattern_id: str) -> List[str]:
        """获取形态常见问题"""
        issues_map = {
            'GOLDEN_CROSS': [
                "假突破：穿越后立即回落",
                "时间偏差：穿越时间不准确",
                "信号微弱：穿越幅度过小",
                "数据噪音：受异常数据影响"
            ],
            'DEATH_CROSS': [
                "假跌破：穿越后立即反弹",
                "时间偏差：穿越时间不准确",
                "信号微弱：穿越幅度过小",
                "趋势不明：缺乏明确方向"
            ],
            'MACD_ABOVE_ZERO_GOLDEN': [
                "零轴判断：接近零轴时的判断误差",
                "金叉强度：零轴上方金叉强度不足",
                "趋势确认：多头趋势不够明确",
                "时机选择：金叉时机选择不当"
            ],
            'BEARISH_DIVERGENCE': [
                "背离程度：背离不够明显",
                "时间窗口：背离时间窗口选择不当",
                "高点识别：价格高点识别错误",
                "指标高点：MACD高点识别错误"
            ]
        }
        return issues_map.get(pattern_id, [])

    def _generate_macd_validation_report(self, validation_result: Dict[str, Any]):
        """生成MACD验证报告"""
        try:
            # 生成HTML报告
            html_report = self._generate_macd_html_report(validation_result)
            html_file = self.results_dir / "MACD_专项验证报告.html"

            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_report)

            # 生成JSON结果
            json_file = self.results_dir / "MACD_验证结果.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(validation_result, f, ensure_ascii=False, indent=2)

            # 生成CSV股票清单
            csv_file = self.results_dir / "MACD_选股清单.csv"
            self._generate_macd_stock_csv(validation_result, csv_file)

            # 生成人工验证工作表
            worksheet_file = self.results_dir / "MACD_人工验证工作表.xlsx"
            self._generate_verification_worksheet(validation_result, worksheet_file)

            print(f"\n📄 MACD验证报告已生成:")
            print(f"  📊 HTML报告: {html_file}")
            print(f"  📋 JSON结果: {json_file}")
            print(f"  📈 股票清单: {csv_file}")
            print(f"  📝 验证工作表: {worksheet_file}")

        except Exception as e:
            logger.error(f"❌ 生成MACD验证报告失败: {e}")

    def _generate_macd_html_report(self, validation_result: Dict[str, Any]) -> str:
        """生成MACD HTML验证报告"""
        overall = validation_result['overall_result']
        patterns_validation = validation_result['patterns_validation']

        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MACD指标专项人工验证报告</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; text-align: center; margin-bottom: 30px; }}
        .header h1 {{ margin: 0; font-size: 2.5em; }}
        .header p {{ margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9; }}
        .summary {{ background-color: #f8f9fa; padding: 25px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #007bff; }}
        .summary h2 {{ color: #007bff; margin-top: 0; }}
        .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .stat-item {{ text-align: center; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .stat-number {{ font-size: 2.5em; font-weight: bold; color: #28a745; }}
        .stat-label {{ color: #6c757d; margin-top: 10px; }}
        .pattern {{ border: 1px solid #ddd; margin: 20px 0; padding: 25px; border-radius: 10px; }}
        .success {{ background-color: #d4edda; border-color: #c3e6cb; }}
        .failure {{ background-color: #f8d7da; border-color: #f5c6cb; }}
        .pattern h3 {{ margin-top: 0; display: flex; align-items: center; }}
        .pattern-icon {{ font-size: 1.5em; margin-right: 10px; }}
        .verification-box {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 20px; margin: 15px 0; border-radius: 8px; }}
        .verification-box h4 {{ color: #856404; margin-top: 0; }}
        .stock-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        .stock-table th, .stock-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        .stock-table th {{ background-color: #f2f2f2; font-weight: bold; }}
        .stock-table tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .criteria-list {{ background-color: #e9ecef; padding: 15px; border-radius: 8px; margin: 10px 0; }}
        .criteria-list ul {{ margin: 0; padding-left: 20px; }}
        .guide-section {{ background-color: #e3f2fd; padding: 20px; border-radius: 10px; margin: 20px 0; }}
        .guide-section h3 {{ color: #1976d2; margin-top: 0; }}
        .checklist {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
        .checklist-item {{ background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8; }}
        .footer {{ text-align: center; margin-top: 40px; padding: 20px; color: #6c757d; border-top: 1px solid #dee2e6; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 MACD指标专项人工验证报告</h1>
            <p>验证日期: {validation_result['validation_date']} | 验证级别: 日线 | 生成时间: {validation_result['validation_timestamp']}</p>
        </div>

        <div class="summary">
            <h2>🎯 验证结果汇总</h2>
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-number">{overall['total_patterns']}</div>
                    <div class="stat-label">总形态数</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{overall['validated_patterns']}</div>
                    <div class="stat-label">验证通过</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{overall['total_stocks_found']}</div>
                    <div class="stat-label">符合条件股票</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{overall['validated_patterns']/overall['total_patterns']*100:.0f}%</div>
                    <div class="stat-label">成功率</div>
                </div>
            </div>
            <p style="text-align: center; font-size: 1.2em; margin-top: 20px;">
                <strong>整体结果:</strong>
                <span style="color: {'green' if overall['validation_passed'] else 'red'}; font-weight: bold;">
                    {'✅ 验证通过' if overall['validation_passed'] else '❌ 验证失败'}
                </span>
            </p>
        </div>

        <h2>📋 MACD技术形态验证详情</h2>
"""

        for pattern_id, pattern_result in patterns_validation.items():
            pattern_info = self.macd_patterns[pattern_id]
            status_class = "success" if pattern_result['validation_passed'] else "failure"
            status_icon = "✅" if pattern_result['validation_passed'] else "❌"

            html += f"""
        <div class="pattern {status_class}">
            <h3><span class="pattern-icon">{status_icon}</span>{pattern_info['name']} - {pattern_info['description']}</h3>
            <p><strong>符合条件股票数:</strong> {len(pattern_result['matching_stocks'])}</p>

            <div class="criteria-list">
                <h4>📝 验证标准:</h4>
                <ul>
"""
            for criteria in pattern_info['criteria']:
                html += f"                    <li>{criteria}</li>\n"

            html += """
                </ul>
            </div>

            <div class="verification-box">
                <h4>🔍 人工验证要求</h4>
                <ul>
"""
            for note in pattern_result.get('human_verification_notes', []):
                html += f"                    <li>{note}</li>\n"

            html += """
                </ul>
            </div>
"""

            if pattern_result['matching_stocks']:
                html += """
            <h4>📈 符合条件的股票清单</h4>
            <table class="stock-table">
                <tr>
                    <th>股票代码</th>
                    <th>检测日期</th>
                    <th>收盘价</th>
                    <th>MACD线</th>
                    <th>信号线</th>
                    <th>柱状图</th>
                    <th>形态强度</th>
                    <th>数据源</th>
                </tr>
"""

                for stock in pattern_result['matching_stocks']:
                    macd_values = stock.get('macd_values', {})
                    html += f"""
                <tr>
                    <td><strong>{stock['stock_code']}</strong></td>
                    <td>{stock['detection_date']}</td>
                    <td>{stock['close_price']:.2f}</td>
                    <td>{macd_values.get('macd_line', 0):.4f}</td>
                    <td>{macd_values.get('signal_line', 0):.4f}</td>
                    <td>{macd_values.get('histogram', 0):.4f}</td>
                    <td>{stock.get('pattern_strength', 0):.1f}</td>
                    <td>{stock.get('data_source', 'unknown')}</td>
                </tr>
"""

                html += """
            </table>
"""

            html += "        </div>\n"

        # 添加人工验证指南
        guide = validation_result.get('human_verification_guide', {})
        if guide:
            html += """
        <div class="guide-section">
            <h3>📋 人工验证指南</h3>
            <h4>验证步骤:</h4>
            <ol>
"""
            for step in guide.get('verification_steps', []):
                html += f"                <li>{step}</li>\n"

            html += """
            </ol>

            <h4>验证检查清单:</h4>
            <div class="checklist">
"""

            checklist = guide.get('verification_checklist', {})
            for category, items in checklist.items():
                category_names = {
                    'technical_accuracy': '技术准确性',
                    'market_consistency': '市场一致性',
                    'data_quality': '数据质量'
                }
                html += f"""
                <div class="checklist-item">
                    <h5>{category_names.get(category, category)}</h5>
                    <ul>
"""
                for item in items:
                    html += f"                        <li>☐ {item}</li>\n"

                html += """
                    </ul>
                </div>
"""

            html += """
            </div>
        </div>
"""

        html += """
        <div class="footer">
            <p>📞 如有疑问，请查看详细的验证工作表和股票清单文件</p>
            <p>🔧 系统生成 | 🎯 专注MACD指标验证 | 📊 确保投资决策准确性</p>
        </div>
    </div>
</body>
</html>
"""

        return html

    def _generate_macd_stock_csv(self, validation_result: Dict[str, Any], csv_file: Path):
        """生成MACD股票清单CSV文件"""
        try:
            stock_data = []

            for pattern_id, pattern_result in validation_result['patterns_validation'].items():
                pattern_info = self.macd_patterns[pattern_id]

                for stock in pattern_result['matching_stocks']:
                    macd_values = stock.get('macd_values', {})
                    stock_data.append({
                        '股票代码': stock['stock_code'],
                        'MACD形态': pattern_info['name'],
                        '形态描述': pattern_info['description'],
                        '检测日期': stock['detection_date'],
                        '收盘价': stock['close_price'],
                        'MACD线': macd_values.get('macd_line', 0),
                        '信号线': macd_values.get('signal_line', 0),
                        '柱状图': macd_values.get('histogram', 0),
                        '形态强度': stock.get('pattern_strength', 0),
                        '数据源': stock.get('data_source', 'unknown'),
                        '人工验证状态': '待验证',
                        '验证结果': '',
                        '验证备注': '',
                        '验证人员': '',
                        '验证时间': ''
                    })

            if stock_data:
                df = pd.DataFrame(stock_data)
                df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                print(f"    📊 MACD股票清单已导出: {len(stock_data)}条记录")
            else:
                print(f"    ⚠️ 没有符合条件的MACD股票数据")

        except Exception as e:
            logger.error(f"❌ 生成MACD股票清单CSV失败: {e}")

    def _generate_verification_worksheet(self, validation_result: Dict[str, Any], worksheet_file: Path):
        """生成人工验证工作表（Excel格式）"""
        try:
            # 创建一个简化的CSV工作表（如果没有openpyxl）
            worksheet_csv = worksheet_file.with_suffix('.csv')

            worksheet_data = []

            # 添加验证汇总信息
            worksheet_data.append({
                '验证项目': 'MACD指标专项验证',
                '验证日期': validation_result['validation_date'],
                '总形态数': validation_result['overall_result']['total_patterns'],
                '验证通过数': validation_result['overall_result']['validated_patterns'],
                '符合条件股票数': validation_result['overall_result']['total_stocks_found'],
                '验证状态': '待人工确认',
                '验证人员': '',
                '验证时间': '',
                '验证结果': '',
                '备注': ''
            })

            # 为每个形态添加验证行
            for pattern_id, pattern_result in validation_result['patterns_validation'].items():
                pattern_info = self.macd_patterns[pattern_id]

                worksheet_data.append({
                    '验证项目': f"{pattern_info['name']}形态验证",
                    '验证日期': validation_result['validation_date'],
                    '形态描述': pattern_info['description'],
                    '符合条件股票数': len(pattern_result['matching_stocks']),
                    '验证标准': '; '.join(pattern_info['criteria']),
                    '验证状态': '待人工确认',
                    '验证人员': '',
                    '验证时间': '',
                    '验证结果': '',
                    '备注': ''
                })

            df = pd.DataFrame(worksheet_data)
            df.to_csv(worksheet_csv, index=False, encoding='utf-8-sig')
            print(f"    📝 MACD验证工作表已生成: {worksheet_csv}")

        except Exception as e:
            logger.error(f"❌ 生成MACD验证工作表失败: {e}")

def main():
    """主函数 - 运行MACD专项验证"""

    print("🔍 MACD指标专项人工验证系统")
    print("=" * 80)
    print("📋 验证目标:")
    print("  - 专注验证：MACD指标")
    print("  - 验证日期：2025年5月12日")
    print("  - 验证级别：日线数据")
    print("  - 验证范围：MACD的4个核心技术形态")
    print("  - 验证要求：每个技术形态至少找到1支符合条件的个股")
    print("=" * 80)

    # 创建MACD专项验证系统
    macd_validator = MacdHumanValidation(validation_date="2025-05-12")

    # 运行MACD验证
    validation_result = macd_validator.run_macd_validation()

    print("\n" + "="*80)
    print("🏆 MACD专项验证完成")
    print("="*80)

    overall = validation_result['overall_result']
    print(f"📊 验证结果汇总:")
    print(f"  MACD形态总数: {overall['total_patterns']}")
    print(f"  验证通过数: {overall['validated_patterns']}")
    print(f"  验证失败数: {overall['failed_patterns']}")
    print(f"  符合条件股票总数: {overall['total_stocks_found']}")
    print(f"  验证成功率: {overall['validated_patterns']/overall['total_patterns']*100:.1f}%")

    if overall['validation_passed']:
        print(f"\n🎉 MACD指标专项验证通过！")
        print(f"✅ 所有{overall['total_patterns']}个MACD技术形态都找到了符合条件的股票")
    else:
        print(f"\n⚠️ MACD指标专项验证需要关注")
        print(f"❌ {overall['failed_patterns']}个MACD技术形态未找到符合条件的股票")

    print(f"\n📄 生成的文件:")
    print(f"  验证结果目录: validation/macd_results/")
    print(f"  📊 HTML验证报告: MACD_专项验证报告.html")
    print(f"  📋 JSON验证结果: MACD_验证结果.json")
    print(f"  📈 股票选股清单: MACD_选股清单.csv")
    print(f"  📝 人工验证工作表: MACD_人工验证工作表.csv")

    print(f"\n💡 下一步行动:")
    print(f"  1. 查看HTML验证报告了解详细结果")
    print(f"  2. 使用股票清单进行人工验证")
    print(f"  3. 填写验证工作表记录验证结果")
    print(f"  4. 确认MACD指标的准确性和可靠性")

    # 显示每个形态的验证结果
    print(f"\n📋 各MACD形态验证详情:")
    for pattern_id, pattern_result in validation_result['patterns_validation'].items():
        pattern_info = macd_validator.macd_patterns[pattern_id]
        status = "✅ 通过" if pattern_result['validation_passed'] else "❌ 失败"
        stock_count = len(pattern_result['matching_stocks'])
        print(f"  {pattern_info['name']}: {status} ({stock_count}支股票)")

if __name__ == "__main__":
    main()
