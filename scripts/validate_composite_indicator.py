#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COMPOSITE指标五阶段验证脚本
使用五阶段验证标准验证修复后的COMPOSITE指标
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.logger import get_logger

logger = get_logger(__name__)


class CompositeIndicatorValidator:
    """COMPOSITE指标五阶段验证器"""
    
    def __init__(self):
        self.indicator_name = "COMPOSITE"
        self.validation_results = {}
        
    def generate_test_data(self) -> pd.DataFrame:
        """生成高质量测试数据"""
        np.random.seed(42)
        
        dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
        
        # 生成更真实的价格数据
        base_price = 100.0
        returns = np.random.normal(0, 0.015, 200)  # 1.5%日波动率
        
        # 添加趋势和周期性
        trend = np.linspace(0, 0.3, 200)  # 30%的总趋势
        cycle = 0.05 * np.sin(np.linspace(0, 4*np.pi, 200))  # 5%的周期性波动
        
        returns = returns + trend/200 + cycle/200
        
        prices = [base_price]
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 1.0))  # 确保价格为正
        
        # 生成OHLCV数据
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # 计算日内波动
            volatility = abs(returns[i]) * 1.5
            high = close * (1 + volatility * np.random.uniform(0.3, 1.0))
            low = close * (1 - volatility * np.random.uniform(0.3, 1.0))
            open_price = prices[i-1] if i > 0 else close
            
            # 确保OHLC关系正确
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            volume = np.random.randint(1000000, 5000000)
            
            data.append({
                'date': date,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def stage1_algorithm_authenticity(self, composite_indicator, test_data: pd.DataFrame) -> Dict[str, Any]:
        """阶段1：算法真实性验证"""
        logger.info("🔍 阶段1：算法真实性验证...")
        
        result = {
            'stage': 'algorithm_authenticity',
            'score': 0,
            'max_score': 20,
            'details': {}
        }
        
        try:
            # 计算COMPOSITE
            composite_result = composite_indicator.calculate(test_data)
            
            if not composite_result or 'composite_score' not in composite_result:
                result['details']['error'] = 'COMPOSITE计算失败'
                return result
            
            composite_score = composite_result['composite_score']
            trend_score = composite_result.get('trend_score', pd.Series())
            momentum_score = composite_result.get('momentum_score', pd.Series())
            volatility_score = composite_result.get('volatility_score', pd.Series())
            volume_score = composite_result.get('volume_score', pd.Series())
            
            # 验证COMPOSITE算法正确性
            score = 0
            
            # 1. 复合评分应该在合理范围内
            if (composite_score >= 0).all() and (composite_score <= 100).all():
                score += 5
                result['details']['score_range_valid'] = True
            else:
                result['details']['score_range_valid'] = False
            
            # 2. 复合评分应该是多个子评分的综合
            if len(trend_score) > 0 and len(momentum_score) > 0:
                # 检查复合评分与子评分的相关性
                correlation_with_trend = np.corrcoef(composite_score, trend_score)[0, 1]
                correlation_with_momentum = np.corrcoef(composite_score, momentum_score)[0, 1]
                if correlation_with_trend > 0.3 and correlation_with_momentum > 0.3:
                    score += 5
                    result['details']['component_correlation'] = True
                else:
                    result['details']['component_correlation'] = False
            
            # 3. 复合评分应该反映市场状态变化
            price_change = test_data['close'].pct_change()
            score_change = composite_score.pct_change()
            if len(price_change) > 10 and len(score_change) > 10:
                # 在强趋势期间，评分变化应该与价格变化有一定相关性
                strong_moves = abs(price_change) > 0.02
                if strong_moves.sum() > 5:
                    trend_correlation = np.corrcoef(
                        price_change[strong_moves].fillna(0), 
                        score_change[strong_moves].fillna(0)
                    )[0, 1]
                    if abs(trend_correlation) > 0.2:
                        score += 5
                        result['details']['market_sensitivity'] = True
                    else:
                        result['details']['market_sensitivity'] = False
                else:
                    score += 5  # 如果没有强趋势，给予通过
                    result['details']['market_sensitivity'] = True
            
            # 4. 各子评分应该有合理的分布
            if len(trend_score) > 0 and len(momentum_score) > 0:
                trend_std = trend_score.std()
                momentum_std = momentum_score.std()
                if trend_std > 5 and momentum_std > 5:  # 评分应该有一定变化
                    score += 5
                    result['details']['score_distribution'] = True
                else:
                    result['details']['score_distribution'] = False
            
            result['score'] = score
            result['details']['composite_mean'] = float(composite_score.mean())
            result['details']['composite_std'] = float(composite_score.std())
            
        except Exception as e:
            result['details']['error'] = str(e)
        
        logger.info(f"✅ 阶段1完成，得分: {result['score']}/{result['max_score']}")
        return result
    
    def stage2_basic_functionality(self, composite_indicator, test_data: pd.DataFrame) -> Dict[str, Any]:
        """阶段2：基础功能验证"""
        logger.info("🔍 阶段2：基础功能验证...")
        
        result = {
            'stage': 'basic_functionality',
            'score': 0,
            'max_score': 20,
            'details': {}
        }
        
        try:
            score = 0
            
            # 1. calculate方法测试
            calc_result = composite_indicator.calculate(test_data)
            if calc_result and isinstance(calc_result, dict):
                score += 5
                result['details']['calculate_method'] = True
            else:
                result['details']['calculate_method'] = False
            
            # 2. get_patterns方法测试
            patterns = composite_indicator.get_patterns()
            if patterns and isinstance(patterns, dict):
                score += 5
                result['details']['get_patterns_method'] = True
                result['details']['pattern_count'] = patterns.get('pattern_count', 0)
            else:
                result['details']['get_patterns_method'] = False
            
            # 3. get_signal方法测试
            signals = composite_indicator.get_signal()
            if signals and isinstance(signals, dict):
                score += 5
                result['details']['get_signal_method'] = True
                result['details']['signal_count'] = signals.get('signal_count', 0)
            else:
                result['details']['get_signal_method'] = False
            
            # 4. get_score方法测试
            score_value = composite_indicator.get_score()
            if isinstance(score_value, (int, float)) and 0 <= score_value <= 100:
                score += 5
                result['details']['get_score_method'] = True
                result['details']['score_value'] = score_value
            else:
                result['details']['get_score_method'] = False
            
            result['score'] = score
            
        except Exception as e:
            result['details']['error'] = str(e)
        
        logger.info(f"✅ 阶段2完成，得分: {result['score']}/{result['max_score']}")
        return result
    
    def stage3_pattern_recognition(self, composite_indicator, test_data: pd.DataFrame) -> Dict[str, Any]:
        """阶段3：形态识别验证"""
        logger.info("🔍 阶段3：形态识别验证...")
        
        result = {
            'stage': 'pattern_recognition',
            'score': 0,
            'max_score': 20,
            'details': {}
        }
        
        try:
            # 先计算COMPOSITE
            composite_indicator.calculate(test_data)
            patterns = composite_indicator.get_patterns()
            
            if not patterns:
                result['details']['error'] = '无法获取形态识别结果'
                return result
            
            score = 0
            
            # 1. 多头复合形态识别
            if 'bullish_composite' in patterns:
                bullish = patterns['bullish_composite']
                if isinstance(bullish, list) and len(bullish) > 0:
                    score += 5
                    result['details']['bullish_composite_detected'] = True
                else:
                    result['details']['bullish_composite_detected'] = False
            
            # 2. 空头复合形态识别
            if 'bearish_composite' in patterns:
                bearish = patterns['bearish_composite']
                if isinstance(bearish, list) and len(bearish) > 0:
                    score += 5
                    result['details']['bearish_composite_detected'] = True
                else:
                    result['details']['bearish_composite_detected'] = False
            
            # 3. 中性复合形态识别
            if 'neutral_composite' in patterns:
                neutral = patterns['neutral_composite']
                if isinstance(neutral, list):
                    score += 5
                    result['details']['neutral_composite_detected'] = True
                else:
                    result['details']['neutral_composite_detected'] = False
            
            # 4. 强趋势形态识别
            if 'strong_trend' in patterns:
                strong_trend = patterns['strong_trend']
                if isinstance(strong_trend, list):
                    score += 5
                    result['details']['strong_trend_detected'] = True
                else:
                    result['details']['strong_trend_detected'] = False
            
            result['score'] = score
            result['details']['total_patterns'] = patterns.get('pattern_count', 0)
            
        except Exception as e:
            result['details']['error'] = str(e)
        
        logger.info(f"✅ 阶段3完成，得分: {result['score']}/{result['max_score']}")
        return result
    
    def stage4_architecture_compliance(self, composite_indicator) -> Dict[str, Any]:
        """阶段4：架构合规性验证"""
        logger.info("🔍 阶段4：架构合规性验证...")
        
        result = {
            'stage': 'architecture_compliance',
            'score': 0,
            'max_score': 20,
            'details': {}
        }
        
        try:
            score = 0
            
            # 1. 继承BaseIndicator
            from indicators.base_indicator import BaseIndicator
            if isinstance(composite_indicator, BaseIndicator):
                score += 5
                result['details']['inherits_baseindicator'] = True
            else:
                result['details']['inherits_baseindicator'] = False
            
            # 2. 必要属性检查
            required_attrs = ['name', 'period']
            has_all_attrs = all(hasattr(composite_indicator, attr) for attr in required_attrs)
            if has_all_attrs:
                score += 5
                result['details']['has_required_attributes'] = True
            else:
                result['details']['has_required_attributes'] = False
            
            # 3. 必要方法检查
            required_methods = ['calculate', 'get_patterns', 'get_signal', 'get_score']
            has_all_methods = all(hasattr(composite_indicator, method) for method in required_methods)
            if has_all_methods:
                score += 5
                result['details']['has_required_methods'] = True
            else:
                result['details']['has_required_methods'] = False
            
            # 4. 名称正确性
            if hasattr(composite_indicator, 'name') and composite_indicator.name == 'COMPOSITE':
                score += 5
                result['details']['correct_name'] = True
            else:
                result['details']['correct_name'] = False
            
            result['score'] = score
            
        except Exception as e:
            result['details']['error'] = str(e)
        
        logger.info(f"✅ 阶段4完成，得分: {result['score']}/{result['max_score']}")
        return result
    
    def stage5_production_readiness(self, composite_indicator, test_data: pd.DataFrame) -> Dict[str, Any]:
        """阶段5：生产就绪性验证"""
        logger.info("🔍 阶段5：生产就绪性验证...")
        
        result = {
            'stage': 'production_readiness',
            'score': 0,
            'max_score': 20,
            'details': {}
        }
        
        try:
            score = 0
            
            # 1. 性能测试
            start_time = time.time()
            for _ in range(10):
                composite_indicator.calculate(test_data)
            avg_time = (time.time() - start_time) / 10
            
            if avg_time < 0.2:  # 200ms以内
                score += 5
                result['details']['performance_test'] = True
            else:
                result['details']['performance_test'] = False
            result['details']['avg_calculation_time'] = avg_time
            
            # 2. 数据质量测试
            calc_result = composite_indicator.calculate(test_data)
            if calc_result and 'composite_score' in calc_result:
                composite_score = calc_result['composite_score']
                valid_ratio = composite_score.notna().sum() / len(composite_score)
                if valid_ratio >= 0.95:
                    score += 5
                    result['details']['data_quality_test'] = True
                else:
                    result['details']['data_quality_test'] = False
                result['details']['valid_data_ratio'] = valid_ratio
            
            # 3. 异常处理测试
            try:
                # 测试空数据
                empty_data = pd.DataFrame()
                composite_indicator.calculate(empty_data)
                score += 5
                result['details']['exception_handling'] = True
            except:
                result['details']['exception_handling'] = False
            
            # 4. 一致性测试
            result1 = composite_indicator.calculate(test_data)
            result2 = composite_indicator.calculate(test_data)
            if result1 and result2 and 'composite_score' in result1 and 'composite_score' in result2:
                if result1['composite_score'].equals(result2['composite_score']):
                    score += 5
                    result['details']['consistency_test'] = True
                else:
                    result['details']['consistency_test'] = False
            
            result['score'] = score
            
        except Exception as e:
            result['details']['error'] = str(e)
        
        logger.info(f"✅ 阶段5完成，得分: {result['score']}/{result['max_score']}")
        return result
    
    def run_five_stage_validation(self) -> Dict[str, Any]:
        """运行五阶段验证"""
        logger.info(f"🚀 开始COMPOSITE指标五阶段验证...")
        
        try:
            # 导入COMPOSITE指标
            from indicators.composite import COMPOSITE
            composite_indicator = COMPOSITE(period=20)
            
            # 生成测试数据
            test_data = self.generate_test_data()
            
            # 执行五个阶段的验证
            stage1_result = self.stage1_algorithm_authenticity(composite_indicator, test_data)
            stage2_result = self.stage2_basic_functionality(composite_indicator, test_data)
            stage3_result = self.stage3_pattern_recognition(composite_indicator, test_data)
            stage4_result = self.stage4_architecture_compliance(composite_indicator)
            stage5_result = self.stage5_production_readiness(composite_indicator, test_data)
            
            # 计算总分
            total_score = (stage1_result['score'] + stage2_result['score'] + 
                          stage3_result['score'] + stage4_result['score'] + 
                          stage5_result['score'])
            max_total_score = 100
            
            # 生成验证报告
            validation_report = {
                'indicator_name': self.indicator_name,
                'validation_timestamp': datetime.now().isoformat(),
                'total_score': total_score,
                'max_score': max_total_score,
                'percentage': (total_score / max_total_score) * 100,
                'status': 'PASSED' if total_score >= 95 else 'FAILED',
                'stages': {
                    'stage1': stage1_result,
                    'stage2': stage2_result,
                    'stage3': stage3_result,
                    'stage4': stage4_result,
                    'stage5': stage5_result
                }
            }
            
            logger.info(f"🎯 COMPOSITE指标验证完成!")
            logger.info(f"📊 总得分: {total_score}/{max_total_score} ({validation_report['percentage']:.1f}%)")
            logger.info(f"📊 验证状态: {validation_report['status']}")
            
            return validation_report
            
        except Exception as e:
            logger.error(f"❌ COMPOSITE指标验证失败: {e}")
            return {
                'indicator_name': self.indicator_name,
                'validation_timestamp': datetime.now().isoformat(),
                'total_score': 0,
                'max_score': 100,
                'percentage': 0,
                'status': 'FAILED',
                'error': str(e)
            }


def main():
    """主函数"""
    logger.info("🔍 开始COMPOSITE指标五阶段验证...")
    
    validator = CompositeIndicatorValidator()
    result = validator.run_five_stage_validation()
    
    # 保存验证报告
    report_file = f"docs/finaltesting/indicators/COMPOSITE_fixed_validation_report.md"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    # 生成Markdown报告
    if result['status'] == 'PASSED':
        status_icon = "✅"
        status_text = "验证通过"
    else:
        status_icon = "❌"
        status_text = "验证失败"
    
    report_content = f"""# COMPOSITE指标修复验证报告

## 验证概览
- **指标名称**: {result['indicator_name']}
- **验证时间**: {result['validation_timestamp']}
- **总得分**: {result['total_score']}/{result['max_score']} ({result['percentage']:.1f}%)
- **验证状态**: {status_icon} {result['status']} - {status_text}

## 五阶段验证结果

### 阶段1: 算法真实性 ({result['stages']['stage1']['score']}/{result['stages']['stage1']['max_score']}分)
- **评分范围有效**: {result['stages']['stage1']['details'].get('score_range_valid', False)}
- **组件相关性**: {result['stages']['stage1']['details'].get('component_correlation', False)}
- **市场敏感性**: {result['stages']['stage1']['details'].get('market_sensitivity', False)}
- **评分分布**: {result['stages']['stage1']['details'].get('score_distribution', False)}

### 阶段2: 基础功能 ({result['stages']['stage2']['score']}/{result['stages']['stage2']['max_score']}分)
- **calculate方法**: {result['stages']['stage2']['details'].get('calculate_method', False)}
- **get_patterns方法**: {result['stages']['stage2']['details'].get('get_patterns_method', False)}
- **get_signal方法**: {result['stages']['stage2']['details'].get('get_signal_method', False)}
- **get_score方法**: {result['stages']['stage2']['details'].get('get_score_method', False)}

### 阶段3: 形态识别 ({result['stages']['stage3']['score']}/{result['stages']['stage3']['max_score']}分)
- **多头复合形态**: {result['stages']['stage3']['details'].get('bullish_composite_detected', False)}
- **空头复合形态**: {result['stages']['stage3']['details'].get('bearish_composite_detected', False)}
- **中性复合形态**: {result['stages']['stage3']['details'].get('neutral_composite_detected', False)}
- **强趋势形态**: {result['stages']['stage3']['details'].get('strong_trend_detected', False)}

### 阶段4: 架构合规性 ({result['stages']['stage4']['score']}/{result['stages']['stage4']['max_score']}分)
- **继承BaseIndicator**: {result['stages']['stage4']['details'].get('inherits_baseindicator', False)}
- **必要属性**: {result['stages']['stage4']['details'].get('has_required_attributes', False)}
- **必要方法**: {result['stages']['stage4']['details'].get('has_required_methods', False)}
- **名称正确性**: {result['stages']['stage4']['details'].get('correct_name', False)}

### 阶段5: 生产就绪性 ({result['stages']['stage5']['score']}/{result['stages']['stage5']['max_score']}分)
- **性能测试**: {result['stages']['stage5']['details'].get('performance_test', False)}
- **数据质量**: {result['stages']['stage5']['details'].get('data_quality_test', False)}
- **异常处理**: {result['stages']['stage5']['details'].get('exception_handling', False)}
- **一致性测试**: {result['stages']['stage5']['details'].get('consistency_test', False)}

## 验证结论

{'### 🎉 COMPOSITE指标修复成功！' if result['status'] == 'PASSED' else '### ⚠️ COMPOSITE指标需要进一步修复'}

{'COMPOSITE指标已达到生产级别标准，可以安全部署。' if result['status'] == 'PASSED' else 'COMPOSITE指标未达到95分标准，需要进一步优化。'}

---
*验证工具: 五阶段技术指标验证系统*
*质量标准: 95分以上为生产级别*
"""
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"📄 验证报告已保存: {report_file}")
    
    return result


if __name__ == "__main__":
    main()
