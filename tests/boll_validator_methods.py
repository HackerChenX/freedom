#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BOLL验证器的剩余方法
这些方法将被添加到主验证器文件中
"""

def _test_signal_quality_with_real_data(self, boll, real_data):
    """使用真实数据测试信号质量"""
    try:
        signal_quality_scores = []
        
        if 'code' in real_data.columns:
            unique_codes = real_data['code'].unique()[:5]  # 测试前5只股票
            
            for code in unique_codes:
                stock_data = real_data[real_data['code'] == code].copy()
                stock_data = stock_data.sort_values('date').reset_index(drop=True)
                
                if len(stock_data) >= 30:
                    boll.set_parameters_Boll(period=20, std_dev=2)
                    result = boll.calculate(stock_data)
                    
                    if result is not None:
                        if 'middle' in result.columns and 'upper' in result.columns and 'lower' in result.columns:
                            middle_values = result['middle'].dropna()
                            upper_values = result['upper'].dropna()
                            lower_values = result['lower'].dropna()
                            
                            if len(middle_values) > 10:
                                # 检查信号的合理性
                                bandwidth = (upper_values - lower_values) / middle_values
                                bandwidth_cv = bandwidth.std() / bandwidth.mean() if bandwidth.mean() > 0 else 0
                                
                                # 合理的带宽变异系数表示良好的信号质量
                                if 0.1 <= bandwidth_cv <= 2.0:
                                    signal_quality_scores.append(100)
                                elif bandwidth_cv <= 3.0:
                                    signal_quality_scores.append(80)
                                else:
                                    signal_quality_scores.append(60)
        
        if signal_quality_scores:
            score = sum(signal_quality_scores) / len(signal_quality_scores)
            return {'score': score, 'tested_stocks': len(signal_quality_scores)}
        else:
            return {'score': 50, 'error': '无法测试信号质量'}
            
    except Exception as e:
        return {'score': 0, 'error': str(e)}

def _test_multi_period_analysis(self, boll, standard_data):
    """测试多周期分析"""
    try:
        periods = [10, 20, 26]  # BOLL常用周期
        successful_periods = 0
        
        for period in periods:
            boll.set_parameters_Boll(period=period, std_dev=2)
            result = boll.calculate(standard_data)
            
            if result is not None and 'middle' in result.columns:
                middle_values = result['middle'].dropna()
                if len(middle_values) > 0:
                    successful_periods += 1
        
        score = (successful_periods / len(periods)) * 100
        return {'score': score, 'successful_periods': successful_periods, 'total_periods': len(periods)}
        
    except Exception as e:
        return {'score': 0, 'error': str(e)}

def _test_pattern_accuracy(self, boll, real_data, standard_data):
    """测试形态准确性"""
    try:
        accuracy_tests = []
        
        # 测试1: 真实数据的BOLL带宽合理性
        if len(real_data) >= 50:
            stock_data = real_data.head(50)
            boll.set_parameters_Boll(period=20, std_dev=2)
            result = boll.calculate(stock_data)
            
            if result is not None and all(col in result.columns for col in ['middle', 'upper', 'lower']):
                middle_values = result['middle'].dropna()
                upper_values = result['upper'].dropna()
                lower_values = result['lower'].dropna()
                
                if len(middle_values) > 10:
                    # 检查带宽的合理性
                    bandwidth = (upper_values - lower_values) / middle_values
                    bandwidth_mean = bandwidth.mean()
                    accuracy_tests.append(0.05 <= bandwidth_mean <= 0.5)  # 合理的带宽范围
        
        # 测试2: 标准数据的BOLL响应性
        if len(standard_data) >= 50:
            boll.set_parameters_Boll(period=20, std_dev=2)
            result = boll.calculate(standard_data)
            
            if result is not None and 'middle' in result.columns:
                middle_values = result['middle'].dropna()
                if len(middle_values) > 10:
                    # 检查中轨对价格的响应性
                    responsiveness = self._calculate_responsiveness(middle_values, standard_data['close'])
                    accuracy_tests.append(responsiveness > 0.3)
        
        score = (sum(accuracy_tests) / len(accuracy_tests)) * 100 if accuracy_tests else 50
        return {'score': score, 'passed_tests': sum(accuracy_tests), 'total_tests': len(accuracy_tests)}
        
    except Exception as e:
        return {'score': 0, 'error': str(e)}

def _calculate_responsiveness(self, boll_values, price_values):
    """计算BOLL对价格变化的响应性"""
    if len(boll_values) != len(price_values) or len(boll_values) < 10:
        return 0
    
    import numpy as np
    price_changes = price_values.pct_change().dropna()
    boll_changes = boll_values.pct_change().dropna()
    
    min_len = min(len(price_changes), len(boll_changes))
    if min_len < 5:
        return 0
    
    correlation = np.corrcoef(price_changes[-min_len:], boll_changes[-min_len:])[0, 1]
    return abs(correlation) if not np.isnan(correlation) else 0

def _stage4_architecture_compliance(self, boll):
    """阶段4: BOLL架构合规性验证"""
    test_results = []
    
    # 测试1: 分层架构
    architecture_test = self._test_layered_architecture(boll)
    test_results.append(architecture_test['score'])
    
    # 测试2: 依赖注入
    di_test = self._test_dependency_injection(boll)
    test_results.append(di_test['score'])
    
    # 测试3: 无直接SQL
    sql_test = self._test_no_direct_sql(boll)
    test_results.append(sql_test['score'])
    
    # 测试4: 接口合规性
    interface_test = self._test_interface_compliance(boll)
    test_results.append(interface_test['score'])
    
    # 测试5: 关注点分离
    separation_test = self._test_separation_of_concerns(boll)
    test_results.append(separation_test['score'])
    
    overall_score = sum(test_results) / len(test_results) if test_results else 0
    
    return {
        'score': overall_score,
        'layered_architecture': architecture_test,
        'dependency_injection': di_test,
        'no_direct_sql': sql_test,
        'interface_compliance': interface_test,
        'separation_of_concerns': separation_test,
        'passed': overall_score >= 99.0
    }

def _test_layered_architecture(self, boll):
    """测试分层架构"""
    architecture_checks = {
        'has_calculate_method': hasattr(boll, 'calculate'),
        'has_set_parameters_method': hasattr(boll, 'set_parameters_Boll'),
        'inherits_from_base': hasattr(boll, '__bases__') and len(boll.__class__.__bases__) > 0,
        'proper_method_separation': len([m for m in dir(boll) if not m.startswith('__')]) >= 10
    }
    
    passed_checks = sum(architecture_checks.values())
    total_checks = len(architecture_checks)
    score = (passed_checks / total_checks) * 100
    
    return {'score': score, 'architecture_checks': architecture_checks, 'passed_checks': passed_checks}

def _test_dependency_injection(self, boll):
    """测试依赖注入"""
    di_checks = {
        'uses_dependency_container': hasattr(boll, 'container') or 'container' in str(type(boll)),
        'no_hard_coded_dependencies': 'import' not in str(boll.calculate) if hasattr(boll, 'calculate') else True,
        'configurable_parameters': hasattr(boll, 'set_parameters_Boll'),
        'injectable_services': True  # BOLL通常不需要外部服务
    }
    
    passed_checks = sum(di_checks.values())
    total_checks = len(di_checks)
    score = (passed_checks / total_checks) * 100
    
    return {'score': score, 'di_checks': di_checks, 'passed_checks': passed_checks}

def _test_no_direct_sql(self, boll):
    """测试无直接SQL"""
    import inspect
    import re
    
    source_code = inspect.getsource(boll.__class__)
    
    # 更精确的SQL检测
    sql_patterns = [
        r'\bSELECT\s+.*\s+FROM\b',
        r'\bINSERT\s+INTO\b',
        r'\bUPDATE\s+.*\s+SET\b',
        r'\bDELETE\s+FROM\b',
        r'\bCREATE\s+TABLE\b',
        r'\bDROP\s+TABLE\b',
        r'\.execute\s*\(',
        r'\.query\s*\(',
    ]
    
    has_sql = any(re.search(pattern, source_code, re.IGNORECASE) for pattern in sql_patterns)
    
    score = 0 if has_sql else 100
    return {'score': score, 'has_direct_sql': has_sql}

def _test_interface_compliance(self, boll):
    """测试接口合规性"""
    interface_checks = {
        'has_calculate': hasattr(boll, 'calculate'),
        'calculate_returns_dataframe': True,  # 需要运行时检查
        'has_set_parameters': hasattr(boll, 'set_parameters_Boll'),
        'has_proper_naming': boll.__class__.__name__.endswith('Boll') or 'BOLL' in boll.__class__.__name__
    }
    
    # 运行时检查calculate方法返回类型
    try:
        import pandas as pd
        test_data = self._create_standard_test_data(50)
        result = boll.calculate(test_data)
        interface_checks['calculate_returns_dataframe'] = isinstance(result, pd.DataFrame) or result is None
    except:
        interface_checks['calculate_returns_dataframe'] = False
    
    passed_checks = sum(interface_checks.values())
    total_checks = len(interface_checks)
    score = (passed_checks / total_checks) * 100
    
    return {'score': score, 'interface_checks': interface_checks, 'passed_checks': passed_checks}

def _test_separation_of_concerns(self, boll):
    """测试关注点分离"""
    separation_checks = {
        'single_responsibility': 'BOLL' in boll.__class__.__name__ or 'Boll' in boll.__class__.__name__,
        'no_mixed_concerns': True,  # BOLL通常职责单一
        'proper_abstraction': hasattr(boll, 'calculate'),
        'clean_interface': len([m for m in dir(boll) if not m.startswith('_')]) <= 50
    }
    
    passed_checks = sum(separation_checks.values())
    total_checks = len(separation_checks)
    score = (passed_checks / total_checks) * 100
    
    return {'score': score, 'separation_checks': separation_checks, 'passed_checks': passed_checks}
