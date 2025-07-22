#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
统一架构验证器

验证所有组件在统一架构下能否正常工作，确保数据与逻辑正确分离
"""

import os
import sys
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Any, List

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from utils.logger import getLogger

logger = getLogger(__name__)


class ProductionModeValidator:
    """统一架构验证器"""
    
    def __init__(self):
        """初始化验证器"""
        self.validation_results = {
            'buypoint_analyzer': None,
            'selection_strategy_tester': None,
            'closed_loop_validator': None,
            'unified_indicator_tester': None
        }
        
        logger.info("统一架构验证器初始化完成")
    
    def run_full_validation(self) -> Dict[str, Any]:
        """运行完整的生产模式验证"""
        logger.info("🚀 开始生产模式完整验证")
        
        validation_start_time = datetime.now()
        
        # 1. 验证BuypointAnalyzer
        logger.info("1️⃣ 验证BuypointAnalyzer生产模式...")
        self.validation_results['buypoint_analyzer'] = self._validate_buypoint_analyzer()
        
        # 2. 验证SelectionStrategyTester
        logger.info("2️⃣ 验证SelectionStrategyTester生产模式...")
        self.validation_results['selection_strategy_tester'] = self._validate_selection_strategy_tester()
        
        # 3. 验证ClosedLoopValidator
        logger.info("3️⃣ 验证ClosedLoopValidator生产模式...")
        self.validation_results['closed_loop_validator'] = self._validate_closed_loop_validator()
        
        # 4. 验证UnifiedIndicatorTester
        logger.info("4️⃣ 验证UnifiedIndicatorTester生产模式...")
        self.validation_results['unified_indicator_tester'] = self._validate_unified_indicator_tester()
        
        validation_end_time = datetime.now()
        execution_time = (validation_end_time - validation_start_time).total_seconds()
        
        # 生成总结报告
        summary = self._generate_validation_summary(execution_time)
        
        logger.info("✅ 生产模式验证完成")
        return summary
    
    def _validate_buypoint_analyzer(self) -> Dict[str, Any]:
        """验证BuypointAnalyzer统一架构"""
        try:
            from components.buypoint_analyzer import BuypointAnalyzer
            
            # 测试统一架构初始化（无模式参数）
            try:
                analyzer = BuypointAnalyzer()
                logger.info("✅ BuypointAnalyzer统一架构初始化成功")
                
                # 创建测试数据
                test_data = self._create_real_stock_data()
                
                # 测试统一计算引擎
                result = analyzer._calculate_indicator(test_data, 'MACD')
                if result is not None:
                    logger.info("✅ BuypointAnalyzer统一计算引擎工作正常")
                    return {'status': 'PASS', 'message': '统一架构工作正常'}
                else:
                    return {'status': 'FAIL', 'message': '统一计算引擎失败'}
                    
            except RuntimeError as e:
                if "引擎" in str(e) or "计算" in str(e):
                    logger.warning(f"⚠️ BuypointAnalyzer统一计算引擎未满足要求: {e}")
                    return {'status': 'REQUIREMENTS_NOT_MET', 'message': str(e)}
                else:
                    raise
                    
        except Exception as e:
            logger.error(f"❌ BuypointAnalyzer验证失败: {e}")
            return {'status': 'ERROR', 'message': str(e)}
    
    def _validate_selection_strategy_tester(self) -> Dict[str, Any]:
        """验证SelectionStrategyTester统一架构"""
        try:
            from components.selection_strategy_tester import SelectionStrategyTester
            
            # 测试统一架构初始化（无模式参数）
            try:
                tester = SelectionStrategyTester()
                logger.info("✅ SelectionStrategyTester统一架构初始化成功")
                
                # 检查是否找到真实选股脚本
                if tester.stock_select_script:
                    logger.info(f"✅ 找到真实选股脚本: {tester.stock_select_script}")
                    return {'status': 'PASS', 'message': '统一架构工作正常'}
                else:
                    return {'status': 'FAIL', 'message': '未找到真实选股脚本'}
                    
            except RuntimeError as e:
                if "脚本" in str(e) or "选股" in str(e):
                    logger.warning(f"⚠️ SelectionStrategyTester选股脚本要求未满足: {e}")
                    return {'status': 'REQUIREMENTS_NOT_MET', 'message': str(e)}
                else:
                    raise
                    
        except Exception as e:
            logger.error(f"❌ SelectionStrategyTester验证失败: {e}")
            return {'status': 'ERROR', 'message': str(e)}
    
    def _validate_closed_loop_validator(self) -> Dict[str, Any]:
        """验证ClosedLoopValidator统一架构"""
        try:
            from components.closed_loop_validator import ClosedLoopValidator
            
            # 测试统一架构初始化（无模式参数）
            try:
                validator = ClosedLoopValidator()
                logger.info("✅ ClosedLoopValidator统一架构初始化成功")
                
                # 测试统一计算引擎
                test_data = self._create_real_stock_data()
                result = validator._calculate_technical_indicators(test_data, 'MACD')
                
                if result is not None and len(result) > 0:
                    logger.info("✅ ClosedLoopValidator统一计算引擎工作正常")
                    return {'status': 'PASS', 'message': '统一架构工作正常'}
                else:
                    return {'status': 'FAIL', 'message': '统一计算引擎失败'}
                    
            except RuntimeError as e:
                if "引擎" in str(e) or "计算" in str(e):
                    logger.warning(f"⚠️ ClosedLoopValidator统一计算引擎未满足要求: {e}")
                    return {'status': 'REQUIREMENTS_NOT_MET', 'message': str(e)}
                else:
                    raise
                    
        except Exception as e:
            logger.error(f"❌ ClosedLoopValidator验证失败: {e}")
            return {'status': 'ERROR', 'message': str(e)}
    
    def _validate_unified_indicator_tester(self) -> Dict[str, Any]:
        """验证UnifiedIndicatorTester统一架构"""
        try:
            from unified_indicator_tester import UnifiedIndicatorTester
            
            # 测试统一架构初始化（无模式参数）
            try:
                tester = UnifiedIndicatorTester()
                logger.info("✅ UnifiedIndicatorTester统一架构初始化成功")
                return {'status': 'PASS', 'message': '统一架构工作正常'}
                    
            except RuntimeError as e:
                if "架构" in str(e) or "初始化" in str(e):
                    logger.warning(f"⚠️ UnifiedIndicatorTester统一架构要求未满足: {e}")
                    return {'status': 'REQUIREMENTS_NOT_MET', 'message': str(e)}
                else:
                    raise
                    
        except Exception as e:
            logger.error(f"❌ UnifiedIndicatorTester验证失败: {e}")
            return {'status': 'ERROR', 'message': str(e)}
    
    def _create_real_stock_data(self) -> pd.DataFrame:
        """创建真实格式的股票数据"""
        import numpy as np
        
        # 生成50天的真实股票数据
        dates = pd.date_range(end=pd.Timestamp.now(), periods=50, freq='D')
        
        # 生成真实的价格序列
        base_price = 10.0
        price_changes = np.random.normal(0, 0.02, 50)  # 2%标准差的价格变动
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(1.0, new_price))  # 确保价格不为负
        
        data = pd.DataFrame({
            'date': dates.strftime('%Y%m%d'),
            'code': ['TEST001'] * 50,
            'name': ['测试股票'] * 50,
            'open': [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices],
            'high': [p * (1 + np.random.uniform(0.005, 0.03)) for p in prices],
            'low': [p * (1 + np.random.uniform(-0.03, -0.005)) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 50),
            'industry': ['科技'] * 50
        })
        
        return data
    
    def _generate_validation_summary(self, execution_time: float) -> Dict[str, Any]:
        """生成验证总结报告"""
        summary = {
            'validation_time': datetime.now().isoformat(),
            'execution_time_seconds': execution_time,
            'components_tested': len(self.validation_results),
            'results': self.validation_results,
            'overall_status': 'UNKNOWN'
        }
        
        # 计算总体状态
        statuses = [result['status'] for result in self.validation_results.values() if result]
        
        if all(status == 'PASS' for status in statuses):
            summary['overall_status'] = 'ALL_PASS'
            summary['message'] = '所有组件都支持生产模式'
        elif any(status == 'PASS' for status in statuses):
            summary['overall_status'] = 'PARTIAL_PASS'
            summary['message'] = '部分组件支持生产模式，需要完善环境配置'
        else:
            summary['overall_status'] = 'ALL_FAIL'
            summary['message'] = '需要完善生产环境配置'
        
        # 生成建议
        summary['recommendations'] = self._generate_recommendations()
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """生成建议"""
        recommendations = []
        
        for component, result in self.validation_results.items():
            if result and result['status'] == 'REQUIREMENTS_NOT_MET':
                if component == 'buypoint_analyzer':
                    recommendations.append("需要安装和配置真实技术指标计算引擎")
                elif component == 'selection_strategy_tester':
                    recommendations.append("需要配置有效的选股脚本路径")
                elif component == 'closed_loop_validator':
                    recommendations.append("需要安装统一指标引擎")
                elif component == 'unified_indicator_tester':
                    recommendations.append("需要完善整体环境配置")
        
        if not recommendations:
            recommendations.append("所有组件都已准备好用于生产环境")
        
        return recommendations


def main():
    """主函数"""
    print("🚀 生产模式验证器")
    print("=" * 50)
    
    validator = ProductionModeValidator()
    summary = validator.run_full_validation()
    
    print("\n📊 验证结果汇总")
    print("=" * 50)
    print(f"总体状态: {summary['overall_status']}")
    print(f"执行时间: {summary['execution_time_seconds']:.2f} 秒")
    print(f"测试组件: {summary['components_tested']} 个")
    print(f"结果描述: {summary['message']}")
    
    print("\n📋 详细结果")
    print("-" * 30)
    for component, result in summary['results'].items():
        if result:
            status_icon = "✅" if result['status'] == 'PASS' else "⚠️" if result['status'] == 'REQUIREMENTS_NOT_MET' else "❌"
            print(f"{status_icon} {component}: {result['status']} - {result['message']}")
    
    print("\n💡 建议")
    print("-" * 30)
    for i, recommendation in enumerate(summary['recommendations'], 1):
        print(f"{i}. {recommendation}")
    
    print("\n" + "=" * 50)
    print("验证完成！")


if __name__ == "__main__":
    main() 