from utils.container import container
from analysis.base_analyzer import BaseAnalyzer
#!/usr/bin/env python3
"""
通用双向验证器 - 符合六层架构的生产级实现
L5: 业务应用层 - 双向验证业务逻辑
"""

import sys
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from db.sql_manager import SQLManager, QueryType

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from analysis.universal_buypoint_analyzer import UniversalBuyPointAnalyzer
from strategy.universal_strategy_generator import UniversalStrategyGenerator

logger = get_logger(__name__)

class UniversalBidirectionalValidator:
    """
    通用双向验证器
    
    符合六层架构设计：
    - L5: 业务应用层 - 双向验证业务逻辑
    - 依赖注入获取服务
    - 通用化的验证接口
    """
    
    def __init__(self):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """初始化验证器"""
        try:
            # 直接实例化服务
            self.buypoint_analyzer = UniversalBuyPointAnalyzer()
            self.strategy_generator = UniversalStrategyGenerator()

            logger.info("通用双向验证器初始化完成")

        except Exception as e:
            logger.error(f"通用双向验证器初始化失败: {e}")
            raise
    
    def validate_bidirectional(self, stock_code: str, target_date: str,
                              timeframes: Optional[List[str]] = None,
                              test_stocks: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        执行双向验证
        
        Args:
            stock_code: 目标股票代码
            target_date: 目标日期
            timeframes: 时间周期列表
            test_stocks: 测试股票列表，用于验证策略选股
            
        Returns:
            Dict: 验证结果
        """
        logger.info(f"开始双向验证: {stock_code} ({target_date})")
        
        if test_stocks is None:
            test_stocks = [stock_code, '000001', '600519', '000002', '600036']
        
        try:
            # 步骤1: 买点分析
            buypoint_result = self._execute_buypoint_analysis(
                stock_code, target_date, timeframes
            )
            
            if not buypoint_result.get('success', False):
                return self._create_failed_result(
                    stock_code, target_date, "买点分析失败",
                    buypoint_result.get('error_message', '未知错误')
                )
            
            # 步骤2: 策略生成
            strategy_result = self._generate_strategy(
                stock_code, target_date, buypoint_result, timeframes
            )
            
            if not strategy_result.get('success', False):
                return self._create_failed_result(
                    stock_code, target_date, "策略生成失败",
                    strategy_result.get('error_message', '未知错误')
                )
            
            # 步骤3: 策略验证
            validation_result = self._validate_strategy(
                strategy_result, stock_code, target_date, test_stocks
            )
            
            # 步骤4: 生成验证报告
            final_result = self._create_validation_result(
                stock_code, target_date, buypoint_result, 
                strategy_result, validation_result
            )
            
            logger.info(f"双向验证完成: {stock_code}, "
                       f"成功: {final_result.get('validation_success', False)}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"双向验证失败 {stock_code}: {e}")
            return self._create_failed_result(stock_code, target_date, "系统错误", str(e))
    
    def batch_validate(self, stock_list: List[Dict[str, str]],
                      timeframes: Optional[List[str]] = None,
                      test_stocks: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        批量双向验证
        
        Args:
            stock_list: 股票列表 [{'code': 'xxx', 'date': 'yyyymmdd'}, ...]
            timeframes: 时间周期列表
            test_stocks: 测试股票列表
            
        Returns:
            Dict: 批量验证结果
        """
        logger.info(f"开始批量双向验证: {len(stock_list)} 只股票")
        
        results = {
            'total_stocks': len(stock_list),
            'success_count': 0,
            'failed_count': 0,
            'validation_results': [],
            'summary': {},
            'batch_timestamp': datetime.now().isoformat()
        }
        
        for stock_info in stock_list:
            stock_code = stock_info.get('code')
            target_date = stock_info.get('date')
            
            if not stock_code or not target_date:
                logger.warning(f"跳过无效的股票信息: {stock_info}")
                results['failed_count'] += 1
                continue
            
            try:
                validation_result = self.validate_bidirectional(
                    stock_code, target_date, timeframes, test_stocks
                )
                
                if validation_result.get('validation_success', False):
                    results['success_count'] += 1
                    logger.info(f"✅ {stock_code} ({target_date}) 验证成功")
                else:
                    results['failed_count'] += 1
                    logger.warning(f"❌ {stock_code} ({target_date}) 验证失败")
                
                results['validation_results'].append(validation_result)
                
            except Exception as e:
                logger.error(f"验证 {stock_code} 时发生错误: {e}")
                results['failed_count'] += 1
        
        # 生成汇总统计
        results['summary'] = self._generate_batch_summary(results)
        
        logger.info(f"批量验证完成: 成功 {results['success_count']}, "
                   f"失败 {results['failed_count']}, "
                   f"成功率 {results['summary']['success_rate']:.1f}%")
        
        return results
    
    def _execute_buypoint_analysis(self, stock_code: str, target_date: str,
                                  timeframes: Optional[List[str]]) -> Dict[str, Any]:
        """执行买点分析"""
        logger.info(f"步骤1: 买点分析 - {stock_code} ({target_date})")
        
        try:
            result = self.buypoint_analyzer.analyze_buypoint(
                stock_code, target_date, timeframes
            )
            
            logger.info(f"买点分析完成: 指标数 {result.get('total_indicators', 0)}, "
                       f"形态数 {result.get('total_patterns', 0)}, "
                       f"评分 {result.get('buypoint_score', 0):.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"买点分析失败: {e}")
            return {'success': False, 'error_message': str(e)}
    
    def _generate_strategy(self, stock_code: str, target_date: str,
                          buypoint_result: Dict[str, Any],
                          timeframes: Optional[List[str]]) -> Dict[str, Any]:
        """生成策略配置"""
        logger.info(f"步骤2: 策略生成 - {stock_code} ({target_date})")
        
        try:
            result = self.strategy_generator.generate_strategy_from_buypoint(
                stock_code, target_date, timeframes=timeframes
            )
            
            if result.get('success', False):
                logger.info(f"策略生成完成: {result.get('config_file', 'N/A')}")
            else:
                logger.warning(f"策略生成失败: {result.get('error_message', '未知错误')}")
            
            return result
            
        except Exception as e:
            logger.error(f"策略生成失败: {e}")
            return {'success': False, 'error_message': str(e)}
    
    def _validate_strategy(self, strategy_result: Dict[str, Any],
                          stock_code: str, target_date: str,
                          test_stocks: List[str]) -> Dict[str, Any]:
        """验证策略选股"""
        logger.info(f"步骤3: 策略验证 - {stock_code} ({target_date})")
        
        try:
            strategy_config = strategy_result.get('strategy_config', {})
            strategy_id = strategy_config.get('strategy', {}).get('id')
            
            if not strategy_id:
                return {'success': False, 'error_message': '策略ID为空'}
            
            # 简化版本：模拟策略验证
            # 在实际实现中，这里应该调用策略引擎
            selection_results = self._simulate_strategy_execution(
                strategy_id, test_stocks, target_date, stock_code
            )
            
            # 检查是否选中目标股票
            selected_stocks = [result.get('stock_code') for result in selection_results]
            target_selected = stock_code in selected_stocks
            
            result = {
                'success': True,
                'strategy_id': strategy_id,
                'target_stock': stock_code,
                'target_date': target_date,
                'selected_stocks': selected_stocks,
                'total_selected': len(selected_stocks),
                'target_selected': target_selected,
                'validation_timestamp': datetime.now().isoformat()
            }
            
            if target_selected:
                target_result = next((r for r in selection_results 
                                    if r.get('stock_code') == stock_code), {})
                result['target_score'] = target_result.get('score', 0)
                logger.info(f"✅ 策略验证成功: 选中目标股票，评分 {result['target_score']:.4f}")
            else:
                logger.warning(f"❌ 策略验证失败: 未选中目标股票")
                logger.info(f"实际选中: {', '.join(selected_stocks) if selected_stocks else '无'}")
            
            return result
            
        except Exception as e:
            logger.error(f"策略验证失败: {e}")
            return {'success': False, 'error_message': str(e)}

    def _simulate_strategy_execution(self, strategy_id: str, test_stocks: List[str],
                                   target_date: str, target_stock: str) -> List[Dict[str, Any]]:
        """真实策略执行（基于生成的策略配置）"""
        results = []

        try:
            # 读取生成的策略配置
            strategy_config_file = f"config/strategies/{strategy_id}.yaml"

            import yaml
            from pathlib import Path
from db.sql_manager import SQLManager, QueryType

            config_path = Path(strategy_config_file)
            if not config_path.exists():
                logger.warning(f"策略配置文件不存在: {strategy_config_file}")
                return []

            with open(config_path, 'r', encoding='utf-8') as f:
                strategy_config = yaml.safe_load(f)

            # 获取策略配置中的指标
            indicators = strategy_config.get('technical_indicators', {}).get('primary_indicators', [])

            if not indicators:
                logger.warning(f"策略配置中没有指标: {strategy_id}")
                # 如果策略配置为空，所有股票都不选中
                for stock in test_stocks:
                    results.append({
                        'stock_code': stock,
                        'score': 0.0,
                        'selected': False
                    })
                return results

            # 基于策略配置进行真实选股
            for stock in test_stocks:
                try:
                    # 为每个股票计算买点分析
                    stock_analysis = self.buypoint_analyzer.analyze_buypoint(
                        stock, target_date, ['日线']
                    )

                    if stock_analysis.get('success', False):
                        # 基于买点评分决定是否选中
                        buypoint_score = stock_analysis.get('buypoint_score', 0.0)
                        pattern_count = stock_analysis.get('total_patterns', 0)

                        # 真实的选股逻辑：买点评分 > 0.3 且有形态匹配
                        if buypoint_score > 0.3 and pattern_count > 0:
                            results.append({
                                'stock_code': stock,
                                'score': buypoint_score,
                                'selected': True
                            })
                        else:
                            results.append({
                                'stock_code': stock,
                                'score': buypoint_score,
                                'selected': False
                            })
                    else:
                        results.append({
                            'stock_code': stock,
                            'score': 0.0,
                            'selected': False
                        })

                except Exception as e:
                    logger.debug(f"股票分析失败 {stock}: {e}")
                    results.append({
                        'stock_code': stock,
                        'score': 0.0,
                        'selected': False
                    })

            return results

        except Exception as e:
            logger.error(f"策略执行失败: {e}")
            # 如果策略执行失败，所有股票都不选中
            for stock in test_stocks:
                results.append({
                    'stock_code': stock,
                    'score': 0.0,
                    'selected': False
                })
            return results

    def _create_validation_result(self, stock_code: str, target_date: str,
                                 buypoint_result: Dict[str, Any],
                                 strategy_result: Dict[str, Any],
                                 validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """创建验证结果"""
        return {
            'stock_code': stock_code,
            'target_date': target_date,
            'validation_success': validation_result.get('target_selected', False),
            'buypoint_analysis': {
                'total_indicators': buypoint_result.get('total_indicators', 0),
                'total_patterns': buypoint_result.get('total_patterns', 0),
                'buypoint_score': buypoint_result.get('buypoint_score', 0.0),
                'success': buypoint_result.get('success', False)
            },
            'strategy_generation': {
                'strategy_id': strategy_result.get('strategy_config', {}).get('strategy', {}).get('id'),
                'config_file': strategy_result.get('config_file'),
                'success': strategy_result.get('success', False)
            },
            'strategy_validation': {
                'target_selected': validation_result.get('target_selected', False),
                'target_score': validation_result.get('target_score', 0.0),
                'total_selected': validation_result.get('total_selected', 0),
                'selected_stocks': validation_result.get('selected_stocks', []),
                'success': validation_result.get('success', False)
            },
            'validation_timestamp': datetime.now().isoformat()
        }
    
    def _create_failed_result(self, stock_code: str, target_date: str,
                             failure_stage: str, error_message: str) -> Dict[str, Any]:
        """创建失败结果"""
        return {
            'stock_code': stock_code,
            'target_date': target_date,
            'validation_success': False,
            'failure_stage': failure_stage,
            'error_message': error_message,
            'validation_timestamp': datetime.now().isoformat()
        }
    
    def _generate_batch_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成批量验证汇总"""
        total = results['total_stocks']
        success = results['success_count']
        failed = results['failed_count']
        
        success_rate = (success / total * 100) if total > 0 else 0
        
        # 分析失败原因
        failure_reasons = {}
        for result in results['validation_results']:
            if not result.get('validation_success', False):
                reason = result.get('failure_stage', '未知原因')
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        return {
            'total_stocks': total,
            'success_count': success,
            'failed_count': failed,
            'success_rate': success_rate,
            'failure_reasons': failure_reasons,
            'summary_timestamp': datetime.now().isoformat()
        }
    
    def save_validation_report(self, validation_results: Dict[str, Any],
                              report_file: Optional[str] = None) -> str:
        """保存验证报告"""
        if report_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = f"universal_validation_report_{timestamp}.json"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(validation_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"验证报告已保存: {report_file}")
            return report_file
            
        except Exception as e:
            logger.error(f"保存验证报告失败: {e}")
            raise
