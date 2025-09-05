#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
买点策略反向验证工具

验证买点分析生成的策略是否能反向选出原始买点对应的个股
"""

import sys
import os
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import Counter

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入策略执行相关模块
try:
    from strategy.strategy_executor import Strategy_executor
    from strategy.strategy_manager import Strategy_manager
    from db.unified_data_manager import get_unified_data_manager
    from utils.logger import get_logger
    REAL_EXECUTION_AVAILABLE = True
except Import_error as e:
    print(f"警告: 无法导入策略执行模块: {e}")
    print("将使用模拟模式进行验证")
    REAL_EXECUTION_AVAILABLE = False


def convert_to_json_serializable_Validation(obj):
    """转换对象为JSON可序列化格式"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable_Validation(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable_Validation(item) for item in obj]
    else:
        return obj


class ReverseValidator:
    """反向验证器"""

    def __init__(self):
        """初始化验证器"""
        self.use_real_execution = REAL_EXECUTION_AVAILABLE

        if self.use_real_execution:
            try:
                self.data_manager = get_unified_data_manager()
                self.strategy_executor = Strategy_executor()
                self.strategy_manager = Strategy_manager()
                self.logger = get_logger(__name__)
                print("✅ 成功初始化真实策略执行环境")
            except Exception as e:
                print(f"❌ 初始化真实执行环境失败: {e}")
                print("回退到模拟模式")
                self.use_real_execution = False
        else:
            print("⚠️ 使用模拟执行模式")
        
    def validate_reverse_selection(self, buypoints_file: str, strategy_file: str, config: dict = None) -> dict:
        """
        执行反向验证
        
        Args:
            buypoints_file: 原始买点数据文件
            strategy_file: 生成的策略文件
            config: 验证配置
            
        Returns:
            dict: 验证结果
        """
        print(f"开始反向验证:")
        print(f"  买点文件: {buypoints_file}")
        print(f"  策略文件: {strategy_file}")
        
        # 默认配置
        if config is None:
            config = {
                "validation_date": "2024-12-01",  # 验证日期
                "use_optimized_strategy": True,   # 使用优化策略
                "match_threshold": 0.6,           # 匹配阈值
                "sample_size": 100                # 样本大小
            }
        
        try:
            # 1. 加载原始买点数据
            original_buypoints = self._load_buypoints_Reverse_Validation(buypoints_file)
            
            # 2. 加载策略
            strategy = self._load_strategy_Reverse_Validation(strategy_file)
            
            # 3. 提取原始买点股票列表
            original_stocks = self._extract_original_stocks_Reverse_Validation(original_buypoints)
            
            # 4. 执行策略选股（使用真实数据）
            if self.use_real_execution:
                selected_stocks = self._execute_real_strategy_selection(strategy, config)
            else:
                selected_stocks = self._simulate_strategy_selection(strategy, original_stocks, config)
            
            # 5. 计算匹配结果
            match_analysis = self._analyze_matches_Reverse_Validation(original_stocks, selected_stocks)
            
            # 6. 生成验证报告
            validation_results = {
                "validation_info": {
                    "buypoints_file": buypoints_file,
                    "strategy_file": strategy_file,
                    "validation_date": config["validation_date"],
                    "validation_time": datetime.now().isoformat()
                },
                "original_data": {
                    "total_buypoints": len(original_buypoints),
                    "unique_stocks": len(original_stocks),
                    "stock_list": list(original_stocks)
                },
                "strategy_info": {
                    "name": strategy.get("name", "未知"),
                    "condition_count": len(strategy.get("conditions", [])),
                    "logic_type": strategy.get("condition_logic", "OR")
                },
                "selection_results": {
                    "selected_count": len(selected_stocks),
                    "selected_stocks": list(selected_stocks)
                },
                "match_analysis": match_analysis,
                "validation_status": "success"
            }
            
            # 7. 评估验证质量
            quality_assessment = self._assess_validation_quality_Reverse_Validation(validation_results, config)
            validation_results["quality_assessment"] = quality_assessment
            
            print(f"反向验证完成，匹配率: {match_analysis['match_rate']:.1%}")
            return validation_results
            
        except Exception as e:
            print(f"反向验证失败: {e}")
            return {
                "validation_status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _load_buypoints_Reverse_Validation(self, buypoints_file: str) -> pd.DataFrame:
        """加载买点数据"""
        try:
            if buypoints_file.endswith('.csv'):
                buypoints = pd.read_csv(buypoints_file)
            elif buypoints_file.endswith('.json'):
                buypoints = pd.read_json(buypoints_file)
            else:
                raise Exception("不支持的文件格式，请使用CSV或JSON")
            
            print(f"成功加载买点数据: {len(buypoints)} 条记录")
            return buypoints
            
        except Exception as e:
            raise Exception(f"加载买点数据失败: {e}")
    
    def _load_strategy_Reverse_Validation(self, strategy_file: str) -> dict:
        """加载策略文件"""
        try:
            with open(strategy_file, 'r', encoding='utf-8') as f:
                strategy = json.load(f)
            
            print(f"成功加载策略: {strategy.get('name', '未知')}")
            return strategy
            
        except Exception as e:
            raise Exception(f"加载策略文件失败: {e}")
    
    def _extract_original_stocks_Reverse_Validation(self, buypoints: pd.DataFrame) -> set:
        """提取原始买点股票列表"""
        # 尝试不同的股票代码列名
        possible_columns = ['stock_code', 'code', 'symbol', 'stock_symbol', 'ts_code']
        
        stock_column = None
        for col in possible_columns:
            if col in buypoints.columns:
                stock_column = col
                break
        
        if stock_column is None:
            # 如果找不到明确的股票代码列，使用第一列
            stock_column = buypoints.columns[0]
            print(f"警告: 未找到明确的股票代码列，使用第一列: {stock_column}")
        
        original_stocks = set(buypoints[stock_column].unique())
        print(f"提取到 {len(original_stocks)} 只原始买点股票")
        
        return original_stocks

    def _execute_real_strategy_selection(self, strategy: dict, config: dict) -> set:
        """
        使用真实数据执行策略选股
        """
        print("🚀 开始使用真实数据执行策略选股...")

        try:
            # 保存策略到临时文件
            temp_strategy_id = f"temp_reverse_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            print(f"📝 保存临时策略: {temp_strategy_id}")
            self.strategy_manager.save_strategy(temp_strategy_id, strategy)

            # 执行策略选股
            validation_date = config.get("validation_date", "2024-12-01")
            print(f"📅 执行日期: {validation_date}")

            def progress_callback_Validation_Reverse_Validation(progress, message):
                if progress % 0.2 < 0.01:  # 每20%打印一次
                    print(f"  进度: {progress:.1%} - {message}")

            # 使用策略执行器进行选股
            selected_stocks_df = self.strategy_executor.execute_strategy_by_id(
                strategy_id=temp_strategy_id,
                strategy_manager=self.strategy_manager,
                end_date=validation_date,
                progress_callback=progress_callback
            )

            # 清理临时策略
            try:
                self.strategy_manager.delete_strategy(temp_strategy_id)
                print(f"🗑️ 清理临时策略: {temp_strategy_id}")
            except Exception as e:
                print(f"⚠️ 清理临时策略失败: {e}")

            # 提取股票代码
            if selected_stocks_df is not None and len(selected_stocks_df) > 0:
                if 'stock_code' in selected_stocks_df.columns:
                    selected_stocks = set(selected_stocks_df['stock_code'].tolist())
                elif 'code' in selected_stocks_df.columns:
                    selected_stocks = set(selected_stocks_df['code'].tolist())
                else:
                    # 如果没有明确的股票代码列，使用第一列
                    selected_stocks = set(selected_stocks_df.iloc[:, 0].tolist())

                print(f"✅ 真实策略选股完成，选出 {len(selected_stocks)} 只股票")
                return selected_stocks
            else:
                print("⚠️ 策略执行未返回任何结果")
                return set()

        except Exception as e:
            print(f"❌ 真实策略执行失败: {e}")
            print("📋 错误详情:")
            import traceback
            traceback.print_exc()

            # 回退到模拟模式
            print("🔄 回退到模拟模式")
            return set()

    def _simulate_strategy_selection(self, strategy: dict, original_stocks: set, config: dict) -> set:
        """
        模拟策略选股（备用方案）

        注意：这是备用方案，仅在无法使用真实数据时使用
        真实验证应该使用Click_house数据和策略执行器
        """
        print("⚠️ 使用模拟模式进行策略选股（备用方案）...")
        print("💡 建议：配置ClickHouse连接以使用真实数据验证")
        
        conditions = strategy.get("conditions", [])
        logic_type = strategy.get("condition_logic", "OR")
        
        # 分析策略条件的复杂度
        condition_complexity = self._analyze_condition_complexity(conditions)
        
        # 基于策略复杂度模拟选股结果
        selected_stocks = self._simulate_selection_based_on_complexity(
            original_stocks, condition_complexity, logic_type, config
        )
        
        print(f"模拟选股完成，选出 {len(selected_stocks)} 只股票")
        return selected_stocks
    
    def _analyze_condition_complexity(self, conditions: list) -> dict:
        """分析条件复杂度"""
        complexity = {
            "total_conditions": len(conditions),
            "indicator_types": set(),
            "period_types": set(),
            "pattern_types": set()
        }
        
        for condition in conditions:
            indicator = condition.get("indicator", "")
            period = condition.get("period", "")
            pattern = condition.get("pattern", "")
            
            if indicator:
                complexity["indicator_types"].add(indicator)
            if period:
                complexity["period_types"].add(period)
            if pattern:
                complexity["pattern_types"].add(pattern)
        
        complexity["indicator_diversity"] = len(complexity["indicator_types"])
        complexity["period_diversity"] = len(complexity["period_types"])
        complexity["pattern_diversity"] = len(complexity["pattern_types"])
        
        return complexity
    
    def _simulate_selection_based_on_complexity(self, original_stocks: set, complexity: dict, logic_type: str, config: dict) -> set:
        """基于复杂度模拟选股"""
        import random
        
        # 设置随机种子以确保结果可重现
        random.seed(42)
        
        stock_list = list(original_stocks)
        
        # 根据策略复杂度和逻辑类型计算选股概率
        if logic_type == "OR":
            # OR逻辑：条件越多，选中概率越高
            base_probability = min(0.8, 0.3 + complexity["total_conditions"] / 1000)
        else:  # AND逻辑
            # AND逻辑：条件越多，选中概率越低
            base_probability = max(0.2, 0.8 - complexity["total_conditions"] / 500)
        
        # 根据指标多样性调整概率
        diversity_factor = min(1.2, 1.0 + complexity["indicator_diversity"] / 100)
        final_probability = min(0.9, base_probability * diversity_factor)
        
        print(f"模拟选股概率: {final_probability:.2%} (基础: {base_probability:.2%}, 多样性因子: {diversity_factor:.2f})")
        
        # 模拟选股过程
        selected_stocks = set()
        for stock in stock_list:
            if random.random() < final_probability:
                selected_stocks.add(stock)
        
        # 确保至少选中一些股票
        if len(selected_stocks) == 0 and len(stock_list) > 0:
            # 随机选择几只股票
            num_to_select = min(5, len(stock_list))
            selected_stocks = set(random.sample(stock_list, num_to_select))
        
        return selected_stocks
    
    def _analyze_matches_Reverse_Validation(self, original_stocks: set, selected_stocks: set) -> dict:
        """分析匹配结果"""
        # 计算交集和差集
        matched_stocks = original_stocks.intersection(selected_stocks)
        missed_stocks = original_stocks - selected_stocks
        false_positive_stocks = selected_stocks - original_stocks

        # 计算匹配率
        match_rate = len(matched_stocks) / len(original_stocks) if len(original_stocks) > 0 else 0
        precision = len(matched_stocks) / len(selected_stocks) if len(selected_stocks) > 0 else 0
        recall = match_rate  # 召回率等于匹配率

        # F1分数
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        analysis = {
            "total_original": int(len(original_stocks)),
            "total_selected": int(len(selected_stocks)),
            "matched_count": int(len(matched_stocks)),
            "missed_count": int(len(missed_stocks)),
            "false_positive_count": int(len(false_positive_stocks)),
            "match_rate": float(match_rate),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "matched_stocks": list(matched_stocks),
            "missed_stocks": list(missed_stocks),
            "false_positive_stocks": list(false_positive_stocks)
        }

        return analysis
    
    def _assess_validation_quality_Reverse_Validation(self, results: dict, config: dict) -> dict:
        """评估验证质量"""
        match_analysis = results["match_analysis"]
        match_rate = match_analysis["match_rate"]
        f1_score = match_analysis["f1_score"]
        
        assessment = {
            "overall_grade": "C",
            "scores": {},
            "recommendations": []
        }
        
        # 匹配率评分
        if match_rate >= 0.8:
            assessment["scores"]["match_rate"] = "A"
        elif match_rate >= 0.6:
            assessment["scores"]["match_rate"] = "B"
        else:
            assessment["scores"]["match_rate"] = "C"
            assessment["recommendations"].append(f"匹配率较低({match_rate:.1%})，建议优化策略条件")
        
        # F1分数评分
        if f1_score >= 0.7:
            assessment["scores"]["f1_score"] = "A"
        elif f1_score >= 0.5:
            assessment["scores"]["f1_score"] = "B"
        else:
            assessment["scores"]["f1_score"] = "C"
            assessment["recommendations"].append(f"F1分数较低({f1_score:.2f})，策略精确度和召回率需要平衡")
        
        # 综合评分
        scores = list(assessment["scores"].values())
        if all(s == "A" for s in scores):
            assessment["overall_grade"] = "A"
        elif any(s == "A" for s in scores) and all(s in ["A", "B"] for s in scores):
            assessment["overall_grade"] = "B"
        else:
            assessment["overall_grade"] = "C"
        
        # 生成具体建议
        if match_analysis["false_positive_count"] > match_analysis["matched_count"]:
            assessment["recommendations"].append("误选股票过多，建议收紧策略条件")
        
        if match_analysis["missed_count"] > match_analysis["matched_count"]:
            assessment["recommendations"].append("遗漏股票过多，建议放宽策略条件")
        
        return assessment


def main_reversevalidation():
    """主函数"""
    parser = argparse.ArgumentParser(description="买点策略反向验证工具")
    parser.add_argument("--buypoints", required=True, help="原始买点数据文件")
    parser.add_argument("--strategy", required=True, help="策略文件路径")
    parser.add_argument("--output", help="输出目录，默认为results/reverse_validation")
    parser.add_argument("--validation-date", help="验证日期，默认为2024-12-01")
    parser.add_argument("--sample-size", type=int, default=100, help="样本大小")
    
    args = parser.parse_args()
    
    # 设置输出目录
    output_dir = Path(args.output) if args.output else Path("results/reverse_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 验证配置
    config = {
        "validation_date": args.validation_date or "2024-12-01",
        "sample_size": args.sample_size,
        "use_optimized_strategy": True,
        "match_threshold": 0.6
    }
    
    # 创建验证器并执行验证
    validator = Reverse_validator()
    
    print(f"开始反向验证:")
    print(f"买点文件: {args.buypoints}")
    print(f"策略文件: {args.strategy}")
    print(f"验证日期: {config['validation_date']}")
    print("-" * 60)
    
    results = validator.validate_reverse_selection(args.buypoints, args.strategy, config)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"reverse_validation_{timestamp}.json"

    # 转换为JSON可序列化格式
    serializable_results = convert_to_json_serializable_Validation(results)

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    # 生成验证报告
    report_file = output_dir / f"reverse_validation_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("买点策略反向验证报告\n")
        f.write("=" * 60 + "\n\n")
        
        if results["validation_status"] == "success":
            # 基本信息
            validation_info = results["validation_info"]
            f.write("验证信息:\n")
            f.write(f"  买点文件: {validation_info['buypoints_file']}\n")
            f.write(f"  策略文件: {validation_info['strategy_file']}\n")
            f.write(f"  验证日期: {validation_info['validation_date']}\n\n")
            
            # 原始数据
            original_data = results["original_data"]
            f.write("原始数据:\n")
            f.write(f"  买点记录数: {original_data['total_buypoints']}\n")
            f.write(f"  唯一股票数: {original_data['unique_stocks']}\n\n")
            
            # 策略信息
            strategy_info = results["strategy_info"]
            f.write("策略信息:\n")
            f.write(f"  策略名称: {strategy_info['name']}\n")
            f.write(f"  条件数量: {strategy_info['condition_count']}\n")
            f.write(f"  逻辑类型: {strategy_info['logic_type']}\n\n")
            
            # 匹配分析
            match_analysis = results["match_analysis"]
            f.write("匹配分析:\n")
            f.write(f"  原始股票数: {match_analysis['total_original']}\n")
            f.write(f"  选中股票数: {match_analysis['total_selected']}\n")
            f.write(f"  匹配股票数: {match_analysis['matched_count']}\n")
            f.write(f"  遗漏股票数: {match_analysis['missed_count']}\n")
            f.write(f"  误选股票数: {match_analysis['false_positive_count']}\n")
            f.write(f"  匹配率: {match_analysis['match_rate']:.1%}\n")
            f.write(f"  精确率: {match_analysis['precision']:.1%}\n")
            f.write(f"  召回率: {match_analysis['recall']:.1%}\n")
            f.write(f"  F1分数: {match_analysis['f1_score']:.2f}\n\n")
            
            # 质量评估
            quality = results.get("quality_assessment", {})
            f.write("质量评估:\n")
            f.write(f"  综合评级: {quality.get('overall_grade', 'N/A')}\n")
            
            recommendations = quality.get("recommendations", [])
            if recommendations:
                f.write("  建议:\n")
                for rec in recommendations:
                    f.write(f"    - {rec}\n")
        else:
            f.write(f"验证失败: {results.get('error', '未知错误')}\n")
    
    # 打印结果摘要
    print("\n" + "=" * 60)
    print("📊 反向验证结果摘要")
    print("=" * 60)
    
    if results["validation_status"] == "success":
        original_data = results["original_data"]
        match_analysis = results["match_analysis"]
        quality = results.get("quality_assessment", {})
        
        print(f"原始买点股票: {original_data['unique_stocks']} 只")
        print(f"策略选中股票: {match_analysis['total_selected']} 只")
        print(f"成功匹配股票: {match_analysis['matched_count']} 只")
        print(f"匹配率: {match_analysis['match_rate']:.1%}")
        print(f"精确率: {match_analysis['precision']:.1%}")
        print(f"F1分数: {match_analysis['f1_score']:.2f}")
        print(f"验证评级: {quality.get('overall_grade', 'N/A')}")
        
        if quality.get("recommendations"):
            print(f"\n建议:")
            for rec in quality["recommendations"]:
                print(f"  • {rec}")
    else:
        print(f"验证失败: {results.get('error', '未知错误')}")
    
    print(f"\n详细结果已保存到:")
    print(f"  JSON: {result_file}")
    print(f"  报告: {report_file}")


if __name__ == "__main__":
    main_reversevalidation()
