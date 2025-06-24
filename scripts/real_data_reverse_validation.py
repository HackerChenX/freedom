#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
真实数据反向验证工具

使用ClickHouse真实数据执行策略选股，验证买点策略的有效性
"""

import sys
import os
import json
import argparse
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from strategy.strategy_executor import StrategyExecutor
from strategy.strategy_manager import StrategyManager
from db.data_manager_adapter import get_data_manager_adapter
from utils.logger import get_logger

logger = get_logger(__name__)


def convert_to_json_serializable(obj):
    """转换对象为JSON可序列化格式"""
    import numpy as np

    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif hasattr(obj, 'item'):  # 处理numpy标量
        return obj.item()
    else:
        return obj


class RealDataReverseValidator:
    """真实数据反向验证器"""
    
    def __init__(self):
        """初始化验证器"""
        try:
            self.data_manager = get_data_manager_adapter()
            self.strategy_executor = StrategyExecutor()
            self.strategy_manager = StrategyManager()
            print("✅ 成功连接到ClickHouse数据库")
            print("✅ 策略执行器初始化完成")
        except Exception as e:
            raise Exception(f"初始化失败: {e}")
    
    def validate_with_real_data(self, buypoints_file: str, strategy_file: str, config: dict = None) -> dict:
        """
        使用真实数据进行反向验证
        
        Args:
            buypoints_file: 原始买点数据文件
            strategy_file: 策略文件路径
            config: 验证配置
            
        Returns:
            dict: 验证结果
        """
        print("🚀 开始真实数据反向验证")
        print("=" * 60)
        
        # 默认配置
        if config is None:
            config = {
                "validation_date": None,  # 将从买点数据中自动提取
                "enable_progress": True,
                "cleanup_temp": True
            }
        
        try:
            # 1. 加载原始买点数据
            original_buypoints = self._load_buypoints(buypoints_file)
            
            # 2. 加载策略
            strategy = self._load_strategy(strategy_file)
            
            # 3. 提取原始买点股票列表
            original_stocks = self._extract_original_stocks(original_buypoints)

            # 3.5. 提取验证日期（如果配置中没有指定）
            if config.get("validation_date") is None:
                config["validation_date"] = self._extract_validation_date(original_buypoints)

            # 4. 使用真实数据执行策略选股
            selected_stocks = self._execute_real_strategy(strategy, config, buypoints_file)
            
            # 5. 计算匹配结果
            match_analysis = self._analyze_matches(original_stocks, selected_stocks)
            
            # 6. 获取选股详细信息
            selection_details = self._get_selection_details(selected_stocks, config)
            
            # 7. 生成验证报告
            validation_results = {
                "validation_info": {
                    "buypoints_file": buypoints_file,
                    "strategy_file": strategy_file,
                    "validation_date": config["validation_date"],
                    "validation_time": datetime.now().isoformat(),
                    "execution_mode": "real_data"
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
                    "selected_stocks": list(selected_stocks),
                    "selection_details": selection_details
                },
                "match_analysis": match_analysis,
                "validation_status": "success"
            }
            
            # 8. 评估验证质量
            quality_assessment = self._assess_validation_quality(validation_results)
            validation_results["quality_assessment"] = quality_assessment
            
            print(f"✅ 真实数据验证完成，匹配率: {match_analysis['match_rate']:.1%}")
            return validation_results
            
        except Exception as e:
            logger.error(f"真实数据验证失败: {e}")
            return {
                "validation_status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "execution_mode": "real_data"
            }
    
    def _load_buypoints(self, buypoints_file: str) -> pd.DataFrame:
        """加载买点数据"""
        try:
            if buypoints_file.endswith('.csv'):
                # 使用utf-8-sig编码处理BOM标记
                buypoints = pd.read_csv(buypoints_file, encoding='utf-8-sig')
            elif buypoints_file.endswith('.json'):
                buypoints = pd.read_json(buypoints_file)
            else:
                raise Exception("不支持的文件格式，请使用CSV或JSON")

            print(f"📊 成功加载买点数据: {len(buypoints)} 条记录")
            return buypoints

        except Exception as e:
            raise Exception(f"加载买点数据失败: {e}")
    
    def _load_strategy(self, strategy_file: str) -> dict:
        """加载策略文件"""
        try:
            with open(strategy_file, 'r', encoding='utf-8') as f:
                strategy = json.load(f)
            
            print(f"📋 成功加载策略: {strategy.get('name', '未知')}")
            print(f"📋 策略条件数: {len(strategy.get('conditions', []))}")
            return strategy
            
        except Exception as e:
            raise Exception(f"加载策略文件失败: {e}")
    
    def _extract_original_stocks(self, buypoints: pd.DataFrame) -> set:
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
            print(f"⚠️ 未找到明确的股票代码列，使用第一列: {stock_column}")
        
        # 转换为字符串类型并格式化为6位股票代码
        def format_stock_code(code):
            """格式化股票代码为6位字符串"""
            code_str = str(code).strip()
            # 如果是数字，补齐到6位
            if code_str.isdigit():
                return code_str.zfill(6)
            return code_str

        original_stocks = set(format_stock_code(code) for code in buypoints[stock_column].unique())
        print(f"📈 提取到 {len(original_stocks)} 只原始买点股票")
        print(f"📈 股票列表: {sorted(list(original_stocks))}")

        return original_stocks

    def _extract_validation_date(self, buypoints_data: pd.DataFrame) -> str:
        """从买点数据中提取验证日期"""
        # 尝试多种可能的日期列名
        date_columns = ['date', 'buypoint_date', 'buy_date', 'trade_date', 'timestamp']

        for date_col in date_columns:
            if date_col in buypoints_data.columns:
                # 使用买点数据中的最新日期作为验证日期
                date_values = buypoints_data[date_col]

                # 处理不同的日期格式
                if date_values.dtype == 'object':
                    # 字符串格式，可能是YYYYMMDD或YYYY-MM-DD
                    first_date = str(date_values.iloc[0])
                    if len(first_date) == 8 and first_date.isdigit():
                        # YYYYMMDD格式
                        dates = pd.to_datetime(date_values, format='%Y%m%d')
                    else:
                        # 其他格式，让pandas自动解析
                        dates = pd.to_datetime(date_values)
                else:
                    # 数值格式，可能是YYYYMMDD
                    first_date = str(date_values.iloc[0])
                    if len(first_date) == 8 and first_date.isdigit():
                        # YYYYMMDD格式的数值
                        dates = pd.to_datetime(date_values.astype(str), format='%Y%m%d')
                    else:
                        # 其他数值格式，可能是Unix时间戳
                        dates = pd.to_datetime(date_values, unit='s')

                latest_date = dates.max()
                validation_date = latest_date.strftime('%Y-%m-%d')
                print(f"📅 从买点数据中提取验证日期 (列: {date_col}): {validation_date}")
                return validation_date

        # 如果没有找到任何日期列，使用默认日期
        default_date = "2024-01-15"
        print(f"⚠️ 买点数据中没有找到日期列 {date_columns}，使用默认日期: {default_date}")
        return default_date

    def _execute_real_strategy(self, strategy: dict, config: dict, buypoints_file: str) -> set:
        """使用真实数据执行策略"""
        print("🔄 开始执行策略选股...")
        
        try:
            # 创建临时策略
            temp_strategy_id = f"real_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            print(f"💾 创建临时策略: {temp_strategy_id}")

            # 转换条件格式：将'indicator'字段转换为'indicator_id'字段
            converted_conditions = []
            for condition in strategy.get("conditions", []):
                converted_condition = condition.copy()

                # 将'indicator'字段转换为'indicator_id'字段
                if 'indicator' in converted_condition:
                    converted_condition['indicator_id'] = converted_condition.pop('indicator')

                # 确保必需字段存在
                if 'period' not in converted_condition:
                    converted_condition['period'] = '15min'  # 默认周期

                converted_conditions.append(converted_condition)

            # 准备策略执行计划（包含StrategyExecutor需要的所有字段）
            strategy_plan = {
                "strategy_id": temp_strategy_id,
                "name": strategy.get("name", temp_strategy_id),
                "description": f"临时验证策略 - {strategy.get('description', '')}",
                "conditions": converted_conditions,
                "condition_logic": strategy.get("condition_logic", "OR"),
                "version": strategy.get("version", "1.0"),
                "created_by": "reverse_validation",
                "tags": ["validation", "temporary"]
            }

            # 直接使用策略配置，不保存到数据库
            validation_date = config.get("validation_date", "2024-12-01")
            print(f"📅 执行策略选股，截止日期: {validation_date}")

            # 设置进度回调
            def progress_callback(progress, message):
                if config.get("enable_progress", True):
                    print(f"  📊 进度: {progress:.1%} - {message}")

            # 修改策略执行计划，只处理原始买点股票
            # 重新加载买点数据来获取原始股票列表
            buypoints_df = self._load_buypoints(buypoints_file)
            original_stocks = self._extract_original_stocks(buypoints_df)

            # 创建包含原始股票的股票列表DataFrame
            original_stocks_df = pd.DataFrame({
                'stock_code': list(original_stocks)
            })

            # 临时修改策略执行器的股票获取方法
            original_get_filtered_stock_list = self.strategy_executor._get_filtered_stock_list
            def custom_get_filtered_stock_list(filters):
                return original_stocks_df

            self.strategy_executor._get_filtered_stock_list = custom_get_filtered_stock_list

            try:
                # 直接使用策略执行计划执行，只处理原始买点股票
                selected_stocks_df = self.strategy_executor.execute_strategy(
                    strategy_plan=strategy_plan,
                    end_date=validation_date,
                    progress_callback=progress_callback
                )
            finally:
                # 恢复原始方法
                self.strategy_executor._get_filtered_stock_list = original_get_filtered_stock_list

            # 无需清理，因为没有保存到数据库
            print("✅ 策略执行完成，无需清理临时文件")
            
            # 提取股票代码
            if selected_stocks_df is not None and len(selected_stocks_df) > 0:
                # 尝试不同的股票代码列名
                stock_columns = ['stock_code', 'code', 'symbol', 'stock_symbol']
                stock_column = None
                
                for col in stock_columns:
                    if col in selected_stocks_df.columns:
                        stock_column = col
                        break
                
                if stock_column is None:
                    # 使用第一列
                    stock_column = selected_stocks_df.columns[0]
                    print(f"⚠️ 使用第一列作为股票代码: {stock_column}")
                
                # 转换为字符串类型并格式化股票代码
                def format_stock_code(code):
                    """格式化股票代码为6位字符串"""
                    code_str = str(code).strip()
                    # 如果是数字，补齐到6位
                    if code_str.isdigit():
                        return code_str.zfill(6)
                    return code_str

                selected_stocks = set(format_stock_code(code) for code in selected_stocks_df[stock_column].tolist())

                print(f"✅ 策略执行完成，选出 {len(selected_stocks)} 只股票")
                print(f"📈 选中股票: {sorted(list(selected_stocks))}")

                return selected_stocks
            else:
                print("⚠️ 策略执行未返回任何结果")
                return set()
                
        except Exception as e:
            logger.error(f"策略执行失败: {e}")
            raise Exception(f"策略执行失败: {e}")
    
    def _get_selection_details(self, selected_stocks: set, config: dict) -> dict:
        """获取选股详细信息"""
        if not selected_stocks:
            return {}
        
        try:
            # 获取股票基本信息
            details = {}
            validation_date = config.get("validation_date", "2024-12-01")
            
            for stock_code in selected_stocks:
                try:
                    # 获取股票基本信息
                    stock_info = self.data_manager.get_stock_basic_info(stock_code)
                    if stock_info:
                        details[stock_code] = {
                            "stock_name": stock_info.get("stock_name", "未知"),
                            "industry": stock_info.get("industry", "未知"),
                            "market": stock_info.get("market", "未知")
                        }
                    else:
                        details[stock_code] = {
                            "stock_name": "未知",
                            "industry": "未知", 
                            "market": "未知"
                        }
                except Exception as e:
                    logger.warning(f"获取股票 {stock_code} 信息失败: {e}")
                    details[stock_code] = {
                        "stock_name": "获取失败",
                        "industry": "获取失败",
                        "market": "获取失败"
                    }
            
            return details
            
        except Exception as e:
            logger.warning(f"获取选股详细信息失败: {e}")
            return {}
    
    def _analyze_matches(self, original_stocks: set, selected_stocks: set) -> dict:
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
            "matched_stocks": sorted(list(matched_stocks)),
            "missed_stocks": sorted(list(missed_stocks)),
            "false_positive_stocks": sorted(list(false_positive_stocks))
        }
        
        return analysis
    
    def _assess_validation_quality(self, results: dict) -> dict:
        """评估验证质量"""
        match_analysis = results["match_analysis"]
        match_rate = match_analysis["match_rate"]
        f1_score = match_analysis["f1_score"]
        precision = match_analysis["precision"]
        
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
        
        # 精确率评分
        if precision >= 0.8:
            assessment["scores"]["precision"] = "A"
        elif precision >= 0.6:
            assessment["scores"]["precision"] = "B"
        else:
            assessment["scores"]["precision"] = "C"
            assessment["recommendations"].append(f"精确率较低({precision:.1%})，策略可能过于宽松")
        
        # F1分数评分
        if f1_score >= 0.8:
            assessment["scores"]["f1_score"] = "A"
        elif f1_score >= 0.6:
            assessment["scores"]["f1_score"] = "B"
        else:
            assessment["scores"]["f1_score"] = "C"
            assessment["recommendations"].append(f"F1分数较低({f1_score:.2f})，需要平衡精确率和召回率")
        
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


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="真实数据反向验证工具")
    parser.add_argument("--buypoints", required=True, help="原始买点数据文件")
    parser.add_argument("--strategy", required=True, help="策略文件路径")
    parser.add_argument("--output", help="输出目录，默认为results/real_validation")
    parser.add_argument("--validation-date", help="验证日期，默认从买点数据中自动提取")
    parser.add_argument("--no-progress", action="store_true", help="禁用进度显示")
    parser.add_argument("--keep-temp", action="store_true", help="保留临时策略文件")
    
    args = parser.parse_args()
    
    # 设置输出目录
    output_dir = Path(args.output) if args.output else Path("results/real_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 验证配置
    config = {
        "validation_date": args.validation_date,  # None表示从买点数据中自动提取
        "enable_progress": not args.no_progress,
        "cleanup_temp": not args.keep_temp
    }
    
    try:
        # 创建验证器并执行验证
        validator = RealDataReverseValidator()
        
        print(f"🎯 真实数据反向验证")
        print(f"买点文件: {args.buypoints}")
        print(f"策略文件: {args.strategy}")
        print(f"验证日期: {config['validation_date']}")
        print("-" * 60)
        
        results = validator.validate_with_real_data(args.buypoints, args.strategy, config)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = output_dir / f"real_validation_{timestamp}.json"

        # 转换为JSON可序列化格式
        serializable_results = convert_to_json_serializable(results)

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        # 生成验证报告
        report_file = output_dir / f"real_validation_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("真实数据反向验证报告\n")
            f.write("=" * 60 + "\n\n")
            
            if results["validation_status"] == "success":
                # 基本信息
                validation_info = results["validation_info"]
                f.write("验证信息:\n")
                f.write(f"  买点文件: {validation_info['buypoints_file']}\n")
                f.write(f"  策略文件: {validation_info['strategy_file']}\n")
                f.write(f"  验证日期: {validation_info['validation_date']}\n")
                f.write(f"  执行模式: {validation_info['execution_mode']}\n\n")
                
                # 原始数据
                original_data = results["original_data"]
                f.write("原始数据:\n")
                f.write(f"  买点记录数: {original_data['total_buypoints']}\n")
                f.write(f"  唯一股票数: {original_data['unique_stocks']}\n")
                f.write(f"  股票列表: {', '.join(original_data['stock_list'])}\n\n")
                
                # 策略信息
                strategy_info = results["strategy_info"]
                f.write("策略信息:\n")
                f.write(f"  策略名称: {strategy_info['name']}\n")
                f.write(f"  条件数量: {strategy_info['condition_count']}\n")
                f.write(f"  逻辑类型: {strategy_info['logic_type']}\n\n")
                
                # 选股结果
                selection_results = results["selection_results"]
                f.write("选股结果:\n")
                f.write(f"  选中股票数: {selection_results['selected_count']}\n")
                f.write(f"  选中股票: {', '.join(selection_results['selected_stocks'])}\n\n")
                
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
                
                if match_analysis['matched_stocks']:
                    f.write(f"  匹配股票: {', '.join(match_analysis['matched_stocks'])}\n")
                if match_analysis['missed_stocks']:
                    f.write(f"  遗漏股票: {', '.join(match_analysis['missed_stocks'])}\n")
                if match_analysis['false_positive_stocks']:
                    f.write(f"  误选股票: {', '.join(match_analysis['false_positive_stocks'])}\n")
                f.write("\n")
                
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
        print("📊 真实数据验证结果摘要")
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
                print(f"\n💡 建议:")
                for rec in quality["recommendations"]:
                    print(f"  • {rec}")
        else:
            print(f"❌ 验证失败: {results.get('error', '未知错误')}")
        
        print(f"\n📁 详细结果已保存到:")
        print(f"  JSON: {result_file}")
        print(f"  报告: {report_file}")
        
    except Exception as e:
        print(f"❌ 验证过程失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
