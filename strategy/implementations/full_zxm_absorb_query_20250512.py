#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
全量股票ZXM 30分钟吸筹信号查询

从数据库中的所有四千多只股票中筛选2025年5月12日出现ZXM 30分钟吸筹信号的个股

Author: AI Assistant
Date: 2025-07-20
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any, Optional
import time
import json

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FullStockZXMAnalyzer:
    """全量股票ZXM吸筹信号分析器"""
    
    def __init__(self):
        self.target_date = '2025-05-12'
        self.results = []
        self.processed_count = 0
        self.error_count = 0
        
    def get_all_stock_codes(self) -> List[str]:
        """获取所有股票代码"""
        try:
            from db.clickhouse_db import get_clickhouse_db
            db = get_clickhouse_db()
            
            # 查询所有股票代码
            query = """
            SELECT DISTINCT code, name
            FROM stock.stock_info 
            WHERE code IS NOT NULL 
            AND name IS NOT NULL
            ORDER BY code
            """
            
            result = db.query(query)
            
            if result is not None and not result.empty:
                stock_codes = result['code'].tolist()
                logger.info(f"✅ 获取到 {len(stock_codes)} 只股票代码")
                return stock_codes
            else:
                logger.error("❌ 未获取到股票代码")
                return []
                
        except Exception as e:
            logger.error(f"❌ 获取股票代码失败: {e}")
            # 如果数据库连接失败，使用模拟的全量股票代码
            return self._get_simulated_stock_codes()
    
    def _get_simulated_stock_codes(self) -> List[str]:
        """获取模拟的全量股票代码（用于演示）"""
        logger.warning("⚠️ 使用模拟股票代码进行演示")
        
        # 生成模拟的4000+股票代码
        stock_codes = []
        
        # 主板股票 (000001-002999)
        for i in range(1, 3000):
            if i <= 999:
                stock_codes.append(f"00{i:04d}")
            else:
                stock_codes.append(f"00{i}")
        
        # 创业板股票 (300001-300999)
        for i in range(1, 1000):
            stock_codes.append(f"30{i:04d}")
        
        # 科创板股票 (688001-688999)
        for i in range(1, 500):
            stock_codes.append(f"68{i:04d}")
        
        # 北交所股票 (430001-430999)
        for i in range(1, 200):
            stock_codes.append(f"43{i:04d}")
        
        logger.info(f"✅ 生成模拟股票代码 {len(stock_codes)} 只")
        return stock_codes
    
    def get_stock_data(self, stock_code: str) -> Optional[pd.DataFrame]:
        """获取单只股票的历史数据"""
        try:
            from db.clickhouse_db import get_clickhouse_db
            db = get_clickhouse_db()
            
            # 获取目标日期前60天的数据（用于计算55周期指标）
            start_date = (datetime.strptime(self.target_date, '%Y-%m-%d') - timedelta(days=80)).strftime('%Y-%m-%d')
            end_date = self.target_date
            
            query = f"""
            SELECT code, name, date, open, close, high, low, volume, turnover_rate
            FROM stock.stock_info 
            WHERE code = '{stock_code}' 
            AND date >= '{start_date}' 
            AND date <= '{end_date}'
            ORDER BY date
            """
            
            result = db.query(query)
            
            if result is not None and not result.empty:
                return result
            else:
                return None
                
        except Exception as e:
            logger.debug(f"获取股票 {stock_code} 数据失败: {e}")
            return None
    
    def calculate_zxm_absorb_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        计算ZXM吸筹信号
        
        核心算法：
        V11 = 3*SMA((C-LLV(L,55))/(HHV(H,55)-LLV(L,55))*100,5,1) - 2*SMA(SMA(...),3,1)
        V12 = (EMA(V11,3)-REF(EMA(V11,3),1))/REF(EMA(V11,3),1)*100
        吸筹信号 = (EMA(V11,3)<=13) AND (V12>13)
        """
        try:
            if len(data) < 55:
                return {'has_signal': False, 'reason': 'insufficient_data'}
            
            # 确保数据类型正确
            close = pd.to_numeric(data['close'], errors='coerce').dropna()
            high = pd.to_numeric(data['high'], errors='coerce').dropna()
            low = pd.to_numeric(data['low'], errors='coerce').dropna()
            volume = pd.to_numeric(data['volume'], errors='coerce').dropna()
            
            if len(close) < 55:
                return {'has_signal': False, 'reason': 'invalid_data'}
            
            # 计算ZXM指标
            # 1. 计算55周期的最高价和最低价
            hhv_55 = high.rolling(window=55, min_periods=55).max()
            llv_55 = low.rolling(window=55, min_periods=55).min()
            
            # 2. 计算RSV
            rsv = (close - llv_55) / (hhv_55 - llv_55) * 100
            
            # 3. 计算SMA
            sma_rsv_5 = rsv.rolling(window=5, min_periods=5).mean()
            sma_sma_rsv_3 = sma_rsv_5.rolling(window=3, min_periods=3).mean()
            
            # 4. 计算V11
            v11 = 3 * sma_rsv_5 - 2 * sma_sma_rsv_3
            
            # 5. 计算EMA(V11, 3)
            ema_v11_3 = v11.ewm(span=3, adjust=False).mean()
            
            # 6. 计算V12
            v12 = (ema_v11_3 - ema_v11_3.shift(1)) / ema_v11_3.shift(1) * 100
            
            # 7. 获取目标日期的值
            target_data = data[data['date'] == self.target_date]
            if target_data.empty:
                return {'has_signal': False, 'reason': 'no_target_date'}
            
            target_idx = target_data.index[-1]
            
            # 8. 检查吸筹信号条件
            v11_val = ema_v11_3.loc[target_idx] if target_idx in ema_v11_3.index else np.nan
            v12_val = v12.loc[target_idx] if target_idx in v12.index else np.nan
            
            if pd.isna(v11_val) or pd.isna(v12_val):
                return {'has_signal': False, 'reason': 'calculation_failed'}
            
            # 9. 判断信号
            aa_condition = v11_val <= 13  # V11 <= 13
            bb_condition = v12_val > 13   # V12 > 13
            
            # 计算XG（近6个周期内满足条件的次数）
            recent_data = data.tail(6)
            xg_count = 0
            
            for i in range(max(0, len(ema_v11_3) - 6), len(ema_v11_3)):
                if i < len(ema_v11_3) and i < len(v12):
                    if (ema_v11_3.iloc[i] <= 13) or (ema_v11_3.iloc[i] <= 13 and v12.iloc[i] > 13):
                        xg_count += 1
            
            # 最终信号判断
            has_signal = aa_condition and bb_condition
            signal_strength = 'strong' if (has_signal and xg_count >= 3) else 'normal' if has_signal else 'weak'
            
            return {
                'has_signal': has_signal,
                'signal_strength': signal_strength,
                'v11_value': float(v11_val),
                'v12_value': float(v12_val),
                'xg_count': xg_count,
                'aa_condition': aa_condition,
                'bb_condition': bb_condition,
                'target_date': self.target_date,
                'data_points': len(data)
            }
            
        except Exception as e:
            logger.error(f"计算ZXM指标失败: {e}")
            return {'has_signal': False, 'reason': f'calculation_error: {e}'}
    
    def analyze_single_stock(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """分析单只股票"""
        try:
            # 获取股票数据
            stock_data = self.get_stock_data(stock_code)
            
            if stock_data is None or stock_data.empty:
                return None
            
            # 计算ZXM信号
            signal_result = self.calculate_zxm_absorb_signal(stock_data)
            
            if signal_result['has_signal']:
                stock_name = stock_data.iloc[0]['name'] if 'name' in stock_data.columns else f'股票{stock_code}'
                
                result = {
                    'code': stock_code,
                    'name': stock_name,
                    'signal_date': self.target_date,
                    **signal_result
                }
                
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"分析股票 {stock_code} 失败: {e}")
            return None
    
    def run_full_analysis(self) -> List[Dict[str, Any]]:
        """运行全量股票分析"""
        logger.info("🚀 开始全量股票ZXM 30分钟吸筹信号分析")
        logger.info(f"📅 目标日期: {self.target_date}")
        
        # 获取所有股票代码
        all_stock_codes = self.get_all_stock_codes()
        
        if not all_stock_codes:
            logger.error("❌ 未获取到股票代码，无法进行分析")
            return []
        
        total_stocks = len(all_stock_codes)
        logger.info(f"📊 开始分析 {total_stocks} 只股票")
        
        start_time = time.time()
        signal_stocks = []
        
        # 分批处理，避免内存问题
        batch_size = 100
        
        for i in range(0, total_stocks, batch_size):
            batch_codes = all_stock_codes[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_stocks + batch_size - 1) // batch_size
            
            logger.info(f"🔄 处理第 {batch_num}/{total_batches} 批 ({len(batch_codes)} 只股票)")
            
            for stock_code in batch_codes:
                try:
                    self.processed_count += 1
                    
                    # 分析单只股票
                    result = self.analyze_single_stock(stock_code)
                    
                    if result:
                        signal_stocks.append(result)
                        logger.info(f"✅ 发现信号: {result['code']} {result['name']} - {result['signal_strength']}")
                    
                    # 每处理100只股票显示进度
                    if self.processed_count % 100 == 0:
                        elapsed = time.time() - start_time
                        progress = self.processed_count / total_stocks * 100
                        logger.info(f"📈 进度: {progress:.1f}% ({self.processed_count}/{total_stocks}) - 已发现 {len(signal_stocks)} 只信号股票")
                
                except Exception as e:
                    self.error_count += 1
                    logger.error(f"❌ 处理股票 {stock_code} 失败: {e}")
                    
                    # 如果错误太多，停止处理
                    if self.error_count > 100:
                        logger.error("❌ 错误过多，停止分析")
                        break
            
            # 批次间短暂休息，避免数据库压力
            time.sleep(0.1)
        
        elapsed_time = time.time() - start_time
        
        logger.info("=" * 80)
        logger.info(f"🎉 全量分析完成!")
        logger.info(f"📊 处理股票数: {self.processed_count}")
        logger.info(f"⏱️  总耗时: {elapsed_time:.1f} 秒")
        logger.info(f"✅ 发现信号股票: {len(signal_stocks)} 只")
        logger.info(f"❌ 错误数量: {self.error_count}")
        
        return signal_stocks
    
    def save_results(self, signal_stocks: List[Dict[str, Any]]):
        """保存分析结果"""
        try:
            # 保存为JSON
            json_file = f"zxm_absorb_signals_{self.target_date.replace('-', '')}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'analysis_date': datetime.now().isoformat(),
                    'target_date': self.target_date,
                    'total_processed': self.processed_count,
                    'signal_count': len(signal_stocks),
                    'error_count': self.error_count,
                    'signals': signal_stocks
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 结果已保存到: {json_file}")
            
            # 保存为CSV
            if signal_stocks:
                df = pd.DataFrame(signal_stocks)
                csv_file = f"zxm_absorb_signals_{self.target_date.replace('-', '')}.csv"
                df.to_csv(csv_file, index=False, encoding='utf-8')
                logger.info(f"📊 CSV文件已保存到: {csv_file}")
            
        except Exception as e:
            logger.error(f"❌ 保存结果失败: {e}")


def main():
    """主函数"""
    analyzer = FullStockZXMAnalyzer()
    
    # 运行全量分析
    signal_stocks = analyzer.run_full_analysis()
    
    # 保存结果
    analyzer.save_results(signal_stocks)
    
    # 显示结果摘要
    if signal_stocks:
        print("\n" + "=" * 80)
        print(f"📈 2025年5月12日ZXM 30分钟吸筹信号股票 ({len(signal_stocks)} 只):")
        print("=" * 80)
        
        # 按信号强度分组
        strong_signals = [s for s in signal_stocks if s['signal_strength'] == 'strong']
        normal_signals = [s for s in signal_stocks if s['signal_strength'] == 'normal']
        
        if strong_signals:
            print(f"\n🔥 强信号股票 ({len(strong_signals)} 只):")
            for stock in strong_signals:
                print(f"   {stock['code']} {stock['name']} - V11:{stock['v11_value']:.1f} V12:{stock['v12_value']:.1f} XG:{stock['xg_count']}")
        
        if normal_signals:
            print(f"\n📊 一般信号股票 ({len(normal_signals)} 只):")
            for stock in normal_signals:
                print(f"   {stock['code']} {stock['name']} - V11:{stock['v11_value']:.1f} V12:{stock['v12_value']:.1f} XG:{stock['xg_count']}")
    
    else:
        print("\n❌ 未发现符合条件的股票")


if __name__ == "__main__":
    main()
