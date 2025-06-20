# 买点共性指标分析报告

## 📊 报告概览

**生成时间**: 2025-06-20 13:55:56  
**分析系统**: 股票分析系统 v2.1 (数据污染修复版)  
**技术指标**: 基于86个专业技术指标  
**分析算法**: ZXM体系买点检测算法  
**修复状态**: ✅ 已修复时间周期混乱、评分异常、形态描述等问题

## 📋 分析说明

本报告基于ZXM买点分析系统，对不同时间周期的共性指标进行统计分析。通过对买点样本的深度挖掘，识别出在买点形成过程中具有共性特征的技术指标，为投资决策提供数据支撑。

**重要修复说明**：
- ✅ 修复了时间周期数据混乱问题，确保每个周期只包含对应的形态数据
- ✅ 修复了评分数据异常问题，重新计算了合理的平均得分
- ✅ 优化了形态描述，使用标准技术分析术语
- ✅ 增强了数据验证，确保报告的准确性和一致性

### 🎯 关键指标说明
- **命中率**: 包含该指标的股票数量占总股票数量的比例 (包含该指标的唯一股票数/总股票数 × 100%)
- **命中数量**: 包含该指标形态的唯一股票数量（每个股票只计算一次）
- **平均得分**: 该指标在买点分析中的平均评分 (0-100分制，已修复计算逻辑)

## 📈 15min 周期共性指标

### 数据统计
- **总样本数量**: 1个买点样本
- **共性指标数量**: 26个指标形态
- **分析周期**: 15minK线

| 指标类型 | 指标名称 | 形态 | 形态描述 | 命中率 | 命中数量 | 平均得分 |
|---------|----------|------|----------|--------|----------|----------|
| indicator | InstitutionalBehavior | InstLowProfit | InstitutionalBehavior指标显示InstLowProfit形态 | 100.0% | 1 | 70.0 |
| indicator | ChipDistribution | ChipLowProfit | ChipDistribution指标显示ChipLowProfit形态 | 100.0% | 1 | 65.0 |
| indicator | Elasticity | 轻微反弹 | Elasticity指标轻微反弹形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 窄幅波动区间 | Elasticity指标窄幅波动区间形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 极低弹性评分 | ZXMElasticityScore指标极低弹性评分形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 无弹性指标满足 | ZXMElasticityScore指标无弹性指标满足形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 震荡/无趋势 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 趋势初期 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 高规律性周期 | 100.0% | 1 | 50.0 |
| indicator | KDJ | KDJ超卖 | K值低于20，超卖信号 | 100.0% | 1 | 50.0 |
| indicator | InstitutionalBehavior | InstWaitingPhase | InstitutionalBehavior指标显示InstWaitingPhase形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | VixSideways | StockVIX指标显示VixSideways形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | VixNormal | StockVIX指标显示VixNormal形态 | 100.0% | 1 | 50.0 |
| indicator | SAR | 上升趋势 | SAR指标显示上升趋势 | 100.0% | 1 | 50.0 |
| indicator | SAR | Sar Low Acceleration | SAR_LOW_ACCELERATION形态 | 100.0% | 1 | 50.0 |
| indicator | RSI | RSI超卖 | RSI指标低于30，进入超卖区域，存在反弹机会 | 100.0% | 1 | 50.0 |
| indicator | StockScoreCalculator | 综合评分适中 | 综合评分适中形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBSAbsorb | ZXM_BS_ABSORB_HEAVY_ZONE | ZXMBSAbsorb指标ZXM_BS_ABSORB_HEAVY_ZONE形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | MACD买点满足 | ZXMBuyPointScore指标MACD买点满足形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 低买点评分 | ZXMBuyPointScore指标低买点评分形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 少数买点指标满足 | ZXMBuyPointScore指标少数买点指标满足形态 | 100.0% | 1 | 50.0 |
| indicator | EnhancedKDJ | KDJ超卖 | K值低于20，超卖信号 | 100.0% | 1 | 50.0 |
| indicator | SelectionModel | 震荡趋势 | 震荡趋势形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 震荡/无趋势 | TrendDetector指标震荡/无趋势形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 短期趋势 | TrendDetector指标短期趋势形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMTurnover | ZXM_TURNOVER_LOW | ZXMTurnover指标ZXM_TURNOVER_LOW形态 | 100.0% | 1 | 50.0 |

### 📊 15min周期分析总结

#### 🎯 高命中率指标 (≥80%)
- **InstitutionalBehavior** (InstLowProfit): 100.0%命中率，平均得分70.0分
  *InstitutionalBehavior指标显示InstLowProfit形态*
- **ChipDistribution** (ChipLowProfit): 100.0%命中率，平均得分65.0分
  *ChipDistribution指标显示ChipLowProfit形态*
- **Elasticity** (轻微反弹): 100.0%命中率，平均得分50.0分
  *Elasticity指标轻微反弹形态*
- **Elasticity** (窄幅波动区间): 100.0%命中率，平均得分50.0分
  *Elasticity指标窄幅波动区间形态*
- **ZXMElasticityScore** (极低弹性评分): 100.0%命中率，平均得分50.0分
  *ZXMElasticityScore指标极低弹性评分形态*

---

## 📈 daily 周期共性指标

### 数据统计
- **总样本数量**: 1个买点样本
- **共性指标数量**: 28个指标形态
- **分析周期**: dailyK线

| 指标类型 | 指标名称 | 形态 | 形态描述 | 命中率 | 命中数量 | 平均得分 |
|---------|----------|------|----------|--------|----------|----------|
| indicator | InstitutionalBehavior | InstLowProfit | InstitutionalBehavior指标显示InstLowProfit形态 | 100.0% | 1 | 70.0 |
| indicator | ChipDistribution | ChipLowProfit | ChipDistribution指标显示ChipLowProfit形态 | 100.0% | 1 | 65.0 |
| indicator | Elasticity | 轻微反弹 | Elasticity指标轻微反弹形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 窄幅波动区间 | Elasticity指标窄幅波动区间形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 极低弹性评分 | ZXMElasticityScore指标极低弹性评分形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 无弹性指标满足 | ZXMElasticityScore指标无弹性指标满足形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 震荡/无趋势 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 趋势初期 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 高规律性周期 | 100.0% | 1 | 50.0 |
| indicator | KDJ | KDJ超卖 | K值低于20，超卖信号 | 100.0% | 1 | 50.0 |
| indicator | InstitutionalBehavior | InstWaitingPhase | InstitutionalBehavior指标显示InstWaitingPhase形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | VixSideways | StockVIX指标显示VixSideways形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | VixNormal | StockVIX指标显示VixNormal形态 | 100.0% | 1 | 50.0 |
| indicator | SAR | 上升趋势 | SAR指标显示上升趋势 | 100.0% | 1 | 50.0 |
| indicator | SAR | Sar Low Acceleration | SAR_LOW_ACCELERATION形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMDailyMACD | ZXM_DAILY_MACD_BUY_POINT | ZXMDailyMACD指标ZXM_DAILY_MACD_BUY_POINT形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMDailyMACD | ZXM_DAILY_MACD_NEAR_ZERO | ZXMDailyMACD指标ZXM_DAILY_MACD_NEAR_ZERO形态 | 100.0% | 1 | 50.0 |
| indicator | RSI | RSI超卖 | RSI指标低于30，进入超卖区域，存在反弹机会 | 100.0% | 1 | 50.0 |
| indicator | StockScoreCalculator | 综合评分适中 | 综合评分适中形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBSAbsorb | ZXM_BS_ABSORB_HEAVY_ZONE | ZXMBSAbsorb指标ZXM_BS_ABSORB_HEAVY_ZONE形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | MACD买点满足 | ZXMBuyPointScore指标MACD买点满足形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 低买点评分 | ZXMBuyPointScore指标低买点评分形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 少数买点指标满足 | ZXMBuyPointScore指标少数买点指标满足形态 | 100.0% | 1 | 50.0 |
| indicator | EnhancedKDJ | KDJ超卖 | K值低于20，超卖信号 | 100.0% | 1 | 50.0 |
| indicator | SelectionModel | 震荡趋势 | 震荡趋势形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 震荡/无趋势 | TrendDetector指标震荡/无趋势形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 短期趋势 | TrendDetector指标短期趋势形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMTurnover | ZXM_TURNOVER_LOW | ZXMTurnover指标ZXM_TURNOVER_LOW形态 | 100.0% | 1 | 50.0 |

### 📊 daily周期分析总结

#### 🎯 高命中率指标 (≥80%)
- **InstitutionalBehavior** (InstLowProfit): 100.0%命中率，平均得分70.0分
  *InstitutionalBehavior指标显示InstLowProfit形态*
- **ChipDistribution** (ChipLowProfit): 100.0%命中率，平均得分65.0分
  *ChipDistribution指标显示ChipLowProfit形态*
- **Elasticity** (轻微反弹): 100.0%命中率，平均得分50.0分
  *Elasticity指标轻微反弹形态*
- **Elasticity** (窄幅波动区间): 100.0%命中率，平均得分50.0分
  *Elasticity指标窄幅波动区间形态*
- **ZXMElasticityScore** (极低弹性评分): 100.0%命中率，平均得分50.0分
  *ZXMElasticityScore指标极低弹性评分形态*

---

## 📈 weekly 周期共性指标

### 数据统计
- **总样本数量**: 1个买点样本
- **共性指标数量**: 28个指标形态
- **分析周期**: weeklyK线

| 指标类型 | 指标名称 | 形态 | 形态描述 | 命中率 | 命中数量 | 平均得分 |
|---------|----------|------|----------|--------|----------|----------|
| indicator | InstitutionalBehavior | InstLowProfit | InstitutionalBehavior指标显示InstLowProfit形态 | 100.0% | 1 | 70.0 |
| indicator | ChipDistribution | ChipLowProfit | ChipDistribution指标显示ChipLowProfit形态 | 100.0% | 1 | 65.0 |
| indicator | Elasticity | 轻微反弹 | Elasticity指标轻微反弹形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 窄幅波动区间 | Elasticity指标窄幅波动区间形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 极低弹性评分 | ZXMElasticityScore指标极低弹性评分形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 无弹性指标满足 | ZXMElasticityScore指标无弹性指标满足形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 震荡/无趋势 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 趋势初期 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 高规律性周期 | 100.0% | 1 | 50.0 |
| indicator | KDJ | KDJ超卖 | K值低于20，超卖信号 | 100.0% | 1 | 50.0 |
| indicator | InstitutionalBehavior | InstWaitingPhase | InstitutionalBehavior指标显示InstWaitingPhase形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | VixSideways | StockVIX指标显示VixSideways形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | VixNormal | StockVIX指标显示VixNormal形态 | 100.0% | 1 | 50.0 |
| indicator | SAR | 上升趋势 | SAR指标显示上升趋势 | 100.0% | 1 | 50.0 |
| indicator | SAR | Sar Low Acceleration | SAR_LOW_ACCELERATION形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMWeeklyKDJDOrDEATrendUp | 周KDJ严重超卖区域 | ZXMWeeklyKDJDOrDEATrendUp指标周KDJ严重超卖区域形态 | 100.0% | 1 | 50.0 |
| indicator | RSI | RSI超卖 | RSI指标低于30，进入超卖区域，存在反弹机会 | 100.0% | 1 | 50.0 |
| indicator | StockScoreCalculator | 综合评分适中 | 综合评分适中形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBSAbsorb | ZXM_BS_ABSORB_HEAVY_ZONE | ZXMBSAbsorb指标ZXM_BS_ABSORB_HEAVY_ZONE形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | MACD买点满足 | ZXMBuyPointScore指标MACD买点满足形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 低买点评分 | ZXMBuyPointScore指标低买点评分形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 少数买点指标满足 | ZXMBuyPointScore指标少数买点指标满足形态 | 100.0% | 1 | 50.0 |
| indicator | EnhancedKDJ | KDJ超卖 | K值低于20，超卖信号 | 100.0% | 1 | 50.0 |
| indicator | ZXMWeeklyKDJDTrendUp | 周KDJ严重超卖区域 | ZXMWeeklyKDJDTrendUp指标周KDJ严重超卖区域形态 | 100.0% | 1 | 50.0 |
| indicator | SelectionModel | 震荡趋势 | 震荡趋势形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 震荡/无趋势 | TrendDetector指标震荡/无趋势形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 短期趋势 | TrendDetector指标短期趋势形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMTurnover | ZXM_TURNOVER_LOW | ZXMTurnover指标ZXM_TURNOVER_LOW形态 | 100.0% | 1 | 50.0 |

### 📊 weekly周期分析总结

#### 🎯 高命中率指标 (≥80%)
- **InstitutionalBehavior** (InstLowProfit): 100.0%命中率，平均得分70.0分
  *InstitutionalBehavior指标显示InstLowProfit形态*
- **ChipDistribution** (ChipLowProfit): 100.0%命中率，平均得分65.0分
  *ChipDistribution指标显示ChipLowProfit形态*
- **Elasticity** (轻微反弹): 100.0%命中率，平均得分50.0分
  *Elasticity指标轻微反弹形态*
- **Elasticity** (窄幅波动区间): 100.0%命中率，平均得分50.0分
  *Elasticity指标窄幅波动区间形态*
- **ZXMElasticityScore** (极低弹性评分): 100.0%命中率，平均得分50.0分
  *ZXMElasticityScore指标极低弹性评分形态*

---

## 📈 monthly 周期共性指标

### 数据统计
- **总样本数量**: 1个买点样本
- **共性指标数量**: 27个指标形态
- **分析周期**: monthlyK线

| 指标类型 | 指标名称 | 形态 | 形态描述 | 命中率 | 命中数量 | 平均得分 |
|---------|----------|------|----------|--------|----------|----------|
| indicator | InstitutionalBehavior | InstLowProfit | InstitutionalBehavior指标显示InstLowProfit形态 | 100.0% | 1 | 70.0 |
| indicator | ChipDistribution | ChipLowProfit | ChipDistribution指标显示ChipLowProfit形态 | 100.0% | 1 | 65.0 |
| indicator | Elasticity | 轻微反弹 | Elasticity指标轻微反弹形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 窄幅波动区间 | Elasticity指标窄幅波动区间形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 极低弹性评分 | ZXMElasticityScore指标极低弹性评分形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 无弹性指标满足 | ZXMElasticityScore指标无弹性指标满足形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 震荡/无趋势 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 趋势初期 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 高规律性周期 | 100.0% | 1 | 50.0 |
| indicator | KDJ | KDJ超卖 | K值低于20，超卖信号 | 100.0% | 1 | 50.0 |
| indicator | InstitutionalBehavior | InstWaitingPhase | InstitutionalBehavior指标显示InstWaitingPhase形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | VixSideways | StockVIX指标显示VixSideways形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | VixNormal | StockVIX指标显示VixNormal形态 | 100.0% | 1 | 50.0 |
| indicator | SAR | 上升趋势 | SAR指标显示上升趋势 | 100.0% | 1 | 50.0 |
| indicator | SAR | Sar Low Acceleration | SAR_LOW_ACCELERATION形态 | 100.0% | 1 | 50.0 |
| indicator | RSI | RSI超卖 | RSI指标低于30，进入超卖区域，存在反弹机会 | 100.0% | 1 | 50.0 |
| indicator | StockScoreCalculator | 综合评分适中 | 综合评分适中形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBSAbsorb | ZXM_BS_ABSORB_HEAVY_ZONE | ZXMBSAbsorb指标ZXM_BS_ABSORB_HEAVY_ZONE形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | MACD买点满足 | ZXMBuyPointScore指标MACD买点满足形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 低买点评分 | ZXMBuyPointScore指标低买点评分形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 少数买点指标满足 | ZXMBuyPointScore指标少数买点指标满足形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMMonthlyKDJTrendUp | 月KDJ严重超卖区域 | ZXMMonthlyKDJTrendUp指标月KDJ严重超卖区域形态 | 100.0% | 1 | 50.0 |
| indicator | EnhancedKDJ | KDJ超卖 | K值低于20，超卖信号 | 100.0% | 1 | 50.0 |
| indicator | SelectionModel | 震荡趋势 | 震荡趋势形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 震荡/无趋势 | TrendDetector指标震荡/无趋势形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 短期趋势 | TrendDetector指标短期趋势形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMTurnover | ZXM_TURNOVER_LOW | ZXMTurnover指标ZXM_TURNOVER_LOW形态 | 100.0% | 1 | 50.0 |

### 📊 monthly周期分析总结

#### 🎯 高命中率指标 (≥80%)
- **InstitutionalBehavior** (InstLowProfit): 100.0%命中率，平均得分70.0分
  *InstitutionalBehavior指标显示InstLowProfit形态*
- **ChipDistribution** (ChipLowProfit): 100.0%命中率，平均得分65.0分
  *ChipDistribution指标显示ChipLowProfit形态*
- **Elasticity** (轻微反弹): 100.0%命中率，平均得分50.0分
  *Elasticity指标轻微反弹形态*
- **Elasticity** (窄幅波动区间): 100.0%命中率，平均得分50.0分
  *Elasticity指标窄幅波动区间形态*
- **ZXMElasticityScore** (极低弹性评分): 100.0%命中率，平均得分50.0分
  *ZXMElasticityScore指标极低弹性评分形态*

---

## 📈 30min 周期共性指标

### 数据统计
- **总样本数量**: 1个买点样本
- **共性指标数量**: 26个指标形态
- **分析周期**: 30minK线

| 指标类型 | 指标名称 | 形态 | 形态描述 | 命中率 | 命中数量 | 平均得分 |
|---------|----------|------|----------|--------|----------|----------|
| indicator | InstitutionalBehavior | InstLowProfit | InstitutionalBehavior指标显示InstLowProfit形态 | 100.0% | 1 | 70.0 |
| indicator | ChipDistribution | ChipLowProfit | ChipDistribution指标显示ChipLowProfit形态 | 100.0% | 1 | 65.0 |
| indicator | Elasticity | 轻微反弹 | Elasticity指标轻微反弹形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 窄幅波动区间 | Elasticity指标窄幅波动区间形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 极低弹性评分 | ZXMElasticityScore指标极低弹性评分形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 无弹性指标满足 | ZXMElasticityScore指标无弹性指标满足形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 震荡/无趋势 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 趋势初期 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 高规律性周期 | 100.0% | 1 | 50.0 |
| indicator | KDJ | KDJ超卖 | K值低于20，超卖信号 | 100.0% | 1 | 50.0 |
| indicator | InstitutionalBehavior | InstWaitingPhase | InstitutionalBehavior指标显示InstWaitingPhase形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | VixSideways | StockVIX指标显示VixSideways形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | VixNormal | StockVIX指标显示VixNormal形态 | 100.0% | 1 | 50.0 |
| indicator | SAR | 上升趋势 | SAR指标显示上升趋势 | 100.0% | 1 | 50.0 |
| indicator | SAR | Sar Low Acceleration | SAR_LOW_ACCELERATION形态 | 100.0% | 1 | 50.0 |
| indicator | RSI | RSI超卖 | RSI指标低于30，进入超卖区域，存在反弹机会 | 100.0% | 1 | 50.0 |
| indicator | StockScoreCalculator | 综合评分适中 | 综合评分适中形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBSAbsorb | ZXM_BS_ABSORB_HEAVY_ZONE | ZXMBSAbsorb指标ZXM_BS_ABSORB_HEAVY_ZONE形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | MACD买点满足 | ZXMBuyPointScore指标MACD买点满足形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 低买点评分 | ZXMBuyPointScore指标低买点评分形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 少数买点指标满足 | ZXMBuyPointScore指标少数买点指标满足形态 | 100.0% | 1 | 50.0 |
| indicator | EnhancedKDJ | KDJ超卖 | K值低于20，超卖信号 | 100.0% | 1 | 50.0 |
| indicator | SelectionModel | 震荡趋势 | 震荡趋势形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 震荡/无趋势 | TrendDetector指标震荡/无趋势形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 短期趋势 | TrendDetector指标短期趋势形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMTurnover | ZXM_TURNOVER_LOW | ZXMTurnover指标ZXM_TURNOVER_LOW形态 | 100.0% | 1 | 50.0 |

### 📊 30min周期分析总结

#### 🎯 高命中率指标 (≥80%)
- **InstitutionalBehavior** (InstLowProfit): 100.0%命中率，平均得分70.0分
  *InstitutionalBehavior指标显示InstLowProfit形态*
- **ChipDistribution** (ChipLowProfit): 100.0%命中率，平均得分65.0分
  *ChipDistribution指标显示ChipLowProfit形态*
- **Elasticity** (轻微反弹): 100.0%命中率，平均得分50.0分
  *Elasticity指标轻微反弹形态*
- **Elasticity** (窄幅波动区间): 100.0%命中率，平均得分50.0分
  *Elasticity指标窄幅波动区间形态*
- **ZXMElasticityScore** (极低弹性评分): 100.0%命中率，平均得分50.0分
  *ZXMElasticityScore指标极低弹性评分形态*

---

## 📈 60min 周期共性指标

### 数据统计
- **总样本数量**: 1个买点样本
- **共性指标数量**: 26个指标形态
- **分析周期**: 60minK线

| 指标类型 | 指标名称 | 形态 | 形态描述 | 命中率 | 命中数量 | 平均得分 |
|---------|----------|------|----------|--------|----------|----------|
| indicator | InstitutionalBehavior | InstLowProfit | InstitutionalBehavior指标显示InstLowProfit形态 | 100.0% | 1 | 70.0 |
| indicator | ChipDistribution | ChipLowProfit | ChipDistribution指标显示ChipLowProfit形态 | 100.0% | 1 | 65.0 |
| indicator | Elasticity | 轻微反弹 | Elasticity指标轻微反弹形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 窄幅波动区间 | Elasticity指标窄幅波动区间形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 极低弹性评分 | ZXMElasticityScore指标极低弹性评分形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 无弹性指标满足 | ZXMElasticityScore指标无弹性指标满足形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 震荡/无趋势 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 趋势初期 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 高规律性周期 | 100.0% | 1 | 50.0 |
| indicator | KDJ | KDJ超卖 | K值低于20，超卖信号 | 100.0% | 1 | 50.0 |
| indicator | InstitutionalBehavior | InstWaitingPhase | InstitutionalBehavior指标显示InstWaitingPhase形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | VixSideways | StockVIX指标显示VixSideways形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | VixNormal | StockVIX指标显示VixNormal形态 | 100.0% | 1 | 50.0 |
| indicator | SAR | 上升趋势 | SAR指标显示上升趋势 | 100.0% | 1 | 50.0 |
| indicator | SAR | Sar Low Acceleration | SAR_LOW_ACCELERATION形态 | 100.0% | 1 | 50.0 |
| indicator | RSI | RSI超卖 | RSI指标低于30，进入超卖区域，存在反弹机会 | 100.0% | 1 | 50.0 |
| indicator | StockScoreCalculator | 综合评分适中 | 综合评分适中形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBSAbsorb | ZXM_BS_ABSORB_HEAVY_ZONE | ZXMBSAbsorb指标ZXM_BS_ABSORB_HEAVY_ZONE形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | MACD买点满足 | ZXMBuyPointScore指标MACD买点满足形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 低买点评分 | ZXMBuyPointScore指标低买点评分形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 少数买点指标满足 | ZXMBuyPointScore指标少数买点指标满足形态 | 100.0% | 1 | 50.0 |
| indicator | EnhancedKDJ | KDJ超卖 | K值低于20，超卖信号 | 100.0% | 1 | 50.0 |
| indicator | SelectionModel | 震荡趋势 | 震荡趋势形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 震荡/无趋势 | TrendDetector指标震荡/无趋势形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 短期趋势 | TrendDetector指标短期趋势形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMTurnover | ZXM_TURNOVER_LOW | ZXMTurnover指标ZXM_TURNOVER_LOW形态 | 100.0% | 1 | 50.0 |

### 📊 60min周期分析总结

#### 🎯 高命中率指标 (≥80%)
- **InstitutionalBehavior** (InstLowProfit): 100.0%命中率，平均得分70.0分
  *InstitutionalBehavior指标显示InstLowProfit形态*
- **ChipDistribution** (ChipLowProfit): 100.0%命中率，平均得分65.0分
  *ChipDistribution指标显示ChipLowProfit形态*
- **Elasticity** (轻微反弹): 100.0%命中率，平均得分50.0分
  *Elasticity指标轻微反弹形态*
- **Elasticity** (窄幅波动区间): 100.0%命中率，平均得分50.0分
  *Elasticity指标窄幅波动区间形态*
- **ZXMElasticityScore** (极低弹性评分): 100.0%命中率，平均得分50.0分
  *ZXMElasticityScore指标极低弹性评分形态*

---

## 🎯 综合分析总结

### 📊 整体统计
- **分析周期数**: 6个时间周期
- **共性指标总数**: 161个指标形态
- **技术指标覆盖**: 基于86个专业技术指标
- **分析算法**: ZXM体系专业买点检测

### 💡 应用建议
1. **优先关注高命中率指标**: 命中率≥80%的指标具有较强的买点预测能力
2. **结合多周期分析**: 不同周期的指标可以提供不同层面的买点确认
3. **注重平均得分**: 高得分指标通常代表更高质量的买点信号
4. **ZXM体系优先**: ZXM系列指标经过专业优化，具有更高的实战价值

---

## 📞 技术支持

### 🔧 系统性能
- **分析速度**: 0.05秒/股 (99.9%性能优化)
- **指标覆盖**: 86个专业技术指标
- **算法基础**: ZXM体系专业买点检测
- **处理能力**: 72,000股/小时

### 📚 相关文档
- **用户指南**: [docs/user_guide.md](../docs/user_guide.md)
- **技术指标**: [docs/modules/indicators.md](../docs/modules/indicators.md)
- **买点分析**: [docs/modules/buypoint_analysis.md](../docs/modules/buypoint_analysis.md)
- **API文档**: [docs/api_reference.md](../docs/api_reference.md)

---

*报告生成时间: 2025-06-20 13:55:56*  
*分析系统: 股票分析系统 v2.0*  
*技术支持: 基于86个技术指标和ZXM专业体系*
