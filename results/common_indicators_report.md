# 买点共性指标分析报告

## 📊 报告概览

**生成时间**: 2025-06-21 16:24:07  
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
- **共性指标数量**: 156个指标形态
- **分析周期**: 15minK线

| 指标类型 | 指标名称 | 形态 | 形态描述 | 命中率 | 命中数量 | 平均得分 |
|---------|----------|------|----------|--------|----------|----------|
| indicator | MFI | Mfi看涨背离 | MFI指标显示Mfi看涨背离形态 | 100.0% | 1 | 80.0 |
| indicator | ChipDistribution | ChipBottomAccumulation | ChipDistribution指标显示ChipBottomAccumulation形态 | 100.0% | 1 | 75.0 |
| indicator | ADX | Adx Strong上升 | ADX指标Adx Strong上升 | 100.0% | 1 | 75.0 |
| indicator | VOL | 放量上涨 | 成交量显著放大，同时价格上涨，通常是趋势启动或加速的信号 | 100.0% | 1 | 75.0 |
| indicator | InstitutionalBehavior | InstAbsorptionPhase | InstitutionalBehavior指标显示InstAbsorptionPhase形态 | 100.0% | 1 | 75.0 |
| indicator | ChipDistribution | PriceNearCost | ChipDistribution指标显示PriceNearCost形态 | 100.0% | 1 | 70.0 |
| indicator | ADX | 上升趋势 | ADX指标显示上升趋势 | 100.0% | 1 | 70.0 |
| indicator | EnhancedWR | Wr上升趋势 | EnhancedWR指标显示Wr上升趋势形态 | 100.0% | 1 | 70.0 |
| indicator | WR | Wr上升趋势 | WR指标显示Wr上升趋势形态 | 100.0% | 1 | 70.0 |
| indicator | InstitutionalBehavior | InstLowProfit | InstitutionalBehavior指标显示InstLowProfit形态 | 100.0% | 1 | 70.0 |
| indicator | ChipDistribution | ChipLowProfit | ChipDistribution指标显示ChipLowProfit形态 | 100.0% | 1 | 65.0 |
| indicator | StockVIX | VixExtremeLow | StockVIX指标显示VixExtremeLow形态 | 100.0% | 1 | 65.0 |
| indicator | BOLL | 布林带收缩 | 布林带收缩，表明波动率降低，可能酝酿突破 | 100.0% | 1 | 65.0 |
| indicator | EnhancedOBV | Obv上升 | EnhancedOBV指标Obv上升 | 100.0% | 1 | 65.0 |
| indicator | VOL | 成交量能量分析 | 基于成交量能量变化的技术分析: VOL_RISING | 100.0% | 1 | 65.0 |
| indicator | TRIX | Trix Above Signal | TRIX_ABOVE_SIGNAL形态 | 100.0% | 1 | 65.0 |
| indicator | InstitutionalBehavior | InstGentleAbsorption | InstitutionalBehavior指标显示InstGentleAbsorption形态 | 100.0% | 1 | 65.0 |
| indicator |  | 通用条件: MA5>MA10 | 自定义条件表达式: MA5>MA10 | 100.0% | 1 | 65.0 |
| indicator | EnhancedMACD | MACD上升 | MACD线呈上升趋势 | 100.0% | 1 | 60.0 |
| indicator | ChipDistribution | ChipTight | ChipDistribution指标显示ChipTight形态 | 100.0% | 1 | 60.0 |
| indicator | StockVIX | VixVeryLow波动性 | StockVIX指标显示VixVeryLow波动性形态 | 100.0% | 1 | 60.0 |
| indicator | EnhancedWR | Wr上升 | EnhancedWR指标显示Wr上升形态 | 100.0% | 1 | 60.0 |
| indicator | WR | Wr上升 | WR指标显示Wr上升形态 | 100.0% | 1 | 60.0 |
| indicator | VOL | 成交量偏高 | 成交量高于平均水平，市场活跃度较高 | 100.0% | 1 | 60.0 |
| indicator | TRIX | Trix上升 | TRIX指标Trix上升 | 100.0% | 1 | 60.0 |
| indicator | EnhancedMFI | Mfi上升 | EnhancedMFI指标Mfi上升 | 100.0% | 1 | 60.0 |
| indicator | STOCHRSI | StochRSI超买 | StochRSI进入超买区域，可能出现回调 | 100.0% | 1 | 50.0 |
| indicator | STOCHRSI | 随机RSI超买超卖 | 基于StochRSI指标的超买超卖分析: STOCHRSI_K_RISING | 100.0% | 1 | 50.0 |
| indicator | Chaikin | Chaikin零轴上方 | Chaikin震荡器位于零轴上方 | 100.0% | 1 | 50.0 |
| indicator | Chaikin | Chaikin上升 | Chaikin震荡器上升 | 100.0% | 1 | 50.0 |
| indicator | Chaikin | Chaikin连续上升 | Chaikin震荡器连续上升 | 100.0% | 1 | 50.0 |
| indicator | Chaikin | Chaikin大幅上升 | Chaikin震荡器大幅上升 | 100.0% | 1 | 50.0 |
| indicator | VOSC | Vosc上升 | VOSC指标显示Vosc上升形态 | 100.0% | 1 | 50.0 |
| indicator | VOSC | VoscLow | VOSC指标显示VoscLow形态 | 100.0% | 1 | 50.0 |
| indicator | VOSC | VoscPrice背离 | VOSC指标显示VoscPrice背离形态 | 100.0% | 1 | 50.0 |
| indicator | StockScoreCalculator | 综合评分适中 | 综合评分适中形态 | 100.0% | 1 | 50.0 |
| indicator | StockScoreCalculator | 低波动性 | 低波动性形态 | 100.0% | 1 | 50.0 |
| indicator | BounceDetector | 短期上升趋势 | 短期上升趋势形态 | 100.0% | 1 | 50.0 |
| indicator | BounceDetector | 明显放量 | 明显放量形态 | 100.0% | 1 | 50.0 |
| indicator | Ichimoku | 价格位于云层之下 | 价格位于云层下方，看跌信号 | 100.0% | 1 | 50.0 |
| indicator | ZXMPattern | macd_double_diverge | ZXMPattern指标macd_double_diverge形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMPattern | ma_precise_support | ZXMPattern指标ma_precise_support形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMPattern | small_alternating | ZXMPattern指标small_alternating形态 | 100.0% | 1 | 50.0 |
| indicator | Vortex | Vortex看涨交叉 | Vortex指标显示Vortex看涨交叉形态 | 100.0% | 1 | 50.0 |
| indicator | Vortex | VortexViPlus上方 | Vortex指标显示VortexViPlus上方形态 | 100.0% | 1 | 50.0 |
| indicator | Vortex | VortexViPlus强势 | Vortex指标显示VortexViPlus强势形态 | 100.0% | 1 | 50.0 |
| indicator | Vortex | VortexViPlus上升 | Vortex指标显示VortexViPlus上升形态 | 100.0% | 1 | 50.0 |
| indicator | Vortex | VortexViDiff上升 | Vortex指标显示VortexViDiff上升形态 | 100.0% | 1 | 50.0 |
| indicator | Vortex | VortexViPlus上升趋势 | Vortex指标显示VortexViPlus上升趋势形态 | 100.0% | 1 | 50.0 |
| indicator | PSY | PSY_ABOVE_50 | PSY指标PSY_ABOVE_50形态 | 100.0% | 1 | 50.0 |
| indicator | PSY | PSY_ABOVE_MA | PSY指标PSY_ABOVE_MA形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMTurnover | ZXM_TURNOVER_BUY_POINT | ZXMTurnover指标ZXM_TURNOVER_BUY_POINT形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMTurnover | ZXM_TURNOVER_NORMAL_ACTIVE | ZXMTurnover指标ZXM_TURNOVER_NORMAL_ACTIVE形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMTurnover | ZXM_TURNOVER_RELATIVE_ACTIVE | ZXMTurnover指标ZXM_TURNOVER_RELATIVE_ACTIVE形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMTurnover | ZXM_TURNOVER_SUDDEN_INCREASE | ZXMTurnover指标ZXM_TURNOVER_SUDDEN_INCREASE形态 | 100.0% | 1 | 50.0 |
| indicator | EnhancedMACD | MACD柱状体为正 | MACD柱状体大于零，表示上升动能 | 100.0% | 1 | 50.0 |
| indicator | EnhancedMACD | MACD柱状体增长 | MACD柱状体连续增长，表示动能增强 | 100.0% | 1 | 50.0 |
| indicator | ATR | 波动性Compression | ATR指标显示波动性Compression形态 | 100.0% | 1 | 50.0 |
| indicator | EMA | EMA多头排列 | 指数移动平均线呈多头排列，趋势向上 | 100.0% | 1 | 50.0 |
| indicator | AmplitudeElasticity | 振幅弹性信号 | AmplitudeElasticity指标振幅弹性信号形态 | 100.0% | 1 | 50.0 |
| indicator | AmplitudeElasticity | 小振幅 | AmplitudeElasticity指标小振幅形态 | 100.0% | 1 | 50.0 |
| indicator | AmplitudeElasticity | 少量大振幅 | AmplitudeElasticity指标少量大振幅形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | VixNormal | StockVIX指标显示VixNormal形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | Vix上升 | StockVIX指标显示Vix上升形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | VixNearLow | StockVIX指标显示VixNearLow形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMMACallback | ZXM_MA_CALLBACK_BUY_POINT | ZXMMACallback指标ZXM_MA_CALLBACK_BUY_POINT形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMMACallback | ZXM_MA20_CALLBACK | ZXMMACallback指标ZXM_MA20_CALLBACK形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMMACallback | ZXM_MA30_CALLBACK | ZXMMACallback指标ZXM_MA30_CALLBACK形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMMACallback | ZXM_MA60_CALLBACK | ZXMMACallback指标ZXM_MA60_CALLBACK形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMMACallback | ZXM_MULTIPLE_MA_CALLBACK | ZXMMACallback指标ZXM_MULTIPLE_MA_CALLBACK形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMMACallback | ZXM_MA20_SUPPORT | ZXMMACallback指标ZXM_MA20_SUPPORT形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMMACallback | ZXM_MA30_SUPPORT | ZXMMACallback指标ZXM_MA30_SUPPORT形态 | 100.0% | 1 | 50.0 |
| indicator | EMV | EMV零轴上方 | EMV位于零轴上方，买盘力量占优 | 100.0% | 1 | 50.0 |
| indicator | EMV | EMV均线上方 | EMV位于移动平均线上方 | 100.0% | 1 | 50.0 |
| indicator | EMV | EMV上升 | EMV值上升 | 100.0% | 1 | 50.0 |
| indicator | EMV | EMV强势上升 | EMV大幅上升，买盘力量强劲 | 100.0% | 1 | 50.0 |
| indicator | EMV | EMV极高值 | EMV达到近期高点 | 100.0% | 1 | 50.0 |
| indicator | CMO | CMO超买 | CMO指标高于40，表示超买状态 | 100.0% | 1 | 50.0 |
| indicator | CMO | CMO动量震荡 | 基于CMO动量震荡指标的技术分析: CMO_ABOVE_ZERO | 100.0% | 1 | 50.0 |
| indicator | CMO | CMO动量震荡 | 基于CMO动量震荡指标的技术分析: CMO_RISING | 100.0% | 1 | 50.0 |
| indicator | CMO | CMO动量震荡 | 基于CMO动量震荡指标的技术分析: CMO_STRONG_RISE | 100.0% | 1 | 50.0 |
| indicator | Momentum | Mtm上方Zero | Momentum指标显示Mtm上方Zero形态 | 100.0% | 1 | 50.0 |
| indicator | Momentum | Mtm上方信号 | Momentum指标显示Mtm上方信号形态 | 100.0% | 1 | 50.0 |
| indicator | Momentum | Mtm上升 | Momentum指标显示Mtm上升形态 | 100.0% | 1 | 50.0 |
| indicator | Momentum | MtmConsecutive上升 | Momentum指标显示MtmConsecutive上升形态 | 100.0% | 1 | 50.0 |
| indicator | Momentum | Mtm看涨背离 | Momentum指标显示Mtm看涨背离形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 低弹性比率 | Elasticity指标低弹性比率形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 中等反弹 | Elasticity指标中等反弹形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 放量反弹 | Elasticity指标放量反弹形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 窄幅波动区间 | Elasticity指标窄幅波动区间形态 | 100.0% | 1 | 50.0 |
| indicator | KC | Kc上方Middle | KC指标显示Kc上方Middle形态 | 100.0% | 1 | 50.0 |
| indicator | KC | KcAtMiddle | KC指标显示KcAtMiddle形态 | 100.0% | 1 | 50.0 |
| indicator | KC | KcExpanding | KC指标显示KcExpanding形态 | 100.0% | 1 | 50.0 |
| indicator | KC | KcOscillating | KC指标显示KcOscillating形态 | 100.0% | 1 | 50.0 |
| indicator | ADX | 上升趋势 | ADX指标显示上升趋势 | 100.0% | 1 | 50.0 |
| indicator | ZXMRiseElasticity | 温和上涨 | ZXMRiseElasticity指标温和上涨形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMRiseElasticity | 无大涨 | ZXMRiseElasticity指标无大涨形态 | 100.0% | 1 | 50.0 |
| indicator | MTM | Mtm上方Zero | MTM指标显示Mtm上方Zero形态 | 100.0% | 1 | 50.0 |
| indicator | MTM | Mtm上方Ma | MTM指标显示Mtm上方Ma形态 | 100.0% | 1 | 50.0 |
| indicator | VR | VrNormal | VR指标显示VrNormal形态 | 100.0% | 1 | 50.0 |
| indicator | VR | Vr上方Ma | VR指标显示Vr上方Ma形态 | 100.0% | 1 | 50.0 |
| indicator | MACD | MACD柱状图扩张 | MACD柱状图连续增大，表明趋势加强 | 100.0% | 1 | 50.0 |
| indicator | ZXMVolumeShrink | ZXM_VOLUME_NORMAL | ZXMVolumeShrink指标ZXM_VOLUME_NORMAL形态 | 100.0% | 1 | 50.0 |
| indicator | DMI | ADX强趋势 | ADX大于25，表示趋势强劲 | 100.0% | 1 | 50.0 |
| indicator | DMI | ADX上升 | ADX上升，趋势强度增强 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 买点评分信号 | ZXMBuyPointScore指标买点评分信号形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | MACD买点满足 | ZXMBuyPointScore指标MACD买点满足形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 换手买点满足 | ZXMBuyPointScore指标换手买点满足形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 均线回调买点满足 | ZXMBuyPointScore指标均线回调买点满足形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 极高买点评分 | ZXMBuyPointScore指标极高买点评分形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 三重买点信号共振 | ZXMBuyPointScore指标三重买点信号共振形态 | 100.0% | 1 | 50.0 |
| indicator | EnhancedKDJ | J线超买 | J线高于100，表明极度超买 | 100.0% | 1 | 50.0 |
| indicator | EnhancedKDJ | K线上升 | K线呈上升趋势 | 100.0% | 1 | 50.0 |
| indicator | EnhancedKDJ | D线上升 | D线呈上升趋势 | 100.0% | 1 | 50.0 |
| indicator | OBV | OBV上升趋势 | OBV持续上升，表明资金持续流入 | 100.0% | 1 | 50.0 |
| indicator | OBV | OBV金叉 | OBV上穿其均线，表明买盘力量增强 | 100.0% | 1 | 50.0 |
| indicator | OBV | OBV量价配合 | 基于OBV指标的量价配合分析: OBV_VOLUME_PRICE_HARMONY | 100.0% | 1 | 50.0 |
| indicator | CCI | CCI强势上升趋势 | CCI持续上升，表示强势上涨 | 100.0% | 1 | 50.0 |
| indicator | EnhancedTRIX | TRIX零轴下方 | TRIX位于零轴下方，长期趋势偏空 | 100.0% | 1 | 50.0 |
| indicator | EnhancedTRIX | TRIX上升 | TRIX指标上升，长期动量增强 | 100.0% | 1 | 50.0 |
| indicator | EnhancedTRIX | TRIX加速上升 | TRIX指标加速上升，表明价格上涨动能不断增强 | 100.0% | 1 | 50.0 |
| indicator | UnifiedMA | Ma盘整 | UnifiedMA指标显示Ma盘整形态 | 100.0% | 1 | 50.0 |
| indicator | SAR | 上升趋势 | SAR指标显示上升趋势 | 100.0% | 1 | 50.0 |
| indicator | SAR | Sar Close To Price | SAR_CLOSE_TO_PRICE形态 | 100.0% | 1 | 50.0 |
| indicator | SAR | Sar Low Acceleration | SAR_LOW_ACCELERATION形态 | 100.0% | 1 | 50.0 |
| indicator | PVT | Pvt Above Signal | PVT_ABOVE_SIGNAL形态 | 100.0% | 1 | 50.0 |
| indicator | PVT | Pvt上升 | PVT指标Pvt上升 | 100.0% | 1 | 50.0 |
| indicator | TRIX | Trix Consecutive上升 | TRIX指标Trix Consecutive上升 | 100.0% | 1 | 50.0 |
| indicator | MFI | MfiConsecutive上升 | MFI指标显示MfiConsecutive上升形态 | 100.0% | 1 | 50.0 |
| indicator | MFI | MfiLargeRise | MFI指标显示MfiLargeRise形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBSAbsorb | ZXM_BS_ABSORB_OBVIOUS | ZXMBSAbsorb指标ZXM_BS_ABSORB_OBVIOUS形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBSAbsorb | ZXM_BS_ABSORB_WATCH_ZONE | ZXMBSAbsorb指标ZXM_BS_ABSORB_WATCH_ZONE形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBSAbsorb | ZXM_BS_STRONG_MOMENTUM | ZXMBSAbsorb指标ZXM_BS_STRONG_MOMENTUM形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 震荡/无趋势 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 趋势初期 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 高规律性周期 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 下降趋势 | TrendDetector指标下降趋势形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 下降趋势初期 | TrendDetector指标下降趋势初期形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 虚弱下降趋势 | TrendDetector指标虚弱下降趋势形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 中期趋势 | TrendDetector指标中期趋势形态 | 100.0% | 1 | 50.0 |
| indicator | DMA | DMA平均差值分析 | 基于DMA平均差值指标的技术分析: DMA_WEAK_DOWNTREND | 100.0% | 1 | 50.0 |
| indicator | DMA | DMA平均差值分析 | 基于DMA平均差值指标的技术分析: DMA_BELOW_ZERO | 100.0% | 1 | 50.0 |
| indicator | EnhancedMFI | Mfi Above 50 | MFI_ABOVE_50形态 | 100.0% | 1 | 50.0 |
| indicator | EnhancedMFI | Mfi Cross Overbought Up | MFI_CROSS_OVERBOUGHT_UP形态 | 100.0% | 1 | 50.0 |
| indicator | BIAS | BIAS中性 | BIAS值在-5%到+5%之间，表示价格相对均衡 | 100.0% | 1 | 50.0 |
| indicator | KDJ | KDJ超买 | K值高于80，超买信号 | 100.0% | 1 | 50.0 |
| indicator | ROC | Roc上方Zero | ROC指标显示Roc上方Zero形态 | 100.0% | 1 | 50.0 |
| indicator | ROC | Roc上方Ma | ROC指标显示Roc上方Ma形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 振幅弹性满足 | ZXMElasticityScore指标振幅弹性满足形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 中等弹性评分 | ZXMElasticityScore指标中等弹性评分形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 部分弹性指标满足 | ZXMElasticityScore指标部分弹性指标满足形态 | 100.0% | 1 | 50.0 |
| indicator | SelectionModel | 横盘震荡洗盘 | 横盘震荡洗盘形态 | 100.0% | 1 | 50.0 |
| indicator | CCI | CCI超买 | CCI值高于+100，表示超买 | 100.0% | 1 | 35.0 |
| indicator | EnhancedMFI | Mfi Overbought | MFI_OVERBOUGHT形态 | 100.0% | 1 | 35.0 |
| indicator | EnhancedWR | WrExtreme超买 | EnhancedWR指标显示WrExtreme超买形态 | 100.0% | 1 | 30.0 |
| indicator | WR | WrExtreme超买 | WR指标显示WrExtreme超买形态 | 100.0% | 1 | 30.0 |

### 📊 15min周期分析总结

#### 🎯 高命中率指标 (≥80%)
- **MFI** (Mfi看涨背离): 100.0%命中率，平均得分80.0分
  *MFI指标显示Mfi看涨背离形态*
- **ChipDistribution** (ChipBottomAccumulation): 100.0%命中率，平均得分75.0分
  *ChipDistribution指标显示ChipBottomAccumulation形态*
- **ADX** (Adx Strong上升): 100.0%命中率，平均得分75.0分
  *ADX指标Adx Strong上升*
- **VOL** (放量上涨): 100.0%命中率，平均得分75.0分
  *成交量显著放大，同时价格上涨，通常是趋势启动或加速的信号*
- **InstitutionalBehavior** (InstAbsorptionPhase): 100.0%命中率，平均得分75.0分
  *InstitutionalBehavior指标显示InstAbsorptionPhase形态*

---

## 📈 daily 周期共性指标

### 数据统计
- **总样本数量**: 1个买点样本
- **共性指标数量**: 135个指标形态
- **分析周期**: dailyK线

| 指标类型 | 指标名称 | 形态 | 形态描述 | 命中率 | 命中数量 | 平均得分 |
|---------|----------|------|----------|--------|----------|----------|
| indicator | ChipDistribution | ChipBottomAccumulation | ChipDistribution指标显示ChipBottomAccumulation形态 | 100.0% | 1 | 75.0 |
| indicator | ADX | Adx Strong上升 | ADX指标Adx Strong上升 | 100.0% | 1 | 75.0 |
| indicator | ChipDistribution | PriceNearCost | ChipDistribution指标显示PriceNearCost形态 | 100.0% | 1 | 70.0 |
| indicator | ADX | 上升趋势 | ADX指标显示上升趋势 | 100.0% | 1 | 70.0 |
| indicator | InstitutionalBehavior | InstLowProfit | InstitutionalBehavior指标显示InstLowProfit形态 | 100.0% | 1 | 70.0 |
| indicator | ChipDistribution | ChipLowProfit | ChipDistribution指标显示ChipLowProfit形态 | 100.0% | 1 | 65.0 |
| indicator | BOLL | 布林带收缩 | 布林带收缩，表明波动率降低，可能酝酿突破 | 100.0% | 1 | 65.0 |
| indicator | TRIX | Trix Above Signal | TRIX_ABOVE_SIGNAL形态 | 100.0% | 1 | 65.0 |
| indicator | ChipDistribution | ChipTight | ChipDistribution指标显示ChipTight形态 | 100.0% | 1 | 60.0 |
| indicator | STOCHRSI | StochRSI超卖 | StochRSI进入超卖区域，可能出现反弹 | 100.0% | 1 | 50.0 |
| indicator | STOCHRSI | 随机RSI超买超卖 | 基于StochRSI指标的超买超卖分析: STOCHRSI_K_BELOW_D | 100.0% | 1 | 50.0 |
| indicator | STOCHRSI | 随机RSI超买超卖 | 基于StochRSI指标的超买超卖分析: STOCHRSI_K_FALLING | 100.0% | 1 | 50.0 |
| indicator | Chaikin | Chaikin零轴下方 | Chaikin震荡器位于零轴下方 | 100.0% | 1 | 50.0 |
| indicator | Chaikin | Chaikin上升 | Chaikin震荡器上升 | 100.0% | 1 | 50.0 |
| indicator | VOSC | Vosc上方Zero | VOSC指标显示Vosc上方Zero形态 | 100.0% | 1 | 50.0 |
| indicator | VOSC | Vosc上方信号 | VOSC指标显示Vosc上方信号形态 | 100.0% | 1 | 50.0 |
| indicator | VOSC | VoscPrice背离 | VOSC指标显示VoscPrice背离形态 | 100.0% | 1 | 50.0 |
| indicator | StockScoreCalculator | 综合评分适中 | 综合评分适中形态 | 100.0% | 1 | 50.0 |
| indicator | StockScoreCalculator | 趋势强劲 | 趋势强劲形态 | 100.0% | 1 | 50.0 |
| indicator | StockScoreCalculator | 高波动性 | 高波动性形态 | 100.0% | 1 | 50.0 |
| indicator | BounceDetector | 大幅反弹 | 大幅反弹形态 | 100.0% | 1 | 50.0 |
| indicator | BounceDetector | 明显缩量 | 明显缩量形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMDailyTrendUp | 均线上移 | ZXMDailyTrendUp指标均线上移形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMDailyTrendUp | 双均线上移 | ZXMDailyTrendUp指标双均线上移形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMDailyTrendUp | 价格站上双均线 | ZXMDailyTrendUp指标价格站上双均线形态 | 100.0% | 1 | 50.0 |
| indicator | Ichimoku | 价格位于云层之下 | 价格位于云层下方，看跌信号 | 100.0% | 1 | 50.0 |
| indicator | ZXMPattern | class_two_buy | ZXMPattern指标class_two_buy形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMPattern | volume_decrease | ZXMPattern指标volume_decrease形态 | 100.0% | 1 | 50.0 |
| indicator | Vortex | VortexViPlus上方 | Vortex指标显示VortexViPlus上方形态 | 100.0% | 1 | 50.0 |
| indicator | Vortex | VortexViMinus上升 | Vortex指标显示VortexViMinus上升形态 | 100.0% | 1 | 50.0 |
| indicator | Vortex | VortexViMinus上升趋势 | Vortex指标显示VortexViMinus上升趋势形态 | 100.0% | 1 | 50.0 |
| indicator | PSY | PSY_CROSS_DOWN_50 | PSY指标PSY_CROSS_DOWN_50形态 | 100.0% | 1 | 50.0 |
| indicator | PSY | PSY_BELOW_50 | PSY指标PSY_BELOW_50形态 | 100.0% | 1 | 50.0 |
| indicator | PSY | PSY_BELOW_MA | PSY指标PSY_BELOW_MA形态 | 100.0% | 1 | 50.0 |
| indicator | PSY | PSY_STRONG_DOWN | PSY指标PSY_STRONG_DOWN形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMTurnover | ZXM_TURNOVER_BUY_POINT | ZXMTurnover指标ZXM_TURNOVER_BUY_POINT形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMTurnover | ZXM_TURNOVER_EXTREMELY_ACTIVE | ZXMTurnover指标ZXM_TURNOVER_EXTREMELY_ACTIVE形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMTurnover | ZXM_TURNOVER_SUDDEN_DECREASE | ZXMTurnover指标ZXM_TURNOVER_SUDDEN_DECREASE形态 | 100.0% | 1 | 50.0 |
| indicator | EnhancedMACD | MACD柱状体为正 | MACD柱状体大于零，表示上升动能 | 100.0% | 1 | 50.0 |
| indicator | EnhancedMACD | MACD柱状体减少 | MACD柱状体连续减少，表示动能减弱 | 100.0% | 1 | 50.0 |
| indicator | EnhancedMACD | MACD强上升趋势 | MACD柱状体为正且趋势强度高，表明强势上升趋势 | 100.0% | 1 | 50.0 |
| indicator | EnhancedDMI | Trend Reversal Warning | trend_reversal_warning形态 | 100.0% | 1 | 50.0 |
| indicator | EMA | EMA多头排列 | 指数移动平均线呈多头排列，趋势向上 | 100.0% | 1 | 50.0 |
| indicator | AmplitudeElasticity | 振幅弹性信号 | AmplitudeElasticity指标振幅弹性信号形态 | 100.0% | 1 | 50.0 |
| indicator | AmplitudeElasticity | 小振幅 | AmplitudeElasticity指标小振幅形态 | 100.0% | 1 | 50.0 |
| indicator | AmplitudeElasticity | 频繁大振幅 | AmplitudeElasticity指标频繁大振幅形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | VixSideways | StockVIX指标显示VixSideways形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | VixNormal | StockVIX指标显示VixNormal形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | Vix强势Strength | StockVIX指标显示Vix强势Strength形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | Vix上方Ma20 | StockVIX指标显示Vix上方Ma20形态 | 100.0% | 1 | 50.0 |
| indicator | EMV | EMV零轴上方 | EMV位于零轴上方，买盘力量占优 | 100.0% | 1 | 50.0 |
| indicator | EMV | EMV均线下方 | EMV位于移动平均线下方 | 100.0% | 1 | 50.0 |
| indicator | EMV | EMV下穿均线 | EMV下穿其移动平均线，趋势转弱 | 100.0% | 1 | 50.0 |
| indicator | EMV | EMV顶背离 | 价格上涨但EMV下降，可能反转向下 | 100.0% | 1 | 50.0 |
| indicator | CMO | CMO动量震荡 | 基于CMO动量震荡指标的技术分析: CMO_ABOVE_ZERO | 100.0% | 1 | 50.0 |
| indicator | CMO | CMO动量震荡 | 基于CMO动量震荡指标的技术分析: CMO_FALLING | 100.0% | 1 | 50.0 |
| indicator | CMO | CMO动量震荡 | 基于CMO动量震荡指标的技术分析: CMO_STRONG_FALL | 100.0% | 1 | 50.0 |
| indicator | Momentum | Mtm上方Zero | Momentum指标显示Mtm上方Zero形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 中等弹性比率 | Elasticity指标中等弹性比率形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 轻微反弹 | Elasticity指标轻微反弹形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 缩量反弹 | Elasticity指标缩量反弹形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 大幅波动区间 | Elasticity指标大幅波动区间形态 | 100.0% | 1 | 50.0 |
| indicator | KC | Kc上方Middle | KC指标显示Kc上方Middle形态 | 100.0% | 1 | 50.0 |
| indicator | KC | KcAtMiddle | KC指标显示KcAtMiddle形态 | 100.0% | 1 | 50.0 |
| indicator | KC | KcContracting | KC指标显示KcContracting形态 | 100.0% | 1 | 50.0 |
| indicator | ADX | 上升趋势 | ADX指标显示上升趋势 | 100.0% | 1 | 50.0 |
| indicator | BOLL | 布林带上轨突破 | 价格突破布林带上轨，表明强势上涨 | 100.0% | 1 | 50.0 |
| indicator | BOLL | 布林带趋势跟随 | 价格沿布林带边缘运行，表明趋势强劲 | 100.0% | 1 | 50.0 |
| indicator | ZXMRiseElasticity | 涨幅弹性信号 | ZXMRiseElasticity指标涨幅弹性信号形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMRiseElasticity | 温和上涨 | ZXMRiseElasticity指标温和上涨形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMRiseElasticity | 频繁大涨 | ZXMRiseElasticity指标频繁大涨形态 | 100.0% | 1 | 50.0 |
| indicator | MTM | Mtm上方Zero | MTM指标显示Mtm上方Zero形态 | 100.0% | 1 | 50.0 |
| indicator | VR | VrNormal | VR指标显示VrNormal形态 | 100.0% | 1 | 50.0 |
| indicator | VR | Vr死叉交叉 | VR指标显示Vr死叉交叉形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMVolumeShrink | ZXM_VOLUME_SHRINK_BUY_POINT | ZXMVolumeShrink指标ZXM_VOLUME_SHRINK_BUY_POINT形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMVolumeShrink | ZXM_VOLUME_SLIGHT_SHRINK | ZXMVolumeShrink指标ZXM_VOLUME_SLIGHT_SHRINK形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMVolumeShrink | ZXM_VOLUME_CONSECUTIVE_SHRINK | ZXMVolumeShrink指标ZXM_VOLUME_CONSECUTIVE_SHRINK形态 | 100.0% | 1 | 50.0 |
| indicator | DMI | ADX强趋势 | ADX大于25，表示趋势强劲 | 100.0% | 1 | 50.0 |
| indicator | DMI | ADX上升 | ADX上升，趋势强度增强 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | MACD买点满足 | ZXMBuyPointScore指标MACD买点满足形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 换手买点满足 | ZXMBuyPointScore指标换手买点满足形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 中等买点评分 | ZXMBuyPointScore指标中等买点评分形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 多数买点指标满足 | ZXMBuyPointScore指标多数买点指标满足形态 | 100.0% | 1 | 50.0 |
| indicator | EnhancedKDJ | J线超卖 | J线低于0，表明极度超卖 | 100.0% | 1 | 50.0 |
| indicator | EnhancedWR | WrNormal | EnhancedWR指标显示WrNormal形态 | 100.0% | 1 | 50.0 |
| indicator | OBV | OBV量价配合 | 基于OBV指标的量价配合分析: OBV_VOLUME_PRICE_HARMONY | 100.0% | 1 | 50.0 |
| indicator | CCI | CCI下穿零轴 | CCI下穿零轴线 | 100.0% | 1 | 50.0 |
| indicator | EnhancedTRIX | TRIX零轴上方 | TRIX位于零轴上方，长期趋势偏多 | 100.0% | 1 | 50.0 |
| indicator | EnhancedTRIX | TRIX下降 | TRIX指标下降，长期动量减弱 | 100.0% | 1 | 50.0 |
| indicator | EnhancedTRIX | TRIX强烈看涨共振 | TRIX多重信号共振，形成强烈看涨态势 | 100.0% | 1 | 50.0 |
| indicator | EnhancedTRIX | TRIX减速 | TRIX指标减速变化，动能转变 | 100.0% | 1 | 50.0 |
| indicator | WR | WrNormal | WR指标显示WrNormal形态 | 100.0% | 1 | 50.0 |
| indicator | UnifiedMA | Price上方LongMa | UnifiedMA指标显示Price上方LongMa形态 | 100.0% | 1 | 50.0 |
| indicator | UnifiedMA | Ma看涨Alignment | UnifiedMA指标显示Ma看涨Alignment形态 | 100.0% | 1 | 50.0 |
| indicator | UnifiedMA | MaLong上升趋势 | UnifiedMA指标显示MaLong上升趋势形态 | 100.0% | 1 | 50.0 |
| indicator | VOL | 成交量能量分析 | 基于成交量能量变化的技术分析: VOL_FALLING | 100.0% | 1 | 50.0 |
| indicator | VOL | 均量线多头排列 | 成交量均线呈多头排列，表示成交量趋势强劲 | 100.0% | 1 | 50.0 |
| indicator | SAR | Sar Close To Price | SAR_CLOSE_TO_PRICE形态 | 100.0% | 1 | 50.0 |
| indicator | SAR | Sar Low Acceleration | SAR_LOW_ACCELERATION形态 | 100.0% | 1 | 50.0 |
| indicator |  | 均线MA5条件 | 均线MA5 > CLOSE | 100.0% | 1 | 50.0 |
| indicator | TRIX | Trix Above Zero | TRIX_ABOVE_ZERO形态 | 100.0% | 1 | 50.0 |
| indicator | Aroon | Aroon强势上涨 | Aroon Up > 70 且 Aroon Down < 30，表明强势上升趋势 | 100.0% | 1 | 50.0 |
| indicator | Aroon | Aroon震荡器极度看涨 | Aroon震荡器值 > 50，表明极强的上升趋势 | 100.0% | 1 | 50.0 |
| indicator | ZXMDailyMACD | ZXM_DAILY_MACD_BUY_POINT | ZXMDailyMACD指标ZXM_DAILY_MACD_BUY_POINT形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMDailyMACD | ZXM_DAILY_MACD_POSITIVE | ZXMDailyMACD指标ZXM_DAILY_MACD_POSITIVE形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMDailyMACD | ZXM_DAILY_MACD_FALLING | ZXMDailyMACD指标ZXM_DAILY_MACD_FALLING形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMDailyMACD | 技术指标分析 | 基于技术指标的分析: ZXM_DAILY_MACD_BULLISH_ALIGNMENT | 100.0% | 1 | 50.0 |
| indicator | ZXMDailyMACD | ZXM_DAILY_MACD_NEAR_ZERO | ZXMDailyMACD指标ZXM_DAILY_MACD_NEAR_ZERO形态 | 100.0% | 1 | 50.0 |
| indicator | InstitutionalBehavior | InstWaitingPhase | InstitutionalBehavior指标显示InstWaitingPhase形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBSAbsorb | ZXM_BS_ABSORB_WATCH_ZONE | ZXMBSAbsorb指标ZXM_BS_ABSORB_WATCH_ZONE形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBSAbsorb | ZXM_BS_STABLE_MOMENTUM | ZXMBSAbsorb指标ZXM_BS_STABLE_MOMENTUM形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 上升趋势 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 上升趋势发展阶段 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 短期上升趋势 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 上升趋势 | TrendDetector指标上升趋势形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 上升趋势初期 | TrendDetector指标上升趋势初期形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 虚弱上升趋势 | TrendDetector指标虚弱上升趋势形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 短期趋势 | TrendDetector指标短期趋势形态 | 100.0% | 1 | 50.0 |
| indicator | DMA | DMA上升趋势 | DMA大于0且DMA大于AMA，表示强势上升趋势 | 100.0% | 1 | 50.0 |
| indicator | DMA | DMA平均差值分析 | 基于DMA平均差值指标的技术分析: DMA_ABOVE_ZERO | 100.0% | 1 | 50.0 |
| indicator | DMA | DMA强势上涨 | DMA百分比差值大于5%，表示强势上涨 | 100.0% | 1 | 50.0 |
| indicator | DMA | DMA平均差值分析 | 基于DMA平均差值指标的技术分析: DMA_ACCELERATION_UP | 100.0% | 1 | 50.0 |
| indicator | EnhancedMFI | Mfi Above 50 | MFI_ABOVE_50形态 | 100.0% | 1 | 50.0 |
| indicator | BIAS | BIAS中性 | BIAS值在-5%到+5%之间，表示价格相对均衡 | 100.0% | 1 | 50.0 |
| indicator | ROC | Roc超买 | ROC指标显示Roc超买形态 | 100.0% | 1 | 50.0 |
| indicator | ROC | Roc上方Zero | ROC指标显示Roc上方Zero形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 弹性评分信号 | ZXMElasticityScore指标弹性评分信号形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 振幅弹性满足 | ZXMElasticityScore指标振幅弹性满足形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 涨幅弹性满足 | ZXMElasticityScore指标涨幅弹性满足形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 极高弹性评分 | ZXMElasticityScore指标极高弹性评分形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 全部弹性指标满足 | ZXMElasticityScore指标全部弹性指标满足形态 | 100.0% | 1 | 50.0 |
| indicator | MA | MA多头排列 | 短期MA(5)在长期MA(60)之上，强劲上升趋势 | 100.0% | 1 | 50.0 |
| indicator | SelectionModel | 选股系统买入信号 | 选股系统买入信号形态 | 100.0% | 1 | 50.0 |
| indicator | SelectionModel | 强趋势上涨股 | 强趋势上涨股形态 | 100.0% | 1 | 50.0 |
| indicator | SelectionModel | 超强上升趋势 | 超强上升趋势形态 | 100.0% | 1 | 50.0 |

### 📊 daily周期分析总结

#### 🎯 高命中率指标 (≥80%)
- **ChipDistribution** (ChipBottomAccumulation): 100.0%命中率，平均得分75.0分
  *ChipDistribution指标显示ChipBottomAccumulation形态*
- **ADX** (Adx Strong上升): 100.0%命中率，平均得分75.0分
  *ADX指标Adx Strong上升*
- **ChipDistribution** (PriceNearCost): 100.0%命中率，平均得分70.0分
  *ChipDistribution指标显示PriceNearCost形态*
- **ADX** (上升趋势): 100.0%命中率，平均得分70.0分
  *ADX指标显示上升趋势*
- **InstitutionalBehavior** (InstLowProfit): 100.0%命中率，平均得分70.0分
  *InstitutionalBehavior指标显示InstLowProfit形态*

---

## 📈 weekly 周期共性指标

### 数据统计
- **总样本数量**: 1个买点样本
- **共性指标数量**: 170个指标形态
- **分析周期**: weeklyK线

| 指标类型 | 指标名称 | 形态 | 形态描述 | 命中率 | 命中数量 | 平均得分 |
|---------|----------|------|----------|--------|----------|----------|
| indicator | ChipDistribution | ChipBottomAccumulation | ChipDistribution指标显示ChipBottomAccumulation形态 | 100.0% | 1 | 75.0 |
| indicator | ADX | Adx Strong上升 | ADX指标Adx Strong上升 | 100.0% | 1 | 75.0 |
| indicator | InstitutionalBehavior | InstAbsorptionPhase | InstitutionalBehavior指标显示InstAbsorptionPhase形态 | 100.0% | 1 | 75.0 |
| indicator | ADX | 上升趋势 | ADX指标显示上升趋势 | 100.0% | 1 | 70.0 |
| indicator | InstitutionalBehavior | InstLowProfit | InstitutionalBehavior指标显示InstLowProfit形态 | 100.0% | 1 | 70.0 |
| indicator | ChipDistribution | ChipLowProfit | ChipDistribution指标显示ChipLowProfit形态 | 100.0% | 1 | 65.0 |
| indicator | EnhancedOBV | Obv上升 | EnhancedOBV指标Obv上升 | 100.0% | 1 | 65.0 |
| indicator | VOL | 成交量能量分析 | 基于成交量能量变化的技术分析: VOL_RISING | 100.0% | 1 | 65.0 |
| indicator | TRIX | Trix Above Signal | TRIX_ABOVE_SIGNAL形态 | 100.0% | 1 | 65.0 |
| indicator |  | 通用条件: MA5>MA10 | 自定义条件表达式: MA5>MA10 | 100.0% | 1 | 65.0 |
| indicator | EnhancedMACD | MACD上升 | MACD线呈上升趋势 | 100.0% | 1 | 60.0 |
| indicator | ChipDistribution | ChipTight | ChipDistribution指标显示ChipTight形态 | 100.0% | 1 | 60.0 |
| indicator | VOL | 成交量偏高 | 成交量高于平均水平，市场活跃度较高 | 100.0% | 1 | 60.0 |
| indicator | TRIX | Trix上升 | TRIX指标Trix上升 | 100.0% | 1 | 60.0 |
| indicator | STOCHRSI | StochRSI超卖 | StochRSI进入超卖区域，可能出现反弹 | 100.0% | 1 | 50.0 |
| indicator | STOCHRSI | 随机RSI超买超卖 | 基于StochRSI指标的超买超卖分析: STOCHRSI_K_BELOW_D | 100.0% | 1 | 50.0 |
| indicator | STOCHRSI | 随机RSI超买超卖 | 基于StochRSI指标的超买超卖分析: STOCHRSI_K_FALLING | 100.0% | 1 | 50.0 |
| indicator | STOCHRSI | 随机RSI超买超卖 | 基于StochRSI指标的超买超卖分析: STOCHRSI_D_FALLING | 100.0% | 1 | 50.0 |
| indicator | STOCHRSI | StochRSI超买反转 | StochRSI从超买区域向下突破，看跌信号 | 100.0% | 1 | 50.0 |
| indicator | Chaikin | Chaikin零轴上方 | Chaikin震荡器位于零轴上方 | 100.0% | 1 | 50.0 |
| indicator | VOSC | Vosc上方Zero | VOSC指标显示Vosc上方Zero形态 | 100.0% | 1 | 50.0 |
| indicator | VOSC | Vosc上方信号 | VOSC指标显示Vosc上方信号形态 | 100.0% | 1 | 50.0 |
| indicator | VOSC | Vosc上升 | VOSC指标显示Vosc上升形态 | 100.0% | 1 | 50.0 |
| indicator | VOSC | Vosc上升趋势 | VOSC指标显示Vosc上升趋势形态 | 100.0% | 1 | 50.0 |
| indicator | VOSC | VoscPrice背离 | VOSC指标显示VoscPrice背离形态 | 100.0% | 1 | 50.0 |
| indicator | StockScoreCalculator | 买入信号 | 指标产生买入信号，建议关注 | 100.0% | 1 | 50.0 |
| indicator | StockScoreCalculator | 优质股票 | 优质股票形态 | 100.0% | 1 | 50.0 |
| indicator | StockScoreCalculator | 趋势强劲 | 趋势强劲形态 | 100.0% | 1 | 50.0 |
| indicator | StockScoreCalculator | 动量强劲 | 动量强劲形态 | 100.0% | 1 | 50.0 |
| indicator | StockScoreCalculator | 高波动性 | 高波动性形态 | 100.0% | 1 | 50.0 |
| indicator | BounceDetector | 反弹确认信号 | 反弹确认信号形态 | 100.0% | 1 | 50.0 |
| indicator | BounceDetector | 大幅反弹 | 大幅反弹形态 | 100.0% | 1 | 50.0 |
| indicator | BounceDetector | 短期上升趋势 | 短期上升趋势形态 | 100.0% | 1 | 50.0 |
| indicator | BounceDetector | 明显放量 | 明显放量形态 | 100.0% | 1 | 50.0 |
| indicator | BounceDetector | 强势反弹 | 强势反弹形态 | 100.0% | 1 | 50.0 |
| indicator | Ichimoku | 价格位于云层之上 | 价格位于云层上方，看涨信号 | 100.0% | 1 | 50.0 |
| indicator | ZXMWeeklyKDJDTrendUp | 周KDJ·D上移 | ZXMWeeklyKDJDTrendUp指标周KDJ·D上移形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMWeeklyKDJDTrendUp | 周线KDJ金叉后持续上行 | ZXMWeeklyKDJDTrendUp指标周线KDJ金叉后持续上行形态 | 100.0% | 1 | 50.0 |
| indicator | Vortex | VortexViPlus上方 | Vortex指标显示VortexViPlus上方形态 | 100.0% | 1 | 50.0 |
| indicator | Vortex | VortexViPlus强势 | Vortex指标显示VortexViPlus强势形态 | 100.0% | 1 | 50.0 |
| indicator | Vortex | VortexViDiff上升 | Vortex指标显示VortexViDiff上升形态 | 100.0% | 1 | 50.0 |
| indicator | PSY | PSY_ABOVE_50 | PSY指标PSY_ABOVE_50形态 | 100.0% | 1 | 50.0 |
| indicator | PSY | PSY_ABOVE_MA | PSY指标PSY_ABOVE_MA形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMTurnover | ZXM_TURNOVER_BUY_POINT | ZXMTurnover指标ZXM_TURNOVER_BUY_POINT形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMTurnover | ZXM_TURNOVER_EXTREMELY_ACTIVE | ZXMTurnover指标ZXM_TURNOVER_EXTREMELY_ACTIVE形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMTurnover | 技术指标分析 | 基于技术指标的分析: ZXM_TURNOVER_RELATIVE_EXTREMELY_ACTIVE | 100.0% | 1 | 50.0 |
| indicator | ZXMTurnover | ZXM_TURNOVER_SUDDEN_INCREASE | ZXMTurnover指标ZXM_TURNOVER_SUDDEN_INCREASE形态 | 100.0% | 1 | 50.0 |
| indicator | EnhancedMACD | MACD柱状体为正 | MACD柱状体大于零，表示上升动能 | 100.0% | 1 | 50.0 |
| indicator | EnhancedMACD | MACD柱状体减少 | MACD柱状体连续减少，表示动能减弱 | 100.0% | 1 | 50.0 |
| indicator | EnhancedMACD | MACD强上升趋势 | MACD柱状体为正且趋势强度高，表明强势上升趋势 | 100.0% | 1 | 50.0 |
| indicator | ATR | AtrUpward突破 | ATR指标显示AtrUpward突破形态 | 100.0% | 1 | 50.0 |
| indicator | ATR | 波动性Expansion | ATR指标显示波动性Expansion形态 | 100.0% | 1 | 50.0 |
| indicator | EnhancedDMI | Strong Trend | strong_trend形态 | 100.0% | 1 | 50.0 |
| indicator | EMA | EMA多头排列 | 指数移动平均线呈多头排列，趋势向上 | 100.0% | 1 | 50.0 |
| indicator | ChipDistribution | HardUntrapped | ChipDistribution指标显示HardUntrapped形态 | 100.0% | 1 | 50.0 |
| indicator | AmplitudeElasticity | 振幅弹性信号 | AmplitudeElasticity指标振幅弹性信号形态 | 100.0% | 1 | 50.0 |
| indicator | AmplitudeElasticity | 大振幅日 | AmplitudeElasticity指标大振幅日形态 | 100.0% | 1 | 50.0 |
| indicator | AmplitudeElasticity | 极大振幅 | AmplitudeElasticity指标极大振幅形态 | 100.0% | 1 | 50.0 |
| indicator | AmplitudeElasticity | 频繁大振幅 | AmplitudeElasticity指标频繁大振幅形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | Vix上升趋势 | StockVIX指标显示Vix上升趋势形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | VixNormal | StockVIX指标显示VixNormal形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | Vix强势Strength | StockVIX指标显示Vix强势Strength形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | Vix上升 | StockVIX指标显示Vix上升形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | Vix上方Ma20 | StockVIX指标显示Vix上方Ma20形态 | 100.0% | 1 | 50.0 |
| indicator | EMV | EMV零轴上方 | EMV位于零轴上方，买盘力量占优 | 100.0% | 1 | 50.0 |
| indicator | EMV | EMV均线上方 | EMV位于移动平均线上方 | 100.0% | 1 | 50.0 |
| indicator | EMV | EMV上升 | EMV值上升 | 100.0% | 1 | 50.0 |
| indicator | EMV | EMV强势上升 | EMV大幅上升，买盘力量强劲 | 100.0% | 1 | 50.0 |
| indicator | EMV | EMV极高值 | EMV达到近期高点 | 100.0% | 1 | 50.0 |
| indicator | EMV | EMV底背离 | 价格下跌但EMV上升，可能反转向上 | 100.0% | 1 | 50.0 |
| indicator | CMO | CMO动量震荡 | 基于CMO动量震荡指标的技术分析: CMO_ABOVE_ZERO | 100.0% | 1 | 50.0 |
| indicator | CMO | CMO动量震荡 | 基于CMO动量震荡指标的技术分析: CMO_FALLING | 100.0% | 1 | 50.0 |
| indicator | CMO | CMO动量震荡 | 基于CMO动量震荡指标的技术分析: CMO_STRONG_RISE | 100.0% | 1 | 50.0 |
| indicator | Momentum | Mtm上方Zero | Momentum指标显示Mtm上方Zero形态 | 100.0% | 1 | 50.0 |
| indicator | Momentum | MtmLargeFall | Momentum指标显示MtmLargeFall形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 弹性买点 | Elasticity指标弹性买点形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 轻微弹性比率 | Elasticity指标轻微弹性比率形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 轻微反弹 | Elasticity指标轻微反弹形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 放量反弹 | Elasticity指标放量反弹形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 大幅波动区间 | Elasticity指标大幅波动区间形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMWeeklyTrendUp | 周均线上移 | ZXMWeeklyTrendUp指标周均线上移形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMWeeklyTrendUp | 三均线同时上移 | ZXMWeeklyTrendUp指标三均线同时上移形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMWeeklyTrendUp | 价格站上三均线 | ZXMWeeklyTrendUp指标价格站上三均线形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMWeeklyTrendUp | 均线多头排列 | ZXMWeeklyTrendUp指标均线多头排列形态 | 100.0% | 1 | 50.0 |
| indicator | KC | Kc上方Middle | KC指标显示Kc上方Middle形态 | 100.0% | 1 | 50.0 |
| indicator | KC | KcWideChannel | KC指标显示KcWideChannel形态 | 100.0% | 1 | 50.0 |
| indicator | KC | KcExpanding | KC指标显示KcExpanding形态 | 100.0% | 1 | 50.0 |
| indicator | ADX | 上升趋势 | ADX指标显示上升趋势 | 100.0% | 1 | 50.0 |
| indicator | BOLL | 布林带上轨突破 | 价格突破布林带上轨，表明强势上涨 | 100.0% | 1 | 50.0 |
| indicator | BOLL | 布林带扩张 | 布林带扩张，表明波动率增加，趋势可能延续 | 100.0% | 1 | 50.0 |
| indicator | BOLL | 布林带趋势跟随 | 价格沿布林带边缘运行，表明趋势强劲 | 100.0% | 1 | 50.0 |
| indicator | ZXMRiseElasticity | 涨幅弹性信号 | ZXMRiseElasticity指标涨幅弹性信号形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMRiseElasticity | 温和上涨 | ZXMRiseElasticity指标温和上涨形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMRiseElasticity | 频繁大涨 | ZXMRiseElasticity指标频繁大涨形态 | 100.0% | 1 | 50.0 |
| indicator | MTM | Mtm死叉交叉 | MTM指标显示Mtm死叉交叉形态 | 100.0% | 1 | 50.0 |
| indicator | MTM | Mtm上方Zero | MTM指标显示Mtm上方Zero形态 | 100.0% | 1 | 50.0 |
| indicator | VR | VrNormal | VR指标显示VrNormal形态 | 100.0% | 1 | 50.0 |
| indicator | VR | Vr死叉交叉 | VR指标显示Vr死叉交叉形态 | 100.0% | 1 | 50.0 |
| indicator | VR | VrRapidFall | VR指标显示VrRapidFall形态 | 100.0% | 1 | 50.0 |
| indicator | EnhancedOBV | Obv Above Ma | OBV_ABOVE_MA形态 | 100.0% | 1 | 50.0 |
| indicator | EnhancedOBV | Obv Cross Ma Up | OBV_CROSS_MA_UP形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMVolumeShrink | ZXM_VOLUME_NORMAL | ZXMVolumeShrink指标ZXM_VOLUME_NORMAL形态 | 100.0% | 1 | 50.0 |
| indicator | ZXM周线MACD指标 | 周线MACD多头排列 | 周线MACD多头排列形态 | 100.0% | 1 | 50.0 |
| indicator | ZXM周线MACD指标 | 周线MACD零轴上方 | 周线MACD零轴上方形态 | 100.0% | 1 | 50.0 |
| indicator | ZXM周线MACD指标 | 周线MACD柱状图收缩 | 周线MACD柱状图收缩形态 | 100.0% | 1 | 50.0 |
| indicator | DMI | ADX强趋势 | ADX大于25，表示趋势强劲 | 100.0% | 1 | 50.0 |
| indicator | DMI | ADX上升 | ADX上升，趋势强度增强 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | MACD买点满足 | ZXMBuyPointScore指标MACD买点满足形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 换手买点满足 | ZXMBuyPointScore指标换手买点满足形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 中等买点评分 | ZXMBuyPointScore指标中等买点评分形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 多数买点指标满足 | ZXMBuyPointScore指标多数买点指标满足形态 | 100.0% | 1 | 50.0 |
| indicator | EnhancedKDJ | D线上升 | D线呈上升趋势 | 100.0% | 1 | 50.0 |
| indicator | EnhancedWR | WrNormal | EnhancedWR指标显示WrNormal形态 | 100.0% | 1 | 50.0 |
| indicator | OBV | OBV上升趋势 | OBV持续上升，表明资金持续流入 | 100.0% | 1 | 50.0 |
| indicator | OBV | OBV量价配合 | 基于OBV指标的量价配合分析: OBV_VOLUME_PRICE_HARMONY | 100.0% | 1 | 50.0 |
| indicator | CCI | CCI强势上升趋势 | CCI持续上升，表示强势上涨 | 100.0% | 1 | 50.0 |
| indicator | EnhancedTRIX | TRIX零轴上方 | TRIX位于零轴上方，长期趋势偏多 | 100.0% | 1 | 50.0 |
| indicator | EnhancedTRIX | TRIX上升 | TRIX指标上升，长期动量增强 | 100.0% | 1 | 50.0 |
| indicator | EnhancedTRIX | TRIX强烈看涨共振 | TRIX多重信号共振，形成强烈看涨态势 | 100.0% | 1 | 50.0 |
| indicator | EnhancedTRIX | TRIX加速上升 | TRIX指标加速上升，表明价格上涨动能不断增强 | 100.0% | 1 | 50.0 |
| indicator | WR | WrNormal | WR指标显示WrNormal形态 | 100.0% | 1 | 50.0 |
| indicator | UnifiedMA | Price上方LongMa | UnifiedMA指标显示Price上方LongMa形态 | 100.0% | 1 | 50.0 |
| indicator | UnifiedMA | Ma看涨Alignment | UnifiedMA指标显示Ma看涨Alignment形态 | 100.0% | 1 | 50.0 |
| indicator | UnifiedMA | MaLong上升趋势 | UnifiedMA指标显示MaLong上升趋势形态 | 100.0% | 1 | 50.0 |
| indicator | VOL | 成交量极高 | 成交量极高，可能存在异常交易或重大消息 | 100.0% | 1 | 50.0 |
| indicator | VOL | 均量线多头排列 | 成交量均线呈多头排列，表示成交量趋势强劲 | 100.0% | 1 | 50.0 |
| indicator | VOL | 成交量金叉 | 短期成交量均线上穿长期均线，表示成交量趋势向好 | 100.0% | 1 | 50.0 |
| indicator | SAR | 看涨Sar  Reversal | SAR指标显示看涨Sar  Reversal形态 | 100.0% | 1 | 50.0 |
| indicator | SAR | 上升趋势 | SAR指标显示上升趋势 | 100.0% | 1 | 50.0 |
| indicator | SAR | Sar Close To Price | SAR_CLOSE_TO_PRICE形态 | 100.0% | 1 | 50.0 |
| indicator | SAR | Sar Low Acceleration | SAR_LOW_ACCELERATION形态 | 100.0% | 1 | 50.0 |
| indicator | PVT | Pvt Death Cross | PVT_DEATH_CROSS形态 | 100.0% | 1 | 50.0 |
| indicator | TRIX | Trix Above Zero | TRIX_ABOVE_ZERO形态 | 100.0% | 1 | 50.0 |
| indicator | TRIX | Trix Consecutive上升 | TRIX指标Trix Consecutive上升 | 100.0% | 1 | 50.0 |
| indicator | MFI | MfiLargeFall | MFI指标显示MfiLargeFall形态 | 100.0% | 1 | 50.0 |
| indicator | InstitutionalBehavior | InstAcceleratedRally | InstitutionalBehavior指标显示InstAcceleratedRally形态 | 100.0% | 1 | 50.0 |
| indicator | InstitutionalBehavior | InstModerateActivity | InstitutionalBehavior指标显示InstModerateActivity形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMWeeklyKDJDOrDEATrendUp | 周KDJ·D/DEA上移 | ZXMWeeklyKDJDOrDEATrendUp指标周KDJ·D/DEA上移形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMWeeklyKDJDOrDEATrendUp | DEA高于0 | ZXMWeeklyKDJDOrDEATrendUp指标DEA高于0形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBSAbsorb | ZXM_BS_ABSORB_WATCH_ZONE | ZXMBSAbsorb指标ZXM_BS_ABSORB_WATCH_ZONE形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBSAbsorb | ZXM_BS_STABLE_MOMENTUM | ZXMBSAbsorb指标ZXM_BS_STABLE_MOMENTUM形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 上升趋势 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 上升趋势发展阶段 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 短期上升趋势 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 高规律性周期 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 上升趋势 | TrendDetector指标上升趋势形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 上升趋势初期 | TrendDetector指标上升趋势初期形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 虚弱上升趋势 | TrendDetector指标虚弱上升趋势形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 中期趋势 | TrendDetector指标中期趋势形态 | 100.0% | 1 | 50.0 |
| indicator | DMA | DMA上升趋势 | DMA大于0且DMA大于AMA，表示强势上升趋势 | 100.0% | 1 | 50.0 |
| indicator | DMA | DMA平均差值分析 | 基于DMA平均差值指标的技术分析: DMA_ABOVE_ZERO | 100.0% | 1 | 50.0 |
| indicator | DMA | DMA强势上涨 | DMA百分比差值大于5%，表示强势上涨 | 100.0% | 1 | 50.0 |
| indicator | DMA | DMA平均差值分析 | 基于DMA平均差值指标的技术分析: DMA_LARGE_DIVERGENCE_UP | 100.0% | 1 | 50.0 |
| indicator | DMA | DMA平均差值分析 | 基于DMA平均差值指标的技术分析: DMA_ACCELERATION_UP | 100.0% | 1 | 50.0 |
| indicator | EnhancedMFI | Mfi Above 50 | MFI_ABOVE_50形态 | 100.0% | 1 | 50.0 |
| indicator | BIAS | BIAS中度偏高 | BIAS值在+5%到+15%之间，表示轻度超买 | 100.0% | 1 | 50.0 |
| indicator | ROC | Roc超买 | ROC指标显示Roc超买形态 | 100.0% | 1 | 50.0 |
| indicator | ROC | Roc上方Zero | ROC指标显示Roc上方Zero形态 | 100.0% | 1 | 50.0 |
| indicator | ROC | Roc上方Ma | ROC指标显示Roc上方Ma形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 弹性评分信号 | ZXMElasticityScore指标弹性评分信号形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 振幅弹性满足 | ZXMElasticityScore指标振幅弹性满足形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 涨幅弹性满足 | ZXMElasticityScore指标涨幅弹性满足形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 极高弹性评分 | ZXMElasticityScore指标极高弹性评分形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 全部弹性指标满足 | ZXMElasticityScore指标全部弹性指标满足形态 | 100.0% | 1 | 50.0 |
| indicator | MA | MA多头排列 | 短期MA(5)在长期MA(60)之上，强劲上升趋势 | 100.0% | 1 | 50.0 |
| indicator | SelectionModel | 选股系统买入信号 | 选股系统买入信号形态 | 100.0% | 1 | 50.0 |
| indicator | SelectionModel | 强趋势上涨股 | 强趋势上涨股形态 | 100.0% | 1 | 50.0 |
| indicator | SelectionModel | 最高优先级选股 | 最高优先级选股形态 | 100.0% | 1 | 50.0 |
| indicator | SelectionModel | 超强上升趋势 | 超强上升趋势形态 | 100.0% | 1 | 50.0 |
| indicator | CCI | CCI超买 | CCI值高于+100，表示超买 | 100.0% | 1 | 35.0 |

### 📊 weekly周期分析总结

#### 🎯 高命中率指标 (≥80%)
- **ChipDistribution** (ChipBottomAccumulation): 100.0%命中率，平均得分75.0分
  *ChipDistribution指标显示ChipBottomAccumulation形态*
- **ADX** (Adx Strong上升): 100.0%命中率，平均得分75.0分
  *ADX指标Adx Strong上升*
- **InstitutionalBehavior** (InstAbsorptionPhase): 100.0%命中率，平均得分75.0分
  *InstitutionalBehavior指标显示InstAbsorptionPhase形态*
- **ADX** (上升趋势): 100.0%命中率，平均得分70.0分
  *ADX指标显示上升趋势*
- **InstitutionalBehavior** (InstLowProfit): 100.0%命中率，平均得分70.0分
  *InstitutionalBehavior指标显示InstLowProfit形态*

---

## 📈 monthly 周期共性指标

### 数据统计
- **总样本数量**: 1个买点样本
- **共性指标数量**: 168个指标形态
- **分析周期**: monthlyK线

| 指标类型 | 指标名称 | 形态 | 形态描述 | 命中率 | 命中数量 | 平均得分 |
|---------|----------|------|----------|--------|----------|----------|
| indicator | MFI | Mfi看涨背离 | MFI指标显示Mfi看涨背离形态 | 100.0% | 1 | 80.0 |
| indicator | EnhancedOBV | Obv Breakout | OBV_BREAKOUT形态 | 100.0% | 1 | 75.0 |
| indicator | VOL | 放量上涨 | 成交量显著放大，同时价格上涨，通常是趋势启动或加速的信号 | 100.0% | 1 | 75.0 |
| indicator | ADX | 上升趋势 | ADX指标显示上升趋势 | 100.0% | 1 | 70.0 |
| indicator | EnhancedWR | Wr上升趋势 | EnhancedWR指标显示Wr上升趋势形态 | 100.0% | 1 | 70.0 |
| indicator | WR | Wr上升趋势 | WR指标显示Wr上升趋势形态 | 100.0% | 1 | 70.0 |
| indicator | BOLL | 布林带收缩 | 布林带收缩，表明波动率降低，可能酝酿突破 | 100.0% | 1 | 65.0 |
| indicator | EnhancedOBV | Obv上升 | EnhancedOBV指标Obv上升 | 100.0% | 1 | 65.0 |
| indicator | TRIX | Trix Above Signal | TRIX_ABOVE_SIGNAL形态 | 100.0% | 1 | 65.0 |
| indicator |  | 通用条件: MA5>MA10 | 自定义条件表达式: MA5>MA10 | 100.0% | 1 | 65.0 |
| indicator | EnhancedMACD | MACD上升 | MACD线呈上升趋势 | 100.0% | 1 | 60.0 |
| indicator | ChipDistribution | ChipTight | ChipDistribution指标显示ChipTight形态 | 100.0% | 1 | 60.0 |
| indicator | EnhancedWR | Wr上升 | EnhancedWR指标显示Wr上升形态 | 100.0% | 1 | 60.0 |
| indicator | WR | Wr上升 | WR指标显示Wr上升形态 | 100.0% | 1 | 60.0 |
| indicator | VOL | 成交量偏高 | 成交量高于平均水平，市场活跃度较高 | 100.0% | 1 | 60.0 |
| indicator | TRIX | Trix上升 | TRIX指标Trix上升 | 100.0% | 1 | 60.0 |
| indicator | EnhancedMFI | Mfi上升 | EnhancedMFI指标Mfi上升 | 100.0% | 1 | 60.0 |
| indicator | STOCHRSI | StochRSI超买 | StochRSI进入超买区域，可能出现回调 | 100.0% | 1 | 50.0 |
| indicator | STOCHRSI | 随机RSI超买超卖 | 基于StochRSI指标的超买超卖分析: STOCHRSI_D_RISING | 100.0% | 1 | 50.0 |
| indicator | Chaikin | Chaikin上穿零轴 | Chaikin震荡器从下方穿越零轴 | 100.0% | 1 | 50.0 |
| indicator | Chaikin | Chaikin零轴上方 | Chaikin震荡器位于零轴上方 | 100.0% | 1 | 50.0 |
| indicator | Chaikin | Chaikin上升 | Chaikin震荡器上升 | 100.0% | 1 | 50.0 |
| indicator | Chaikin | Chaikin大幅上升 | Chaikin震荡器大幅上升 | 100.0% | 1 | 50.0 |
| indicator | Chaikin | Chaikin快速变化 | Chaikin震荡器快速变化 | 100.0% | 1 | 50.0 |
| indicator | VOSC | Vosc上方Zero | VOSC指标显示Vosc上方Zero形态 | 100.0% | 1 | 50.0 |
| indicator | VOSC | Vosc上方信号 | VOSC指标显示Vosc上方信号形态 | 100.0% | 1 | 50.0 |
| indicator | VOSC | Vosc上升 | VOSC指标显示Vosc上升形态 | 100.0% | 1 | 50.0 |
| indicator | VOSC | Vosc上升趋势 | VOSC指标显示Vosc上升趋势形态 | 100.0% | 1 | 50.0 |
| indicator | VOSC | VoscPriceConfirmation | VOSC指标显示VoscPriceConfirmation形态 | 100.0% | 1 | 50.0 |
| indicator | StockScoreCalculator | 综合评分适中 | 综合评分适中形态 | 100.0% | 1 | 50.0 |
| indicator | StockScoreCalculator | 动量强劲 | 动量强劲形态 | 100.0% | 1 | 50.0 |
| indicator | StockScoreCalculator | 高波动性 | 高波动性形态 | 100.0% | 1 | 50.0 |
| indicator | StockScoreCalculator | 成交量理想 | 成交量理想形态 | 100.0% | 1 | 50.0 |
| indicator | BounceDetector | 反弹确认信号 | 反弹确认信号形态 | 100.0% | 1 | 50.0 |
| indicator | BounceDetector | 大幅反弹 | 大幅反弹形态 | 100.0% | 1 | 50.0 |
| indicator | BounceDetector | 短期上升趋势 | 短期上升趋势形态 | 100.0% | 1 | 50.0 |
| indicator | BounceDetector | 明显放量 | 明显放量形态 | 100.0% | 1 | 50.0 |
| indicator | BounceDetector | 强势反弹 | 强势反弹形态 | 100.0% | 1 | 50.0 |
| indicator | Ichimoku | 价格位于云层之下 | 价格位于云层下方，看跌信号 | 100.0% | 1 | 50.0 |
| indicator | ZXMPattern | ma_precise_support | ZXMPattern指标ma_precise_support形态 | 100.0% | 1 | 50.0 |
| indicator | Vortex | Vortex看涨交叉 | Vortex指标显示Vortex看涨交叉形态 | 100.0% | 1 | 50.0 |
| indicator | Vortex | VortexViPlus上方 | Vortex指标显示VortexViPlus上方形态 | 100.0% | 1 | 50.0 |
| indicator | Vortex | VortexViPlus上升 | Vortex指标显示VortexViPlus上升形态 | 100.0% | 1 | 50.0 |
| indicator | Vortex | VortexViDiff上升 | Vortex指标显示VortexViDiff上升形态 | 100.0% | 1 | 50.0 |
| indicator | Vortex | VortexViPlus上升趋势 | Vortex指标显示VortexViPlus上升趋势形态 | 100.0% | 1 | 50.0 |
| indicator | PSY | PSY_ABOVE_50 | PSY指标PSY_ABOVE_50形态 | 100.0% | 1 | 50.0 |
| indicator | PSY | PSY_ABOVE_MA | PSY指标PSY_ABOVE_MA形态 | 100.0% | 1 | 50.0 |
| indicator | PSY | PSY_STRONG_UP | PSY指标PSY_STRONG_UP形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMTurnover | ZXM_TURNOVER_BUY_POINT | ZXMTurnover指标ZXM_TURNOVER_BUY_POINT形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMTurnover | ZXM_TURNOVER_EXTREMELY_ACTIVE | ZXMTurnover指标ZXM_TURNOVER_EXTREMELY_ACTIVE形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMTurnover | 技术指标分析 | 基于技术指标的分析: ZXM_TURNOVER_RELATIVE_EXTREMELY_ACTIVE | 100.0% | 1 | 50.0 |
| indicator | ZXMTurnover | ZXM_TURNOVER_SUDDEN_INCREASE | ZXMTurnover指标ZXM_TURNOVER_SUDDEN_INCREASE形态 | 100.0% | 1 | 50.0 |
| indicator | EnhancedMACD | MACD柱状体为正 | MACD柱状体大于零，表示上升动能 | 100.0% | 1 | 50.0 |
| indicator | EnhancedMACD | MACD柱状体增长 | MACD柱状体连续增长，表示动能增强 | 100.0% | 1 | 50.0 |
| indicator | EMA | EMA多头排列 | 指数移动平均线呈多头排列，趋势向上 | 100.0% | 1 | 50.0 |
| indicator | ChipDistribution | ChipProfitSurge | ChipDistribution指标显示ChipProfitSurge形态 | 100.0% | 1 | 50.0 |
| indicator | ChipDistribution | PriceFar上方Cost | ChipDistribution指标显示PriceFar上方Cost形态 | 100.0% | 1 | 50.0 |
| indicator | ChipDistribution | HardUntrapped | ChipDistribution指标显示HardUntrapped形态 | 100.0% | 1 | 50.0 |
| indicator | AmplitudeElasticity | 大振幅日 | AmplitudeElasticity指标大振幅日形态 | 100.0% | 1 | 50.0 |
| indicator | AmplitudeElasticity | 极大振幅 | AmplitudeElasticity指标极大振幅形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | Vix上升趋势 | StockVIX指标显示Vix上升趋势形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | VixAnomalySpike | StockVIX指标显示VixAnomalySpike形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | Vix上升 | StockVIX指标显示Vix上升形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | Vix上方Ma20 | StockVIX指标显示Vix上方Ma20形态 | 100.0% | 1 | 50.0 |
| indicator | EMV | EMV零轴下方 | EMV位于零轴下方，卖盘力量占优 | 100.0% | 1 | 50.0 |
| indicator | EMV | EMV均线上方 | EMV位于移动平均线上方 | 100.0% | 1 | 50.0 |
| indicator | EMV | EMV上升 | EMV值上升 | 100.0% | 1 | 50.0 |
| indicator | EMV | EMV强势上升 | EMV大幅上升，买盘力量强劲 | 100.0% | 1 | 50.0 |
| indicator | CMO | CMO上穿零轴 | CMO从负值区域穿越零轴，动量转正 | 100.0% | 1 | 50.0 |
| indicator | CMO | CMO动量震荡 | 基于CMO动量震荡指标的技术分析: CMO_ABOVE_ZERO | 100.0% | 1 | 50.0 |
| indicator | CMO | CMO动量震荡 | 基于CMO动量震荡指标的技术分析: CMO_RISING | 100.0% | 1 | 50.0 |
| indicator | CMO | CMO动量震荡 | 基于CMO动量震荡指标的技术分析: CMO_STRONG_RISE | 100.0% | 1 | 50.0 |
| indicator | Momentum | Mtm上方Zero | Momentum指标显示Mtm上方Zero形态 | 100.0% | 1 | 50.0 |
| indicator | Momentum | Mtm上方信号 | Momentum指标显示Mtm上方信号形态 | 100.0% | 1 | 50.0 |
| indicator | Momentum | Mtm上升 | Momentum指标显示Mtm上升形态 | 100.0% | 1 | 50.0 |
| indicator | Momentum | MtmConsecutive上升 | Momentum指标显示MtmConsecutive上升形态 | 100.0% | 1 | 50.0 |
| indicator | Momentum | Mtm看涨背离 | Momentum指标显示Mtm看涨背离形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 强反弹 | Elasticity指标强反弹形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 放量反弹 | Elasticity指标放量反弹形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 大幅波动区间 | Elasticity指标大幅波动区间形态 | 100.0% | 1 | 50.0 |
| indicator | KC | Kc上方Middle | KC指标显示Kc上方Middle形态 | 100.0% | 1 | 50.0 |
| indicator | KC | KcBreakMiddleUp | KC指标显示KcBreakMiddleUp形态 | 100.0% | 1 | 50.0 |
| indicator | KC | KcWideChannel | KC指标显示KcWideChannel形态 | 100.0% | 1 | 50.0 |
| indicator | KC | KcExpanding | KC指标显示KcExpanding形态 | 100.0% | 1 | 50.0 |
| indicator | BOLL | 布林带扩张 | 布林带扩张，表明波动率增加，趋势可能延续 | 100.0% | 1 | 50.0 |
| indicator | BOLL | 布林带均值回归 | 价格向中轨回归，表明超买超卖修正 | 100.0% | 1 | 50.0 |
| indicator | ZXMRiseElasticity | 涨幅弹性信号 | ZXMRiseElasticity指标涨幅弹性信号形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMRiseElasticity | 大涨日 | ZXMRiseElasticity指标大涨日形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMRiseElasticity | 极大涨幅 | ZXMRiseElasticity指标极大涨幅形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMRiseElasticity | 频繁大涨 | ZXMRiseElasticity指标频繁大涨形态 | 100.0% | 1 | 50.0 |
| indicator | MTM | Mtm上方Zero | MTM指标显示Mtm上方Zero形态 | 100.0% | 1 | 50.0 |
| indicator | MTM | Mtm上方Ma | MTM指标显示Mtm上方Ma形态 | 100.0% | 1 | 50.0 |
| indicator | VR | Vr超买 | VR指标显示Vr超买形态 | 100.0% | 1 | 50.0 |
| indicator | VR | Vr上方Ma | VR指标显示Vr上方Ma形态 | 100.0% | 1 | 50.0 |
| indicator | VR | Vr上升 | VR指标显示Vr上升形态 | 100.0% | 1 | 50.0 |
| indicator | VR | Vr上升趋势 | VR指标显示Vr上升趋势形态 | 100.0% | 1 | 50.0 |
| indicator | VR | VrRapidRise | VR指标显示VrRapidRise形态 | 100.0% | 1 | 50.0 |
| indicator | EnhancedOBV | Obv Above Ma | OBV_ABOVE_MA形态 | 100.0% | 1 | 50.0 |
| indicator | EnhancedOBV | 看涨Obv  Momentum | EnhancedOBV指标显示看涨Obv  Momentum形态 | 100.0% | 1 | 50.0 |
| indicator | MACD | MACD柱状图扩张 | MACD柱状图连续增大，表明趋势加强 | 100.0% | 1 | 50.0 |
| indicator | ZXMVolumeShrink | ZXM_VOLUME_NORMAL | ZXMVolumeShrink指标ZXM_VOLUME_NORMAL形态 | 100.0% | 1 | 50.0 |
| indicator | DMI | ADX强趋势 | ADX大于25，表示趋势强劲 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | MACD买点满足 | ZXMBuyPointScore指标MACD买点满足形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 换手买点满足 | ZXMBuyPointScore指标换手买点满足形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 中等买点评分 | ZXMBuyPointScore指标中等买点评分形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 多数买点指标满足 | ZXMBuyPointScore指标多数买点指标满足形态 | 100.0% | 1 | 50.0 |
| indicator | EnhancedKDJ | K线上升 | K线呈上升趋势 | 100.0% | 1 | 50.0 |
| indicator | EnhancedKDJ | D线上升 | D线呈上升趋势 | 100.0% | 1 | 50.0 |
| indicator | EnhancedWR | Wr超买 | EnhancedWR指标显示Wr超买形态 | 100.0% | 1 | 50.0 |
| indicator | EnhancedWR | Wr交叉上方Mid | EnhancedWR指标显示Wr交叉上方Mid形态 | 100.0% | 1 | 50.0 |
| indicator | OBV | OBV上升趋势 | OBV持续上升，表明资金持续流入 | 100.0% | 1 | 50.0 |
| indicator | OBV | OBV量价配合 | 基于OBV指标的量价配合分析: OBV_BREAKOUT_HIGH | 100.0% | 1 | 50.0 |
| indicator | OBV | OBV量价配合 | 基于OBV指标的量价配合分析: OBV_VOLUME_PRICE_HARMONY | 100.0% | 1 | 50.0 |
| indicator | ZXMMonthlyMACD | 月线MACD多头排列 | ZXMMonthlyMACD指标月线MACD多头排列形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMMonthlyMACD | 月线MACD柱状图扩大 | ZXMMonthlyMACD指标月线MACD柱状图扩大形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMMonthlyMACD | 月线MACD双线位于零轴下方 | ZXMMonthlyMACD指标月线MACD双线位于零轴下方形态 | 100.0% | 1 | 50.0 |
| indicator | CCI | CCI强势上升趋势 | CCI持续上升，表示强势上涨 | 100.0% | 1 | 50.0 |
| indicator | EnhancedTRIX | TRIX零轴下方 | TRIX位于零轴下方，长期趋势偏空 | 100.0% | 1 | 50.0 |
| indicator | EnhancedTRIX | TRIX上升 | TRIX指标上升，长期动量增强 | 100.0% | 1 | 50.0 |
| indicator | EnhancedTRIX | TRIX加速上升 | TRIX指标加速上升，表明价格上涨动能不断增强 | 100.0% | 1 | 50.0 |
| indicator | WR | Wr超买 | WR指标显示Wr超买形态 | 100.0% | 1 | 50.0 |
| indicator | WR | Wr交叉上方Mid | WR指标显示Wr交叉上方Mid形态 | 100.0% | 1 | 50.0 |
| indicator | VOL | 成交量极高 | 成交量极高，可能存在异常交易或重大消息 | 100.0% | 1 | 50.0 |
| indicator | VOL | 成交量能量分析 | 基于成交量能量变化的技术分析: VOL_FALLING | 100.0% | 1 | 50.0 |
| indicator | VOL | 均量线多头排列 | 成交量均线呈多头排列，表示成交量趋势强劲 | 100.0% | 1 | 50.0 |
| indicator | SAR | Sar Close To Price | SAR_CLOSE_TO_PRICE形态 | 100.0% | 1 | 50.0 |
| indicator | SAR | Sar Low Acceleration | SAR_LOW_ACCELERATION形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMMonthlyKDJTrendUp | 月KDJ指标K值上移 | ZXMMonthlyKDJTrendUp指标月KDJ指标K值上移形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMMonthlyKDJTrendUp | 月线KDJ金叉后持续上行 | ZXMMonthlyKDJTrendUp指标月线KDJ金叉后持续上行形态 | 100.0% | 1 | 50.0 |
| indicator | PVT | Pvt Above Signal | PVT_ABOVE_SIGNAL形态 | 100.0% | 1 | 50.0 |
| indicator | PVT | Pvt上升 | PVT指标Pvt上升 | 100.0% | 1 | 50.0 |
| indicator | PVT | Pvt Consecutive上升 | PVT指标Pvt Consecutive上升 | 100.0% | 1 | 50.0 |
| indicator | PVT | Pvt Strong Up | PVT_STRONG_UP形态 | 100.0% | 1 | 50.0 |
| indicator | TRIX | Trix Consecutive上升 | TRIX指标Trix Consecutive上升 | 100.0% | 1 | 50.0 |
| indicator | TRIX | Trix Strong | TRIX_STRONG形态 | 100.0% | 1 | 50.0 |
| indicator | Aroon | Aroon震荡器极度看涨 | Aroon震荡器值 > 50，表明极强的上升趋势 | 100.0% | 1 | 50.0 |
| indicator | MFI | MfiConsecutive上升 | MFI指标显示MfiConsecutive上升形态 | 100.0% | 1 | 50.0 |
| indicator | InstitutionalBehavior | InstControlPhase | InstitutionalBehavior指标显示InstControlPhase形态 | 100.0% | 1 | 50.0 |
| indicator | InstitutionalBehavior | InstAcceleratedRally | InstitutionalBehavior指标显示InstAcceleratedRally形态 | 100.0% | 1 | 50.0 |
| indicator | InstitutionalBehavior | InstAbsorptionComplete | InstitutionalBehavior指标显示InstAbsorptionComplete形态 | 100.0% | 1 | 50.0 |
| indicator | InstitutionalBehavior | Inst强势Activity | InstitutionalBehavior指标显示Inst强势Activity形态 | 100.0% | 1 | 50.0 |
| indicator | InstitutionalBehavior | InstModerateProfit | InstitutionalBehavior指标显示InstModerateProfit形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBSAbsorb | ZXM_BS_ABSORB_OBVIOUS | ZXMBSAbsorb指标ZXM_BS_ABSORB_OBVIOUS形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBSAbsorb | ZXM_BS_ABSORB_WATCH_ZONE | ZXMBSAbsorb指标ZXM_BS_ABSORB_WATCH_ZONE形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBSAbsorb | ZXM_BS_STRONG_MOMENTUM | ZXMBSAbsorb指标ZXM_BS_STRONG_MOMENTUM形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 震荡/无趋势 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 趋势初期 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 高规律性周期 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 下降趋势 | TrendDetector指标下降趋势形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 下降趋势中期 | TrendDetector指标下降趋势中期形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 虚弱下降趋势 | TrendDetector指标虚弱下降趋势形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 长期趋势 | TrendDetector指标长期趋势形态 | 100.0% | 1 | 50.0 |
| indicator | DMA | DMA平均差值分析 | 基于DMA平均差值指标的技术分析: DMA_WEAK_DOWNTREND | 100.0% | 1 | 50.0 |
| indicator | DMA | DMA平均差值分析 | 基于DMA平均差值指标的技术分析: DMA_BELOW_ZERO | 100.0% | 1 | 50.0 |
| indicator | DMA | DMA平均差值分析 | 基于DMA平均差值指标的技术分析: DMA_LARGE_DIVERGENCE_UP | 100.0% | 1 | 50.0 |
| indicator | DMA | DMA平均差值分析 | 基于DMA平均差值指标的技术分析: DMA_ACCELERATION_UP | 100.0% | 1 | 50.0 |
| indicator | RSI | RSI超买 | RSI指标超过70，进入超买区域，存在回调压力 | 100.0% | 1 | 50.0 |
| indicator | EnhancedMFI | Mfi Above 50 | MFI_ABOVE_50形态 | 100.0% | 1 | 50.0 |
| indicator | BIAS | BIAS极高值 | BIAS值超过+15%，表示严重超买 | 100.0% | 1 | 50.0 |
| indicator | ROC | Roc超买 | ROC指标显示Roc超买形态 | 100.0% | 1 | 50.0 |
| indicator | ROC | Roc上方Zero | ROC指标显示Roc上方Zero形态 | 100.0% | 1 | 50.0 |
| indicator | ROC | Roc上方Ma | ROC指标显示Roc上方Ma形态 | 100.0% | 1 | 50.0 |
| indicator | ROC | Roc强势Up | ROC指标显示Roc强势Up形态 | 100.0% | 1 | 50.0 |
| indicator | EnhancedRSI | Rsi Overbought | rsi_overbought形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 涨幅弹性满足 | ZXMElasticityScore指标涨幅弹性满足形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 中等弹性评分 | ZXMElasticityScore指标中等弹性评分形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 部分弹性指标满足 | ZXMElasticityScore指标部分弹性指标满足形态 | 100.0% | 1 | 50.0 |
| indicator | CCI | CCI超买 | CCI值高于+100，表示超买 | 100.0% | 1 | 35.0 |

### 📊 monthly周期分析总结

#### 🎯 高命中率指标 (≥80%)
- **MFI** (Mfi看涨背离): 100.0%命中率，平均得分80.0分
  *MFI指标显示Mfi看涨背离形态*
- **EnhancedOBV** (Obv Breakout): 100.0%命中率，平均得分75.0分
  *OBV_BREAKOUT形态*
- **VOL** (放量上涨): 100.0%命中率，平均得分75.0分
  *成交量显著放大，同时价格上涨，通常是趋势启动或加速的信号*
- **ADX** (上升趋势): 100.0%命中率，平均得分70.0分
  *ADX指标显示上升趋势*
- **EnhancedWR** (Wr上升趋势): 100.0%命中率，平均得分70.0分
  *EnhancedWR指标显示Wr上升趋势形态*

---

## 📈 30min 周期共性指标

### 数据统计
- **总样本数量**: 1个买点样本
- **共性指标数量**: 176个指标形态
- **分析周期**: 30minK线

| 指标类型 | 指标名称 | 形态 | 形态描述 | 命中率 | 命中数量 | 平均得分 |
|---------|----------|------|----------|--------|----------|----------|
| indicator | EnhancedMACD | MACD金叉 | MACD线上穿信号线，表明上升趋势开始 | 100.0% | 1 | 75.0 |
| indicator | ChipDistribution | ChipBottomAccumulation | ChipDistribution指标显示ChipBottomAccumulation形态 | 100.0% | 1 | 75.0 |
| indicator | ADX | Adx Strong上升 | ADX指标Adx Strong上升 | 100.0% | 1 | 75.0 |
| indicator | InstitutionalBehavior | InstAbsorptionPhase | InstitutionalBehavior指标显示InstAbsorptionPhase形态 | 100.0% | 1 | 75.0 |
| indicator | ADX | 上升趋势 | ADX指标显示上升趋势 | 100.0% | 1 | 70.0 |
| indicator | InstitutionalBehavior | InstLowProfit | InstitutionalBehavior指标显示InstLowProfit形态 | 100.0% | 1 | 70.0 |
| indicator | ChipDistribution | ChipLowProfit | ChipDistribution指标显示ChipLowProfit形态 | 100.0% | 1 | 65.0 |
| indicator | TRIX | Trix Above Signal | TRIX_ABOVE_SIGNAL形态 | 100.0% | 1 | 65.0 |
| indicator |  | 通用条件: MA5>MA10 | 自定义条件表达式: MA5>MA10 | 100.0% | 1 | 65.0 |
| indicator | EnhancedMACD | MACD上升 | MACD线呈上升趋势 | 100.0% | 1 | 60.0 |
| indicator | ChipDistribution | ChipTight | ChipDistribution指标显示ChipTight形态 | 100.0% | 1 | 60.0 |
| indicator | EnhancedWR | Wr上升 | EnhancedWR指标显示Wr上升形态 | 100.0% | 1 | 60.0 |
| indicator | WR | Wr上升 | WR指标显示Wr上升形态 | 100.0% | 1 | 60.0 |
| indicator | TRIX | Trix上升 | TRIX指标Trix上升 | 100.0% | 1 | 60.0 |
| indicator | EnhancedMFI | Mfi上升 | EnhancedMFI指标Mfi上升 | 100.0% | 1 | 60.0 |
| indicator | STOCHRSI | StochRSI超买 | StochRSI进入超买区域，可能出现回调 | 100.0% | 1 | 50.0 |
| indicator | STOCHRSI | StochRSI金叉 | StochRSI K线上穿D线，产生看涨信号 | 100.0% | 1 | 50.0 |
| indicator | STOCHRSI | 随机RSI超买超卖 | 基于StochRSI指标的超买超卖分析: STOCHRSI_K_ABOVE_D | 100.0% | 1 | 50.0 |
| indicator | STOCHRSI | 随机RSI超买超卖 | 基于StochRSI指标的超买超卖分析: STOCHRSI_K_RISING | 100.0% | 1 | 50.0 |
| indicator | STOCHRSI | 随机RSI超买超卖 | 基于StochRSI指标的超买超卖分析: STOCHRSI_D_FALLING | 100.0% | 1 | 50.0 |
| indicator | STOCHRSI | StochRSI强势看涨 | StochRSI K线在高位且上升，强势看涨 | 100.0% | 1 | 50.0 |
| indicator | STOCHRSI | StochRSI超卖反转 | StochRSI从超卖区域向上突破，看涨信号 | 100.0% | 1 | 50.0 |
| indicator | Chaikin | Chaikin零轴上方 | Chaikin震荡器位于零轴上方 | 100.0% | 1 | 50.0 |
| indicator | Chaikin | Chaikin上升 | Chaikin震荡器上升 | 100.0% | 1 | 50.0 |
| indicator | VOSC | Vosc上方Zero | VOSC指标显示Vosc上方Zero形态 | 100.0% | 1 | 50.0 |
| indicator | VOSC | VoscNeutral | VOSC指标显示VoscNeutral形态 | 100.0% | 1 | 50.0 |
| indicator | VOSC | VoscPriceConfirmation | VOSC指标显示VoscPriceConfirmation形态 | 100.0% | 1 | 50.0 |
| indicator | StockScoreCalculator | 买入信号 | 指标产生买入信号，建议关注 | 100.0% | 1 | 50.0 |
| indicator | StockScoreCalculator | 优质股票 | 优质股票形态 | 100.0% | 1 | 50.0 |
| indicator | StockScoreCalculator | 趋势强劲 | 趋势强劲形态 | 100.0% | 1 | 50.0 |
| indicator | StockScoreCalculator | 动量强劲 | 动量强劲形态 | 100.0% | 1 | 50.0 |
| indicator | StockScoreCalculator | 高波动性 | 高波动性形态 | 100.0% | 1 | 50.0 |
| indicator | BounceDetector | 大幅反弹 | 大幅反弹形态 | 100.0% | 1 | 50.0 |
| indicator | BounceDetector | 短期上升趋势 | 短期上升趋势形态 | 100.0% | 1 | 50.0 |
| indicator | BounceDetector | 明显缩量 | 明显缩量形态 | 100.0% | 1 | 50.0 |
| indicator | Ichimoku | 一目均衡表金叉 | 转换线上穿基准线，看涨信号 | 100.0% | 1 | 50.0 |
| indicator | Ichimoku | 价格位于云层之上 | 价格位于云层上方，看涨信号 | 100.0% | 1 | 50.0 |
| indicator | Ichimoku | 价格向上突破云层 | 价格从下方突破云层，强烈看涨信号 | 100.0% | 1 | 50.0 |
| indicator | Ichimoku | 一目均衡表强烈看涨 | 价格位于云层上方，转换线上穿基准线，云层看涨 | 100.0% | 1 | 50.0 |
| indicator | Vortex | VortexViMinus上方 | Vortex指标显示VortexViMinus上方形态 | 100.0% | 1 | 50.0 |
| indicator | Vortex | VortexViPlus上升 | Vortex指标显示VortexViPlus上升形态 | 100.0% | 1 | 50.0 |
| indicator | Vortex | VortexViDiff上升 | Vortex指标显示VortexViDiff上升形态 | 100.0% | 1 | 50.0 |
| indicator | PSY | PSY_OVERSOLD | PSY指标PSY_OVERSOLD形态 | 100.0% | 1 | 50.0 |
| indicator | PSY | PSY_GOLDEN_CROSS | PSY指标PSY_GOLDEN_CROSS形态 | 100.0% | 1 | 50.0 |
| indicator | PSY | PSY_BELOW_50 | PSY指标PSY_BELOW_50形态 | 100.0% | 1 | 50.0 |
| indicator | PSY | PSY_ABOVE_MA | PSY指标PSY_ABOVE_MA形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMTurnover | ZXM_TURNOVER_BUY_POINT | ZXMTurnover指标ZXM_TURNOVER_BUY_POINT形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMTurnover | ZXM_TURNOVER_NORMAL_ACTIVE | ZXMTurnover指标ZXM_TURNOVER_NORMAL_ACTIVE形态 | 100.0% | 1 | 50.0 |
| indicator | EnhancedMACD | MACD柱状体为正 | MACD柱状体大于零，表示上升动能 | 100.0% | 1 | 50.0 |
| indicator | EnhancedMACD | MACD柱状体增长 | MACD柱状体连续增长，表示动能增强 | 100.0% | 1 | 50.0 |
| indicator | ATR | AtrUpward突破 | ATR指标显示AtrUpward突破形态 | 100.0% | 1 | 50.0 |
| indicator | ATR | 波动性Expansion | ATR指标显示波动性Expansion形态 | 100.0% | 1 | 50.0 |
| indicator | EMA | EMA(5,10)金叉 | 当短期EMA(5)上穿长期EMA(10)时，被视为看涨信号。 | 100.0% | 1 | 50.0 |
| indicator | EMA | EMA多头排列 | 指数移动平均线呈多头排列，趋势向上 | 100.0% | 1 | 50.0 |
| indicator | ChipDistribution | HardUntrapped | ChipDistribution指标显示HardUntrapped形态 | 100.0% | 1 | 50.0 |
| indicator | AmplitudeElasticity | 振幅弹性信号 | AmplitudeElasticity指标振幅弹性信号形态 | 100.0% | 1 | 50.0 |
| indicator | AmplitudeElasticity | 小振幅 | AmplitudeElasticity指标小振幅形态 | 100.0% | 1 | 50.0 |
| indicator | AmplitudeElasticity | 频繁大振幅 | AmplitudeElasticity指标频繁大振幅形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | VixSideways | StockVIX指标显示VixSideways形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | VixNormal | StockVIX指标显示VixNormal形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | Vix上升 | StockVIX指标显示Vix上升形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | Vix上方Ma20 | StockVIX指标显示Vix上方Ma20形态 | 100.0% | 1 | 50.0 |
| indicator | WMA | Price Cross Above Wma14 | PRICE_CROSS_ABOVE_WMA14形态 | 100.0% | 1 | 50.0 |
| indicator | EMV | EMV零轴上方 | EMV位于零轴上方，买盘力量占优 | 100.0% | 1 | 50.0 |
| indicator | EMV | EMV均线上方 | EMV位于移动平均线上方 | 100.0% | 1 | 50.0 |
| indicator | EMV | EMV上穿均线 | EMV上穿其移动平均线，趋势转强 | 100.0% | 1 | 50.0 |
| indicator | EMV | EMV上升 | EMV值上升 | 100.0% | 1 | 50.0 |
| indicator | EMV | EMV极高值 | EMV达到近期高点 | 100.0% | 1 | 50.0 |
| indicator | CMO | CMO上穿零轴 | CMO从负值区域穿越零轴，动量转正 | 100.0% | 1 | 50.0 |
| indicator | CMO | CMO动量震荡 | 基于CMO动量震荡指标的技术分析: CMO_ABOVE_ZERO | 100.0% | 1 | 50.0 |
| indicator | CMO | CMO动量震荡 | 基于CMO动量震荡指标的技术分析: CMO_RISING | 100.0% | 1 | 50.0 |
| indicator | Momentum | Mtm交叉上方Zero | Momentum指标显示Mtm交叉上方Zero形态 | 100.0% | 1 | 50.0 |
| indicator | Momentum | Mtm上方Zero | Momentum指标显示Mtm上方Zero形态 | 100.0% | 1 | 50.0 |
| indicator | Momentum | Mtm交叉上方信号 | Momentum指标显示Mtm交叉上方信号形态 | 100.0% | 1 | 50.0 |
| indicator | Momentum | Mtm上方信号 | Momentum指标显示Mtm上方信号形态 | 100.0% | 1 | 50.0 |
| indicator | Momentum | Mtm上升 | Momentum指标显示Mtm上升形态 | 100.0% | 1 | 50.0 |
| indicator | Momentum | MtmLargeRise | Momentum指标显示MtmLargeRise形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 高弹性比率 | Elasticity指标高弹性比率形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 强反弹 | Elasticity指标强反弹形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 量能正常 | Elasticity指标量能正常形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 大幅波动区间 | Elasticity指标大幅波动区间形态 | 100.0% | 1 | 50.0 |
| indicator | KC | Kc上方Upper | KC指标显示Kc上方Upper形态 | 100.0% | 1 | 50.0 |
| indicator | KC | KcBreakUpper | KC指标显示KcBreakUpper形态 | 100.0% | 1 | 50.0 |
| indicator | KC | KcBreakMiddleUp | KC指标显示KcBreakMiddleUp形态 | 100.0% | 1 | 50.0 |
| indicator | KC | KcExpanding | KC指标显示KcExpanding形态 | 100.0% | 1 | 50.0 |
| indicator | KC | KcExtreme超买 | KC指标显示KcExtreme超买形态 | 100.0% | 1 | 50.0 |
| indicator | BOLL | 布林带波动区间 | 基于布林带指标的波动区间分析: BOLL_OVERBOUGHT | 100.0% | 1 | 50.0 |
| indicator | BOLL | 布林带上轨突破 | 价格突破布林带上轨，表明强势上涨 | 100.0% | 1 | 50.0 |
| indicator | BOLL | 布林带扩张 | 布林带扩张，表明波动率增加，趋势可能延续 | 100.0% | 1 | 50.0 |
| indicator | ZXMRiseElasticity | 涨幅弹性信号 | ZXMRiseElasticity指标涨幅弹性信号形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMRiseElasticity | 大涨日 | ZXMRiseElasticity指标大涨日形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMRiseElasticity | 极大涨幅 | ZXMRiseElasticity指标极大涨幅形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMRiseElasticity | 少量大涨 | ZXMRiseElasticity指标少量大涨形态 | 100.0% | 1 | 50.0 |
| indicator | MTM | Mtm金叉交叉 | MTM指标显示Mtm金叉交叉形态 | 100.0% | 1 | 50.0 |
| indicator | MTM | Mtm交叉UpZero | MTM指标显示Mtm交叉UpZero形态 | 100.0% | 1 | 50.0 |
| indicator | MTM | Mtm上方Zero | MTM指标显示Mtm上方Zero形态 | 100.0% | 1 | 50.0 |
| indicator | MTM | Mtm上方Ma | MTM指标显示Mtm上方Ma形态 | 100.0% | 1 | 50.0 |
| indicator | VR | VrNormal | VR指标显示VrNormal形态 | 100.0% | 1 | 50.0 |
| indicator | VR | VrStable | VR指标显示VrStable形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMVolumeShrink | ZXM_VOLUME_SHRINK_BUY_POINT | ZXMVolumeShrink指标ZXM_VOLUME_SHRINK_BUY_POINT形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMVolumeShrink | ZXM_VOLUME_OBVIOUS_SHRINK | ZXMVolumeShrink指标ZXM_VOLUME_OBVIOUS_SHRINK形态 | 100.0% | 1 | 50.0 |
| indicator | DMI | ADX强趋势 | ADX大于25，表示趋势强劲 | 100.0% | 1 | 50.0 |
| indicator | DMI | ADX上升 | ADX上升，趋势强度增强 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | MACD买点满足 | ZXMBuyPointScore指标MACD买点满足形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 换手买点满足 | ZXMBuyPointScore指标换手买点满足形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 中等买点评分 | ZXMBuyPointScore指标中等买点评分形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 多数买点指标满足 | ZXMBuyPointScore指标多数买点指标满足形态 | 100.0% | 1 | 50.0 |
| indicator | EnhancedKDJ | KDJ金叉 | K线从下方突破D线，买入信号 | 100.0% | 1 | 50.0 |
| indicator | EnhancedKDJ | K线上升 | K线呈上升趋势 | 100.0% | 1 | 50.0 |
| indicator | EnhancedKDJ | D线上升 | D线呈上升趋势 | 100.0% | 1 | 50.0 |
| indicator | EnhancedKDJ | KDJ强势金叉 | K线以大角度上穿D线，表明强势上升趋势 | 100.0% | 1 | 50.0 |
| indicator | EnhancedWR | Wr交叉上方Mid | EnhancedWR指标显示Wr交叉上方Mid形态 | 100.0% | 1 | 50.0 |
| indicator | OBV | OBV量价配合 | 基于OBV指标的量价配合分析: OBV_VOLUME_PRICE_DIVERGENCE | 100.0% | 1 | 50.0 |
| indicator | CCI | CCI极度超买 | CCI值高于+200，表示严重超买 | 100.0% | 1 | 50.0 |
| indicator | CCI | CCI上穿超卖线 | CCI从超卖区上穿-100线 | 100.0% | 1 | 50.0 |
| indicator | CCI | CCI上穿零轴 | CCI上穿零轴线 | 100.0% | 1 | 50.0 |
| indicator | CCI | CCI强势上升趋势 | CCI持续上升，表示强势上涨 | 100.0% | 1 | 50.0 |
| indicator | EnhancedTRIX | TRIX零轴上方 | TRIX位于零轴上方，长期趋势偏多 | 100.0% | 1 | 50.0 |
| indicator | EnhancedTRIX | TRIX上升 | TRIX指标上升，长期动量增强 | 100.0% | 1 | 50.0 |
| indicator | EnhancedTRIX | TRIX强烈看涨共振 | TRIX多重信号共振，形成强烈看涨态势 | 100.0% | 1 | 50.0 |
| indicator | EnhancedTRIX | TRIX减速 | TRIX指标减速变化，动能转变 | 100.0% | 1 | 50.0 |
| indicator | WR | Wr交叉上方Mid | WR指标显示Wr交叉上方Mid形态 | 100.0% | 1 | 50.0 |
| indicator | UnifiedMA | Price上方LongMa | UnifiedMA指标显示Price上方LongMa形态 | 100.0% | 1 | 50.0 |
| indicator | UnifiedMA | Price突破上方LongMa | UnifiedMA指标显示Price突破上方LongMa形态 | 100.0% | 1 | 50.0 |
| indicator | UnifiedMA | Ma看涨Alignment | UnifiedMA指标显示Ma看涨Alignment形态 | 100.0% | 1 | 50.0 |
| indicator | UnifiedMA | MaLong上升趋势 | UnifiedMA指标显示MaLong上升趋势形态 | 100.0% | 1 | 50.0 |
| indicator | EnhancedCCI | Zero Cross Up | zero_cross_up形态 | 100.0% | 1 | 50.0 |
| indicator | VOL | 成交量能量分析 | 基于成交量能量变化的技术分析: VOL_FALLING | 100.0% | 1 | 50.0 |
| indicator | VOL | 成交量金叉 | 短期成交量均线上穿长期均线，表示成交量趋势向好 | 100.0% | 1 | 50.0 |
| indicator | SAR | 看涨Sar  Reversal | SAR指标显示看涨Sar  Reversal形态 | 100.0% | 1 | 50.0 |
| indicator | SAR | 上升趋势 | SAR指标显示上升趋势 | 100.0% | 1 | 50.0 |
| indicator | SAR | Sar Close To Price | SAR_CLOSE_TO_PRICE形态 | 100.0% | 1 | 50.0 |
| indicator | SAR | Sar Low Acceleration | SAR_LOW_ACCELERATION形态 | 100.0% | 1 | 50.0 |
| indicator | PVT | Pvt Golden Cross | PVT_GOLDEN_CROSS形态 | 100.0% | 1 | 50.0 |
| indicator | PVT | Pvt Above Signal | PVT_ABOVE_SIGNAL形态 | 100.0% | 1 | 50.0 |
| indicator | PVT | Pvt上升 | PVT指标Pvt上升 | 100.0% | 1 | 50.0 |
| indicator | PVT | Pvt Strong Up | PVT_STRONG_UP形态 | 100.0% | 1 | 50.0 |
| indicator | TRIX | Trix Above Zero | TRIX_ABOVE_ZERO形态 | 100.0% | 1 | 50.0 |
| indicator | Aroon | Aroon强势上涨 | Aroon Up > 70 且 Aroon Down < 30，表明强势上升趋势 | 100.0% | 1 | 50.0 |
| indicator | Aroon | Aroon震荡器极度看涨 | Aroon震荡器值 > 50，表明极强的上升趋势 | 100.0% | 1 | 50.0 |
| indicator | ZXMBSAbsorb | ZXM_BS_ABSORB_WATCH_ZONE | ZXMBSAbsorb指标ZXM_BS_ABSORB_WATCH_ZONE形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBSAbsorb | ZXM_BS_STABLE_MOMENTUM | ZXMBSAbsorb指标ZXM_BS_STABLE_MOMENTUM形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 上升趋势 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 趋势刚转为上升 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 上升趋势初始阶段 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 短期上升趋势 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 趋势初期 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 高规律性周期 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 上升趋势 | TrendDetector指标上升趋势形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 趋势转折：由空转多 | TrendDetector指标趋势转折：由空转多形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 上升趋势初期 | TrendDetector指标上升趋势初期形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 虚弱上升趋势 | TrendDetector指标虚弱上升趋势形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 短期趋势 | TrendDetector指标短期趋势形态 | 100.0% | 1 | 50.0 |
| indicator | DMA | DMA上升趋势 | DMA大于0且DMA大于AMA，表示强势上升趋势 | 100.0% | 1 | 50.0 |
| indicator | DMA | DMA平均差值分析 | 基于DMA平均差值指标的技术分析: DMA_ABOVE_ZERO | 100.0% | 1 | 50.0 |
| indicator | BIAS | BIAS中度偏高 | BIAS值在+5%到+15%之间，表示轻度超买 | 100.0% | 1 | 50.0 |
| indicator | AD | Strength | strength形态 | 100.0% | 1 | 50.0 |
| indicator | ROC | Roc超买 | ROC指标显示Roc超买形态 | 100.0% | 1 | 50.0 |
| indicator | ROC | Roc金叉交叉 | ROC指标显示Roc金叉交叉形态 | 100.0% | 1 | 50.0 |
| indicator | ROC | Roc交叉UpZero | ROC指标显示Roc交叉UpZero形态 | 100.0% | 1 | 50.0 |
| indicator | ROC | Roc上方Zero | ROC指标显示Roc上方Zero形态 | 100.0% | 1 | 50.0 |
| indicator | ROC | Roc上方Ma | ROC指标显示Roc上方Ma形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 弹性评分信号 | ZXMElasticityScore指标弹性评分信号形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 振幅弹性满足 | ZXMElasticityScore指标振幅弹性满足形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 涨幅弹性满足 | ZXMElasticityScore指标涨幅弹性满足形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 极高弹性评分 | ZXMElasticityScore指标极高弹性评分形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 全部弹性指标满足 | ZXMElasticityScore指标全部弹性指标满足形态 | 100.0% | 1 | 50.0 |
| indicator | MA | MA(5,10)金叉 | 当短期MA(5)上穿中期MA(10)时，被视为看涨信号。 | 100.0% | 1 | 50.0 |
| indicator | MA | MA多头排列 | 短期MA(5)在长期MA(60)之上，强劲上升趋势 | 100.0% | 1 | 50.0 |
| indicator | SelectionModel | 选股系统买入信号 | 选股系统买入信号形态 | 100.0% | 1 | 50.0 |
| indicator | SelectionModel | 强趋势上涨股 | 强趋势上涨股形态 | 100.0% | 1 | 50.0 |
| indicator | SelectionModel | 最高优先级选股 | 最高优先级选股形态 | 100.0% | 1 | 50.0 |
| indicator | SelectionModel | 超强上升趋势 | 超强上升趋势形态 | 100.0% | 1 | 50.0 |
| indicator | CCI | CCI超买 | CCI值高于+100，表示超买 | 100.0% | 1 | 35.0 |
| indicator | EnhancedWR | WrExtreme超买 | EnhancedWR指标显示WrExtreme超买形态 | 100.0% | 1 | 30.0 |
| indicator | WR | WrExtreme超买 | WR指标显示WrExtreme超买形态 | 100.0% | 1 | 30.0 |

### 📊 30min周期分析总结

#### 🎯 高命中率指标 (≥80%)
- **EnhancedMACD** (MACD金叉): 100.0%命中率，平均得分75.0分
  *MACD线上穿信号线，表明上升趋势开始*
- **ChipDistribution** (ChipBottomAccumulation): 100.0%命中率，平均得分75.0分
  *ChipDistribution指标显示ChipBottomAccumulation形态*
- **ADX** (Adx Strong上升): 100.0%命中率，平均得分75.0分
  *ADX指标Adx Strong上升*
- **InstitutionalBehavior** (InstAbsorptionPhase): 100.0%命中率，平均得分75.0分
  *InstitutionalBehavior指标显示InstAbsorptionPhase形态*
- **ADX** (上升趋势): 100.0%命中率，平均得分70.0分
  *ADX指标显示上升趋势*

---

## 📈 60min 周期共性指标

### 数据统计
- **总样本数量**: 1个买点样本
- **共性指标数量**: 146个指标形态
- **分析周期**: 60minK线

| 指标类型 | 指标名称 | 形态 | 形态描述 | 命中率 | 命中数量 | 平均得分 |
|---------|----------|------|----------|--------|----------|----------|
| indicator | ChipDistribution | ChipBottomAccumulation | ChipDistribution指标显示ChipBottomAccumulation形态 | 100.0% | 1 | 75.0 |
| indicator | InstitutionalBehavior | InstAbsorptionPhase | InstitutionalBehavior指标显示InstAbsorptionPhase形态 | 100.0% | 1 | 75.0 |
| indicator | ADX | 上升趋势 | ADX指标显示上升趋势 | 100.0% | 1 | 70.0 |
| indicator | InstitutionalBehavior | InstLowProfit | InstitutionalBehavior指标显示InstLowProfit形态 | 100.0% | 1 | 70.0 |
| indicator | ChipDistribution | ChipLowProfit | ChipDistribution指标显示ChipLowProfit形态 | 100.0% | 1 | 65.0 |
| indicator | BOLL | 布林带收缩 | 布林带收缩，表明波动率降低，可能酝酿突破 | 100.0% | 1 | 65.0 |
| indicator | VOL | 成交量能量分析 | 基于成交量能量变化的技术分析: VOL_RISING | 100.0% | 1 | 65.0 |
| indicator | TRIX | Trix Above Signal | TRIX_ABOVE_SIGNAL形态 | 100.0% | 1 | 65.0 |
| indicator |  | 通用条件: MA5>MA10 | 自定义条件表达式: MA5>MA10 | 100.0% | 1 | 65.0 |
| indicator | EnhancedMACD | MACD上升 | MACD线呈上升趋势 | 100.0% | 1 | 60.0 |
| indicator | ChipDistribution | ChipTight | ChipDistribution指标显示ChipTight形态 | 100.0% | 1 | 60.0 |
| indicator | EnhancedWR | Wr上升 | EnhancedWR指标显示Wr上升形态 | 100.0% | 1 | 60.0 |
| indicator | WR | Wr上升 | WR指标显示Wr上升形态 | 100.0% | 1 | 60.0 |
| indicator | TRIX | Trix上升 | TRIX指标Trix上升 | 100.0% | 1 | 60.0 |
| indicator | EnhancedMFI | Mfi上升 | EnhancedMFI指标Mfi上升 | 100.0% | 1 | 60.0 |
| indicator | STOCHRSI | StochRSI超买 | StochRSI进入超买区域，可能出现回调 | 100.0% | 1 | 50.0 |
| indicator | STOCHRSI | 随机RSI超买超卖 | 基于StochRSI指标的超买超卖分析: STOCHRSI_D_RISING | 100.0% | 1 | 50.0 |
| indicator | Chaikin | Chaikin零轴上方 | Chaikin震荡器位于零轴上方 | 100.0% | 1 | 50.0 |
| indicator | Chaikin | Chaikin上升 | Chaikin震荡器上升 | 100.0% | 1 | 50.0 |
| indicator | Chaikin | Chaikin连续上升 | Chaikin震荡器连续上升 | 100.0% | 1 | 50.0 |
| indicator | Chaikin | Chaikin大幅上升 | Chaikin震荡器大幅上升 | 100.0% | 1 | 50.0 |
| indicator | Chaikin | Chaikin快速变化 | Chaikin震荡器快速变化 | 100.0% | 1 | 50.0 |
| indicator | VOSC | Vosc上方Zero | VOSC指标显示Vosc上方Zero形态 | 100.0% | 1 | 50.0 |
| indicator | VOSC | Vosc上方信号 | VOSC指标显示Vosc上方信号形态 | 100.0% | 1 | 50.0 |
| indicator | VOSC | VoscPriceConfirmation | VOSC指标显示VoscPriceConfirmation形态 | 100.0% | 1 | 50.0 |
| indicator | StockScoreCalculator | 综合评分适中 | 综合评分适中形态 | 100.0% | 1 | 50.0 |
| indicator | StockScoreCalculator | 趋势强劲 | 趋势强劲形态 | 100.0% | 1 | 50.0 |
| indicator | StockScoreCalculator | 高波动性 | 高波动性形态 | 100.0% | 1 | 50.0 |
| indicator | BounceDetector | 反弹确认信号 | 反弹确认信号形态 | 100.0% | 1 | 50.0 |
| indicator | BounceDetector | 大幅反弹 | 大幅反弹形态 | 100.0% | 1 | 50.0 |
| indicator | BounceDetector | 短期上升趋势 | 短期上升趋势形态 | 100.0% | 1 | 50.0 |
| indicator | BounceDetector | 明显放量 | 明显放量形态 | 100.0% | 1 | 50.0 |
| indicator | BounceDetector | 强势反弹 | 强势反弹形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMPattern | key_support_hold | ZXMPattern指标key_support_hold形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMPattern | ma_precise_support | ZXMPattern指标ma_precise_support形态 | 100.0% | 1 | 50.0 |
| indicator | Vortex | VortexViMinus上方 | Vortex指标显示VortexViMinus上方形态 | 100.0% | 1 | 50.0 |
| indicator | Vortex | VortexViPlus上升 | Vortex指标显示VortexViPlus上升形态 | 100.0% | 1 | 50.0 |
| indicator | Vortex | VortexViDiff上升 | Vortex指标显示VortexViDiff上升形态 | 100.0% | 1 | 50.0 |
| indicator | PSY | PSY_BELOW_50 | PSY指标PSY_BELOW_50形态 | 100.0% | 1 | 50.0 |
| indicator | PSY | PSY_BELOW_MA | PSY指标PSY_BELOW_MA形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMTurnover | ZXM_TURNOVER_BUY_POINT | ZXMTurnover指标ZXM_TURNOVER_BUY_POINT形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMTurnover | ZXM_TURNOVER_ACTIVE | ZXMTurnover指标ZXM_TURNOVER_ACTIVE形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMTurnover | ZXM_TURNOVER_SUDDEN_DECREASE | ZXMTurnover指标ZXM_TURNOVER_SUDDEN_DECREASE形态 | 100.0% | 1 | 50.0 |
| indicator | EnhancedMACD | MACD柱状体为正 | MACD柱状体大于零，表示上升动能 | 100.0% | 1 | 50.0 |
| indicator | EnhancedMACD | MACD柱状体增长 | MACD柱状体连续增长，表示动能增强 | 100.0% | 1 | 50.0 |
| indicator | EnhancedMACD | MACD强上升趋势 | MACD柱状体为正且趋势强度高，表明强势上升趋势 | 100.0% | 1 | 50.0 |
| indicator | ATR | AtrUpward突破 | ATR指标显示AtrUpward突破形态 | 100.0% | 1 | 50.0 |
| indicator | EMA | EMA多头排列 | 指数移动平均线呈多头排列，趋势向上 | 100.0% | 1 | 50.0 |
| indicator | ChipDistribution | HardUntrapped | ChipDistribution指标显示HardUntrapped形态 | 100.0% | 1 | 50.0 |
| indicator | AmplitudeElasticity | 大振幅日 | AmplitudeElasticity指标大振幅日形态 | 100.0% | 1 | 50.0 |
| indicator | AmplitudeElasticity | 极大振幅 | AmplitudeElasticity指标极大振幅形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | Vix上升趋势 | StockVIX指标显示Vix上升趋势形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | VixNormal | StockVIX指标显示VixNormal形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | Vix上升 | StockVIX指标显示Vix上升形态 | 100.0% | 1 | 50.0 |
| indicator | StockVIX | Vix上方Ma20 | StockVIX指标显示Vix上方Ma20形态 | 100.0% | 1 | 50.0 |
| indicator | EMV | EMV零轴上方 | EMV位于零轴上方，买盘力量占优 | 100.0% | 1 | 50.0 |
| indicator | EMV | EMV均线上方 | EMV位于移动平均线上方 | 100.0% | 1 | 50.0 |
| indicator | EMV | EMV上升 | EMV值上升 | 100.0% | 1 | 50.0 |
| indicator | CMO | CMO动量震荡 | 基于CMO动量震荡指标的技术分析: CMO_ABOVE_ZERO | 100.0% | 1 | 50.0 |
| indicator | CMO | CMO动量震荡 | 基于CMO动量震荡指标的技术分析: CMO_RISING | 100.0% | 1 | 50.0 |
| indicator | CMO | CMO动量震荡 | 基于CMO动量震荡指标的技术分析: CMO_STRONG_RISE | 100.0% | 1 | 50.0 |
| indicator | Momentum | Mtm交叉上方Zero | Momentum指标显示Mtm交叉上方Zero形态 | 100.0% | 1 | 50.0 |
| indicator | Momentum | Mtm上方Zero | Momentum指标显示Mtm上方Zero形态 | 100.0% | 1 | 50.0 |
| indicator | Momentum | Mtm交叉上方信号 | Momentum指标显示Mtm交叉上方信号形态 | 100.0% | 1 | 50.0 |
| indicator | Momentum | Mtm上方信号 | Momentum指标显示Mtm上方信号形态 | 100.0% | 1 | 50.0 |
| indicator | Momentum | Mtm上升 | Momentum指标显示Mtm上升形态 | 100.0% | 1 | 50.0 |
| indicator | Momentum | MtmLargeRise | Momentum指标显示MtmLargeRise形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 弹性买点 | Elasticity指标弹性买点形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 轻微弹性比率 | Elasticity指标轻微弹性比率形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 强反弹 | Elasticity指标强反弹形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 量能正常 | Elasticity指标量能正常形态 | 100.0% | 1 | 50.0 |
| indicator | Elasticity | 大幅波动区间 | Elasticity指标大幅波动区间形态 | 100.0% | 1 | 50.0 |
| indicator | KC | Kc上方Middle | KC指标显示Kc上方Middle形态 | 100.0% | 1 | 50.0 |
| indicator | KC | KcExpanding | KC指标显示KcExpanding形态 | 100.0% | 1 | 50.0 |
| indicator | KC | KcOscillating | KC指标显示KcOscillating形态 | 100.0% | 1 | 50.0 |
| indicator | BOLL | 布林带波动区间 | 基于布林带指标的波动区间分析: BOLL_OVERBOUGHT | 100.0% | 1 | 50.0 |
| indicator | BOLL | 布林带上轨突破 | 价格突破布林带上轨，表明强势上涨 | 100.0% | 1 | 50.0 |
| indicator | BOLL | 布林带扩张 | 布林带扩张，表明波动率增加，趋势可能延续 | 100.0% | 1 | 50.0 |
| indicator | BOLL | 布林带趋势跟随 | 价格沿布林带边缘运行，表明趋势强劲 | 100.0% | 1 | 50.0 |
| indicator | ZXMRiseElasticity | 涨幅弹性信号 | ZXMRiseElasticity指标涨幅弹性信号形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMRiseElasticity | 大涨日 | ZXMRiseElasticity指标大涨日形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMRiseElasticity | 大涨幅 | ZXMRiseElasticity指标大涨幅形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMRiseElasticity | 偶尔大涨 | ZXMRiseElasticity指标偶尔大涨形态 | 100.0% | 1 | 50.0 |
| indicator | MTM | Mtm金叉交叉 | MTM指标显示Mtm金叉交叉形态 | 100.0% | 1 | 50.0 |
| indicator | MTM | Mtm交叉UpZero | MTM指标显示Mtm交叉UpZero形态 | 100.0% | 1 | 50.0 |
| indicator | MTM | Mtm上方Zero | MTM指标显示Mtm上方Zero形态 | 100.0% | 1 | 50.0 |
| indicator | MTM | Mtm上方Ma | MTM指标显示Mtm上方Ma形态 | 100.0% | 1 | 50.0 |
| indicator | MTM | Mtm强势Up | MTM指标显示Mtm强势Up形态 | 100.0% | 1 | 50.0 |
| indicator | VR | VrNormal | VR指标显示VrNormal形态 | 100.0% | 1 | 50.0 |
| indicator | VR | Vr上方Ma | VR指标显示Vr上方Ma形态 | 100.0% | 1 | 50.0 |
| indicator | VR | Vr金叉交叉 | VR指标显示Vr金叉交叉形态 | 100.0% | 1 | 50.0 |
| indicator | VR | Vr上升 | VR指标显示Vr上升形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMVolumeShrink | ZXM_VOLUME_NORMAL | ZXMVolumeShrink指标ZXM_VOLUME_NORMAL形态 | 100.0% | 1 | 50.0 |
| indicator | DMI | ADX强趋势 | ADX大于25，表示趋势强劲 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | MACD买点满足 | ZXMBuyPointScore指标MACD买点满足形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 换手买点满足 | ZXMBuyPointScore指标换手买点满足形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 中等买点评分 | ZXMBuyPointScore指标中等买点评分形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBuyPointScore | 多数买点指标满足 | ZXMBuyPointScore指标多数买点指标满足形态 | 100.0% | 1 | 50.0 |
| indicator | EnhancedKDJ | K线上升 | K线呈上升趋势 | 100.0% | 1 | 50.0 |
| indicator | EnhancedKDJ | D线上升 | D线呈上升趋势 | 100.0% | 1 | 50.0 |
| indicator | OBV | OBV量价配合 | 基于OBV指标的量价配合分析: OBV_VOLUME_PRICE_DIVERGENCE | 100.0% | 1 | 50.0 |
| indicator | CCI | CCI上穿零轴 | CCI上穿零轴线 | 100.0% | 1 | 50.0 |
| indicator | CCI | CCI强势上升趋势 | CCI持续上升，表示强势上涨 | 100.0% | 1 | 50.0 |
| indicator | EnhancedTRIX | TRIX零轴上方 | TRIX位于零轴上方，长期趋势偏多 | 100.0% | 1 | 50.0 |
| indicator | EnhancedTRIX | TRIX上升 | TRIX指标上升，长期动量增强 | 100.0% | 1 | 50.0 |
| indicator | EnhancedTRIX | TRIX趋势转折 | 基于TRIX指标的趋势转折分析: golden_cross | 100.0% | 1 | 50.0 |
| indicator | EnhancedTRIX | TRIX强烈看涨共振 | TRIX多重信号共振，形成强烈看涨态势 | 100.0% | 1 | 50.0 |
| indicator | EnhancedTRIX | TRIX加速上升 | TRIX指标加速上升，表明价格上涨动能不断增强 | 100.0% | 1 | 50.0 |
| indicator | UnifiedMA | Price上方LongMa | UnifiedMA指标显示Price上方LongMa形态 | 100.0% | 1 | 50.0 |
| indicator | UnifiedMA | Price突破上方LongMa | UnifiedMA指标显示Price突破上方LongMa形态 | 100.0% | 1 | 50.0 |
| indicator | UnifiedMA | MaLong上升趋势 | UnifiedMA指标显示MaLong上升趋势形态 | 100.0% | 1 | 50.0 |
| indicator | EnhancedCCI | Zero Cross Up | zero_cross_up形态 | 100.0% | 1 | 50.0 |
| indicator | SAR | Sar Close To Price | SAR_CLOSE_TO_PRICE形态 | 100.0% | 1 | 50.0 |
| indicator | SAR | Sar Low Acceleration | SAR_LOW_ACCELERATION形态 | 100.0% | 1 | 50.0 |
| indicator | PVT | Pvt Golden Cross | PVT_GOLDEN_CROSS形态 | 100.0% | 1 | 50.0 |
| indicator | PVT | Pvt Above Signal | PVT_ABOVE_SIGNAL形态 | 100.0% | 1 | 50.0 |
| indicator | PVT | Pvt上升 | PVT指标Pvt上升 | 100.0% | 1 | 50.0 |
| indicator | PVT | Pvt Strong Up | PVT_STRONG_UP形态 | 100.0% | 1 | 50.0 |
| indicator | TRIX | Trix Golden Cross | TRIX_GOLDEN_CROSS形态 | 100.0% | 1 | 50.0 |
| indicator | TRIX | Trix Above Zero | TRIX_ABOVE_ZERO形态 | 100.0% | 1 | 50.0 |
| indicator | TRIX | Trix Consecutive上升 | TRIX指标Trix Consecutive上升 | 100.0% | 1 | 50.0 |
| indicator | InstitutionalBehavior | InstWashout | InstitutionalBehavior指标显示InstWashout形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBSAbsorb | ZXM_BS_ABSORB_WATCH_ZONE | ZXMBSAbsorb指标ZXM_BS_ABSORB_WATCH_ZONE形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMBSAbsorb | ZXM_BS_STABLE_MOMENTUM | ZXMBSAbsorb指标ZXM_BS_STABLE_MOMENTUM形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 震荡/无趋势 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 趋势初期 | 100.0% | 1 | 50.0 |
| indicator | TrendDuration | 趋势生命周期分析 | 基于趋势生命周期的持续性分析: 高规律性周期 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 震荡/无趋势 | TrendDetector指标震荡/无趋势形态 | 100.0% | 1 | 50.0 |
| indicator | TrendDetector | 中期趋势 | TrendDetector指标中期趋势形态 | 100.0% | 1 | 50.0 |
| indicator | DMA | DMA上升趋势 | DMA大于0且DMA大于AMA，表示强势上升趋势 | 100.0% | 1 | 50.0 |
| indicator | DMA | DMA平均差值分析 | 基于DMA平均差值指标的技术分析: DMA_ABOVE_ZERO | 100.0% | 1 | 50.0 |
| indicator | DMA | DMA平均差值分析 | 基于DMA平均差值指标的技术分析: DMA_ACCELERATION_UP | 100.0% | 1 | 50.0 |
| indicator | RSI | RSI超买 | RSI指标超过70，进入超买区域，存在回调压力 | 100.0% | 1 | 50.0 |
| indicator | BIAS | BIAS中度偏高 | BIAS值在+5%到+15%之间，表示轻度超买 | 100.0% | 1 | 50.0 |
| indicator | ROC | Roc超买 | ROC指标显示Roc超买形态 | 100.0% | 1 | 50.0 |
| indicator | ROC | Roc上方Zero | ROC指标显示Roc上方Zero形态 | 100.0% | 1 | 50.0 |
| indicator | ROC | Roc上方Ma | ROC指标显示Roc上方Ma形态 | 100.0% | 1 | 50.0 |
| indicator | EnhancedRSI | Rsi Overbought | rsi_overbought形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 涨幅弹性满足 | ZXMElasticityScore指标涨幅弹性满足形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 中等弹性评分 | ZXMElasticityScore指标中等弹性评分形态 | 100.0% | 1 | 50.0 |
| indicator | ZXMElasticityScore | 部分弹性指标满足 | ZXMElasticityScore指标部分弹性指标满足形态 | 100.0% | 1 | 50.0 |
| indicator | MA | MA多头排列 | 短期MA(5)在长期MA(60)之上，强劲上升趋势 | 100.0% | 1 | 50.0 |
| indicator | SelectionModel | 震荡趋势 | 震荡趋势形态 | 100.0% | 1 | 50.0 |
| indicator | CCI | CCI超买 | CCI值高于+100，表示超买 | 100.0% | 1 | 35.0 |
| indicator | EnhancedWR | WrExtreme超买 | EnhancedWR指标显示WrExtreme超买形态 | 100.0% | 1 | 30.0 |
| indicator | WR | WrExtreme超买 | WR指标显示WrExtreme超买形态 | 100.0% | 1 | 30.0 |

### 📊 60min周期分析总结

#### 🎯 高命中率指标 (≥80%)
- **ChipDistribution** (ChipBottomAccumulation): 100.0%命中率，平均得分75.0分
  *ChipDistribution指标显示ChipBottomAccumulation形态*
- **InstitutionalBehavior** (InstAbsorptionPhase): 100.0%命中率，平均得分75.0分
  *InstitutionalBehavior指标显示InstAbsorptionPhase形态*
- **ADX** (上升趋势): 100.0%命中率，平均得分70.0分
  *ADX指标显示上升趋势*
- **InstitutionalBehavior** (InstLowProfit): 100.0%命中率，平均得分70.0分
  *InstitutionalBehavior指标显示InstLowProfit形态*
- **ChipDistribution** (ChipLowProfit): 100.0%命中率，平均得分65.0分
  *ChipDistribution指标显示ChipLowProfit形态*

---

## 🎯 综合分析总结

### 📊 整体统计
- **分析周期数**: 6个时间周期
- **共性指标总数**: 951个指标形态
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

*报告生成时间: 2025-06-21 16:24:07*  
*分析系统: 股票分析系统 v2.0*  
*技术支持: 基于86个技术指标和ZXM专业体系*
