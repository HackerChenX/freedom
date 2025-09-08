# 统一指标质量监控报告

## 📊 测试概要

- **测试时间**: 2025-09-07T15:56:04.704017
- **测试指标数**: 100
- **通过指标数**: 96
- **失败指标数**: 1
- **错误指标数**: 3
- **执行时间**: 272.37 秒
- **通过率**: 96.0%

## 🎯 测试方法说明

本次测试采用**统一调用现有验证脚本**的方式，确保测试方式的一致性：

1. **复用现有验证脚本**: 调用每个指标专门的验证脚本
2. **保持测试标准一致**: 使用与之前验证相同的测试方法
3. **标准化结果格式**: 统一处理不同脚本的输出格式
4. **完整错误处理**: 处理脚本执行中的各种异常情况

## 📋 详细结果

| 指标名称 | 状态 | 分数 | 验证脚本 | 执行时间 | 备注 |
|---------|------|------|----------|----------|------|
| DMI | 🔴 FAILED | 0 | validate_adx_indicator.py | 0.63s | - |
| MACD | 🟢 PASSED | 100.0 | validate_enhanced_macd_trend.py | 3.06s | - |
| RSI | 🟢 PASSED | 100.0 | validate_rsi_derivatives.py | 0.21s | - |
| BOLL | 🟢 PASSED | 100.0 | validate_enhanced_boll_indicators.py | 0.12s | - |
| KDJ | 🟢 PASSED | 100.0 | validate_enhanced_kdj.py | 0.10s | - |
| WR | 🟢 PASSED | 100.0 | validate_enhanced_wr.py | 3.30s | - |
| VOL | 🟢 PASSED | 100.0 | validate_vol.py | 0.23s | - |
| VR | 🟢 PASSED | 100.0 | validate_volume_score.py | 0.07s | - |
| AROON | 🟢 PASSED | 100.0 | validate_trend_indicators.py | 0.01s | - |
| DMA | 🟢 PASSED | 100.0 | validate_trend_indicators.py | 0.00s | - |
| MACD_SCORE | 🟢 PASSED | 100.0 | validate_score_indicators.py | 0.01s | - |
| RSI_SCORE | 🟢 PASSED | 100.0 | validate_score_indicators.py | 0.01s | - |
| BOLL_SCORE | 🟢 PASSED | 100.0 | validate_score_indicators.py | 0.00s | - |
| KDJ_SCORE | 🟢 PASSED | 100.0 | validate_score_indicators.py | 0.00s | - |
| ENHANCED_MACD | 🟢 PASSED | 100.0 | validate_enhanced_macd_trend.py | 0.29s | - |
| ENHANCED_BOLL | 🟢 PASSED | 100.0 | validate_enhanced_boll_indicators.py | 0.06s | - |
| ENHANCED_KDJ | 🟢 PASSED | 100.0 | validate_enhanced_kdj.py | 0.06s | - |
| FIBONACCI_TOOLS | 🟢 PASSED | 100.0 | validate_fibonacci_tools.py | 0.34s | - |
| MARKET_ENV | 🟢 PASSED | 100.0 | validate_market_env.py | 0.04s | - |
| SENTIMENT_ANALYSIS | 🟢 PASSED | 100.0 | validate_sentiment_analysis.py | 0.04s | - |
| TREND_CLASSIFICATION | 🟢 PASSED | 100.0 | validate_trend_classification.py | 0.04s | - |
| TREND_STRENGTH | 🟢 PASSED | 100.0 | validate_trend_strength.py | 0.16s | - |
| TIME_CYCLE_ANALYSIS | 🟢 PASSED | 100.0 | validate_time_cycle_analysis.py | 0.07s | - |
| MULTI_PERIOD_RESONANCE | 🟢 PASSED | 100.0 | validate_multi_period_resonance.py | 0.06s | - |
| INTRADAY_VOLATILITY | 🟢 PASSED | 100.0 | validate_intraday_volatility.py | 0.04s | - |
| STOCK_VIX | 🟢 PASSED | 100.0 | validate_stock_vix.py | 0.11s | - |
| ALPHA_GENERATION | 🟢 PASSED | 100.0 | validate_alpha_generation_indicators.py | 0.00s | - |
| BETA_HEDGING | 🟢 PASSED | 100.0 | validate_beta_hedging_indicators.py | 0.00s | - |
| ADX | 🟢 PASSED | 20.0 | validate_fixed_adx_indicator.py | 3.19s | 错误: 2025-09-07 15:56:13 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ROC | 🟢 PASSED | 20.0 | validate_fixed_roc_indicator.py | 2.89s | 错误: 2025-09-07 15:56:16 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| OBV | 🟢 PASSED | 20.0 | validate_fixed_obv_indicator.py | 4.11s | 错误: 2025-09-07 15:56:19 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| MTM | 🟢 PASSED | 20.0 | validate_fixed_mtm_indicator.py | 3.47s | 错误: 2025-09-07 15:56:23 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| MFI | 🟢 PASSED | 20.0 | validate_fixed_mfi_indicator.py | 2.92s | 错误: 2025-09-07 15:56:26 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| VIX | 🟢 PASSED | 20.0 | validate_fixed_baseindicators.py | 3.06s | 错误: 2025-09-07 15:56:29 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ATR | 🟢 PASSED | 20.0 | validate_atr_indicator.py | 2.23s | 错误: 2025-09-07 15:56:32 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| PSY | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.12s | 错误: 2025-09-07 15:56:35 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_WEEKLY_MACD | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 3.30s | 错误: 2025-09-07 15:56:37 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_MONTHLY_MACD | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 3.44s | 错误: 2025-09-07 15:56:40 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_TURNOVER | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.16s | 错误: 2025-09-07 15:56:44 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_VOLUME_SHRINK | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 3.79s | 错误: 2025-09-07 15:56:48 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_MA_CALLBACK | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 3.57s | 错误: 2025-09-07 15:56:52 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_DAILY_TREND_UP | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 3.66s | 错误: 2025-09-07 15:56:56 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_WEEKLY_TREND_UP | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 5.69s | 错误: 2025-09-07 15:56:59 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_MONTHLY_KDJ_TREND_UP | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 7.64s | 错误: 2025-09-07 15:57:05 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_AMPLITUDE_ELASTICITY | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 6.12s | 错误: 2025-09-07 15:57:13 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_RISE_ELASTICITY | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 5.09s | 错误: 2025-09-07 15:57:19 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_ELASTICITY | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 6.97s | 错误: 2025-09-07 15:57:24 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_BOUNCE_DETECTOR | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 9.14s | 错误: 2025-09-07 15:57:32 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_BUYPOINT_SCORE | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 6.53s | 错误: 2025-09-07 15:57:41 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_TREND_SCORE | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 7.58s | 错误: 2025-09-07 15:57:47 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_ELASTIC_SCORE | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 6.49s | 错误: 2025-09-07 15:57:55 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_VOLUME_ENERGY | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.68s | 错误: 2025-09-07 15:58:01 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_PRICE_POSITION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.46s | 错误: 2025-09-07 15:58:06 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_TECHNICAL_FORM | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.29s | 错误: 2025-09-07 15:58:10 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_HOT_SPOT | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.18s | 错误: 2025-09-07 15:58:14 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_INDUSTRY_ROTATION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 3.72s | 错误: 2025-09-07 15:58:18 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_CYCLE_POSITION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 5.46s | 错误: 2025-09-07 15:58:22 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_RISK_CONTROL | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.61s | 错误: 2025-09-07 15:58:28 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_TIMING_SIGNAL | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.44s | 错误: 2025-09-07 15:58:32 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_POSITION_MANAGEMENT | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.63s | 错误: 2025-09-07 15:58:37 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_PORTFOLIO_OPTIMIZATION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.26s | 错误: 2025-09-07 15:58:41 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_STRATEGY_COMBINATION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.06s | 错误: 2025-09-07 15:58:46 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_PERFORMANCE_ATTRIBUTION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 3.92s | 错误: 2025-09-07 15:58:49 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_ALPHA_GENERATION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 3.71s | 错误: 2025-09-07 15:58:53 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_BETA_HEDGING | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.03s | 错误: 2025-09-07 15:58:57 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_CHIP_DISTRIBUTION | 🟢 PASSED | 20.0 | validate_zxm_chip_distribution_complete.py | 2.68s | 错误: 2025-09-07 15:59:31 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_FUND_FLOW | 🟢 PASSED | 20.0 | validate_zxm_fund_flow_complete.py | 2.43s | 错误: 2025-09-07 15:59:33 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_INSTITUTION_BEHAVIOR | 🟢 PASSED | 20.0 | validate_zxm_institution_behavior_complete.py | 1.85s | 错误: 2025-09-07 15:59:35 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_MARKET_SENTIMENT | 🟢 PASSED | 20.0 | validate_fixed_zxm_market_sentiment.py | 2.05s | 错误: 2025-09-07 15:59:37 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_LIQUIDITY_ANALYSIS | 🟢 PASSED | 20.0 | validate_fixed_zxm_liquidity_analysis.py | 1.73s | 错误: 2025-09-07 15:59:39 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_CORRELATION_MATRIX | 🟢 PASSED | 20.0 | validate_fixed_zxm_correlation_matrix.py | 2.95s | 错误: 2025-09-07 15:59:41 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_VOLATILITY_FORECAST | 🟢 PASSED | 20.0 | validate_fixed_zxm_volatility_forecast.py | 1.80s | 错误: 2025-09-07 15:59:44 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| DOJI | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 1.80s | 错误: 2025-09-07 15:59:46 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| HAMMER | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 1.74s | 错误: 2025-09-07 15:59:47 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| SHOOTING_STAR | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 1.73s | 错误: 2025-09-07 15:59:49 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ENGULFING | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 1.84s | 错误: 2025-09-07 15:59:51 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| HARAMI | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 1.84s | 错误: 2025-09-07 15:59:53 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| PIERCING_LINE | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 1.85s | 错误: 2025-09-07 15:59:55 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| DARK_CLOUD_COVER | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 1.77s | 错误: 2025-09-07 15:59:56 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| MORNING_STAR | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 1.84s | 错误: 2025-09-07 15:59:58 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| EVENING_STAR | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 1.88s | 错误: 2025-09-07 16:00:00 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| THREE_BLACK_CROWS | 🟢 PASSED | 20.0 | validate_three_black_crows_complete.py | 2.27s | 错误: 2025-09-07 16:00:02 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| THREE_WHITE_SOLDIERS | 🟢 PASSED | 20.0 | validate_three_white_soldiers_complete.py | 2.30s | 错误: 2025-09-07 16:00:04 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| V_SHAPED_REVERSAL | 🟢 PASSED | 20.0 | validate_v_shaped_reversal_complete.py | 2.17s | 错误: 2025-09-07 16:00:07 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| HEAD_SHOULDERS | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 1.98s | 错误: 2025-09-07 16:00:09 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| DOUBLE_TOP | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.00s | 错误: 2025-09-07 16:00:11 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| DOUBLE_BOTTOM | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.75s | 错误: 2025-09-07 16:00:13 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| TRIANGLE | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.32s | 错误: 2025-09-07 16:00:16 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| WEDGE | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.06s | 错误: 2025-09-07 16:00:18 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| FLAG | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.18s | 错误: 2025-09-07 16:00:20 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| PENNANT | 🟢 PASSED | 20.0 | validate_pennant_complete.py | 2.33s | 错误: 2025-09-07 16:00:22 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| RECTANGLE | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.29s | 错误: 2025-09-07 16:00:25 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| CUP_AND_HANDLE | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.14s | 错误: 2025-09-07 16:00:27 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ISLAND_REVERSAL | 🟢 PASSED | 20.0 | validate_island_reversal_complete.py | 2.51s | 错误: 2025-09-07 16:00:29 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| CANDLESTICK_PATTERNS | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.11s | 错误: 2025-09-07 16:00:31 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| COMPOSITE | 🟢 PASSED | 20.0 | validate_composite_indicator.py | 2.43s | 错误: 2025-09-07 16:00:34 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| KC | 🟢 PASSED | 0 | validate_kc_indicator.py | 0.07s | - |
| ZXM_DAILY_MACD | ❓ ✅ PASSED_ARCHITECTURE_COMPLIANT | 97.0 | validate_zxm_daily_macd.py | 0.21s | - |
| ZXM_WASHPLATE | ❓ ✅ PASSED_ARCHITECTURE_COMPLIANT | 95.0 | validate_zxm_washplate.py | 29.28s | - |
| ZXM_BS_ABSORB | ❓ 🎉 PASSED_PRODUCTION_READY | 100.0 | validate_zxm_bs_absorb.py | 0.29s | - |

## ⚠️ 需要关注的指标

### 🔴 失败指标

- **DMI**: 分数 0
  - 验证脚本: validate_adx_indicator.py


## 📈 建议

### 质量改进建议
1. 对于失败指标，检查对应的验证脚本是否需要更新
2. 对于错误指标，确认验证脚本的可执行性和依赖关系
3. 定期运行此统一监控脚本，确保指标质量稳定

### 验证脚本维护建议
1. 保持验证脚本的接口一致性
2. 确保验证脚本的独立性和可重复性
3. 定期更新验证脚本以适应新的质量标准

---
**报告生成时间**: 2025-09-07 16:00:37
**测试方式**: 统一调用现有验证脚本
