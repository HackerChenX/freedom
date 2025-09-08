# 统一指标质量监控报告

## 📊 测试概要

- **测试时间**: 2025-09-07T14:08:07.132711
- **测试指标数**: 100
- **通过指标数**: 97
- **失败指标数**: 0
- **错误指标数**: 3
- **执行时间**: 145.78 秒
- **通过率**: 97.0%

## 🎯 测试方法说明

本次测试采用**统一调用现有验证脚本**的方式，确保测试方式的一致性：

1. **复用现有验证脚本**: 调用每个指标专门的验证脚本
2. **保持测试标准一致**: 使用与之前验证相同的测试方法
3. **标准化结果格式**: 统一处理不同脚本的输出格式
4. **完整错误处理**: 处理脚本执行中的各种异常情况

## 📋 详细结果

| 指标名称 | 状态 | 分数 | 验证脚本 | 执行时间 | 备注 |
|---------|------|------|----------|----------|------|
| MACD | 🟢 PASSED | 100.0 | validate_enhanced_macd_trend.py | 1.05s | - |
| RSI | 🟢 PASSED | 100.0 | validate_rsi_derivatives.py | 0.10s | - |
| BOLL | 🟢 PASSED | 100.0 | validate_enhanced_boll_indicators.py | 0.05s | - |
| KDJ | 🟢 PASSED | 100.0 | validate_enhanced_kdj.py | 0.04s | - |
| WR | 🟢 PASSED | 100.0 | validate_enhanced_wr.py | 1.51s | - |
| VOL | 🟢 PASSED | 100.0 | validate_vol.py | 0.08s | - |
| VR | 🟢 PASSED | 100.0 | validate_volume_score.py | 0.05s | - |
| AROON | 🟢 PASSED | 100.0 | validate_trend_indicators.py | 0.00s | - |
| DMA | 🟢 PASSED | 100.0 | validate_trend_indicators.py | 0.00s | - |
| MACD_SCORE | 🟢 PASSED | 100.0 | validate_score_indicators.py | 0.00s | - |
| RSI_SCORE | 🟢 PASSED | 100.0 | validate_score_indicators.py | 0.00s | - |
| BOLL_SCORE | 🟢 PASSED | 100.0 | validate_score_indicators.py | 0.01s | - |
| KDJ_SCORE | 🟢 PASSED | 100.0 | validate_score_indicators.py | 0.00s | - |
| ENHANCED_MACD | 🟢 PASSED | 100.0 | validate_enhanced_macd_trend.py | 0.20s | - |
| ENHANCED_BOLL | 🟢 PASSED | 100.0 | validate_enhanced_boll_indicators.py | 0.04s | - |
| ENHANCED_KDJ | 🟢 PASSED | 100.0 | validate_enhanced_kdj.py | 0.04s | - |
| FIBONACCI_TOOLS | 🟢 PASSED | 100.0 | validate_fibonacci_tools.py | 0.21s | - |
| MARKET_ENV | 🟢 PASSED | 100.0 | validate_market_env.py | 0.03s | - |
| SENTIMENT_ANALYSIS | 🟢 PASSED | 100.0 | validate_sentiment_analysis.py | 0.03s | - |
| TREND_CLASSIFICATION | 🟢 PASSED | 100.0 | validate_trend_classification.py | 0.05s | - |
| TREND_STRENGTH | 🟢 PASSED | 100.0 | validate_trend_strength.py | 0.11s | - |
| TIME_CYCLE_ANALYSIS | 🟢 PASSED | 100.0 | validate_time_cycle_analysis.py | 0.03s | - |
| MULTI_PERIOD_RESONANCE | 🟢 PASSED | 100.0 | validate_multi_period_resonance.py | 0.04s | - |
| INTRADAY_VOLATILITY | 🟢 PASSED | 100.0 | validate_intraday_volatility.py | 0.03s | - |
| STOCK_VIX | 🟢 PASSED | 100.0 | validate_stock_vix.py | 0.08s | - |
| ALPHA_GENERATION | 🟢 PASSED | 100.0 | validate_alpha_generation_indicators.py | 0.00s | - |
| BETA_HEDGING | 🟢 PASSED | 100.0 | validate_beta_hedging_indicators.py | 0.00s | - |
| ADX | 🟢 PASSED | 20.0 | validate_fixed_adx_indicator.py | 1.57s | 错误: 2025-09-07 14:08:10 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ROC | 🟢 PASSED | 20.0 | validate_fixed_roc_indicator.py | 1.50s | 错误: 2025-09-07 14:08:12 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| OBV | 🟢 PASSED | 20.0 | validate_fixed_obv_indicator.py | 1.68s | 错误: 2025-09-07 14:08:13 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| MTM | 🟢 PASSED | 20.0 | validate_fixed_mtm_indicator.py | 1.58s | 错误: 2025-09-07 14:08:15 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| MFI | 🟢 PASSED | 20.0 | validate_fixed_mfi_indicator.py | 1.71s | 错误: 2025-09-07 14:08:17 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| VIX | 🟢 PASSED | 20.0 | validate_fixed_baseindicators.py | 1.72s | 错误: 2025-09-07 14:08:18 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ATR | 🟢 PASSED | 20.0 | validate_atr_indicator.py | 1.50s | 错误: 2025-09-07 14:08:20 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| PSY | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 1.38s | 错误: 2025-09-07 14:08:22 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_WEEKLY_MACD | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 2.02s | 错误: 2025-09-07 14:08:23 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_MONTHLY_MACD | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 2.40s | 错误: 2025-09-07 14:08:25 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_TURNOVER | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 2.29s | 错误: 2025-09-07 14:08:28 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_VOLUME_SHRINK | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 2.15s | 错误: 2025-09-07 14:08:30 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_MA_CALLBACK | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 2.07s | 错误: 2025-09-07 14:08:33 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_DAILY_TREND_UP | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 2.28s | 错误: 2025-09-07 14:08:35 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_WEEKLY_TREND_UP | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 2.25s | 错误: 2025-09-07 14:08:37 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_MONTHLY_KDJ_TREND_UP | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 2.16s | 错误: 2025-09-07 14:08:40 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_AMPLITUDE_ELASTICITY | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 2.02s | 错误: 2025-09-07 14:08:42 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_RISE_ELASTICITY | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 2.03s | 错误: 2025-09-07 14:08:44 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_ELASTICITY | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 2.23s | 错误: 2025-09-07 14:08:46 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_BOUNCE_DETECTOR | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 2.01s | 错误: 2025-09-07 14:08:48 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_BUYPOINT_SCORE | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 2.01s | 错误: 2025-09-07 14:08:51 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_TREND_SCORE | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 2.00s | 错误: 2025-09-07 14:08:53 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_ELASTIC_SCORE | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 2.00s | 错误: 2025-09-07 14:08:55 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_VOLUME_ENERGY | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 2.07s | 错误: 2025-09-07 14:08:57 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_PRICE_POSITION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 2.08s | 错误: 2025-09-07 14:08:59 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_TECHNICAL_FORM | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 2.15s | 错误: 2025-09-07 14:09:01 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_HOT_SPOT | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 2.06s | 错误: 2025-09-07 14:09:03 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_INDUSTRY_ROTATION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 2.42s | 错误: 2025-09-07 14:09:06 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_CYCLE_POSITION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 2.43s | 错误: 2025-09-07 14:09:08 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_RISK_CONTROL | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 2.06s | 错误: 2025-09-07 14:09:11 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_TIMING_SIGNAL | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 2.04s | 错误: 2025-09-07 14:09:13 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_POSITION_MANAGEMENT | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 2.21s | 错误: 2025-09-07 14:09:15 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_PORTFOLIO_OPTIMIZATION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 2.15s | 错误: 2025-09-07 14:09:17 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_STRATEGY_COMBINATION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 2.14s | 错误: 2025-09-07 14:09:20 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_PERFORMANCE_ATTRIBUTION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 2.00s | 错误: 2025-09-07 14:09:22 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_ALPHA_GENERATION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 2.16s | 错误: 2025-09-07 14:09:24 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_BETA_HEDGING | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 2.34s | 错误: 2025-09-07 14:09:26 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_CHIP_DISTRIBUTION | 🟢 PASSED | 20.0 | validate_zxm_chip_distribution_complete.py | 1.43s | 错误: 2025-09-07 14:09:43 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_FUND_FLOW | 🟢 PASSED | 20.0 | validate_zxm_fund_flow_complete.py | 1.64s | 错误: 2025-09-07 14:09:44 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_INSTITUTION_BEHAVIOR | 🟢 PASSED | 20.0 | validate_zxm_institution_behavior_complete.py | 1.59s | 错误: 2025-09-07 14:09:46 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_MARKET_SENTIMENT | 🟢 PASSED | 20.0 | validate_fixed_zxm_market_sentiment.py | 1.72s | 错误: 2025-09-07 14:09:47 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_LIQUIDITY_ANALYSIS | 🟢 PASSED | 20.0 | validate_fixed_zxm_liquidity_analysis.py | 1.57s | 错误: 2025-09-07 14:09:49 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_CORRELATION_MATRIX | 🟢 PASSED | 20.0 | validate_fixed_zxm_correlation_matrix.py | 2.18s | 错误: 2025-09-07 14:09:51 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_VOLATILITY_FORECAST | 🟢 PASSED | 20.0 | validate_fixed_zxm_volatility_forecast.py | 1.37s | 错误: 2025-09-07 14:09:53 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| DOJI | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 1.41s | 错误: 2025-09-07 14:09:54 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| HAMMER | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 1.66s | 错误: 2025-09-07 14:09:56 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| SHOOTING_STAR | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 1.61s | 错误: 2025-09-07 14:09:58 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ENGULFING | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 1.52s | 错误: 2025-09-07 14:09:59 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| HARAMI | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 1.53s | 错误: 2025-09-07 14:10:01 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| PIERCING_LINE | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 1.47s | 错误: 2025-09-07 14:10:02 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| DARK_CLOUD_COVER | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 1.48s | 错误: 2025-09-07 14:10:04 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| MORNING_STAR | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 1.48s | 错误: 2025-09-07 14:10:05 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| EVENING_STAR | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 1.54s | 错误: 2025-09-07 14:10:07 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| THREE_BLACK_CROWS | 🟢 PASSED | 20.0 | validate_three_black_crows_complete.py | 1.88s | 错误: 2025-09-07 14:10:08 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| THREE_WHITE_SOLDIERS | 🟢 PASSED | 20.0 | validate_three_white_soldiers_complete.py | 1.76s | 错误: 2025-09-07 14:10:10 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| V_SHAPED_REVERSAL | 🟢 PASSED | 20.0 | validate_v_shaped_reversal_complete.py | 1.51s | 错误: 2025-09-07 14:10:12 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| HEAD_SHOULDERS | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 1.40s | 错误: 2025-09-07 14:10:13 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| DOUBLE_TOP | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 1.49s | 错误: 2025-09-07 14:10:15 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| DOUBLE_BOTTOM | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 1.61s | 错误: 2025-09-07 14:10:16 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| TRIANGLE | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 1.63s | 错误: 2025-09-07 14:10:18 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| WEDGE | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 1.53s | 错误: 2025-09-07 14:10:19 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| FLAG | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 1.52s | 错误: 2025-09-07 14:10:21 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| PENNANT | 🟢 PASSED | 20.0 | validate_pennant_complete.py | 1.48s | 错误: 2025-09-07 14:10:23 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| RECTANGLE | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 1.42s | 错误: 2025-09-07 14:10:24 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| CUP_AND_HANDLE | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 1.51s | 错误: 2025-09-07 14:10:25 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ISLAND_REVERSAL | 🟢 PASSED | 20.0 | validate_island_reversal_complete.py | 1.82s | 错误: 2025-09-07 14:10:27 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| CANDLESTICK_PATTERNS | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 1.53s | 错误: 2025-09-07 14:10:29 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| COMPOSITE | 🟢 PASSED | 20.0 | validate_composite_indicator.py | 1.60s | 错误: 2025-09-07 14:10:31 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| DMI | 🟢 PASSED | 0 | validate_adx_indicator.py | 0.36s | - |
| KC | 🟢 PASSED | 0 | validate_kc_indicator.py | 0.04s | - |
| ZXM_WASHPLATE | ❓ ✅ PASSED_ARCHITECTURE_COMPLIANT | 98.0 | validate_zxm_washplate.py | 14.36s | - |
| ZXM_DAILY_MACD | ❓ ✅ PASSED_ARCHITECTURE_COMPLIANT | 97.0 | validate_zxm_daily_macd.py | 0.16s | - |
| ZXM_BS_ABSORB | ❓ 🎉 PASSED_PRODUCTION_READY | 100.0 | validate_zxm_bs_absorb.py | 0.18s | - |

## ⚠️ 需要关注的指标


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
**报告生成时间**: 2025-09-07 14:10:32
**测试方式**: 统一调用现有验证脚本
