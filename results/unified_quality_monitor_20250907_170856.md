# 统一指标质量监控报告

## 📊 测试概要

- **测试时间**: 2025-09-07T17:02:42.262206
- **测试指标数**: 132
- **通过指标数**: 102
- **失败指标数**: 27
- **错误指标数**: 3
- **执行时间**: 374.61 秒
- **通过率**: 77.3%

## 🎯 测试方法说明

本次测试采用**统一调用现有验证脚本**的方式，确保测试方式的一致性：

1. **复用现有验证脚本**: 调用每个指标专门的验证脚本
2. **保持测试标准一致**: 使用与之前验证相同的测试方法
3. **标准化结果格式**: 统一处理不同脚本的输出格式
4. **完整错误处理**: 处理脚本执行中的各种异常情况

## 📋 详细结果

| 指标名称 | 状态 | 分数 | 验证脚本 | 执行时间 | 备注 |
|---------|------|------|----------|----------|------|
| EMA | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| WMA | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| CCI | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.00s | - |
| BIAS | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.11s | - |
| EMV | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.00s | - |
| AD | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| FORCE_INDEX | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| CHAIKIN | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.00s | - |
| PVT | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.00s | - |
| VORTEX | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.00s | - |
| VOSC | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.00s | - |
| STDDEV | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.03s | - |
| VOLATILITY | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| CHAIKIN_VOLATILITY | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.00s | - |
| GARMAN_KLASS | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.00s | - |
| CMO | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| ULTIMATE | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.00s | - |
| RSIMA | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.00s | - |
| PSAR | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.03s | - |
| SUPERTREND | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| ELLIOTT_WAVE | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| ENHANCED_CCI | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| ENHANCED_TRIX | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| TRIX | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.00s | - |
| SAR | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.00s | - |
| MOMENTUM | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.00s | - |
| ROC_OSCILLATOR | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.00s | - |
| MACD | 🟢 PASSED | 100.0 | validate_enhanced_macd_trend.py | 1.17s | - |
| RSI | 🟢 PASSED | 100.0 | validate_rsi_derivatives.py | 0.43s | - |
| BOLL | 🟢 PASSED | 100.0 | validate_enhanced_boll_indicators.py | 0.21s | - |
| KDJ | 🟢 PASSED | 100.0 | validate_enhanced_kdj.py | 0.16s | - |
| WR | 🟢 PASSED | 100.0 | validate_enhanced_wr.py | 5.97s | - |
| VOL | 🟢 PASSED | 100.0 | validate_vol.py | 0.20s | - |
| ADX | 🟢 PASSED | 100.0 | validate_fixed_adx_indicator.py | 25.83s | 错误: 2025-09-07 17:03:14 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分100.0分 |
| ROC | 🟢 PASSED | 100.0 | validate_fixed_roc_indicator.py | 9.49s | 错误: 2025-09-07 17:03:31 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分100.0分 |
| OBV | 🟢 PASSED | 100.0 | validate_fixed_obv_indicator.py | 18.49s | 错误: 2025-09-07 17:03:40 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分100.0分 |
| MTM | 🟢 PASSED | 100.0 | validate_fixed_mtm_indicator.py | 8.15s | 错误: 2025-09-07 17:03:58 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分100.0分 |
| MFI | 🟢 PASSED | 100.0 | validate_fixed_mfi_indicator.py | 4.32s | 错误: 2025-09-07 17:04:05 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分100.0分 |
| VIX | 🟢 PASSED | 100.0 | validate_fixed_baseindicators.py | 4.15s | 错误: 2025-09-07 17:04:09 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分100.0分 |
| VR | 🟢 PASSED | 100.0 | validate_volume_score.py | 0.10s | - |
| ATR | 🟢 PASSED | 100.0 | validate_atr_indicator.py | 3.64s | 错误: 2025-09-07 17:04:13 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分100.0分 |
| AROON | 🟢 PASSED | 100.0 | validate_trend_indicators.py | 0.01s | - |
| DMA | 🟢 PASSED | 100.0 | validate_trend_indicators.py | 0.03s | - |
| ZXM_MARKET_SENTIMENT | 🟢 PASSED | 100.0 | validate_fixed_zxm_market_sentiment.py | 2.32s | 错误: 2025-09-07 17:07:32 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分100.0分 |
| ZXM_LIQUIDITY_ANALYSIS | 🟢 PASSED | 100.0 | validate_fixed_zxm_liquidity_analysis.py | 2.82s | 错误: 2025-09-07 17:07:34 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分100.0分 |
| ZXM_CORRELATION_MATRIX | 🟢 PASSED | 100.0 | validate_fixed_zxm_correlation_matrix.py | 5.15s | 错误: 2025-09-07 17:07:37 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分100.0分 |
| ZXM_VOLATILITY_FORECAST | 🟢 PASSED | 100.0 | validate_fixed_zxm_volatility_forecast.py | 4.53s | 错误: 2025-09-07 17:07:43 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分100.0分 |
| THREE_BLACK_CROWS | 🟢 PASSED | 100.0 | validate_three_black_crows_complete.py | 2.60s | 错误: 2025-09-07 17:08:16 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分100.0分 |
| THREE_WHITE_SOLDIERS | 🟢 PASSED | 100.0 | validate_three_white_soldiers_complete.py | 3.08s | 错误: 2025-09-07 17:08:19 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分100.0分 |
| MACD_SCORE | 🟢 PASSED | 100.0 | validate_score_indicators.py | 0.02s | - |
| RSI_SCORE | 🟢 PASSED | 100.0 | validate_score_indicators.py | 0.00s | - |
| BOLL_SCORE | 🟢 PASSED | 100.0 | validate_score_indicators.py | 0.00s | - |
| KDJ_SCORE | 🟢 PASSED | 100.0 | validate_score_indicators.py | 0.01s | - |
| ENHANCED_MACD | 🟢 PASSED | 100.0 | validate_enhanced_macd_trend.py | 0.49s | - |
| ENHANCED_BOLL | 🟢 PASSED | 100.0 | validate_enhanced_boll_indicators.py | 0.10s | - |
| ENHANCED_KDJ | 🟢 PASSED | 100.0 | validate_enhanced_kdj.py | 0.11s | - |
| COMPOSITE | 🟢 PASSED | 100.0 | validate_composite_indicator.py | 2.38s | 错误: 2025-09-07 17:08:54 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分100.0分 |
| FIBONACCI_TOOLS | 🟢 PASSED | 100.0 | validate_fibonacci_tools.py | 0.28s | - |
| MARKET_ENV | 🟢 PASSED | 100.0 | validate_market_env.py | 0.04s | - |
| SENTIMENT_ANALYSIS | 🟢 PASSED | 100.0 | validate_sentiment_analysis.py | 0.04s | - |
| TREND_CLASSIFICATION | 🟢 PASSED | 100.0 | validate_trend_classification.py | 0.05s | - |
| TREND_STRENGTH | 🟢 PASSED | 100.0 | validate_trend_strength.py | 0.23s | - |
| TIME_CYCLE_ANALYSIS | 🟢 PASSED | 100.0 | validate_time_cycle_analysis.py | 0.05s | - |
| MULTI_PERIOD_RESONANCE | 🟢 PASSED | 100.0 | validate_multi_period_resonance.py | 0.07s | - |
| INTRADAY_VOLATILITY | 🟢 PASSED | 100.0 | validate_intraday_volatility.py | 0.05s | - |
| STOCK_VIX | 🟢 PASSED | 100.0 | validate_stock_vix.py | 0.13s | - |
| ALPHA_GENERATION | 🟢 PASSED | 100.0 | validate_alpha_generation_indicators.py | 0.00s | - |
| BETA_HEDGING | 🟢 PASSED | 100.0 | validate_beta_hedging_indicators.py | 0.00s | - |
| ZXM_CHIP_DISTRIBUTION | 🟢 PASSED | 95.0 | validate_zxm_chip_distribution_complete.py | 2.45s | 错误: 2025-09-07 17:07:24 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分95.0分 |
| ZXM_FUND_FLOW | 🟢 PASSED | 95.0 | validate_zxm_fund_flow_complete.py | 2.49s | 错误: 2025-09-07 17:07:27 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分95.0分 |
| ZXM_INSTITUTION_BEHAVIOR | 🟢 PASSED | 95.0 | validate_zxm_institution_behavior_complete.py | 2.24s | 错误: 2025-09-07 17:07:29 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分95.0分 |
| V_SHAPED_REVERSAL | 🟢 PASSED | 95.0 | validate_v_shaped_reversal_complete.py | 2.66s | 错误: 2025-09-07 17:08:22 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分95.0分 |
| PENNANT | 🟢 PASSED | 95.0 | validate_pennant_complete.py | 2.57s | 错误: 2025-09-07 17:08:40 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分95.0分 |
| ISLAND_REVERSAL | 🟢 PASSED | 95.0 | validate_island_reversal_complete.py | 2.31s | 错误: 2025-09-07 17:08:48 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分95.0分 |
| PSY | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 7.92s | 错误: 2025-09-07 17:04:20 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| DOJI | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 6.04s | 错误: 2025-09-07 17:07:48 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| HAMMER | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 3.51s | 错误: 2025-09-07 17:07:53 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| SHOOTING_STAR | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 3.31s | 错误: 2025-09-07 17:07:57 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| ENGULFING | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 2.56s | 错误: 2025-09-07 17:07:59 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| HARAMI | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 4.41s | 错误: 2025-09-07 17:08:02 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| PIERCING_LINE | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 2.95s | 错误: 2025-09-07 17:08:07 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| DARK_CLOUD_COVER | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 2.24s | 错误: 2025-09-07 17:08:09 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| MORNING_STAR | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 2.34s | 错误: 2025-09-07 17:08:11 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| EVENING_STAR | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 2.27s | 错误: 2025-09-07 17:08:14 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| HEAD_SHOULDERS | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 2.14s | 错误: 2025-09-07 17:08:24 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| DOUBLE_TOP | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 2.58s | 错误: 2025-09-07 17:08:26 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| DOUBLE_BOTTOM | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 2.94s | 错误: 2025-09-07 17:08:29 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| TRIANGLE | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 2.48s | 错误: 2025-09-07 17:08:32 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| WEDGE | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 3.56s | 错误: 2025-09-07 17:08:35 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| FLAG | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 2.08s | 错误: 2025-09-07 17:08:38 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| RECTANGLE | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 2.91s | 错误: 2025-09-07 17:08:43 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| CUP_AND_HANDLE | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 2.68s | 错误: 2025-09-07 17:08:46 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| CANDLESTICK_PATTERNS | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 2.29s | 错误: 2025-09-07 17:08:51 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| ZXM_WEEKLY_MACD | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 6.19s | 错误: 2025-09-07 17:04:25 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_MONTHLY_MACD | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 13.24s | 错误: 2025-09-07 17:04:31 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_TURNOVER | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 6.95s | 错误: 2025-09-07 17:04:46 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_VOLUME_SHRINK | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 8.85s | 错误: 2025-09-07 17:04:53 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_MA_CALLBACK | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 5.88s | 错误: 2025-09-07 17:05:01 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_DAILY_TREND_UP | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 5.23s | 错误: 2025-09-07 17:05:07 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_WEEKLY_TREND_UP | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 3.94s | 错误: 2025-09-07 17:05:12 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_MONTHLY_KDJ_TREND_UP | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 3.65s | 错误: 2025-09-07 17:05:16 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_AMPLITUDE_ELASTICITY | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 3.58s | 错误: 2025-09-07 17:05:20 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_RISE_ELASTICITY | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 6.13s | 错误: 2025-09-07 17:05:24 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_ELASTICITY | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 4.09s | 错误: 2025-09-07 17:05:30 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_BOUNCE_DETECTOR | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 3.69s | 错误: 2025-09-07 17:05:34 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_BUYPOINT_SCORE | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 3.97s | 错误: 2025-09-07 17:05:37 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_TREND_SCORE | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 3.19s | 错误: 2025-09-07 17:05:41 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_ELASTIC_SCORE | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 2.95s | 错误: 2025-09-07 17:05:44 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_VOLUME_ENERGY | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 3.16s | 错误: 2025-09-07 17:05:47 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_PRICE_POSITION | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 3.20s | 错误: 2025-09-07 17:05:51 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_TECHNICAL_FORM | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 3.09s | 错误: 2025-09-07 17:05:54 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_HOT_SPOT | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 2.99s | 错误: 2025-09-07 17:05:57 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_INDUSTRY_ROTATION | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 3.22s | 错误: 2025-09-07 17:06:00 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_CYCLE_POSITION | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 3.38s | 错误: 2025-09-07 17:06:03 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_RISK_CONTROL | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 4.57s | 错误: 2025-09-07 17:06:07 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_TIMING_SIGNAL | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 8.75s | 错误: 2025-09-07 17:06:13 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_POSITION_MANAGEMENT | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 5.96s | 错误: 2025-09-07 17:06:21 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_PORTFOLIO_OPTIMIZATION | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 4.09s | 错误: 2025-09-07 17:06:26 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_STRATEGY_COMBINATION | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 4.63s | 错误: 2025-09-07 17:06:30 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_PERFORMANCE_ATTRIBUTION | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 4.25s | 错误: 2025-09-07 17:06:34 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_ALPHA_GENERATION | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 3.71s | 错误: 2025-09-07 17:06:39 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_BETA_HEDGING | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 3.85s | 错误: 2025-09-07 17:06:43 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| MA | 🟢 PASSED | 0 | validate_unified_ma_indicator_strict.py | 6.95s | - |
| DMI | 🟢 PASSED | 0 | validate_adx_indicator.py | 3.96s | - |
| KC | 🟢 PASSED | 0 | validate_kc_indicator.py | 0.08s | - |
| SYNERGY | 🟢 PASSED | 0 | validate_synergy_indicator_strict.py | 0.04s | - |
| UNIFIED_MA | 🟢 PASSED | 0 | validate_unified_ma_indicator_strict.py | 0.03s | - |
| STOCHRSI | 🟢 PASSED | 0 | validate_enhanced_stochrsi.py | 0.05s | - |
| ENHANCED_STOCHRSI | 🟢 PASSED | 0 | validate_enhanced_stochrsi.py | 0.05s | - |
| ZXM_DAILY_MACD | ❓ ✅ PASSED_ARCHITECTURE_COMPLIANT | 97.0 | validate_zxm_daily_macd.py | 0.68s | - |
| ZXM_WASHPLATE | ❓ ✅ PASSED_ARCHITECTURE_COMPLIANT | 95.0 | validate_zxm_washplate.py | 38.10s | - |
| ZXM_BS_ABSORB | ❓ 🎉 PASSED_PRODUCTION_READY | 100.0 | validate_zxm_bs_absorb.py | 0.73s | - |

## ⚠️ 需要关注的指标

### 🔴 失败指标

- **EMA**: 分数 0
  - 验证脚本: validate_enhanced_indicators.py
- **WMA**: 分数 0
  - 验证脚本: validate_enhanced_indicators.py
- **CCI**: 分数 0
  - 验证脚本: validate_enhanced_indicators.py
- **BIAS**: 分数 0
  - 验证脚本: validate_enhanced_indicators.py
- **EMV**: 分数 0
  - 验证脚本: validate_enhanced_indicators.py
- **AD**: 分数 0
  - 验证脚本: validate_enhanced_indicators.py
- **FORCE_INDEX**: 分数 0
  - 验证脚本: validate_enhanced_indicators.py
- **CHAIKIN**: 分数 0
  - 验证脚本: validate_enhanced_indicators.py
- **PVT**: 分数 0
  - 验证脚本: validate_enhanced_indicators.py
- **VORTEX**: 分数 0
  - 验证脚本: validate_enhanced_indicators.py
- **VOSC**: 分数 0
  - 验证脚本: validate_enhanced_indicators.py
- **STDDEV**: 分数 0
  - 验证脚本: validate_enhanced_indicators.py
- **VOLATILITY**: 分数 0
  - 验证脚本: validate_enhanced_indicators.py
- **CHAIKIN_VOLATILITY**: 分数 0
  - 验证脚本: validate_enhanced_indicators.py
- **GARMAN_KLASS**: 分数 0
  - 验证脚本: validate_enhanced_indicators.py
- **CMO**: 分数 0
  - 验证脚本: validate_enhanced_indicators.py
- **ULTIMATE**: 分数 0
  - 验证脚本: validate_enhanced_indicators.py
- **RSIMA**: 分数 0
  - 验证脚本: validate_enhanced_indicators.py
- **PSAR**: 分数 0
  - 验证脚本: validate_enhanced_indicators.py
- **SUPERTREND**: 分数 0
  - 验证脚本: validate_enhanced_indicators.py
- **ELLIOTT_WAVE**: 分数 0
  - 验证脚本: validate_enhanced_indicators.py
- **ENHANCED_CCI**: 分数 0
  - 验证脚本: validate_enhanced_indicators.py
- **ENHANCED_TRIX**: 分数 0
  - 验证脚本: validate_enhanced_indicators.py
- **TRIX**: 分数 0
  - 验证脚本: validate_enhanced_indicators.py
- **SAR**: 分数 0
  - 验证脚本: validate_enhanced_indicators.py
- **MOMENTUM**: 分数 0
  - 验证脚本: validate_enhanced_indicators.py
- **ROC_OSCILLATOR**: 分数 0
  - 验证脚本: validate_enhanced_indicators.py


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
**报告生成时间**: 2025-09-07 17:08:56
**测试方式**: 统一调用现有验证脚本
