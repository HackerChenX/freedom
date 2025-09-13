# 统一指标质量监控报告

## 📊 测试概要

- **测试时间**: 2025-09-13T15:16:45.973578
- **测试指标数**: 132
- **通过指标数**: 129
- **失败指标数**: 0
- **错误指标数**: 3
- **执行时间**: 298.97 秒
- **通过率**: 97.7%

## 🎯 测试方法说明

本次测试采用**统一调用现有验证脚本**的方式，确保测试方式的一致性：

1. **复用现有验证脚本**: 调用每个指标专门的验证脚本
2. **保持测试标准一致**: 使用与之前验证相同的测试方法
3. **标准化结果格式**: 统一处理不同脚本的输出格式
4. **完整错误处理**: 处理脚本执行中的各种异常情况

## 📋 详细结果

| 指标名称 | 状态 | 分数 | 验证脚本 | 执行时间 | 备注 |
|---------|------|------|----------|----------|------|
| MA | 🟢 PASSED | 100 | validate_unified_ma_indicator_strict.py | 2.07s | - |
| EMA | 🟢 PASSED | 100 | validate_enhanced_indicators.py | 0.01s | - |
| WMA | 🟢 PASSED | 100 | validate_enhanced_indicators.py | 0.02s | - |
| MACD | 🟢 PASSED | 100.0 | validate_enhanced_macd_trend.py | 0.30s | - |
| RSI | 🟢 PASSED | 100.0 | validate_rsi_derivatives.py | 0.13s | - |
| BOLL | 🟢 PASSED | 100.0 | validate_enhanced_boll_indicators.py | 0.07s | - |
| KDJ | 🟢 PASSED | 100.0 | validate_enhanced_kdj.py | 0.07s | - |
| WR | 🟢 PASSED | 100.0 | validate_enhanced_wr.py | 3.56s | - |
| CCI | 🟢 PASSED | 100 | validate_enhanced_indicators.py | 0.04s | - |
| VOL | 🟢 PASSED | 100.0 | validate_vol.py | 0.13s | - |
| BIAS | 🟢 PASSED | 100 | validate_enhanced_indicators.py | 0.02s | - |
| ADX | 🟢 PASSED | 100.0 | validate_fixed_adx_indicator.py | 2.47s | 错误: 2025-09-13 15:16:53 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分100.0分 |
| ROC | 🟢 PASSED | 100.0 | validate_fixed_roc_indicator.py | 1.90s | 错误: 2025-09-13 15:16:55 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分100.0分 |
| OBV | 🟢 PASSED | 100.0 | validate_fixed_obv_indicator.py | 1.99s | 错误: 2025-09-13 15:16:57 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分100.0分 |
| MTM | 🟢 PASSED | 100.0 | validate_fixed_mtm_indicator.py | 1.89s | 错误: 2025-09-13 15:16:59 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分100.0分 |
| MFI | 🟢 PASSED | 100.0 | validate_fixed_mfi_indicator.py | 2.00s | 错误: 2025-09-13 15:17:01 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分100.0分 |
| KC | 🟢 PASSED | 100.0 | validate_kc_indicator.py | 0.05s | - |
| VIX | 🟢 PASSED | 100.0 | validate_fixed_baseindicators.py | 1.92s | 错误: 2025-09-13 15:17:03 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分100.0分 |
| UNIFIED_MA | 🟢 PASSED | 100 | validate_unified_ma_indicator_strict.py | 0.01s | - |
| VR | 🟢 PASSED | 100.0 | validate_volume_score.py | 0.05s | - |
| EMV | 🟢 PASSED | 100 | validate_enhanced_indicators.py | 0.01s | - |
| AD | 🟢 PASSED | 100 | validate_enhanced_indicators.py | 0.01s | - |
| FORCE_INDEX | 🟢 PASSED | 100 | validate_enhanced_indicators.py | 0.01s | - |
| CHAIKIN | 🟢 PASSED | 100 | validate_enhanced_indicators.py | 0.01s | - |
| PVT | 🟢 PASSED | 100 | validate_enhanced_indicators.py | 0.01s | - |
| VORTEX | 🟢 PASSED | 100 | validate_enhanced_indicators.py | 0.02s | - |
| VOSC | 🟢 PASSED | 100 | validate_enhanced_indicators.py | 0.01s | - |
| ATR | 🟢 PASSED | 100.0 | validate_atr_indicator.py | 1.93s | 错误: 2025-09-13 15:17:05 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分100.0分 |
| STDDEV | 🟢 PASSED | 100 | validate_enhanced_indicators.py | 0.01s | - |
| VOLATILITY | 🟢 PASSED | 100 | validate_enhanced_indicators.py | 0.01s | - |
| CHAIKIN_VOLATILITY | 🟢 PASSED | 100 | validate_enhanced_indicators.py | 0.01s | - |
| GARMAN_KLASS | 🟢 PASSED | 100 | validate_enhanced_indicators.py | 0.01s | - |
| CMO | 🟢 PASSED | 100 | validate_enhanced_indicators.py | 0.01s | - |
| ULTIMATE | 🟢 PASSED | 100 | validate_enhanced_indicators.py | 0.02s | - |
| STOCHRSI | 🟢 PASSED | 100 | validate_enhanced_stochrsi.py | 0.02s | - |
| RSIMA | 🟢 PASSED | 100 | validate_enhanced_indicators.py | 0.01s | - |
| AROON | 🟢 PASSED | 100.0 | validate_trend_indicators.py | 0.00s | - |
| DMA | 🟢 PASSED | 100.0 | validate_trend_indicators.py | 0.00s | - |
| PSAR | 🟢 PASSED | 100 | validate_enhanced_indicators.py | 0.00s | - |
| SUPERTREND | 🟢 PASSED | 100 | validate_enhanced_indicators.py | 0.03s | - |
| ELLIOTT_WAVE | 🟢 PASSED | 100 | validate_enhanced_indicators.py | 0.08s | - |
| ZXM_MARKET_SENTIMENT | 🟢 PASSED | 100.0 | validate_fixed_zxm_market_sentiment.py | 2.75s | 错误: 2025-09-13 15:20:31 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分100.0分 |
| ZXM_LIQUIDITY_ANALYSIS | 🟢 PASSED | 100.0 | validate_fixed_zxm_liquidity_analysis.py | 2.72s | 错误: 2025-09-13 15:20:34 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分100.0分 |
| ZXM_CORRELATION_MATRIX | 🟢 PASSED | 100.0 | validate_fixed_zxm_correlation_matrix.py | 4.16s | 错误: 2025-09-13 15:20:37 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分100.0分 |
| ZXM_VOLATILITY_FORECAST | 🟢 PASSED | 100.0 | validate_fixed_zxm_volatility_forecast.py | 2.18s | 错误: 2025-09-13 15:20:41 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分100.0分 |
| THREE_BLACK_CROWS | 🟢 PASSED | 100.0 | validate_three_black_crows_complete.py | 3.11s | 错误: 2025-09-13 15:21:05 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分100.0分 |
| THREE_WHITE_SOLDIERS | 🟢 PASSED | 100.0 | validate_three_white_soldiers_complete.py | 3.09s | 错误: 2025-09-13 15:21:08 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分100.0分 |
| MACD_SCORE | 🟢 PASSED | 100.0 | validate_score_indicators.py | 0.01s | - |
| RSI_SCORE | 🟢 PASSED | 100.0 | validate_score_indicators.py | 0.00s | - |
| BOLL_SCORE | 🟢 PASSED | 100.0 | validate_score_indicators.py | 0.00s | - |
| KDJ_SCORE | 🟢 PASSED | 100.0 | validate_score_indicators.py | 0.00s | - |
| ENHANCED_MACD | 🟢 PASSED | 100.0 | validate_enhanced_macd_trend.py | 0.28s | - |
| ENHANCED_CCI | 🟢 PASSED | 100 | validate_enhanced_indicators.py | 0.03s | - |
| ENHANCED_STOCHRSI | 🟢 PASSED | 100 | validate_enhanced_stochrsi.py | 0.02s | - |
| ENHANCED_BOLL | 🟢 PASSED | 100.0 | validate_enhanced_boll_indicators.py | 0.06s | - |
| ENHANCED_KDJ | 🟢 PASSED | 100.0 | validate_enhanced_kdj.py | 0.06s | - |
| ENHANCED_TRIX | 🟢 PASSED | 100 | validate_enhanced_indicators.py | 0.04s | - |
| COMPOSITE | 🟢 PASSED | 100.0 | validate_composite_indicator.py | 2.34s | 错误: 2025-09-13 15:21:42 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分100.0分 |
| TRIX | 🟢 PASSED | 100 | validate_enhanced_indicators.py | 0.02s | - |
| SAR | 🟢 PASSED | 100 | validate_enhanced_indicators.py | 0.00s | - |
| MOMENTUM | 🟢 PASSED | 100 | validate_enhanced_indicators.py | 0.02s | - |
| ROC_OSCILLATOR | 🟢 PASSED | 100 | validate_enhanced_indicators.py | 0.02s | - |
| FIBONACCI_TOOLS | 🟢 PASSED | 100.0 | validate_fibonacci_tools.py | 0.33s | - |
| MARKET_ENV | 🟢 PASSED | 100.0 | validate_market_env.py | 0.04s | - |
| SENTIMENT_ANALYSIS | 🟢 PASSED | 100.0 | validate_sentiment_analysis.py | 0.04s | - |
| TREND_CLASSIFICATION | 🟢 PASSED | 100.0 | validate_trend_classification.py | 0.04s | - |
| TREND_STRENGTH | 🟢 PASSED | 100.0 | validate_trend_strength.py | 0.18s | - |
| TIME_CYCLE_ANALYSIS | 🟢 PASSED | 100.0 | validate_time_cycle_analysis.py | 0.04s | - |
| MULTI_PERIOD_RESONANCE | 🟢 PASSED | 100.0 | validate_multi_period_resonance.py | 0.07s | - |
| INTRADAY_VOLATILITY | 🟢 PASSED | 100.0 | validate_intraday_volatility.py | 0.06s | - |
| STOCK_VIX | 🟢 PASSED | 100.0 | validate_stock_vix.py | 0.14s | - |
| ALPHA_GENERATION | 🟢 PASSED | 100.0 | validate_alpha_generation_indicators.py | 0.00s | - |
| BETA_HEDGING | 🟢 PASSED | 100.0 | validate_beta_hedging_indicators.py | 0.00s | - |
| ZXM_CHIP_DISTRIBUTION | 🟢 PASSED | 95.0 | validate_zxm_chip_distribution_complete.py | 3.25s | 错误: 2025-09-13 15:20:23 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分95.0分 |
| ZXM_FUND_FLOW | 🟢 PASSED | 95.0 | validate_zxm_fund_flow_complete.py | 3.21s | 错误: 2025-09-13 15:20:25 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分95.0分 |
| ZXM_INSTITUTION_BEHAVIOR | 🟢 PASSED | 95.0 | validate_zxm_institution_behavior_complete.py | 2.48s | 错误: 2025-09-13 15:20:29 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分95.0分 |
| V_SHAPED_REVERSAL | 🟢 PASSED | 95.0 | validate_v_shaped_reversal_complete.py | 3.22s | 错误: 2025-09-13 15:21:12 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分95.0分 |
| PENNANT | 🟢 PASSED | 95.0 | validate_pennant_complete.py | 2.75s | 错误: 2025-09-13 15:21:30 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分95.0分 |
| ISLAND_REVERSAL | 🟢 PASSED | 95.0 | validate_island_reversal_complete.py | 2.49s | 错误: 2025-09-13 15:21:37 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分95.0分 |
| DMI | 🟢 PASSED | 90.0 | validate_adx_indicator.py | 0.46s | - |
| PSY | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 1.76s | 错误: 2025-09-13 15:17:07 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| DOJI | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 2.22s | 错误: 2025-09-13 15:20:43 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| HAMMER | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 2.38s | 错误: 2025-09-13 15:20:45 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| SHOOTING_STAR | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 2.35s | 错误: 2025-09-13 15:20:48 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| ENGULFING | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 2.20s | 错误: 2025-09-13 15:20:50 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| HARAMI | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 2.95s | 错误: 2025-09-13 15:20:52 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| PIERCING_LINE | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 2.61s | 错误: 2025-09-13 15:20:55 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| DARK_CLOUD_COVER | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 2.49s | 错误: 2025-09-13 15:20:58 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| MORNING_STAR | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 2.54s | 错误: 2025-09-13 15:21:00 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| EVENING_STAR | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 2.61s | 错误: 2025-09-13 15:21:03 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| HEAD_SHOULDERS | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 2.80s | 错误: 2025-09-13 15:21:15 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| DOUBLE_TOP | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 2.69s | 错误: 2025-09-13 15:21:18 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| DOUBLE_BOTTOM | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 2.36s | 错误: 2025-09-13 15:21:20 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| TRIANGLE | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 2.73s | 错误: 2025-09-13 15:21:23 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| WEDGE | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 2.19s | 错误: 2025-09-13 15:21:25 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| FLAG | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 2.37s | 错误: 2025-09-13 15:21:27 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| RECTANGLE | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 2.10s | 错误: 2025-09-13 15:21:33 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| CUP_AND_HANDLE | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 2.16s | 错误: 2025-09-13 15:21:35 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| CANDLESTICK_PATTERNS | 🟢 PASSED | 85.0 | validate_pattern_indicators_95.py | 1.98s | 错误: 2025-09-13 15:21:39 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分85.0分 |
| SYNERGY | 🟢 PASSED | 80 | validate_synergy_indicator_strict.py | 0.01s | - |
| ZXM_WEEKLY_MACD | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 2.29s | 错误: 2025-09-13 15:17:09 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_MONTHLY_MACD | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 2.24s | 错误: 2025-09-13 15:17:12 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_TURNOVER | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 2.45s | 错误: 2025-09-13 15:17:14 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_VOLUME_SHRINK | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 2.79s | 错误: 2025-09-13 15:17:17 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_MA_CALLBACK | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 2.57s | 错误: 2025-09-13 15:17:19 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_DAILY_TREND_UP | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 2.55s | 错误: 2025-09-13 15:17:22 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_WEEKLY_TREND_UP | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 2.45s | 错误: 2025-09-13 15:17:24 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_MONTHLY_KDJ_TREND_UP | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 2.48s | 错误: 2025-09-13 15:17:27 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_AMPLITUDE_ELASTICITY | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 2.56s | 错误: 2025-09-13 15:17:29 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_RISE_ELASTICITY | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 3.01s | 错误: 2025-09-13 15:17:32 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_ELASTICITY | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 4.44s | 错误: 2025-09-13 15:17:35 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_BOUNCE_DETECTOR | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 3.60s | 错误: 2025-09-13 15:17:40 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_BUYPOINT_SCORE | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 3.31s | 错误: 2025-09-13 15:17:43 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_TREND_SCORE | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 3.65s | 错误: 2025-09-13 15:17:47 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_ELASTIC_SCORE | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 3.84s | 错误: 2025-09-13 15:17:50 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_VOLUME_ENERGY | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 3.93s | 错误: 2025-09-13 15:17:54 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_PRICE_POSITION | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 3.69s | 错误: 2025-09-13 15:17:58 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_TECHNICAL_FORM | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 4.12s | 错误: 2025-09-13 15:18:02 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_HOT_SPOT | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 4.83s | 错误: 2025-09-13 15:18:06 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_INDUSTRY_ROTATION | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 7.57s | 错误: 2025-09-13 15:18:11 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_CYCLE_POSITION | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 13.00s | 错误: 2025-09-13 15:18:19 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_RISK_CONTROL | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 16.36s | 错误: 2025-09-13 15:18:35 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_TIMING_SIGNAL | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 6.69s | 错误: 2025-09-13 15:18:49 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_POSITION_MANAGEMENT | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 6.07s | 错误: 2025-09-13 15:18:55 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_PORTFOLIO_OPTIMIZATION | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 6.12s | 错误: 2025-09-13 15:19:01 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_STRATEGY_COMBINATION | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 9.06s | 错误: 2025-09-13 15:19:07 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_PERFORMANCE_ATTRIBUTION | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 8.86s | 错误: 2025-09-13 15:19:17 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_ALPHA_GENERATION | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 8.08s | 错误: 2025-09-13 15:19:25 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_BETA_HEDGING | 🟢 PASSED | 40.0 | validate_all_zxm_indicators_95.py | 6.86s | 错误: 2025-09-13 15:19:33 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分40.0分 |
| ZXM_DAILY_MACD | ❓ ✅ PASSED_ARCHITECTURE_COMPLIANT | 97.0 | validate_zxm_daily_macd.py | 0.16s | - |
| ZXM_WASHPLATE | ❓ ✅ PASSED_ARCHITECTURE_COMPLIANT | 95.0 | validate_zxm_washplate.py | 42.78s | - |
| ZXM_BS_ABSORB | ❓ 🎉 PASSED_PRODUCTION_READY | 100.0 | validate_zxm_bs_absorb.py | 0.17s | - |

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
**报告生成时间**: 2025-09-13 15:21:44
**测试方式**: 统一调用现有验证脚本
