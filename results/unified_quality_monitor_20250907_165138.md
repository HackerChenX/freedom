# 统一指标质量监控报告

## 📊 测试概要

- **测试时间**: 2025-09-07T16:37:45.036970
- **测试指标数**: 132
- **通过指标数**: 102
- **失败指标数**: 27
- **错误指标数**: 3
- **执行时间**: 833.95 秒
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
| EMA | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.05s | - |
| WMA | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| CCI | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.18s | - |
| BIAS | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.06s | - |
| EMV | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.02s | - |
| AD | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| FORCE_INDEX | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| CHAIKIN | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| PVT | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| VORTEX | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| VOSC | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| STDDEV | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.17s | - |
| VOLATILITY | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.03s | - |
| CHAIKIN_VOLATILITY | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.08s | - |
| GARMAN_KLASS | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.02s | - |
| CMO | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.02s | - |
| ULTIMATE | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.02s | - |
| RSIMA | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.02s | - |
| PSAR | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.38s | - |
| SUPERTREND | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.10s | - |
| ELLIOTT_WAVE | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.05s | - |
| ENHANCED_CCI | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| ENHANCED_TRIX | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| TRIX | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| SAR | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| MOMENTUM | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| ROC_OSCILLATOR | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| MACD | 🟢 PASSED | 100.0 | validate_enhanced_macd_trend.py | 1.02s | - |
| RSI | 🟢 PASSED | 100.0 | validate_rsi_derivatives.py | 0.45s | - |
| BOLL | 🟢 PASSED | 100.0 | validate_enhanced_boll_indicators.py | 0.19s | - |
| KDJ | 🟢 PASSED | 100.0 | validate_enhanced_kdj.py | 0.18s | - |
| WR | 🟢 PASSED | 100.0 | validate_enhanced_wr.py | 7.27s | - |
| VOL | 🟢 PASSED | 100.0 | validate_vol.py | 0.68s | - |
| VR | 🟢 PASSED | 100.0 | validate_volume_score.py | 0.41s | - |
| AROON | 🟢 PASSED | 100.0 | validate_trend_indicators.py | 0.08s | - |
| DMA | 🟢 PASSED | 100.0 | validate_trend_indicators.py | 0.05s | - |
| MACD_SCORE | 🟢 PASSED | 100.0 | validate_score_indicators.py | 0.03s | - |
| RSI_SCORE | 🟢 PASSED | 100.0 | validate_score_indicators.py | 0.01s | - |
| BOLL_SCORE | 🟢 PASSED | 100.0 | validate_score_indicators.py | 0.01s | - |
| KDJ_SCORE | 🟢 PASSED | 100.0 | validate_score_indicators.py | 0.01s | - |
| ENHANCED_MACD | 🟢 PASSED | 100.0 | validate_enhanced_macd_trend.py | 0.74s | - |
| ENHANCED_BOLL | 🟢 PASSED | 100.0 | validate_enhanced_boll_indicators.py | 0.13s | - |
| ENHANCED_KDJ | 🟢 PASSED | 100.0 | validate_enhanced_kdj.py | 0.13s | - |
| FIBONACCI_TOOLS | 🟢 PASSED | 100.0 | validate_fibonacci_tools.py | 0.69s | - |
| MARKET_ENV | 🟢 PASSED | 100.0 | validate_market_env.py | 0.10s | - |
| SENTIMENT_ANALYSIS | 🟢 PASSED | 100.0 | validate_sentiment_analysis.py | 0.10s | - |
| TREND_CLASSIFICATION | 🟢 PASSED | 100.0 | validate_trend_classification.py | 0.10s | - |
| TREND_STRENGTH | 🟢 PASSED | 100.0 | validate_trend_strength.py | 0.46s | - |
| TIME_CYCLE_ANALYSIS | 🟢 PASSED | 100.0 | validate_time_cycle_analysis.py | 0.09s | - |
| MULTI_PERIOD_RESONANCE | 🟢 PASSED | 100.0 | validate_multi_period_resonance.py | 0.15s | - |
| INTRADAY_VOLATILITY | 🟢 PASSED | 100.0 | validate_intraday_volatility.py | 0.12s | - |
| STOCK_VIX | 🟢 PASSED | 100.0 | validate_stock_vix.py | 0.34s | - |
| ALPHA_GENERATION | 🟢 PASSED | 100.0 | validate_alpha_generation_indicators.py | 0.01s | - |
| BETA_HEDGING | 🟢 PASSED | 100.0 | validate_beta_hedging_indicators.py | 0.01s | - |
| ADX | 🟢 PASSED | 20.0 | validate_fixed_adx_indicator.py | 13.68s | 错误: 2025-09-07 16:38:08 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ROC | 🟢 PASSED | 20.0 | validate_fixed_roc_indicator.py | 10.05s | 错误: 2025-09-07 16:38:21 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| OBV | 🟢 PASSED | 20.0 | validate_fixed_obv_indicator.py | 12.25s | 错误: 2025-09-07 16:38:31 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| MTM | 🟢 PASSED | 20.0 | validate_fixed_mtm_indicator.py | 14.13s | 错误: 2025-09-07 16:38:42 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| MFI | 🟢 PASSED | 20.0 | validate_fixed_mfi_indicator.py | 24.15s | 错误: 2025-09-07 16:39:02 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| VIX | 🟢 PASSED | 20.0 | validate_fixed_baseindicators.py | 15.55s | 错误: 2025-09-07 16:39:23 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ATR | 🟢 PASSED | 20.0 | validate_atr_indicator.py | 14.45s | 错误: 2025-09-07 16:39:38 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| PSY | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 44.36s | 错误: 2025-09-07 16:39:59 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_WEEKLY_MACD | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 39.18s | 错误: 2025-09-07 16:40:46 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_MONTHLY_MACD | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 55.38s | 错误: 2025-09-07 16:41:32 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_TURNOVER | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 33.27s | 错误: 2025-09-07 16:42:20 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_VOLUME_SHRINK | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 16.09s | 错误: 2025-09-07 16:42:50 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_MA_CALLBACK | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 24.53s | 错误: 2025-09-07 16:43:07 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_DAILY_TREND_UP | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 14.41s | 错误: 2025-09-07 16:43:32 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_WEEKLY_TREND_UP | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 14.47s | 错误: 2025-09-07 16:43:46 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_MONTHLY_KDJ_TREND_UP | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 13.18s | 错误: 2025-09-07 16:44:01 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_AMPLITUDE_ELASTICITY | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 13.67s | 错误: 2025-09-07 16:44:13 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_RISE_ELASTICITY | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 9.75s | 错误: 2025-09-07 16:44:27 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_ELASTICITY | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 12.44s | 错误: 2025-09-07 16:44:36 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_BOUNCE_DETECTOR | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 11.14s | 错误: 2025-09-07 16:44:49 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_BUYPOINT_SCORE | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 9.50s | 错误: 2025-09-07 16:45:00 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_TREND_SCORE | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 12.00s | 错误: 2025-09-07 16:45:11 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_ELASTIC_SCORE | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 11.73s | 错误: 2025-09-07 16:45:21 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_VOLUME_ENERGY | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 12.07s | 错误: 2025-09-07 16:45:34 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_PRICE_POSITION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 11.00s | 错误: 2025-09-07 16:45:47 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_TECHNICAL_FORM | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 7.52s | 错误: 2025-09-07 16:45:56 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_HOT_SPOT | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 6.77s | 错误: 2025-09-07 16:46:03 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_INDUSTRY_ROTATION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 11.19s | 错误: 2025-09-07 16:46:11 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_CYCLE_POSITION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 10.48s | 错误: 2025-09-07 16:46:22 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_RISK_CONTROL | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 7.74s | 错误: 2025-09-07 16:46:32 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_TIMING_SIGNAL | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 6.58s | 错误: 2025-09-07 16:46:39 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_POSITION_MANAGEMENT | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 8.71s | 错误: 2025-09-07 16:46:46 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_PORTFOLIO_OPTIMIZATION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 14.36s | 错误: 2025-09-07 16:46:55 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_STRATEGY_COMBINATION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 13.97s | 错误: 2025-09-07 16:47:11 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_PERFORMANCE_ATTRIBUTION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 12.76s | 错误: 2025-09-07 16:47:25 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_ALPHA_GENERATION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 9.44s | 错误: 2025-09-07 16:47:36 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_BETA_HEDGING | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 13.75s | 错误: 2025-09-07 16:47:46 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_CHIP_DISTRIBUTION | 🟢 PASSED | 20.0 | validate_zxm_chip_distribution_complete.py | 8.56s | 错误: 2025-09-07 16:49:12 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_FUND_FLOW | 🟢 PASSED | 20.0 | validate_zxm_fund_flow_complete.py | 12.66s | 错误: 2025-09-07 16:49:21 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_INSTITUTION_BEHAVIOR | 🟢 PASSED | 20.0 | validate_zxm_institution_behavior_complete.py | 6.18s | 错误: 2025-09-07 16:49:33 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_MARKET_SENTIMENT | 🟢 PASSED | 20.0 | validate_fixed_zxm_market_sentiment.py | 7.91s | 错误: 2025-09-07 16:49:38 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_LIQUIDITY_ANALYSIS | 🟢 PASSED | 20.0 | validate_fixed_zxm_liquidity_analysis.py | 3.26s | 错误: 2025-09-07 16:49:45 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_CORRELATION_MATRIX | 🟢 PASSED | 20.0 | validate_fixed_zxm_correlation_matrix.py | 6.22s | 错误: 2025-09-07 16:49:49 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_VOLATILITY_FORECAST | 🟢 PASSED | 20.0 | validate_fixed_zxm_volatility_forecast.py | 3.38s | 错误: 2025-09-07 16:49:55 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| DOJI | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 3.44s | 错误: 2025-09-07 16:49:58 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| HAMMER | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 3.35s | 错误: 2025-09-07 16:50:02 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| SHOOTING_STAR | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 3.55s | 错误: 2025-09-07 16:50:05 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ENGULFING | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 4.68s | 错误: 2025-09-07 16:50:09 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| HARAMI | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 3.45s | 错误: 2025-09-07 16:50:13 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| PIERCING_LINE | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 4.19s | 错误: 2025-09-07 16:50:17 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| DARK_CLOUD_COVER | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 3.45s | 错误: 2025-09-07 16:50:21 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| MORNING_STAR | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 3.40s | 错误: 2025-09-07 16:50:25 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| EVENING_STAR | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 3.23s | 错误: 2025-09-07 16:50:28 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| THREE_BLACK_CROWS | 🟢 PASSED | 20.0 | validate_three_black_crows_complete.py | 4.44s | 错误: 2025-09-07 16:50:31 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| THREE_WHITE_SOLDIERS | 🟢 PASSED | 20.0 | validate_three_white_soldiers_complete.py | 5.12s | 错误: 2025-09-07 16:50:36 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| V_SHAPED_REVERSAL | 🟢 PASSED | 20.0 | validate_v_shaped_reversal_complete.py | 5.08s | 错误: 2025-09-07 16:50:41 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| HEAD_SHOULDERS | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 4.55s | 错误: 2025-09-07 16:50:46 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| DOUBLE_TOP | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 8.46s | 错误: 2025-09-07 16:50:51 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| DOUBLE_BOTTOM | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 3.95s | 错误: 2025-09-07 16:50:59 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| TRIANGLE | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 3.42s | 错误: 2025-09-07 16:51:03 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| WEDGE | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 3.05s | 错误: 2025-09-07 16:51:06 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| FLAG | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 3.07s | 错误: 2025-09-07 16:51:09 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| PENNANT | 🟢 PASSED | 20.0 | validate_pennant_complete.py | 3.31s | 错误: 2025-09-07 16:51:12 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| RECTANGLE | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 3.07s | 错误: 2025-09-07 16:51:15 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| CUP_AND_HANDLE | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 3.53s | 错误: 2025-09-07 16:51:19 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ISLAND_REVERSAL | 🟢 PASSED | 20.0 | validate_island_reversal_complete.py | 4.61s | 错误: 2025-09-07 16:51:22 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| CANDLESTICK_PATTERNS | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 4.18s | 错误: 2025-09-07 16:51:27 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| COMPOSITE | 🟢 PASSED | 20.0 | validate_composite_indicator.py | 5.23s | 错误: 2025-09-07 16:51:33 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| MA | 🟢 PASSED | 0 | validate_unified_ma_indicator_strict.py | 5.80s | - |
| DMI | 🟢 PASSED | 0 | validate_adx_indicator.py | 2.56s | - |
| KC | 🟢 PASSED | 0 | validate_kc_indicator.py | 0.86s | - |
| SYNERGY | 🟢 PASSED | 0 | validate_synergy_indicator_strict.py | 0.25s | - |
| UNIFIED_MA | 🟢 PASSED | 0 | validate_unified_ma_indicator_strict.py | 0.16s | - |
| STOCHRSI | 🟢 PASSED | 0 | validate_enhanced_stochrsi.py | 0.21s | - |
| ENHANCED_STOCHRSI | 🟢 PASSED | 0 | validate_enhanced_stochrsi.py | 0.05s | - |
| ZXM_DAILY_MACD | ❓ ✅ PASSED_ARCHITECTURE_COMPLIANT | 97.0 | validate_zxm_daily_macd.py | 3.75s | - |
| ZXM_WASHPLATE | ❓ ✅ PASSED_ARCHITECTURE_COMPLIANT | 95.0 | validate_zxm_washplate.py | 71.46s | - |
| ZXM_BS_ABSORB | ❓ 🎉 PASSED_PRODUCTION_READY | 100.0 | validate_zxm_bs_absorb.py | 1.54s | - |

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
**报告生成时间**: 2025-09-07 16:51:39
**测试方式**: 统一调用现有验证脚本
