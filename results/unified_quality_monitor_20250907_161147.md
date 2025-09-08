# 统一指标质量监控报告

## 📊 测试概要

- **测试时间**: 2025-09-07T16:06:15.955402
- **测试指标数**: 132
- **通过指标数**: 101
- **失败指标数**: 28
- **错误指标数**: 3
- **执行时间**: 331.65 秒
- **通过率**: 76.5%

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
| CCI | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| BIAS | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| SYNERGY | 🔴 FAILED | 0 | validate_synergy_indicator_strict.py | 0.02s | - |
| EMV | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.00s | - |
| AD | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| FORCE_INDEX | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| CHAIKIN | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| PVT | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| VORTEX | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| VOSC | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| STDDEV | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| VOLATILITY | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| CHAIKIN_VOLATILITY | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| GARMAN_KLASS | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| CMO | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| ULTIMATE | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| RSIMA | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| PSAR | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| SUPERTREND | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| ELLIOTT_WAVE | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| ENHANCED_CCI | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| ENHANCED_TRIX | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| TRIX | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| SAR | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.00s | - |
| MOMENTUM | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.00s | - |
| ROC_OSCILLATOR | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.00s | - |
| MACD | 🟢 PASSED | 100.0 | validate_enhanced_macd_trend.py | 0.73s | - |
| RSI | 🟢 PASSED | 100.0 | validate_rsi_derivatives.py | 0.35s | - |
| BOLL | 🟢 PASSED | 100.0 | validate_enhanced_boll_indicators.py | 0.14s | - |
| KDJ | 🟢 PASSED | 100.0 | validate_enhanced_kdj.py | 0.17s | - |
| WR | 🟢 PASSED | 100.0 | validate_enhanced_wr.py | 6.49s | - |
| VOL | 🟢 PASSED | 100.0 | validate_vol.py | 0.28s | - |
| VR | 🟢 PASSED | 100.0 | validate_volume_score.py | 0.12s | - |
| AROON | 🟢 PASSED | 100.0 | validate_trend_indicators.py | 0.01s | - |
| DMA | 🟢 PASSED | 100.0 | validate_trend_indicators.py | 0.01s | - |
| MACD_SCORE | 🟢 PASSED | 100.0 | validate_score_indicators.py | 0.02s | - |
| RSI_SCORE | 🟢 PASSED | 100.0 | validate_score_indicators.py | 0.01s | - |
| BOLL_SCORE | 🟢 PASSED | 100.0 | validate_score_indicators.py | 0.00s | - |
| KDJ_SCORE | 🟢 PASSED | 100.0 | validate_score_indicators.py | 0.00s | - |
| ENHANCED_MACD | 🟢 PASSED | 100.0 | validate_enhanced_macd_trend.py | 0.46s | - |
| ENHANCED_BOLL | 🟢 PASSED | 100.0 | validate_enhanced_boll_indicators.py | 0.09s | - |
| ENHANCED_KDJ | 🟢 PASSED | 100.0 | validate_enhanced_kdj.py | 0.09s | - |
| FIBONACCI_TOOLS | 🟢 PASSED | 100.0 | validate_fibonacci_tools.py | 0.37s | - |
| MARKET_ENV | 🟢 PASSED | 100.0 | validate_market_env.py | 0.05s | - |
| SENTIMENT_ANALYSIS | 🟢 PASSED | 100.0 | validate_sentiment_analysis.py | 0.05s | - |
| TREND_CLASSIFICATION | 🟢 PASSED | 100.0 | validate_trend_classification.py | 0.05s | - |
| TREND_STRENGTH | 🟢 PASSED | 100.0 | validate_trend_strength.py | 0.20s | - |
| TIME_CYCLE_ANALYSIS | 🟢 PASSED | 100.0 | validate_time_cycle_analysis.py | 0.05s | - |
| MULTI_PERIOD_RESONANCE | 🟢 PASSED | 100.0 | validate_multi_period_resonance.py | 0.09s | - |
| INTRADAY_VOLATILITY | 🟢 PASSED | 100.0 | validate_intraday_volatility.py | 0.07s | - |
| STOCK_VIX | 🟢 PASSED | 100.0 | validate_stock_vix.py | 0.14s | - |
| ALPHA_GENERATION | 🟢 PASSED | 100.0 | validate_alpha_generation_indicators.py | 0.01s | - |
| BETA_HEDGING | 🟢 PASSED | 100.0 | validate_beta_hedging_indicators.py | 0.00s | - |
| ADX | 🟢 PASSED | 20.0 | validate_fixed_adx_indicator.py | 5.79s | 错误: 2025-09-07 16:06:30 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ROC | 🟢 PASSED | 20.0 | validate_fixed_roc_indicator.py | 4.31s | 错误: 2025-09-07 16:06:36 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| OBV | 🟢 PASSED | 20.0 | validate_fixed_obv_indicator.py | 4.04s | 错误: 2025-09-07 16:06:40 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| MTM | 🟢 PASSED | 20.0 | validate_fixed_mtm_indicator.py | 3.91s | 错误: 2025-09-07 16:06:44 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| MFI | 🟢 PASSED | 20.0 | validate_fixed_mfi_indicator.py | 4.05s | 错误: 2025-09-07 16:06:48 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| VIX | 🟢 PASSED | 20.0 | validate_fixed_baseindicators.py | 4.17s | 错误: 2025-09-07 16:06:52 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ATR | 🟢 PASSED | 20.0 | validate_atr_indicator.py | 3.89s | 错误: 2025-09-07 16:06:56 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| PSY | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 3.75s | 错误: 2025-09-07 16:07:00 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_WEEKLY_MACD | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 6.24s | 错误: 2025-09-07 16:07:05 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_MONTHLY_MACD | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 6.38s | 错误: 2025-09-07 16:07:11 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_TURNOVER | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 6.54s | 错误: 2025-09-07 16:07:18 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_VOLUME_SHRINK | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 5.52s | 错误: 2025-09-07 16:07:24 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_MA_CALLBACK | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 5.33s | 错误: 2025-09-07 16:07:29 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_DAILY_TREND_UP | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 6.11s | 错误: 2025-09-07 16:07:35 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_WEEKLY_TREND_UP | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 7.30s | 错误: 2025-09-07 16:07:42 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_MONTHLY_KDJ_TREND_UP | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 6.66s | 错误: 2025-09-07 16:07:49 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_AMPLITUDE_ELASTICITY | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 5.77s | 错误: 2025-09-07 16:07:56 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_RISE_ELASTICITY | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 6.66s | 错误: 2025-09-07 16:08:02 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_ELASTICITY | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 5.11s | 错误: 2025-09-07 16:08:08 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_BOUNCE_DETECTOR | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 5.51s | 错误: 2025-09-07 16:08:13 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_BUYPOINT_SCORE | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 5.86s | 错误: 2025-09-07 16:08:18 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_TREND_SCORE | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 7.35s | 错误: 2025-09-07 16:08:25 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_ELASTIC_SCORE | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 5.95s | 错误: 2025-09-07 16:08:32 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_VOLUME_ENERGY | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 5.64s | 错误: 2025-09-07 16:08:38 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_PRICE_POSITION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 5.95s | 错误: 2025-09-07 16:08:43 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_TECHNICAL_FORM | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.58s | 错误: 2025-09-07 16:08:49 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_HOT_SPOT | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.71s | 错误: 2025-09-07 16:08:53 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_INDUSTRY_ROTATION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.54s | 错误: 2025-09-07 16:08:58 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_CYCLE_POSITION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 5.04s | 错误: 2025-09-07 16:09:03 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_RISK_CONTROL | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 5.29s | 错误: 2025-09-07 16:09:08 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_TIMING_SIGNAL | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.81s | 错误: 2025-09-07 16:09:13 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_POSITION_MANAGEMENT | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.16s | 错误: 2025-09-07 16:09:18 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_PORTFOLIO_OPTIMIZATION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.44s | 错误: 2025-09-07 16:09:22 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_STRATEGY_COMBINATION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.05s | 错误: 2025-09-07 16:09:27 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_PERFORMANCE_ATTRIBUTION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 3.76s | 错误: 2025-09-07 16:09:30 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_ALPHA_GENERATION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 3.76s | 错误: 2025-09-07 16:09:34 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_BETA_HEDGING | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 3.62s | 错误: 2025-09-07 16:09:38 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_CHIP_DISTRIBUTION | 🟢 PASSED | 20.0 | validate_zxm_chip_distribution_complete.py | 9.66s | 错误: 2025-09-07 16:10:17 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_FUND_FLOW | 🟢 PASSED | 20.0 | validate_zxm_fund_flow_complete.py | 3.38s | 错误: 2025-09-07 16:10:22 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_INSTITUTION_BEHAVIOR | 🟢 PASSED | 20.0 | validate_zxm_institution_behavior_complete.py | 2.94s | 错误: 2025-09-07 16:10:25 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_MARKET_SENTIMENT | 🟢 PASSED | 20.0 | validate_fixed_zxm_market_sentiment.py | 2.99s | 错误: 2025-09-07 16:10:28 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_LIQUIDITY_ANALYSIS | 🟢 PASSED | 20.0 | validate_fixed_zxm_liquidity_analysis.py | 2.72s | 错误: 2025-09-07 16:10:31 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_CORRELATION_MATRIX | 🟢 PASSED | 20.0 | validate_fixed_zxm_correlation_matrix.py | 3.84s | 错误: 2025-09-07 16:10:34 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_VOLATILITY_FORECAST | 🟢 PASSED | 20.0 | validate_fixed_zxm_volatility_forecast.py | 2.43s | 错误: 2025-09-07 16:10:38 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| DOJI | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.20s | 错误: 2025-09-07 16:10:40 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| HAMMER | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.30s | 错误: 2025-09-07 16:10:42 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| SHOOTING_STAR | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.38s | 错误: 2025-09-07 16:10:45 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ENGULFING | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.04s | 错误: 2025-09-07 16:10:47 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| HARAMI | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.12s | 错误: 2025-09-07 16:10:49 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| PIERCING_LINE | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.18s | 错误: 2025-09-07 16:10:51 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| DARK_CLOUD_COVER | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.30s | 错误: 2025-09-07 16:10:53 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| MORNING_STAR | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.45s | 错误: 2025-09-07 16:10:56 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| EVENING_STAR | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.26s | 错误: 2025-09-07 16:10:58 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| THREE_BLACK_CROWS | 🟢 PASSED | 20.0 | validate_three_black_crows_complete.py | 2.81s | 错误: 2025-09-07 16:11:00 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| THREE_WHITE_SOLDIERS | 🟢 PASSED | 20.0 | validate_three_white_soldiers_complete.py | 3.50s | 错误: 2025-09-07 16:11:03 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| V_SHAPED_REVERSAL | 🟢 PASSED | 20.0 | validate_v_shaped_reversal_complete.py | 2.75s | 错误: 2025-09-07 16:11:07 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| HEAD_SHOULDERS | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.51s | 错误: 2025-09-07 16:11:09 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| DOUBLE_TOP | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.70s | 错误: 2025-09-07 16:11:12 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| DOUBLE_BOTTOM | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.64s | 错误: 2025-09-07 16:11:15 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| TRIANGLE | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.60s | 错误: 2025-09-07 16:11:17 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| WEDGE | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 3.71s | 错误: 2025-09-07 16:11:20 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| FLAG | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 3.94s | 错误: 2025-09-07 16:11:24 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| PENNANT | 🟢 PASSED | 20.0 | validate_pennant_complete.py | 3.03s | 错误: 2025-09-07 16:11:28 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| RECTANGLE | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 3.52s | 错误: 2025-09-07 16:11:31 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| CUP_AND_HANDLE | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.68s | 错误: 2025-09-07 16:11:34 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ISLAND_REVERSAL | 🟢 PASSED | 20.0 | validate_island_reversal_complete.py | 3.34s | 错误: 2025-09-07 16:11:37 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| CANDLESTICK_PATTERNS | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.63s | 错误: 2025-09-07 16:11:40 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| COMPOSITE | 🟢 PASSED | 20.0 | validate_composite_indicator.py | 3.27s | 错误: 2025-09-07 16:11:44 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| MA | 🟢 PASSED | 0 | validate_unified_ma_indicator_strict.py | 3.86s | - |
| DMI | 🟢 PASSED | 0 | validate_adx_indicator.py | 1.15s | - |
| KC | 🟢 PASSED | 0 | validate_kc_indicator.py | 0.09s | - |
| UNIFIED_MA | 🟢 PASSED | 0 | validate_unified_ma_indicator_strict.py | 0.02s | - |
| STOCHRSI | 🟢 PASSED | 0 | validate_enhanced_stochrsi.py | 0.05s | - |
| ENHANCED_STOCHRSI | 🟢 PASSED | 0 | validate_enhanced_stochrsi.py | 0.06s | - |
| ZXM_DAILY_MACD | ❓ ✅ PASSED_ARCHITECTURE_COMPLIANT | 97.0 | validate_zxm_daily_macd.py | 0.46s | - |
| ZXM_WASHPLATE | ❓ ✅ PASSED_ARCHITECTURE_COMPLIANT | 95.0 | validate_zxm_washplate.py | 30.70s | - |
| ZXM_BS_ABSORB | ❓ 🎉 PASSED_PRODUCTION_READY | 100.0 | validate_zxm_bs_absorb.py | 0.42s | - |

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
- **SYNERGY**: 分数 0
  - 验证脚本: validate_synergy_indicator_strict.py
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
**报告生成时间**: 2025-09-07 16:11:47
**测试方式**: 统一调用现有验证脚本
