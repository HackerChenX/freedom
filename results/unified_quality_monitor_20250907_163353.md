# 统一指标质量监控报告

## 📊 测试概要

- **测试时间**: 2025-09-07T16:26:00.830871
- **测试指标数**: 132
- **通过指标数**: 102
- **失败指标数**: 27
- **错误指标数**: 3
- **执行时间**: 472.21 秒
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
| CCI | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| BIAS | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.00s | - |
| EMV | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.00s | - |
| AD | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.00s | - |
| FORCE_INDEX | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| CHAIKIN | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.00s | - |
| PVT | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.00s | - |
| VORTEX | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| VOSC | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.00s | - |
| STDDEV | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| VOLATILITY | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.00s | - |
| CHAIKIN_VOLATILITY | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.00s | - |
| GARMAN_KLASS | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.00s | - |
| CMO | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.00s | - |
| ULTIMATE | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.00s | - |
| RSIMA | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| PSAR | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| SUPERTREND | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| ELLIOTT_WAVE | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.00s | - |
| ENHANCED_CCI | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.04s | - |
| ENHANCED_TRIX | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.03s | - |
| TRIX | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.04s | - |
| SAR | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| MOMENTUM | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.02s | - |
| ROC_OSCILLATOR | 🔴 FAILED | 0 | validate_enhanced_indicators.py | 0.01s | - |
| MACD | 🟢 PASSED | 100.0 | validate_enhanced_macd_trend.py | 0.63s | - |
| RSI | 🟢 PASSED | 100.0 | validate_rsi_derivatives.py | 0.27s | - |
| BOLL | 🟢 PASSED | 100.0 | validate_enhanced_boll_indicators.py | 0.10s | - |
| KDJ | 🟢 PASSED | 100.0 | validate_enhanced_kdj.py | 0.10s | - |
| WR | 🟢 PASSED | 100.0 | validate_enhanced_wr.py | 4.64s | - |
| VOL | 🟢 PASSED | 100.0 | validate_vol.py | 0.28s | - |
| VR | 🟢 PASSED | 100.0 | validate_volume_score.py | 0.07s | - |
| AROON | 🟢 PASSED | 100.0 | validate_trend_indicators.py | 0.01s | - |
| DMA | 🟢 PASSED | 100.0 | validate_trend_indicators.py | 0.01s | - |
| MACD_SCORE | 🟢 PASSED | 100.0 | validate_score_indicators.py | 0.37s | - |
| RSI_SCORE | 🟢 PASSED | 100.0 | validate_score_indicators.py | 0.03s | - |
| BOLL_SCORE | 🟢 PASSED | 100.0 | validate_score_indicators.py | 0.07s | - |
| KDJ_SCORE | 🟢 PASSED | 100.0 | validate_score_indicators.py | 0.04s | - |
| ENHANCED_MACD | 🟢 PASSED | 100.0 | validate_enhanced_macd_trend.py | 2.33s | - |
| ENHANCED_BOLL | 🟢 PASSED | 100.0 | validate_enhanced_boll_indicators.py | 0.24s | - |
| ENHANCED_KDJ | 🟢 PASSED | 100.0 | validate_enhanced_kdj.py | 0.21s | - |
| FIBONACCI_TOOLS | 🟢 PASSED | 100.0 | validate_fibonacci_tools.py | 3.23s | - |
| MARKET_ENV | 🟢 PASSED | 100.0 | validate_market_env.py | 0.26s | - |
| SENTIMENT_ANALYSIS | 🟢 PASSED | 100.0 | validate_sentiment_analysis.py | 0.32s | - |
| TREND_CLASSIFICATION | 🟢 PASSED | 100.0 | validate_trend_classification.py | 0.20s | - |
| TREND_STRENGTH | 🟢 PASSED | 100.0 | validate_trend_strength.py | 1.12s | - |
| TIME_CYCLE_ANALYSIS | 🟢 PASSED | 100.0 | validate_time_cycle_analysis.py | 0.26s | - |
| MULTI_PERIOD_RESONANCE | 🟢 PASSED | 100.0 | validate_multi_period_resonance.py | 0.37s | - |
| INTRADAY_VOLATILITY | 🟢 PASSED | 100.0 | validate_intraday_volatility.py | 0.41s | - |
| STOCK_VIX | 🟢 PASSED | 100.0 | validate_stock_vix.py | 0.97s | - |
| ALPHA_GENERATION | 🟢 PASSED | 100.0 | validate_alpha_generation_indicators.py | 0.08s | - |
| BETA_HEDGING | 🟢 PASSED | 100.0 | validate_beta_hedging_indicators.py | 0.03s | - |
| ADX | 🟢 PASSED | 20.0 | validate_fixed_adx_indicator.py | 3.84s | 错误: 2025-09-07 16:26:12 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ROC | 🟢 PASSED | 20.0 | validate_fixed_roc_indicator.py | 2.87s | 错误: 2025-09-07 16:26:15 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| OBV | 🟢 PASSED | 20.0 | validate_fixed_obv_indicator.py | 3.28s | 错误: 2025-09-07 16:26:18 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| MTM | 🟢 PASSED | 20.0 | validate_fixed_mtm_indicator.py | 3.17s | 错误: 2025-09-07 16:26:22 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| MFI | 🟢 PASSED | 20.0 | validate_fixed_mfi_indicator.py | 3.28s | 错误: 2025-09-07 16:26:25 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| VIX | 🟢 PASSED | 20.0 | validate_fixed_baseindicators.py | 3.02s | 错误: 2025-09-07 16:26:28 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ATR | 🟢 PASSED | 20.0 | validate_atr_indicator.py | 2.54s | 错误: 2025-09-07 16:26:31 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| PSY | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.50s | 错误: 2025-09-07 16:26:34 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_WEEKLY_MACD | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 3.49s | 错误: 2025-09-07 16:26:36 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_MONTHLY_MACD | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 7.63s | 错误: 2025-09-07 16:26:40 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_TURNOVER | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 5.85s | 错误: 2025-09-07 16:26:49 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_VOLUME_SHRINK | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 5.45s | 错误: 2025-09-07 16:26:54 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_MA_CALLBACK | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.22s | 错误: 2025-09-07 16:26:59 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_DAILY_TREND_UP | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.15s | 错误: 2025-09-07 16:27:03 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_WEEKLY_TREND_UP | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.26s | 错误: 2025-09-07 16:27:08 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_MONTHLY_KDJ_TREND_UP | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.39s | 错误: 2025-09-07 16:27:12 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_AMPLITUDE_ELASTICITY | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 5.45s | 错误: 2025-09-07 16:27:16 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_RISE_ELASTICITY | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 5.02s | 错误: 2025-09-07 16:27:22 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_ELASTICITY | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.86s | 错误: 2025-09-07 16:27:27 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_BOUNCE_DETECTOR | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 5.86s | 错误: 2025-09-07 16:27:32 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_BUYPOINT_SCORE | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 5.77s | 错误: 2025-09-07 16:27:38 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_TREND_SCORE | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 5.83s | 错误: 2025-09-07 16:27:43 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_ELASTIC_SCORE | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.19s | 错误: 2025-09-07 16:27:49 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_VOLUME_ENERGY | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.08s | 错误: 2025-09-07 16:27:53 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_PRICE_POSITION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.08s | 错误: 2025-09-07 16:27:57 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_TECHNICAL_FORM | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.32s | 错误: 2025-09-07 16:28:01 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_HOT_SPOT | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.17s | 错误: 2025-09-07 16:28:06 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_INDUSTRY_ROTATION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 3.53s | 错误: 2025-09-07 16:28:10 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_CYCLE_POSITION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.61s | 错误: 2025-09-07 16:28:13 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_RISK_CONTROL | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 7.89s | 错误: 2025-09-07 16:28:18 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_TIMING_SIGNAL | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.71s | 错误: 2025-09-07 16:28:26 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_POSITION_MANAGEMENT | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.40s | 错误: 2025-09-07 16:28:31 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_PORTFOLIO_OPTIMIZATION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.21s | 错误: 2025-09-07 16:28:35 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_STRATEGY_COMBINATION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 5.29s | 错误: 2025-09-07 16:28:40 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_PERFORMANCE_ATTRIBUTION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.11s | 错误: 2025-09-07 16:28:45 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_ALPHA_GENERATION | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 4.58s | 错误: 2025-09-07 16:28:49 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_BETA_HEDGING | 🟢 PASSED | 20.0 | validate_all_zxm_indicators_95.py | 5.05s | 错误: 2025-09-07 16:28:54 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_CHIP_DISTRIBUTION | 🟢 PASSED | 20.0 | validate_zxm_chip_distribution_complete.py | 2.75s | 错误: 2025-09-07 16:29:32 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_FUND_FLOW | 🟢 PASSED | 20.0 | validate_zxm_fund_flow_complete.py | 2.62s | 错误: 2025-09-07 16:29:34 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_INSTITUTION_BEHAVIOR | 🟢 PASSED | 20.0 | validate_zxm_institution_behavior_complete.py | 2.19s | 错误: 2025-09-07 16:29:37 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_MARKET_SENTIMENT | 🟢 PASSED | 20.0 | validate_fixed_zxm_market_sentiment.py | 2.59s | 错误: 2025-09-07 16:29:39 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_LIQUIDITY_ANALYSIS | 🟢 PASSED | 20.0 | validate_fixed_zxm_liquidity_analysis.py | 2.28s | 错误: 2025-09-07 16:29:42 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_CORRELATION_MATRIX | 🟢 PASSED | 20.0 | validate_fixed_zxm_correlation_matrix.py | 3.71s | 错误: 2025-09-07 16:29:44 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ZXM_VOLATILITY_FORECAST | 🟢 PASSED | 20.0 | validate_fixed_zxm_volatility_forecast.py | 2.60s | 错误: 2025-09-07 16:29:48 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| DOJI | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 3.24s | 错误: 2025-09-07 16:29:51 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| HAMMER | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.50s | 错误: 2025-09-07 16:29:54 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| SHOOTING_STAR | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.55s | 错误: 2025-09-07 16:29:56 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ENGULFING | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.52s | 错误: 2025-09-07 16:29:59 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| HARAMI | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 4.30s | 错误: 2025-09-07 16:30:01 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| PIERCING_LINE | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.59s | 错误: 2025-09-07 16:30:05 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| DARK_CLOUD_COVER | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.47s | 错误: 2025-09-07 16:30:08 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| MORNING_STAR | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.51s | 错误: 2025-09-07 16:30:10 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| EVENING_STAR | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 2.42s | 错误: 2025-09-07 16:30:13 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| THREE_BLACK_CROWS | 🟢 PASSED | 20.0 | validate_three_black_crows_complete.py | 3.94s | 错误: 2025-09-07 16:30:15 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| THREE_WHITE_SOLDIERS | 🟢 PASSED | 20.0 | validate_three_white_soldiers_complete.py | 4.29s | 错误: 2025-09-07 16:30:20 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| V_SHAPED_REVERSAL | 🟢 PASSED | 20.0 | validate_v_shaped_reversal_complete.py | 3.95s | 错误: 2025-09-07 16:30:24 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| HEAD_SHOULDERS | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 29.41s | 错误: 2025-09-07 16:30:40 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| DOUBLE_TOP | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 12.27s | 错误: 2025-09-07 16:31:01 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| DOUBLE_BOTTOM | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 5.62s | 错误: 2025-09-07 16:31:10 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| TRIANGLE | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 6.59s | 错误: 2025-09-07 16:31:17 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| WEDGE | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 6.94s | 错误: 2025-09-07 16:31:22 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| FLAG | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 11.27s | 错误: 2025-09-07 16:31:31 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| PENNANT | 🟢 PASSED | 20.0 | validate_pennant_complete.py | 9.01s | 错误: 2025-09-07 16:31:42 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| RECTANGLE | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 11.82s | 错误: 2025-09-07 16:31:53 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| CUP_AND_HANDLE | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 10.48s | 错误: 2025-09-07 16:32:04 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| ISLAND_REVERSAL | 🟢 PASSED | 20.0 | validate_island_reversal_complete.py | 21.84s | 错误: 2025-09-07 16:32:15 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| CANDLESTICK_PATTERNS | 🟢 PASSED | 20.0 | validate_pattern_indicators_95.py | 29.45s | 错误: 2025-09-07 16:32:40 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| COMPOSITE | 🟢 PASSED | 20.0 | validate_composite_indicator.py | 39.78s | 错误: 2025-09-07 16:33:12 [INFO] root: 日志系统初始化完成
2025-09...; 验证通过，得分20.0分 |
| MA | 🟢 PASSED | 0 | validate_unified_ma_indicator_strict.py | 3.19s | - |
| DMI | 🟢 PASSED | 0 | validate_adx_indicator.py | 0.85s | - |
| KC | 🟢 PASSED | 0 | validate_kc_indicator.py | 0.10s | - |
| SYNERGY | 🟢 PASSED | 0 | validate_synergy_indicator_strict.py | 0.01s | - |
| UNIFIED_MA | 🟢 PASSED | 0 | validate_unified_ma_indicator_strict.py | 0.01s | - |
| STOCHRSI | 🟢 PASSED | 0 | validate_enhanced_stochrsi.py | 0.03s | - |
| ENHANCED_STOCHRSI | 🟢 PASSED | 0 | validate_enhanced_stochrsi.py | 0.30s | - |
| ZXM_DAILY_MACD | ❓ ✅ PASSED_ARCHITECTURE_COMPLIANT | 97.0 | validate_zxm_daily_macd.py | 0.26s | - |
| ZXM_WASHPLATE | ❓ ✅ PASSED_ARCHITECTURE_COMPLIANT | 95.0 | validate_zxm_washplate.py | 33.28s | - |
| ZXM_BS_ABSORB | ❓ 🎉 PASSED_PRODUCTION_READY | 100.0 | validate_zxm_bs_absorb.py | 0.39s | - |

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
**报告生成时间**: 2025-09-07 16:33:53
**测试方式**: 统一调用现有验证脚本
