# 统一指标质量监控报告

## 📊 测试概要

- **测试时间**: 2025-09-04T20:40:12.076407
- **测试指标数**: 25
- **通过指标数**: 47
- **失败指标数**: 1
- **错误指标数**: 2
- **执行时间**: 13.25 秒
- **通过率**: 188.0%

## 🎯 测试方法说明

本次测试采用**统一调用现有验证脚本**的方式，确保测试方式的一致性：

1. **复用现有验证脚本**: 调用每个指标专门的验证脚本
2. **保持测试标准一致**: 使用与之前验证相同的测试方法
3. **标准化结果格式**: 统一处理不同脚本的输出格式
4. **完整错误处理**: 处理脚本执行中的各种异常情况

## 📋 详细结果

| 指标名称 | 状态 | 分数 | 验证脚本 | 执行时间 | 备注 |
|---------|------|------|----------|----------|------|
| DMI | 🔴 FAILED | 0 | validate_adx_indicator.py | 0.11s | - |
| MACD | 🟢 PASSED | 100.0 | validate_enhanced_macd_trend.py | 0.08s | - |
| RSI | 🟢 PASSED | 100.0 | validate_rsi_derivatives.py | 0.03s | - |
| BOLL | 🟢 PASSED | 100.0 | validate_enhanced_boll_indicators.py | 0.01s | - |
| KDJ | 🟢 PASSED | 100.0 | validate_enhanced_kdj.py | 0.01s | - |
| WR | 🟢 PASSED | 100.0 | validate_enhanced_wr.py | 0.76s | - |
| VOL | 🟢 PASSED | 100.0 | validate_vol.py | 0.03s | - |
| VR | 🟢 PASSED | 100.0 | validate_volume_score.py | 0.02s | - |
| AROON | 🟢 PASSED | 100.0 | validate_trend_indicators.py | 0.00s | - |
| DMA | 🟢 PASSED | 100.0 | validate_trend_indicators.py | 0.00s | - |
| MA | 🟢 PASSED | 95 | validate_unified_ma_indicator_strict.py | 0.35s | 验证通过 |
| ADX | 🟢 PASSED | 95 | validate_fixed_adx_indicator.py | 0.37s | 验证通过 |
| ROC | 🟢 PASSED | 95 | validate_fixed_roc_indicator.py | 0.36s | 验证通过 |
| OBV | 🟢 PASSED | 95 | validate_fixed_obv_indicator.py | 0.36s | 验证通过 |
| MTM | 🟢 PASSED | 95 | validate_fixed_mtm_indicator.py | 0.30s | 验证通过 |
| MFI | 🟢 PASSED | 95 | validate_fixed_mfi_indicator.py | 0.43s | 验证通过 |
| VIX | 🟢 PASSED | 95 | validate_fixed_baseindicators.py | 0.37s | 验证通过 |
| SYNERGY | 🟢 PASSED | 95 | validate_synergy_indicator_strict.py | 0.30s | 验证通过 |
| UNIFIED_MA | 🟢 PASSED | 95 | validate_unified_ma_indicator_strict.py | 0.44s | 验证通过 |
| ATR | 🟢 PASSED | 95 | validate_atr_indicator.py | 0.45s | 验证通过 |
| PSY | 🟢 PASSED | 95 | validate_pattern_indicators_95.py | 0.45s | 验证通过 |
| ZXM_WEEKLY_MACD | 🟢 PASSED | 95 | validate_all_zxm_indicators_95.py | 0.70s | 验证通过 |
| ZXM_MONTHLY_MACD | 🟢 PASSED | 95 | validate_all_zxm_indicators_95.py | 0.97s | 验证通过 |
| ZXM_TURNOVER | 🟢 PASSED | 95 | validate_all_zxm_indicators_95.py | 0.86s | 验证通过 |
| ZXM_VOLUME_SHRINK | 🟢 PASSED | 95 | validate_all_zxm_indicators_95.py | 0.55s | 验证通过 |
| ZXM_MA_CALLBACK | 🟢 PASSED | 95 | validate_all_zxm_indicators_95.py | 0.51s | 验证通过 |
| ZXM_DAILY_TREND_UP | 🟢 PASSED | 95 | validate_all_zxm_indicators_95.py | 0.74s | 验证通过 |
| ZXM_WEEKLY_TREND_UP | 🟢 PASSED | 95 | validate_all_zxm_indicators_95.py | 0.51s | 验证通过 |
| ZXM_MONTHLY_KDJ_TREND_UP | 🟢 PASSED | 95 | validate_all_zxm_indicators_95.py | 0.53s | 验证通过 |
| ZXM_AMPLITUDE_ELASTICITY | 🟢 PASSED | 95 | validate_all_zxm_indicators_95.py | 0.52s | 验证通过 |
| ZXM_RISE_ELASTICITY | 🟢 PASSED | 95 | validate_all_zxm_indicators_95.py | 0.49s | 验证通过 |
| ZXM_ELASTICITY | 🟢 PASSED | 95 | validate_all_zxm_indicators_95.py | 0.55s | 验证通过 |
| ZXM_BOUNCE_DETECTOR | 🟢 PASSED | 95 | validate_all_zxm_indicators_95.py | 0.50s | 验证通过 |
| ZXM_BUYPOINT_SCORE | 🟢 PASSED | 95 | validate_all_zxm_indicators_95.py | 0.52s | 验证通过 |
| ZXM_TREND_SCORE | 🟢 PASSED | 95 | validate_all_zxm_indicators_95.py | 0.51s | 验证通过 |
| ZXM_ELASTIC_SCORE | 🟢 PASSED | 95 | validate_all_zxm_indicators_95.py | 0.52s | 验证通过 |
| ZXM_VOLUME_ENERGY | 🟢 PASSED | 95 | validate_all_zxm_indicators_95.py | 0.84s | 验证通过 |
| ZXM_PRICE_POSITION | 🟢 PASSED | 95 | validate_all_zxm_indicators_95.py | 0.46s | 验证通过 |
| ZXM_TECHNICAL_FORM | 🟢 PASSED | 95 | validate_all_zxm_indicators_95.py | 0.49s | 验证通过 |
| ZXM_HOT_SPOT | 🟢 PASSED | 95 | validate_all_zxm_indicators_95.py | 0.52s | 验证通过 |
| ZXM_INDUSTRY_ROTATION | 🟢 PASSED | 95 | validate_all_zxm_indicators_95.py | 0.52s | 验证通过 |
| ZXM_CYCLE_POSITION | 🟢 PASSED | 95 | validate_all_zxm_indicators_95.py | 0.47s | 验证通过 |
| ZXM_RISK_CONTROL | 🟢 PASSED | 95 | validate_all_zxm_indicators_95.py | 0.57s | 验证通过 |
| ZXM_TIMING_SIGNAL | 🟢 PASSED | 95 | validate_all_zxm_indicators_95.py | 0.48s | 验证通过 |
| ZXM_POSITION_MANAGEMENT | 🟢 PASSED | 95 | validate_all_zxm_indicators_95.py | 0.47s | 验证通过 |
| ZXM_PORTFOLIO_OPTIMIZATION | 🟢 PASSED | 95 | validate_all_zxm_indicators_95.py | 0.51s | 验证通过 |
| ZXM_STRATEGY_COMBINATION | 🟢 PASSED | 95 | validate_all_zxm_indicators_95.py | 0.53s | 验证通过 |
| KC | 🟢 PASSED | 0 | validate_kc_indicator.py | 0.01s | - |
| ZXM_DAILY_MACD | ❓ ✅ PASSED_ARCHITECTURE_COMPLIANT | 97.0 | validate_zxm_daily_macd.py | 0.06s | - |
| ZXM_BS_ABSORB | ❓ 🎉 PASSED_PRODUCTION_READY | 100.0 | validate_zxm_bs_absorb.py | 0.06s | - |

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
**报告生成时间**: 2025-09-04 20:40:32
**测试方式**: 统一调用现有验证脚本
