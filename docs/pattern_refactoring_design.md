# Pattern Naming System Refactoring Design

## Overview
This document outlines the design for refactoring the pattern naming system from a centralized `COMPLETE_INDICATOR_PATTERNS_MAP` approach to a decentralized system where each indicator defines its own pattern names and descriptions.

## Current Architecture Problems

### 1. Centralized Mapping Complexity
- `COMPLETE_INDICATOR_PATTERNS_MAP` contains 50+ indicators with hundreds of pattern definitions
- Difficult to maintain and update
- Creates tight coupling between analysis components and pattern definitions
- Inconsistent pattern naming across different indicators

### 2. Mixed Implementation Approaches
- Some indicators (KDJ, RSI, BOLL, MACD) already use proper `register_patterns()` 
- Others (TRIX, ROC, CMO, VOL, etc.) still rely on centralized mapping
- Creates confusion and maintenance overhead

### 3. Pattern Retrieval Complexity
- `get_precise_pattern_info()` method has complex fallback logic
- Multiple pattern matching strategies (direct, case-insensitive, fuzzy)
- Hard to debug and maintain

## New Decentralized Architecture

### 1. Core Principles
- **Self-Contained Indicators**: Each indicator responsible for its own pattern metadata
- **Consistent Registration**: All indicators use `register_pattern_to_registry()` method
- **PatternRegistry as Single Source**: All pattern lookups go through PatternRegistry
- **Backward Compatibility**: Existing functionality preserved during transition

### 2. Pattern Registration Standards

#### Standard Pattern Registration Format
```python
def register_patterns(self):
    """Register indicator patterns to global registry"""
    self.register_pattern_to_registry(
        pattern_id="INDICATOR_PATTERN_NAME",
        display_name="Clear Chinese Technical Name",
        description="Detailed technical description in Chinese",
        pattern_type="BULLISH|BEARISH|NEUTRAL",
        default_strength="VERY_STRONG|STRONG|MEDIUM|WEAK|VERY_WEAK",
        score_impact=float,  # -100 to +100
        polarity="POSITIVE|NEGATIVE|NEUTRAL"
    )
```

#### Pattern Naming Standards
- **Pattern IDs**: Use `INDICATOR_SPECIFIC_PATTERN` format (e.g., `TRIX_ABOVE_ZERO`)
- **Display Names**: Clear Chinese technical terms with indicator prefixes (e.g., `TRIX零轴上方`)
- **Descriptions**: Specific technical analysis descriptions avoiding vague terms
- **Avoid Generic Terms**: No `技术形态`, `未知形态`, `中等股票` etc.

### 3. Implementation Strategy

#### Phase 1: Core Infrastructure
1. Enhance PatternRegistry to handle all pattern lookups
2. Update buypoint_batch_analyzer to use PatternRegistry exclusively
3. Create pattern migration utilities

#### Phase 2: Indicator Refactoring (Priority Order)
1. **P0 Indicators**: Already complete (KDJ, RSI, BOLL, MACD)
2. **P1 Indicators**: TRIX, EMA, DMI (partially complete)
3. **P2 Indicators**: ROC, CMO, oscillators
4. **P3 Indicators**: Volume, volatility, composite indicators

#### Phase 3: Cleanup and Validation
1. Remove COMPLETE_INDICATOR_PATTERNS_MAP dependencies
2. Simplify get_precise_pattern_info() method
3. Run comprehensive tests

### 4. Pattern Registry Enhancement

#### Enhanced Lookup Methods
```python
class PatternRegistry:
    def get_pattern_by_indicator_and_id(self, indicator_name: str, pattern_id: str) -> Optional[Dict]:
        """Direct pattern lookup by indicator and pattern ID"""
        
    def get_all_patterns_for_indicator(self, indicator_name: str) -> List[Dict]:
        """Get all patterns for a specific indicator"""
        
    def search_patterns_by_name(self, pattern_name: str) -> List[Dict]:
        """Search patterns by display name or description"""
```

#### Fallback Strategy
```python
def get_pattern_info_with_fallback(self, indicator_name: str, pattern_id: str) -> Dict:
    """
    1. Try PatternRegistry lookup
    2. If not found, try legacy mapping (during transition)
    3. Generate default pattern info if all else fails
    """
```

### 5. Migration Process

#### Step 1: Identify Indicators to Migrate
- Extract all indicators from COMPLETE_INDICATOR_PATTERNS_MAP
- Categorize by priority (P0/P1/P2/P3)
- Create migration checklist

#### Step 2: Pattern Migration Template
```python
def migrate_indicator_patterns(indicator_class, pattern_mapping):
    """
    Template for migrating patterns from centralized mapping
    to indicator-specific registration
    """
    for pattern_id, pattern_info in pattern_mapping.items():
        indicator_class.register_pattern_to_registry(
            pattern_id=pattern_id,
            display_name=pattern_info['name'],
            description=pattern_info['description'],
            pattern_type=infer_pattern_type(pattern_info),
            score_impact=infer_score_impact(pattern_info),
            polarity=infer_polarity(pattern_info)
        )
```

#### Step 3: Validation Framework
```python
def validate_pattern_migration(indicator_name: str):
    """
    Validate that migrated patterns produce same results
    as centralized mapping
    """
    # Compare old vs new pattern retrieval
    # Ensure backward compatibility
    # Validate pattern completeness
```

## Benefits of New Architecture

### 1. Maintainability
- Each indicator owns its pattern definitions
- Easier to update and extend patterns
- Clear separation of concerns

### 2. Consistency
- Standardized pattern registration across all indicators
- Consistent naming conventions
- Unified pattern metadata structure

### 3. Extensibility
- Easy to add new patterns to existing indicators
- Simple to create new indicators with patterns
- Better support for custom indicators

### 4. Performance
- Direct pattern lookups through PatternRegistry
- Reduced complexity in pattern resolution
- Better caching opportunities

## Implementation Timeline

### Week 1: Infrastructure
- Enhance PatternRegistry
- Update buypoint_batch_analyzer pattern retrieval
- Create migration utilities

### Week 2-3: P1/P2 Indicator Migration
- Migrate TRIX, EMA, ROC, CMO
- Migrate STOCHRSI, WR, CCI, PSY
- Validate each migration

### Week 4: P3 Indicators and Cleanup
- Migrate volume and volatility indicators
- Remove COMPLETE_INDICATOR_PATTERNS_MAP
- Final testing and validation

## Risk Mitigation

### 1. Backward Compatibility
- Keep legacy mapping during transition
- Gradual migration with fallback support
- Comprehensive testing at each step

### 2. Pattern Consistency
- Automated validation of pattern names
- Standardized migration templates
- Review process for pattern quality

### 3. Performance Impact
- Benchmark pattern lookup performance
- Optimize PatternRegistry if needed
- Monitor system performance during migration

## Success Criteria

1. **Complete Migration**: All indicators use register_patterns() method
2. **Functionality Preserved**: All existing reports work unchanged
3. **Code Simplification**: Reduced complexity in pattern retrieval logic
4. **Maintainability Improved**: Easier to add/modify patterns
5. **Performance Maintained**: No degradation in analysis speed
