CREATE TABLE stock.stock_info_new (
    code String,
    name String,
    date Date,
    level String,
    open Float64,
    close Float64,
    high Float64,
    low Float64,
    volume Float64,
    turnover_rate Float64,
    price_change Float64,
    price_range Float64,
    industry String DEFAULT '',
    datetime DateTime DEFAULT now(),
    seq UInt32 DEFAULT 0
) ENGINE = ReplacingMergeTree()
PRIMARY KEY (code, level, date, datetime, seq)
ORDER BY (code, level, date, datetime, seq); 