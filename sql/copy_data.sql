INSERT INTO stock.stock_info_new (
    code, name, date, level, open, close, high, low, 
    volume, turnover_rate, price_change, price_range, industry
)
SELECT 
    code, name, date, level, open, close, high, low, 
    volume, turnover_rate, price_change, price_range, industry
FROM stock.stock_info; 