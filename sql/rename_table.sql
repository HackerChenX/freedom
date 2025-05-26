-- 先重命名旧表
RENAME TABLE stock.stock_info TO stock.stock_info_old;
-- 再将新表重命名为原名
RENAME TABLE stock.stock_info_new TO stock.stock_info; 