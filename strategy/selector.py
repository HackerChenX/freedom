class StockSelector:
    def __init__(self, data_manager, strategy_manager):
        self.data_manager = data_manager
        self.strategy_manager = strategy_manager

    def select_stocks(self, stock_list, start_date, end_date, strategy_config, use_cache=True):
        pass 