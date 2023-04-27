import formula


def stock_select():
    formula.init("301110", "20230201", "20230315")
    print("缩量", formula.缩量())
    print("日均线", formula.日均线上移())
    print("弹性", formula.弹性())
    print("DIFF上移", formula.DIFF上移("日"))
    print("DEA上移", formula.DEA上移("日"))
    print("MACD上移", formula.MACD上移("日"))
    print("周K上移", formula.K上移("周"))
    print("周D上移", formula.D上移("周"))
    print("周J上移", formula.J上移("周"))
    print("日K上移", formula.K上移("日"))
    print("日D上移", formula.D上移("日"))
    print("日J上移", formula.J上移("日"))
    print("月K上移", formula.K上移("月"))
    print("月D上移", formula.D上移("月"))
    print("月J上移", formula.J上移("月"))
    print("换手率大于6", formula.换手率大于(6))
    print("macd小于0.8", formula.MACD小于("日", 0.8))
    print("回踩均线", formula.回踩均线("日"))


if __name__ == '__main__':
    stock_select()
