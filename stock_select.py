import formula


def stock_select():
    formula.init("301110", "20230201", "20230315")
    print("缩量", formula.缩量())
    print("日均线", formula.日均线上移())
    print("弹性", formula.弹性())


if __name__ == '__main__':
    stock_select()
