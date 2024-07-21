class maidanglao():

    def __init__(self, a):  # 初始化模块的自我变量赋值行为

        print("今晚吃三碗饭")
        self.a = a
        print(a)

    def jiafa(self):
        print(self.a + 3)  # 如果需要让上面的变量a参与计算，需要在a的前方加入self.a

    def jianfa(self, b):
        print(100 - 9)
        print("这是对b的减法{}".format(b - self.a))  # 在jianfa()中添加变量，并让a与b一起参与计算

    def chufa(self):
        print(6 / 1)


if __name__ == "__main__":
    n = maidanglao(100)  # 实例化对象
    n.jiafa()
    n.jianfa(800)  # 函数的调用
    n.chufa()

    # 运行结果：
    #
    # 今晚吃三碗饭
    # 100
    # 103
    # 91
    # 这是对b的减法700
    # 6.0
