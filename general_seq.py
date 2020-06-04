from itertools import chain

def rule_compose(rl1, rl2):
    return [
        list(chain.from_iterable(rl2[x] for x in tup))
        for tup in rl1
    ]

class GeneralSeq:
    def __init__(self, rule):
        assert(len(rule[0]) > 1 and rule[0][0] == 0)
        self.symbol_num = len(rule)
        self._rule = rule
        self._rule_composed = rule

    def generate(self, n):
        while len(self._rule_composed[0]) < n:
            self._rule_composed = rule_compose(self._rule, self._rule_composed)
        return self._rule_composed[0][:n]

tm_gen = GeneralSeq([[0,1],[1,0]])
fw_gen = GeneralSeq([[0,1],[0]])

if __name__ == "__main__":
    def thue_morse(n):
        result = [0]
        for _ in range(n):
            result = result + [1-x for x in result]
        return result

    def fibo_word(n):
        a,b = [0], [0,1]
        if n == 0: return a
        for _ in range(n-1):
            a,b = b, b+a
        return b

    tm = thue_morse(15)
    print(tm == tm_gen.generate(len(tm)))
    fw = fibo_word(15)
    print(fw == fw_gen.generate(len(fw)))

    print(GeneralSeq([[0,1], [1,2], [2,0]]).generate(50))
