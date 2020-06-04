import torch
import torch.nn.functional as F

def log_plus(a,b):
    # return torch.log(torch.exp(a) + torch.exp(b))

    m = torch.max(a,b)
    ex, ey = (torch.exp(x-m) for x in (a,b))
    return m+torch.log(ex + ey)

class LogValZero:
    def val(self):
        return torch.zeros(1)
    def __add__(self, other):
        return other
    def __mul__(self, other):
        return self

class LogVal:
    def __init__(self, lval):
        self.lval = lval
    def val(self):
        return torch.exp(self.lval)

    def __add__(self, other):
        if isinstance(other, LogValZero): return self
        elif isinstance(other, LogVal):
            return LogVal(log_plus(self.lval, other.lval))
        else: raise Exception("unexpected type {}".format(type(other)))
    def __mul__(self, other):
        if isinstance(other, LogValZero): return other
        elif isinstance(other, LogVal):
            return LogVal(self.lval + other.lval)
        else: raise Exception("unexpected type {}".format(type(other)))

class LogProb:
    def __init__(self, sigmoid_arg, neg = None):
        if neg is None:
            self.pos = LogVal(F.logsigmoid(sigmoid_arg))
            self.neg = LogVal(F.logsigmoid(-sigmoid_arg))
        else:
            self.pos = sigmoid_arg
            self.neg = neg

    def pos_val(self):
        return self.pos.val()
    def neg_val(self):
        return self.neg.val()

    def __mul__(self, other):
        #print('--------')
        #print("{} = {} + {}".format(self.pos_val() + self.neg_val(),
        #      self.pos.val(), self.neg.val()))
        #print("{} = {} + {}".format(other.pos_val() + other.neg_val(),
        #      other.pos.val(), other.neg.val()))
        pos = self.pos * other.pos
        neg = self.neg + self.pos * other.neg
        #print("{} = {} + {}".format(pos.val() + neg.val(), pos.val(), neg.val()))
        return LogProb(pos, neg)

def log_prob_true():
    return LogProb(LogVal(torch.zeros(1)), LogValZero())
