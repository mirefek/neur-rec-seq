class ArrayEnv:
    def __init__(self, data):
        self.start_data = list(data)
        self.reset()
    def swap(self, i0, i1):
        self.instructions.append((0, (i0, i1)))
        self.data[i0], self.data[i1] = self.data[i1], self.data[i0]
        return None
    def less_than(self, i0, i1):
        self.instructions.append((1, (i0, i1)))
        x,y = self.data[i0], self.data[i1]
        return x is not None and y is not None and x < y
    def stack_push(self, i):
        self.instructions.append((2, (i,)))
        if self.data[i] is None: return
        self.stack.append(self.data[i])
        self.data[i] = None
    def stack_pop(self, i):
        self.instructions.append((3, (i,)))
        if not self.stack or self.data[i] is not None: return
        self.data[i] = self.stack.pop()

    def instr_by_index(self, i):
        return [self.swap, self.less_than, self.stack_push, self.stack_pop][i]
    def instr_name(self, i):
        return self.instr_by_index(i).__name__
    def run_instr(self, instruction):
        t, args = instruction
        return self.instr_by_index(t)(*args)
    def reset(self):
        self.data = list(data)
        self.instructions = []
        self.stack = []
    def __len__(self):
        return len(self.data)
    def __str__(self):
        return str(self.data)
    def __repr__(self):
        return "ArrayEnv({})".format(repr(self.data))

def quick_sort_segment(a, i_min, i_max):
    if i_max - i_min <= 0: return
    i0 = i_min
    i1 = i_max
    while i0 < i1:
        if a.less_than(i0, i0+1):
            a.swap(i0+1,i1)
            i1 -= 1
        else:
            a.swap(i0, i0+1)
            i0 += 1

    quick_sort_segment(a, i_min, i0-1)
    quick_sort_segment(a, i0+1, i_max)

def merge_sort_segment(a, i_min, i_max):
    if i_max <= i_min: return
    i_mid = (i_min + i_max) // 2
    merge_sort_segment(a, i_min, i_mid)
    merge_sort_segment(a, i_mid+1, i_max)

    i0 = i_min
    i1 = i_mid+1
    while i0 <= i_mid or i1 <= i_max:
        if i1 > i_max or (i0 <= i_mid and a.less_than(i0, i1)):
            a.stack_push(i0)
            i0 += 1
        else:
            a.stack_push(i1)
            i1 += 1
    for i in range(i_max, i_min-1, -1):
        a.stack_pop(i)

def quick_sort(a):
    quick_sort_segment(a, 0, len(a)-1)
def merge_sort(a):
    merge_sort_segment(a, 0, len(a)-1)

if __name__ == "__main__":
    import random

    data = [random.randint(0,999) for _ in range(20)]
    a = ArrayEnv(data)
    print(a)
    #quick_sort(a)
    merge_sort(a)
    instructions = a.instructions
    print(a)
    a.reset()
    print(a)
    for instr in instructions:
        t, args = instr
        print(a.instr_name(t), args)
        res = a.run_instr(instr)
        if res is None: print(a)
        else: print(res)
    print(a)
