import numpy as np

def AND(x1,x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

def NAND(x1,x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def XOR(x1, x2):
    return AND(NAND(x1, x2),OR(x1,x2))

def main():
    print(f"AND(0, 0) == {AND(0, 0)}")
    print(f"AND(0, 1) == {AND(0, 1)}")
    print(f"AND(1, 0) == {AND(1, 0)}")
    print(f"AND(1, 1) == {AND(1, 1)}")
    print()
    print(f"NAND(0, 0) == {NAND(0, 0)}")
    print(f"NAND(0, 1) == {NAND(0, 1)}")
    print(f"NAND(1, 0) == {NAND(1, 0)}")
    print(f"NAND(1, 1) == {NAND(1, 1)}")
    print()
    print(f"OR(0, 0) == {OR(0, 0)}")
    print(f"OR(0, 1) == {OR(0, 1)}")
    print(f"OR(1, 0) == {OR(1, 0)}")
    print(f"OR(1, 1) == {OR(1, 1)}")
    print()
    print(f"XOR(0, 0) == {XOR(0, 0)}")
    print(f"XOR(0, 1) == {XOR(0, 1)}")
    print(f"XOR(1, 0) == {XOR(1, 0)}")
    print(f"XOR(1, 1) == {XOR(1, 1)}")



if __name__ == "__main__":
    main()
