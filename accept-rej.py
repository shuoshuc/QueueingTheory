#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math

num_samples = 50

def f(x):
    return 30 * (np.power(x, 2) - 2 * np.power(x, 3) + np.power(x, 4))

def main():
    sum_x, cnt = 0, 0
    while cnt < num_samples:
        x = np.random.uniform()
        p = np.random.uniform()
        #u = round(np.random.uniform(), 2)
        fx = f(x)
        if p < (8*f(x)/15):
            sum_x += x
            cnt += 1
            print(f"sample={cnt}, x={x}, f(x)={f(x)}")
    print(f"mean={sum_x / cnt}")

if __name__ == "__main__":
    main()
