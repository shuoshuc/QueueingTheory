#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math

# p for Geometric distribution.
p = 0.2
num_samples = 5000

sum_x = 0
for _ in range(num_samples):
    u = np.random.uniform()
    #u = round(np.random.uniform(), 2)
    x = math.ceil(np.log(1 - u) / np.log(1 - p))
    sum_x += x
    print(f"u={u}, x={x}")
print(f"mean={sum_x / num_samples}")
