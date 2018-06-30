# -*- coding: utf-8 -*-
"""
@CREATETIME: 13/06/2018 08:36 
@AUTHOR: Chans
@VERSION: 
"""

import copy

def linearComputer(init_vec,n):
    if n <= 2:
        return 1
    current2A = init_vec
    prev2A = copy.deepcopy(current2A)
    for ind in range(n - 2) :
        current2A[0] = 0*prev2A[0] + 1*prev2A[ 1 ]
        current2A[1] = 1*prev2A[0] + 1*prev2A[ 1 ]
        prev2A = copy.deepcopy(current2A)
    return prev2A

print( linearComputer([1,1],10))