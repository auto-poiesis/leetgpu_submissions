import torch
import triton
import triton.language as tl

### CRUX - Say that we are given 2 array of a 10000 elements each and we have a block size of 1000. Each SM/Block processes 1000 elements, our grid consists of 10 such blocks, and within each block
### the addition operation is conducted in parallel and collected to compute c. 

@triton.jit
def vector_add_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0) # pid of the block. where AM I?
    block_start = pid * BLOCK_SIZE # starting index of the block. Eg -pid = 0,start=0, pid = 1, start = 1*1024 = 1024. So the first pid covers 0:1023, next pid convers 1024:2047 ....
    offsets = block_start + tl.arange(0,BLOCK_SIZE) # the actual indices that pid processes
    mask = offsets < n_elements # Boolean mask to make sure that we process only elements inside the offset
    a_vals = tl.load(a + offsets,mask = mask) # load the vector elements inside the offset into gpu
    b_vals = tl.load(b + offsets,mask = mask) 
    result = a_vals + b_vals # the actual operation
    tl.store(c + offsets, result, mask=mask) # parallel operation, where we take each offset, do the operations simultaneoulsy and compute the result. 

# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 1024 # 1 Block = 1 SM = 1 PI, and each SM contains threads. Each PI processes 1024 elements of a vector
    grid = (triton.cdiv(N, BLOCK_SIZE),) # the minimum no of blocks you need to solve the problem for a vector of shape N
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE) # Note - grid is NOT related to SMs at all. This is just to tell Triton how much computation you need to solve the problem