Quantization cuda kernel

March 21, first version, with cutlass and cccl library being used.
1. functional working, initial version, no_validation code yet, only fp8 being tested
2. to verify the memory and check other bugs with help of compute-sanitizer 
3. TODO, add verification code for fp8 and fp4
4. to profile with nsight-compute, ncu -f -o xxx.ncu-rep --set=full bin_XXX
