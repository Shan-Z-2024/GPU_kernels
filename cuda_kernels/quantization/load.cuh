#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

template<const int AM, const int AK, typename T>
__device__ void loadGToS(float* A, float* As, const int M, const int N, const int rowStride, const int innerRow, const int innerCol){
	//load column-major matrix A, with 2D block size (AM, AK)
	// M is row-num and N is col-num
	// As[AM, AK], every thread load 4 elements
	// every transaction 128-bytes, every warp needs 4 transaction line
	// every thread will be mapping to different innerRow and innerCol
	// A is already in block-level mapping
	for (uint offset = 0; offset < AM; offset += rowStride) {
	    reinterpret_cast<float4 *>(
		&As[(innerRow + offset) * AK + innerCol * 4])[0] =
		reinterpret_cast<const float4 *>(
		    &A[(innerRow + offset) * N + innerCol * 4])[0];
	    // asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
	    //     : "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 0]),
	    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 1]),
	    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 2]),
	    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 3])
	    //     : "l"(&B[(innerRowB + offset) * N + innerColB * 4]));
  }

}

template<const int NUM_PER_THREAD, const int NUM_THREADS, typename T>
__device__ void loadSToR(T* As, T* reg_vals, const int offsets){
	//load from shared memory As to register reg_vals with offset as offsets
	// for every local thread, NUM_PER_THREAD is being loaded
	for (uint idx =0; idx < NUM_PER_THREAD; idx++){
		reg_vals[idx] = As[offsets + idx * NUM_THREADS];
	}
}

//TODO, store function from register to shared memory and then to global memory, or from register to global mmeory
