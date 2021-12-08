#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "nolocal_gemm_v4_post_layer.hpp"
namespace caffe {
	
	
	
	template <typename Dtype>
	__global__ void nonlocal4_constrain_forward(const int nthreads, Dtype* const weight, 
		const int inner_shape, const int width, bool code) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tw = index % inner_shape;
			int th = (index / inner_shape) % inner_shape;
			if (code) {
				if (th / width + th%width <= tw / width + tw%width)
					weight[index] = 0;
			}
			else {
				if (th / width + th%width < tw / width + tw%width)
					weight[index] = 0;
			}

		}
	}
	template <typename Dtype>
	__global__ void nonlocal4_post(const int nthreads, Dtype* const data, Dtype* const dist,
		const Dtype mean, const Dtype max_distance, const int inner_shape) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int tidx = index*inner_shape;
			data[tidx] = mean;
			dist[tidx] = max_distance;

		}
	}
	template <typename Dtype>
	__global__ void nonlocal4_div_forward(const int nthreads, const Dtype* const input, const Dtype* const sum, Dtype* const output, 
	    const int inner_shape) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int th = (index / inner_shape) % inner_shape;
			int tn = (index / inner_shape) / inner_shape;
			if (sum[tn*inner_shape + th] > 0)
				output[index] = input[index] / sum[tn*inner_shape + th];
			else
				output[index] = input[index];
		}
	}
	template <typename Dtype>
	void NolocalGemmV4PostLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype * bottom_data = bottom[0]->gpu_data();
		const Dtype * bottom_wt = bottom[1]->mutable_gpu_data();
		Dtype * wt = tmp_.mutable_gpu_data();
		Dtype * top_data = top[0]->mutable_gpu_data();
		int count = bottom[1]->count();
		caffe_gpu_exp(count, bottom_wt, wt);
		nonlocal4_constrain_forward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
			(count, wt, inner_shape_, width_, code_);
		caffe_gpu_set(inner_shape_, Dtype(1.), ones_.mutable_gpu_data());
		caffe_gpu_gemv(CblasNoTrans, num_*channel_*inner_shape_, inner_shape_, Dtype(1.0), wt, 
			ones_.gpu_data(), Dtype(0), sdata_.mutable_gpu_data());
		nonlocal4_div_forward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
			(count, wt, sdata_.gpu_data(),tmp_.mutable_gpu_diff(), inner_shape_);
		caffe_gpu_gemm_batch2(CblasNoTrans, CblasNoTrans, inner_shape_,1, inner_shape_, Dtype(1.0),
			tmp_.gpu_diff(),bottom_data, Dtype(0), top_data, inner_shape_*inner_shape_, inner_shape_,
			inner_shape_, num_*ngroup_);
		caffe_gpu_mul(count, tmp_.gpu_diff(), bottom_wt, bottom[1]->mutable_gpu_diff());
		caffe_gpu_gemv(CblasNoTrans, num_*channel_*inner_shape_, inner_shape_, Dtype(1.0), bottom[1]->gpu_diff(),
			ones_.gpu_data(), Dtype(0), top[1]->mutable_gpu_data());
		count = num_*channel_;
		nonlocal4_post<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
			(count, top[0]->mutable_gpu_data(), top[1]->mutable_gpu_data(), mean_, -margin_*margin_, width_*height_);
	}

	
	template <typename Dtype>
	__global__ void nonlocal4_distance_backward(const int nthreads, const Dtype* const top_diff, const Dtype*  const da,
		const Dtype* const db, Dtype* const da_diff, Dtype* const db_diff,
	    const int inner_shape) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int ph = (index / inner_shape) % inner_shape;
			int pn = index / inner_shape / inner_shape;
			int pidx = pn*inner_shape + ph;
			if (db[index] > 0)
				da_diff[index] = log(db[index])*top_diff[pidx];
			else
				da_diff[index] = 0;
			db_diff[index] = da[index] * top_diff[pidx];
		}
	}
	template <typename Dtype>
	__global__ void nonlocal4_softmax_backward(const int nthreads, const Dtype* const top_diff,	const Dtype* const top_data,
		const Dtype * const sdiff,Dtype* const bottom_diff, const int inner_shape) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int ph = (index / inner_shape) % inner_shape;
			int pn = index / inner_shape / inner_shape;
			int pidx = pn*inner_shape + ph;
			bottom_diff[index] = bottom_diff[index] + (top_diff[index] - sdiff[pidx])*top_data[index];
		}
	}
	template <typename Dtype>
	void NolocalGemmV4PostLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		
		const Dtype * top_diff = top[0]->gpu_diff();
		const Dtype * bottom_data = bottom[0]->gpu_data();
		int count = bottom[1]->count();
		caffe_gpu_memcpy(count * sizeof(Dtype), tmp_.gpu_data(), bottom[1]->mutable_gpu_data());
		caffe_gpu_memcpy(count * sizeof(Dtype), tmp_.gpu_diff(), tmp_.mutable_gpu_data());
		nonlocal4_distance_backward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
			(count, top[1]->gpu_diff(), tmp_.gpu_data(), bottom[1]->gpu_data(),tmp_.mutable_gpu_diff(),
				bottom[1]->mutable_gpu_diff(), inner_shape_);
		caffe_gpu_gemm_batch2(CblasTrans, CblasNoTrans, inner_shape_, 1, inner_shape_, Dtype(1.0),
			tmp_.gpu_data(), top_diff, Dtype(0), bottom[0]->mutable_gpu_diff(), inner_shape_*inner_shape_, inner_shape_,
			inner_shape_, num_*ngroup_);
		caffe_gpu_gemm_batch2(CblasNoTrans, CblasTrans,  inner_shape_, inner_shape_, 1, Dtype(1.0),
			top_diff, bottom_data, Dtype(1.0), tmp_.mutable_gpu_diff(), inner_shape_, inner_shape_,
			inner_shape_*inner_shape_, num_*ngroup_);
		//caffe_gpu_memcpy(count * sizeof(Dtype), tmp_.gpu_diff(), bottom[1]->mutable_gpu_diff());
		//caffe_gpu_set(count, Dtype(1. / (2 * 4 * 16 * 16)), tmp_.mutable_gpu_diff());
		//caffe_gpu_set(count, Dtype(0), bottom[1]->mutable_gpu_diff());
		
		caffe_gpu_gemm_batch2(CblasNoTrans, CblasNoTrans, 1, 1, inner_shape_, Dtype(1.0),
			tmp_.gpu_data(), tmp_.gpu_diff(), Dtype(0), sdata_.mutable_gpu_diff(), inner_shape_, inner_shape_,
			1, num_*ngroup_*inner_shape_);
		nonlocal4_softmax_backward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
			(count, tmp_.gpu_diff(),  tmp_.gpu_data(), sdata_.gpu_diff(),bottom[1]->mutable_gpu_diff(),inner_shape_);
		

	}

	INSTANTIATE_LAYER_GPU_FUNCS(NolocalGemmV4PostLayer);

}  // namespace caffe
