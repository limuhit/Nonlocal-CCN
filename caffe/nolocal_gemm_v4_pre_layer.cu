#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "nolocal_gemm_v4_pre_layer.hpp"
namespace caffe {
	
	
	template <typename Dtype>
	__global__ void nonlocal4_distance_forward(const int nthreads, Dtype* const top_data, const Dtype* const bottom_data,
		const int channel, const int inner_shape, const int len) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int ts = index % inner_shape;
			int tg = (index / inner_shape) % channel;
			int tn = index / inner_shape / channel;
			int ta = ts / len;
			int tb = ts % len;
			Dtype tmp = bottom_data[(tn*channel + tg)*len + ta] - bottom_data[(tn*channel + tg)*len + tb];
			top_data[index] = -tmp*tmp;
		}
	}
	
	template <typename Dtype>
	void NolocalGemmV4PreLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype * data = bottom[0]->gpu_data();
		Dtype * top_data = top[0]->mutable_gpu_data();
		int count = top[0]->count();
		nonlocal4_distance_forward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
			(count, top_data, data, channel_, inner_shape_*inner_shape_, inner_shape_);
		
	}
	template <typename Dtype>
	__global__ void nonlocal4_distance_backward(const int nthreads, Dtype* const top_diff, const Dtype* const bottom_data,
		const int channel, const int inner_shape, const int len) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int ts = index % inner_shape;
			int tg = (index / inner_shape) % channel;
			int tn = index / inner_shape / channel;
			int ta = ts / len;
			int tb = ts % len;
			if (ta > tb) {
				Dtype tmp = 2 * (bottom_data[(tn*channel + tg)*len + tb] - bottom_data[(tn*channel + tg)*len + ta]);
				int tid = ((tn*channel + tg)*len + tb)*len + ta;
				tmp = (top_diff[index] + top_diff[tid])*tmp;
				top_diff[index] = tmp;
				top_diff[tid] = -tmp;
			}
			else if (ta == tb) {
				top_diff[index] = 0;
			}

		}
	}
	template <typename Dtype>
	void NolocalGemmV4PreLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int count = top[0]->count();
		nonlocal4_distance_backward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
			(count, top[0]->mutable_gpu_diff(), bottom[0]->gpu_data(), channel_, inner_shape_*inner_shape_, inner_shape_);
		caffe_gpu_set(inner_shape_, Dtype(1.), ones_.mutable_gpu_data());
		caffe_gpu_gemv(CblasNoTrans, num_*channel_*inner_shape_, inner_shape_, Dtype(1.0), top[0]->gpu_diff(),
			ones_.gpu_data(), Dtype(0), bottom[0]->mutable_gpu_diff());
	
	}

	INSTANTIATE_LAYER_GPU_FUNCS(NolocalGemmV4PreLayer);

}  // namespace caffe
