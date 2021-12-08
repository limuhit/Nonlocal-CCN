#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "nolocal_gemm_v4_pre_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void NolocalGemmV4PreLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		
	}
	template <typename Dtype>
	void  NolocalGemmV4PreLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		num_ = bottom[0]->num();
		channel_ = bottom[0]->channels();
		height_ = bottom[0]->height();
		width_ = bottom[0]->width();
		inner_shape_ = height_*width_;
		top[0]->Reshape(num_, channel_, inner_shape_, inner_shape_);
		sdata_.Reshape(num_,channel_,height_,width_);
		ones_.Reshape(1, 1, height_, width_);
	}

	template <typename Dtype>
	void  NolocalGemmV4PreLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {



	}

	template <typename Dtype>
	void  NolocalGemmV4PreLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {

	}
#ifdef CPU_ONLY
	STUB_GPU(NolocalGemmV4Layer);
#endif

	INSTANTIATE_CLASS(NolocalGemmV4PreLayer);
	REGISTER_LAYER_CLASS(NolocalGemmV4Pre);

}  // namespace caffe
