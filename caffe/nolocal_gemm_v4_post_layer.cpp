#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "nolocal_gemm_v4_post_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void NolocalGemmV4PostLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		ngroup_ = this->layer_param_.nonlocal_param().groups();
		order_ = this->layer_param_.nonlocal_param().order();
		code_ = this->layer_param_.nonlocal_param().code();
		mean_ = this->layer_param_.nonlocal_param().mean();
		margin_ = this->layer_param_.nonlocal_param().margin();
	}
	template <typename Dtype>
	void  NolocalGemmV4PostLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		num_ = bottom[0]->num();
		channel_ = bottom[0]->channels();
		height_ = bottom[0]->height();
		width_ = bottom[0]->width();
		inner_shape_ = height_*width_;
		channel_per_group_ = channel_ / ngroup_;
		top[0]->Reshape(num_, channel_, height_, width_);
		top[1]->Reshape(num_, channel_, height_, width_);
		sdata_.Reshape(num_,channel_, height_, width_);
		ones_.Reshape(inner_shape_, 1, 1, 1);
		tmp_.ReshapeLike(*bottom[1]);
	}

	template <typename Dtype>
	void  NolocalGemmV4PostLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {



	}

	template <typename Dtype>
	void  NolocalGemmV4PostLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {

	}
#ifdef CPU_ONLY
	STUB_GPU(NolocalGemmV4PostLayer);
#endif

	INSTANTIATE_CLASS(NolocalGemmV4PostLayer);
	REGISTER_LAYER_CLASS(NolocalGemmV4Post);

}  // namespace caffe
