#ifndef CAFFE_NOLOCAL_GEMM_V4_POST_LAYER_HPP_
#define CAFFE_NOLCOAL_GEMM_V4_POST_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	template <typename Dtype>
	class NolocalGemmV4PostLayer : public Layer<Dtype> {
	public:
		explicit NolocalGemmV4PostLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual inline const char* type() const { return "NolocalGemmV4Post"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 2; }
	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		int height_, width_, channel_, num_;
		int ngroup_, channel_per_group_;
		int inner_shape_;
		Blob<Dtype> weight_, tmp_, sdata_, ones_;
		int order_;
		bool code_;
		float mean_, margin_;

	};
}

#endif  
