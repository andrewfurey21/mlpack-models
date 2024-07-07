#ifndef MODELS_MODELS_YOLOV3_TINY_YOLOV3_TINY_HPP
#define MODELS_MODELS_YOLOV3_TINY_YOLOV3_TINY_HPP

#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/layer/add.hpp>
#include <mlpack/methods/ann/layer/batch_norm.hpp>
#include <mlpack/methods/ann/layer/convolution.hpp>
#include <mlpack/methods/ann/layer/identity.hpp>
#include <mlpack/methods/ann/layer/leaky_relu.hpp>
#include <mlpack/methods/ann/layer/max_pooling.hpp>
#include <mlpack/methods/ann/layer/multi_layer.hpp>

#include "yolov3_tiny_loss.hpp"
#include "yolov3_tiny_layer.hpp"

#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>

namespace mlpack {
namespace models {

template<typename MatType>
class YoloV3Tiny {
public:
  YoloV3Tiny(const std::vector<double> anchors,
	     const size_t inputWidth = 416,
	     const size_t inputHeight = 416,
	     const size_t inputChannels = 3,
	     const size_t numClasses = 80
	     ) :
	inputWidth(inputWidth),
	inputHeight(inputHeight),
	inputChannels(inputChannels),
	numClasses(numClasses)
  {
    model.Add(ConvolutionalBlock(3, 16));//0
    model.Add(PoolingBlock(2, 2));//1
    //model.Add(ConvolutionalBlock(3, 32));//2
  }


  ~YoloV3Tiny() {}

  FFN<YoloV3TinyLoss<>, RandomInitialization>& Model() { return model; }

private:
  Layer<MatType>* PoolingBlock(const size_t kernel, const size_t stride) {
    MultiLayer<MatType>* poolingBlock = new MultiLayer<MatType>();
    poolingBlock->template Add<MaxPooling>(kernel, kernel, stride, stride);
    return poolingBlock;
  }

  Layer<MatType>* ConvolutionalBlock(const size_t kernelSize,
			     const size_t filters,
			     const size_t stride = 1,
			     const size_t padding = 1,
			     const bool batchNorm = true,
			     const bool activate = true) {
    MultiLayer<MatType>* convBlock = new MultiLayer<MatType>();
    convBlock->template Add<Convolution>(filters, kernelSize, kernelSize, stride, stride, padding, padding, "none", false);
    if (batchNorm) {
      convBlock->template Add<BatchNorm>();
    }
    // // TODO: bias
    if (activate) {
      convBlock->template Add<LeakyReLU>(0.1f);
    }
    return convBlock;
  }

  Layer<MatType>* UpsampleBlock() {}

  Layer<MatType>* RouteBlock() {}

  Layer<MatType>* YOLOBlock(const std::vector<size_t> mask) {}

  FFN<YoloV3TinyLoss<MatType>, RandomInitialization> model;

  size_t inputWidth;
  size_t inputHeight;
  size_t inputChannels;
  size_t numClasses;

  std::vector<double> anchors;
};

} // namespace models
} // namespace mlpack

#endif
