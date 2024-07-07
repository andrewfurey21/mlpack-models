#ifndef MODELS_MODELS_YOLOV3_TINY_YOLOV3_TINY_HPP
#define MODELS_MODELS_YOLOV3_TINY_YOLOV3_TINY_HPP

#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/layer/add.hpp>
#include <mlpack/methods/ann/layer/batch_norm.hpp>
#include <mlpack/methods/ann/layer/convolution.hpp>
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
    AddConvolutionalBlock(3, 16);
    AddPoolingBlock(2, 2);
    AddConvolutionalBlock(3, 32);
    AddPoolingBlock(2, 2);
    AddConvolutionalBlock(3, 64);
    AddPoolingBlock(2, 2);
    AddConvolutionalBlock(3, 128);
    AddPoolingBlock(2, 2);
    AddConvolutionalBlock(3, 256);//save
    AddPoolingBlock(2, 2);
    AddConvolutionalBlock(3, 512);
    AddPoolingBlock(2, 2);
    AddConvolutionalBlock(3, 1024);
    AddConvolutionalBlock(1, 256);//save
    AddConvolutionalBlock(3, 512);
    AddConvolutionalBlock(1, 255);
    model.Add(mlpack::YoloLayer<arma::mat>);

  }

  ~YoloV3Tiny() {}

private:
  void AddPoolingBlock(const size_t kernel, const size_t stride) {
    model.Add(MaxPooling(kernel, kernel, stride, stride));
  }

  void AddConvolutionalBlock(const size_t kernelSize,
			     const size_t filters,
			     const size_t stride = 1,
			     const size_t padding = 1,
			     const bool batchNorm = true,
			     const bool activate = true) {
    model.Add(Convolution(filters, kernelSize, kernelSize, stride, stride, padding, padding, "none", false));
    if (batchNorm) {
      model.Add(BatchNorm());
    }
    model.Add(Add());//biases
    if (activate) {
      model.Add(LeakyReLU(0.1));
    }
  }

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
