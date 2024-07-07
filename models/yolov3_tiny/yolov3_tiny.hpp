#ifndef MODELS_MODELS_YOLOV3_TINY_YOLOV3_TINY_HPP
#define MODELS_MODELS_YOLOV3_TINY_YOLOV3_TINY_HPP

#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/layer/convolution.hpp>
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
  YoloV3Tiny(const std::vector<size_t> mask,
	     const std::vector<double> anchors,
	     const size_t inputWidth = 416,
	     const size_t inputHeight = 416,
	     const size_t inputChannels = 3,
	     const size_t numClasses = 80
	     ) :
	inputWidth(inputWidth),
	inputHeight(inputHeight),
	inputChannels(inputChannels),
	numClasses(numClasses)
	{}

  ~YoloV3Tiny() {}

private:
  void AddPoolingBlock(const size_t kernel, const size_t stride) {
    model.Add(MaxPooling(kernel, kernel, stride, stride));
  }

  FFN<YoloV3TinyLoss<MatType>, RandomInitialization> model;

  size_t inputWidth;
  size_t inputHeight;
  size_t inputChannels;
  size_t numClasses;

  std::vector<size_t> mask;
  std::vector<double> anchors;
};

} // namespace models
} // namespace mlpack

#endif
