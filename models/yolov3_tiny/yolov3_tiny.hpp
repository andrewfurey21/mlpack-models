#ifndef MODELS_MODELS_YOLOV3_TINY_YOLOV3_TINY_HPP
#define MODELS_MODELS_YOLOV3_TINY_YOLOV3_TINY_HPP

#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/layer/add.hpp>
#include <mlpack/methods/ann/layer/batch_norm.hpp>
#include <mlpack/methods/ann/layer/convolution.hpp>
#include <mlpack/methods/ann/layer/leaky_relu.hpp>
#include <mlpack/methods/ann/layer/max_pooling.hpp>
#include <mlpack/methods/ann/layer/multi_layer.hpp>
#include <mlpack/methods/ann/layer/padding.hpp>

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
    layers.resize(24);
    layers[0] = ConvolutionalBlock(3, 16);
    layers[1] = PoolingBlock(2, 2);
    layers[2] = ConvolutionalBlock(3, 32);
    layers[3] = PoolingBlock(2, 2);
    layers[4] = ConvolutionalBlock(3, 64);
    layers[5] = PoolingBlock(2, 2);
    layers[6] = ConvolutionalBlock(3, 128);
    layers[7] = PoolingBlock(2, 2);
    layers[8] = ConvolutionalBlock(3, 256);
    layers[9] = PoolingBlock(2, 2);
    layers[10]= ConvolutionalBlock(3, 512);
    layers[11]= PoolingBlock(2, 1, 0.5, 0.5);
    layers[12]= ConvolutionalBlock(3, 1024);
    layers[13]= ConvolutionalBlock(3, 256);
    layers[14]= ConvolutionalBlock(3, 512);
    layers[15]= ConvolutionalBlock(3, 255, 1, 0, false, false);
    layers[16]= YOLOv3Block({3, 4, 5});
    layers[17]= RouteBlock(13);
    layers[18]= ConvolutionalBlock(1, 128);
    layers[19]= UpsampleBlock(2);
    layers[20]= RouteBlock(19, 8);
    layers[21]= ConvolutionalBlock(3, 256);
    layers[21]= ConvolutionalBlock(3, 255);
    layers[23]= YOLOv3Block({0, 1, 2});
  }


  ~YoloV3Tiny() {}

  FFN<YoloV3TinyLoss<>, RandomInitialization>& Model() { return model; }

private:
  Layer<MatType>* PoolingBlock(const size_t kernel, const size_t stride, const double paddingW=0, const double paddingH=0) {
    MultiLayer<MatType>* poolingBlock = new MultiLayer<MatType>();
    poolingBlock->template Add<Padding>(std::ceil(paddingW), std::floor(paddingW), std::ceil(paddingH), std::floor(paddingH));
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

  Layer<MatType>* UpsampleBlock(const size_t scaleFactor) {}

  Layer<MatType>* RouteBlock() {}

  Layer<MatType>* YOLOv3Block(const std::vector<size_t> mask) {}

  FFN<YoloV3TinyLoss<MatType>, RandomInitialization> model;

  size_t inputWidth;
  size_t inputHeight;
  size_t inputChannels;
  size_t numClasses;

  std::vector<double> anchors;

  std::vector<Layer<MatType>*> layers;
};

} // namespace models
} // namespace mlpack

#endif
