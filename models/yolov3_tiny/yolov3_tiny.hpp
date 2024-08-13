#ifndef MODELS_MODELS_YOLOV3_TINY_YOLOV3_TINY_HPP
#define MODELS_MODELS_YOLOV3_TINY_YOLOV3_TINY_HPP

#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/layer/add.hpp>
#include <mlpack/methods/ann/layer/batch_norm.hpp>
#include <mlpack/methods/ann/layer/convolution.hpp>
#include <mlpack/methods/ann/layer/leaky_relu.hpp>
#include <mlpack/methods/ann/layer/max_pooling.hpp>
#include <mlpack/methods/ann/layer/multi_layer.hpp>
#include <mlpack/methods/ann/layer/concat.hpp>
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
	anchors(anchors),
	inputWidth(inputWidth),
	inputHeight(inputHeight),
	inputChannels(inputChannels),
	numClasses(numClasses)
  {
    large->InputDimensions() = std::vector<size_t>({inputWidth, inputHeight, inputChannels, 1});
    // small.InputDimensions() = std::vector<size_t>({inputWidth, inputHeight, inputChannels, 1});
   
    large->template Add<Convolution>(16, 3, 3, 1, 1, 1, 1, "none", false);//0
    large->template Add<BatchNorm>();
    large->template Add<LeakyReLU>(0.1f);

    large->template Add<MaxPooling>(2, 2, 2, 2);//1

    large->template Add<Convolution>(32, 3, 3, 1, 1, 1, 1, "none", false);//2
    large->template Add<BatchNorm>();
    large->template Add<LeakyReLU>(0.1f);

    large->template Add<MaxPooling>(2, 2, 2, 2);//3
    
    large->template Add<Convolution>(64, 3, 3, 1, 1, 1, 1, "none", false);//4
    large->template Add<BatchNorm>();
    large->template Add<LeakyReLU>(0.1f);

    large->template Add<MaxPooling>(2, 2, 2, 2);//5
    
    large->template Add<Convolution>(128, 3, 3, 1, 1, 1, 1, "none", false);//6
    large->template Add<BatchNorm>();
    large->template Add<LeakyReLU>(0.1f);
    
    large->template Add<MaxPooling>(2, 2, 2, 2);//7
    
    large->template Add<Convolution>(256, 3, 3, 1, 1, 1, 1, "none", false);//8
    large->template Add<BatchNorm>();
    large->template Add<LeakyReLU>(0.1f);

    large->template Add<MaxPooling>(2, 2, 2, 2);//9
    
    large->template Add<Convolution>(512, 3, 3, 1, 1, 1, 1, "none", false);//10
    large->template Add<BatchNorm>();
    large->template Add<LeakyReLU>(0.1f);

    large->template Add<Padding>(1, 0, 1, 0);
    large->template Add<MaxPooling>(2, 2, 1, 1);//11
    
    large->template Add<Convolution>(1024, 3, 3, 1, 1, 1, 1, "none", false);//12
    large->template Add<BatchNorm>();
    large->template Add<LeakyReLU>(0.1f);

    large->template Add<Convolution>(256, 3, 3, 1, 1, 1, 1, "none", false);//13
    large->template Add<BatchNorm>();
    large->template Add<LeakyReLU>(0.1f);

    large->template Add<Convolution>(512, 3, 3, 1, 1, 1, 1, "none", false);//14
    large->template Add<BatchNorm>();
    large->template Add<LeakyReLU>(0.1f);

    large->template Add<Convolution>(255, 1, 1, 1, 1, 1, 1, "none", true);//15

    large->template Add<YOLOv3Layer<MatType>>({3, 4, 5}, anchors);

    // small.template Add(ConvolutionalBlock(3, 16));//0
    // small.template Add(PoolingBlock(2, 2));//1
    // small.template Add(ConvolutionalBlock(3, 32));//2
    // small.template Add(PoolingBlock(2, 2));//3
    // small.template Add(ConvolutionalBlock(3, 64));//4
    // small.template Add(PoolingBlock(2, 2));//5
    // small.template Add(ConvolutionalBlock(3, 128));//6
    // small.template Add(PoolingBlock(2, 2));//7
    //
    // MultiLayer<MatType>* sequential = new MultiLayer<MatType>();
    // sequential->template Add(ConvolutionalBlock(3, 256));//8
    // sequential->template Add(PoolingBlock(2, 2));//9
    // sequential->template Add(ConvolutionalBlock(3, 512));//10
    // sequential->template Add(PoolingBlock(2, 1, 0.5, 0.5));//11
    // sequential->template Add(ConvolutionalBlock(3, 1024));//12
    // sequential->template Add(ConvolutionalBlock(1, 256, 1, 0));//13
    // sequential->template Add(ConvolutionalBlock(1, 128, 1, 0));//18
    // sequential->template Add(UpsampleBlock(2.0f));//19
    //
    // MultiLayer<MatType>* layer8= new MultiLayer<MatType>();
    // layer8->template Add(ConvolutionalBlock(3, 256));//8
    //
    // Concat* concatBlock = new Concat(2);
    // concatBlock->Add(sequential);
    // concatBlock->Add(layer8);
    //
    // small.template Add(concatBlock);//20
    // small.template Add(ConvolutionalBlock(3, 256));//21
    // small.template Add(ConvolutionalBlock(1, 255, 1, 0, false, false));//22
    // small.template Add(YOLOv3Block({0, 1, 2}));//23
    //
    // small.Reset();
    // large.Reset();
  }

  ~YoloV3Tiny() { delete large; }

  void Predict(const MatType& predictors,
	       MatType& results) {
    large->Forward(predictors, results);
  }

  void printModel() {
    printf("Large object detections:\n");
    for (size_t i = 0; i < large->Network().size(); i++) {
      auto layer = large->Network()[i];
      printLayer(layer, i);
    }
  }

private:
  MultiLayer<MatType>* UpsampleBlock(const double scaleFactor) {
    MultiLayer<MatType>* upsampleBlock = new MultiLayer<MatType>();
    std::vector<double> scaleFactors = {scaleFactor, scaleFactor};
    upsampleBlock->template Add<NearestInterpolation>(scaleFactors);
    return upsampleBlock;
  }

  MultiLayer<MatType>* large = new MultiLayer<MatType>();
  // MultiLayer<MatType>* small = new MultiLayer<MatType>();

  YoloV3TinyLoss<MatType> outputLayer;

  size_t inputWidth;
  size_t inputHeight;
  size_t inputChannels;
  size_t numClasses;

  std::vector<double> anchors;

  void printLayer(mlpack::Layer<arma::mat>* layer, size_t layerIndex) {
    int width = layer->OutputDimensions()[0];
    int height = layer->OutputDimensions()[1];
    int channels = layer->OutputDimensions()[2];
    int batch = layer->OutputDimensions()[3];
    printf("Layer %2d output shape:  %3d x %3d x %4d x %3d\n", (int)layerIndex, width, height, channels, batch);
  }
};

} // namespace models
} // namespace mlpack

#endif
