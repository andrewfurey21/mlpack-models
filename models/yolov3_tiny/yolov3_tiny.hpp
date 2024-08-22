#ifndef MODELS_MODELS_YOLOV3_TINY_YOLOV3_TINY_HPP
#define MODELS_MODELS_YOLOV3_TINY_YOLOV3_TINY_HPP

#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/layer/batch_norm.hpp>
#include <mlpack/methods/ann/layer/convolution.hpp>
#include <mlpack/methods/ann/layer/leaky_relu.hpp>
#include <mlpack/methods/ann/layer/max_pooling.hpp>
#include <mlpack/methods/ann/layer/multi_layer.hpp>
#include <mlpack/methods/ann/layer/concat.hpp>
#include <mlpack/methods/ann/layer/nearest_interpolation.hpp>
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
  YoloV3Tiny(
	     const size_t inputWidth = 416,
	     const size_t inputHeight = 416,
	     const size_t inputChannels = 3,
	     const size_t numClasses = 80
	     ) :
	inputWidth(inputWidth),
	inputHeight(inputHeight),
	inputChannels(inputChannels),
	numClasses(numClasses),
	batchSize(1)
  {
    large = new FFN<YoloV3TinyLoss<MatType>, RandomInitialization>();
    large->InputDimensions() = std::vector<size_t>({inputWidth, inputHeight, inputChannels, batchSize});

    //0
    large->template Add<Convolution>(16, 3, 3, 1, 1, 1, 1, "none", false);//0
    large->template Add<BatchNorm>();//1
    large->template Add<LeakyReLU>(0.1f);//2

    //1
    large->template Add<MaxPooling>(2, 2, 2, 2);//3

    //2
    large->template Add<Convolution>(32, 3, 3, 1, 1, 1, 1, "none", false);//4
    large->template Add<BatchNorm>();//5
    large->template Add<LeakyReLU>(0.1f);//6

    //3
    large->template Add<MaxPooling>(2, 2, 2, 2);//7
   
    //4
    large->template Add<Convolution>(64, 3, 3, 1, 1, 1, 1, "none", false);//8
    large->template Add<BatchNorm>();//9
    large->template Add<LeakyReLU>(0.1f);//10

    //5
    large->template Add<MaxPooling>(2, 2, 2, 2);//11
   
    //6
    large->template Add<Convolution>(128, 3, 3, 1, 1, 1, 1, "none", false);//12
    large->template Add<BatchNorm>();//13
    large->template Add<LeakyReLU>(0.1f);//14
   
    //7
    large->template Add<MaxPooling>(2, 2, 2, 2);//15
   
    //8
    large->template Add<Convolution>(256, 3, 3, 1, 1, 1, 1, "none", false);//16
    large->template Add<BatchNorm>();//17
    large->template Add<LeakyReLU>(0.1f);//18

    //9
    large->template Add<MaxPooling>(2, 2, 2, 2);//19
   
    //10
    large->template Add<Convolution>(512, 3, 3, 1, 1, 1, 1, "none", false);//20
    large->template Add<BatchNorm>();//21
    large->template Add<LeakyReLU>(0.1f);//22

    //11
    large->template Add<Padding>(1, 0, 1, 0);//23
    large->template Add<MaxPooling>(2, 2, 1, 1);//24
   
    //12
    large->template Add<Convolution>(1024, 3, 3, 1, 1, 1, 1, "none", false);//25
    large->template Add<BatchNorm>();//26
    large->template Add<LeakyReLU>(0.1f);//27

    //13
    large->template Add<Convolution>(256, 1, 1, 1, 1, 0, 0, "none", false);//28
    large->template Add<BatchNorm>();//29
    large->template Add<LeakyReLU>(0.1f);//30

    //14
    large->template Add<Convolution>(512, 3, 3, 1, 1, 1, 1, "none", false);//31
    large->template Add<BatchNorm>();//32
    large->template Add<LeakyReLU>(0.1f);//33

    //15
    large->template Add<Convolution>(255, 1, 1, 1, 1, 0, 0, "none", true);//34

    //16
    std::vector<size_t> largeMask = {3, 4, 5};
    large->template Add<YOLOv3Layer<MatType>>();//35
    large->Reset();

    // Upsampled detections
    // small = new FFN...
    //small->InputDimensions() = std::vector<size_t>({inputWidth, inputHeight, inputChannels, batchSize});
   //  MultiLayer<MatType>* layer19 = new MultiLayer<MatType>();
   //  layer19->template Add<Convolution>(16, 3, 3, 1, 1, 1, 1, "none", false);//0
   //  layer19->template Add<BatchNorm>();
   //  layer19->template Add<LeakyReLU>(0.1f);
   //  
   //  layer19->template Add<MaxPooling>(2, 2, 2, 2);//1
   //  
   //  layer19->template Add<Convolution>(32, 3, 3, 1, 1, 1, 1, "none", false);//2
   //  layer19->template Add<BatchNorm>();
   //  layer19->template Add<LeakyReLU>(0.1f);
   // 
   //  layer19->template Add<MaxPooling>(2, 2, 2, 2);//3
   // 
   //  layer19->template Add<Convolution>(64, 3, 3, 1, 1, 1, 1, "none", false);//4
   //  layer19->template Add<BatchNorm>();
   //  layer19->template Add<LeakyReLU>(0.1f);
   // 
   //  layer19->template Add<MaxPooling>(2, 2, 2, 2);//5
   // 
   //  layer19->template Add<Convolution>(128, 3, 3, 1, 1, 1, 1, "none", false);//6
   //  layer19->template Add<BatchNorm>();
   //  layer19->template Add<LeakyReLU>(0.1f);
   // 
   //  layer19->template Add<MaxPooling>(2, 2, 2, 2);//7
   // 
   //  layer19->template Add<Convolution>(256, 3, 3, 1, 1, 1, 1, "none", false);//8
   //  layer19->template Add<BatchNorm>();
   //  layer19->template Add<LeakyReLU>(0.1f);
   //
   //  layer19->template Add<MaxPooling>(2, 2, 2, 2);//9
   //
   //  layer19->template Add<Convolution>(512, 3, 3, 1, 1, 1, 1, "none", false);//10
   //  layer19->template Add<BatchNorm>();
   //  layer19->template Add<LeakyReLU>(0.1f);
   //
   //  layer19->template Add<Padding>(1, 0, 1, 0);
   //  layer19->template Add<MaxPooling>(2, 2, 1, 1);//11
   //
   //  layer19->template Add<Convolution>(1024, 3, 3, 1, 1, 1, 1, "none", false);//12
   //  layer19->template Add<BatchNorm>();
   //  layer19->template Add<LeakyReLU>(0.1f);
   //
   //  layer19->template Add<Convolution>(256, 3, 3, 1, 1, 1, 1, "none", false);//13
   //  layer19->template Add<BatchNorm>();
   //  layer19->template Add<LeakyReLU>(0.1f);
   //
   //  layer19->template Add<Convolution>(128, 1, 1, 1, 1, 0, 0, "none", false);//18
   //  layer19->template Add<BatchNorm>();
   //  layer19->template Add<LeakyReLU>(0.1f);
   //
   //  std::vector<double> scaleFactors = {2.0f, 2.0f};
   //  layer19->template Add<NearestInterpolation>(scaleFactors);
   //  
   //  //Layer8
   //  MultiLayer<MatType>* layer8 = new MultiLayer<MatType>();
   //  layer8->template Add<Convolution>(16, 3, 3, 1, 1, 1, 1, "none", false);//0
   //  layer8->template Add<BatchNorm>();
   //  layer8->template Add<LeakyReLU>(0.1f);
   //  
   //  layer8->template Add<MaxPooling>(2, 2, 2, 2);//1
   //  
   //  layer8->template Add<Convolution>(32, 3, 3, 1, 1, 1, 1, "none", false);//2
   //  layer8->template Add<BatchNorm>();
   //  layer8->template Add<LeakyReLU>(0.1f);
   // 
   //  layer8->template Add<MaxPooling>(2, 2, 2, 2);//3
   // 
   //  layer8->template Add<Convolution>(64, 3, 3, 1, 1, 1, 1, "none", false);//4
   //  layer8->template Add<BatchNorm>();
   //  layer8->template Add<LeakyReLU>(0.1f);
   // 
   //  layer8->template Add<MaxPooling>(2, 2, 2, 2);//5
   // 
   //  layer8->template Add<Convolution>(128, 3, 3, 1, 1, 1, 1, "none", false);//6
   //  layer8->template Add<BatchNorm>();
   //  layer8->template Add<LeakyReLU>(0.1f);
   // 
   //  layer8->template Add<MaxPooling>(2, 2, 2, 2);//7
   // 
   //  layer8->template Add<Convolution>(256, 3, 3, 1, 1, 1, 1, "none", false);//8
   //  layer8->template Add<BatchNorm>();
   //  layer8->template Add<LeakyReLU>(0.1f);
   //
   //  ConcatType<MatType>* layer20 = new ConcatType<MatType>(2);
   //  layer20->template Add(layer19);
   //  layer20->template Add(layer8);
   //
   //  small->template Add(layer20);
   //
   //  small->template Add<Convolution>(256, 3, 3, 1, 1, 1, 1, "none", false);//21
   //  small->template Add<BatchNorm>();
   //  small->template Add<LeakyReLU>(0.1f);
   //
   //  small->template Add<Convolution>(255, 1, 1, 1, 1, 0, 0, "none", true);//22
   //
   //  std::vector<size_t> smallMask = {0, 1, 2};
   //  small->template Add<YOLOv3Layer<MatType>>();
   //
   //  small->Reset();
  }

  ~YoloV3Tiny() { };

  void Predict(const MatType& predictors,
	       MatType& results) {
    large->Predict(predictors, results);
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
  
  FFN<YoloV3TinyLoss<MatType>, RandomInitialization>* large;
  FFN<YoloV3TinyLoss<MatType>, RandomInitialization>* small;

  YoloV3TinyLoss<MatType> outputLayer;//for training only.

  size_t inputWidth;
  size_t inputHeight;
  size_t inputChannels;
  size_t numClasses;
  size_t batchSize;

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
