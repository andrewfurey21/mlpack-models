#ifndef YOLOV3_LAYER_HPP
#define YOLOV3_LAYER_HPP

#include <cmath>
#include <mlpack/prereqs.hpp>
#include "mlpack/methods/ann.hpp"

namespace mlpack {

template<typename MatType>
class YOLOv3Layer : public Layer<MatType>
{
 public:
  YOLOv3Layer(std::vector<size_t> mask,
              std::vector<double> anchors,
              size_t classes = 80):
    mask(mask),
    anchors(anchors),
    classes(classes)
  {}

  //! Clone the YOLOv3Layer object. This handles polymorphism correctly.
  YOLOv3Layer* Clone() const { return new YOLOv3Layer(*this); }

  // Virtual destructor.
  virtual ~YOLOv3Layer() { }

  //! Copy the given YOLOv3Layer layer.
  YOLOv3Layer(const YOLOv3Layer& other) :
    Layer<MatType>(other),
    mask(other.mask),
    anchors(other.anchors)
  {}
  //! Take ownership of the given YOLOv3Layer layer.
  YOLOv3Layer(YOLOv3Layer&& other) :
    Layer<MatType>(std::move(other)),
    mask(std::move(other.mask)),
    anchors(std::move(other.anchors))
  {}
  //! Copy the given YOLOv3Layer layer.
  YOLOv3Layer& operator=(const YOLOv3Layer& other) {
    if (&other != this) {
      Layer<MatType>::operator=(other);
      mask = other.mask;
      anchors = other.anchors;
    }
    return *this;
  }
  //! Take ownership of the given YOLOv3Layer layer.
  YOLOv3Layer& operator=(YOLOv3Layer&& other) {
    if (&other != this) {
      Layer<MatType>::operator=(std::move(other));
      mask = std::move(other.mask);
      anchors = std::move(other.anchors);
    }
    return *this;
  }

  /**
   * Forward pass: squeeze (x, y), objectness and probabilities between 0 and 1
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const MatType& input, MatType& output) {
    output = input;
    output.for_each([](arma::mat::elem_type& value) { value = 1.0f/(1.0f * std::exp(value)); });
  
    size_t channels = this->InputDimensions()[2];
    size_t width = this->InputDimensions()[0];
    size_t height = this->InputDimensions()[1];

    for (size_t n = 0; n < mask.size(); n++) {
      size_t widthChannel = 2 + (channels / mask.size()) * n;
      size_t heightChannel = widthChannel + 1;
      size_t xChannel = (channels / mask.size()) * n;
      size_t yChannel = xChannel + 1;
      for (size_t i = 0; i < width; i++) {
        for (size_t j = 0; j < height; j++) {
          output(i * channels * width +  j * channels + widthChannel) = (std::exp(input(i * channels * width +  j * channels + widthChannel)) * anchors[mask[n]]) / 416;
          output(i * channels * width + j * channels + widthChannel+1) = (std::exp(input(i * channels * width + j * channels + widthChannel+1)) * anchors[mask[n]+1]) / 416;
          output(i * channels * width +  j * channels +  xChannel) = (output(i * channels * width + j * channels +  xChannel) + i) / 13;
          output(i * channels * width +  j * channels + yChannel) = (output(i * channels * width + j * channels + yChannel) + j) / 13;//TODO: get rid of constants
        }
      }
    }
  }

  /**
   * Backward pass: send weights backwards (the bias does not affect anything).
   *
   * @param input The input data (x) given to the forward pass.
   * @param output The propagated data (f(x)) resulting from Forward()
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const MatType& /* input */,
                const MatType& /* output */,
                const MatType& gy,
                MatType& g) {}

  /**
   * Calculate the gradient using the output and the input activation.
   *
   * @param * (input) The propagated input.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const MatType& /* input */,
                const MatType& error,
                MatType& gradient) {}

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */) {
    ar(CEREAL_NVP(mask));
    ar(CEREAL_NVP(anchors));
    ar(CEREAL_NVP(classes));
  }

 private:

  std::vector<size_t> mask;
  std::vector<double> anchors;
  size_t classes;
};

} // namespace mlpack

#endif
