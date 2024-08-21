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
   * @param input
   * @param output
   */
  void Forward(const MatType& input, MatType& output) {
    size_t width = this->InputDimensions()[0];
    size_t height = this->InputDimensions()[1];
    size_t channels = this->InputDimensions()[2];
    size_t batchSize = this->InputDimensions()[3];

    auto f = [](const arma::mat::elem_type& value) { return 1.0f/(1.0f * std::exp(-value));};

    output = input;
    output.submat(0, 0, width * height * 2 - 1, batchSize - 1).transform(f);//x, y
    output.submat(width * height * 4, 0, output.n_rows - 1, batchSize - 1).transform(f);//obj, probs
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
                MatType& g) {
  }

  /**
   * Calculate the gradient using the output and the input activation.
   *
   * @param * (input) The propagated input.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const MatType& /* input */,
                const MatType& error,
                MatType& gradient) {
  }

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
