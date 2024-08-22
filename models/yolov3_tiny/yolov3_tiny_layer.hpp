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
  YOLOv3Layer()
  {}

  //! Clone the YOLOv3Layer object. This handles polymorphism correctly.
  YOLOv3Layer* Clone() const { return new YOLOv3Layer(*this); }

  // Virtual destructor.
  virtual ~YOLOv3Layer() { }

  //! Copy the given YOLOv3Layer layer.
  YOLOv3Layer(const YOLOv3Layer& other) :
    Layer<MatType>(other)
  {}
  //! Take ownership of the given YOLOv3Layer layer.
  YOLOv3Layer(YOLOv3Layer&& other) :
    Layer<MatType>(std::move(other))
  {}
  //! Copy the given YOLOv3Layer layer.
  YOLOv3Layer& operator=(const YOLOv3Layer& other) {
    if (&other != this) {
      Layer<MatType>::operator=(other);
    }
    return *this;
  }
  //! Take ownership of the given YOLOv3Layer layer.
  YOLOv3Layer& operator=(YOLOv3Layer&& other) {
    if (&other != this) {
      Layer<MatType>::operator=(std::move(other));
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
   * Backward pass: send weights backwards
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
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */) {}
};

} // namespace mlpack

#endif
