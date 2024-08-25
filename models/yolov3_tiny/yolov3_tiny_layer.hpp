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
  YOLOv3Layer(std::vector<std::pair<size_t, size_t>> anchors,
              size_t width = 416,
              size_t height = 416,
              size_t classes = 80) : 
    anchors(anchors),
    width(width),
    height(height),
    classes(classes)
  {}

  //! Clone the YOLOv3Layer object. This handles polymorphism correctly.
  YOLOv3Layer* Clone() const { return new YOLOv3Layer(*this); }

  // Virtual destructor.
  virtual ~YOLOv3Layer() { }

  //! Copy the given YOLOv3Layer layer.
  YOLOv3Layer(const YOLOv3Layer& other) :
    Layer<MatType>(other),
    anchors(other.anchors),
    width(width),
    height(height),
    classes(classes)
  {}
  //! Take ownership of the given YOLOv3Layer layer.
  YOLOv3Layer(YOLOv3Layer&& other) :
    Layer<MatType>(std::move(other)),
    anchors(std::move(other.anchors)),
    width(std::move(other.width)),
    height(std::move(other.height)),
    classes(std::move(other.classes))
  {}
  //! Copy the given YOLOv3Layer layer.
  YOLOv3Layer& operator=(const YOLOv3Layer& other) {
    if (&other != this) {
      Layer<MatType>::operator=(other);
      anchors = other.anchors;
      width = other.width;
      height = other.height;
      classes = other.classes;
    }
    return *this;
  }
  //! Take ownership of the given YOLOv3Layer layer.
  YOLOv3Layer& operator=(YOLOv3Layer&& other) {
    if (&other != this) {
      Layer<MatType>::operator=(std::move(other));
      anchors = other.anchors;
      width = other.width;
      height = other.height;
      classes = other.classes;
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

    auto f = [](const arma::mat::elem_type& value) { 
      return 1.0f/(1.0f * std::exp(-value));
    };

    output = input;
    output.submat(0, 0, width * height * 2 - 1, batchSize - 1).transform(f);//x, y
    output.submat(width * height * 4, 0, output.n_rows - 1, batchSize - 1).transform(f);//obj, probs

    // for (size_t mask = 0; mask < anchors.size(); mask++) {
    //   size_t anchorWidth = anchors[mask].first;
    //   size_t anchorHeight = anchors[mask].second;
    //   
    //   auto x = [](const arma::mat::elem_type& value) {
    //     
    //   };
    //   
    // }
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

    // auto f = [](const arma::mat::elem_type& value) { return 1.0f/(1.0f * std::exp(-value));};
    // auto df = [=](const arma::mat::elem_type& value) {
    //   return f(value) * (1 - f(value));
    // };
  }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */) {}

private:
  size_t outputIndex(size_t mask, size_t cell, size_t offset, size_t width, size_t height, size_t classes) {
    assert(mask >= 0 && mask < 3);//3 should be const?
    assert(cell >= 0 && cell < width * height);
    assert(offset >= 0 && offset < classes);
    return mask * width * height * (4 + 1 + classes) + offset * width * height + cell;
  }
  // Anchors are (width,height) pairs
  std::vector<std::pair<size_t, size_t>> anchors;
  // Width of input image
  size_t width;
  // Height of input image
  size_t height;
  // Number of classes
  size_t classes;
};

} // namespace mlpack

#endif
