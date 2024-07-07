#ifndef MLPACK_METHODS_ANN_LAYER_ADD_HPP
#define MLPACK_METHODS_ANN_LAYER_ADD_HPP

#include <mlpack/prereqs.hpp>
#include "mlpack/methods/ann.hpp"

namespace mlpack {

template<typename MatType>
class YoloLayer : public Layer<MatType>
{
 public:
  YoloLayer();

  //! Clone the YoloLayer object. This handles polymorphism correctly.
  YoloLayer* Clone() const { return new YoloLayer(*this); }

  // Virtual destructor.
  virtual ~YoloLayer() { }

  //! Copy the given YoloLayer layer.
  YoloLayer(const YoloLayer& other);
  //! Take ownership of the given YoloLayer layer.
  YoloLayer(YoloLayer&& other);
  //! Copy the given YoloLayer layer.
  YoloLayer& operator=(const YoloLayer& other);
  //! Take ownership of the given YoloLayer layer.
  YoloLayer& operator=(YoloLayer&& other);

  /**
   * Forward pass: add the bias to the input.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const MatType& input, MatType& output) {}

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

  void ComputeOutputDimensions() {}

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */) {}

 private:
};

} // namespace mlpack

#endif
