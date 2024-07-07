/**
 * @file methods/ann/layer/add.hpp
 * @author Marcus Edel
 *
 * Definition of the Add class that applies a bias term to the incoming data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ADD_HPP
#define MLPACK_METHODS_ANN_LAYER_ADD_HPP

#include <mlpack/prereqs.hpp>
#include "mlpack/methods/ann.hpp"

namespace mlpack {

template<typename MatType>
class Yolo : public Layer<MatType>
{
 public:
  Yolo();

  //! Clone the Yolo object. This handles polymorphism correctly.
  Yolo* Clone() const { return new Yolo(*this); }

  // Virtual destructor.
  virtual ~Yolo() { }

  //! Copy the given Yolo layer.
  Yolo(const Yolo& other);
  //! Take ownership of the given Yolo layer.
  Yolo(Yolo&& other);
  //! Copy the given Yolo layer.
  Yolo& operator=(const Yolo& other);
  //! Take ownership of the given Yolo layer.
  Yolo& operator=(Yolo&& other);

  /**
   * Forward pass: add the bias to the input.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const MatType& input, MatType& output);

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
                MatType& g);

  /**
   * Calculate the gradient using the output and the input activation.
   *
   * @param * (input) The propagated input.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const MatType& /* input */,
                const MatType& error,
                MatType& gradient);

  void ComputeOutputDimensions();

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
};

} // namespace mlpack

#endif
