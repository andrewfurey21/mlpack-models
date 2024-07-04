/**
 * @author Andrew Furey
 *
 * Definition of the YOLO layer for object detection
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef YOLO_LAYER_HPP
#define YOLO_LAYER_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann.hpp>

namespace mlpack {

/*
 * Implementation of the yolo layer for object detection. 
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType>
class Yolo : public Layer<MatType>
{
 public:
  /**
   * Create the Yolo object.  The output size of the layer will be the same
   * as the input size.
   */
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
   * Forward pass
   */
  void Forward(const MatType& input, MatType& output);

  /**
   * Backward pass
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

  //! Return the weights of the network.
  const MatType& Parameters() const { return weights; }
  //! Modify the weights of the network.
  MatType& Parameters() { return weights; }

  //! Compute the output dimensions of the layer, based on the internal values
  //! of `InputDimensions()`.
  void ComputeOutputDimensions();

  //! Set the weights of the layer to use the given memory.
  void SetWeights(const MatType& weightsIn);

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored weight object.
  MatType weights;
}; // Yolo class
} // namespace mlpack

#endif
