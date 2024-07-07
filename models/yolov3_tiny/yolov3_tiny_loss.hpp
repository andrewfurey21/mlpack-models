#ifndef YOLO_LOSS_HPP
#define YOLO_LOSS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

template<typename MatType = arma::mat>
class YoloV3TinyLoss
{
 public:
  YoloV3TinyLoss();

  typename MatType::elem_type Forward(const MatType& prediction,
                                      const MatType& target);

  void Backward(const MatType& prediction,
                const MatType& target,
                MatType& loss);

  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
};


} // namespace mlpack

#endif
