#ifndef YOLO_LOSS_HPP
#define YOLO_LOSS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

template<typename MatType = arma::mat>
class YoloV3TinyLoss
{
 public:
  YoloV3TinyLoss(size_t width = 13,
                 size_t height = 13,
                 double confidence = 0.5, 
                 double localizaiton = 0.9, 
                 double classifcation = 0.6,
                 size_t classes = 80) :
    width(width),
    height(height),
    confidence(confidence),
    localization(localization),
    classification(classification),
    classes(classes)
                                    {}

  typename MatType::elem_type Forward(const MatType& prediction,
                                      const MatType& target) {
    
    typename MatType::elem_type localizationLoss = 0;
    typename MatType::elem_type classificationLoss = 0;
    typename MatType::elem_type confidenceLoss = 0;

    MatType xyPred = prediction.submat(0, prediction.n_cols - 1, width * height * 2 - 1, prediction.n_cols - 1);
    MatType xyTarg = target.submat(0, prediction.n_cols - 1, width * height * 2 - 1, target.n_cols - 1);

    MatType whPred = prediction.submat(width * height * 2, prediction.n_cols - 1, width * height * 4 - 1, prediction.n_cols - 1);
    MatType whTarg = target.submat(width * height * 2, prediction.n_cols - 1, width * height * 4 - 1, target.n_cols - 1);
    localizationLoss = localizationLossFn.Forward(xyPred, xyTarg) + localizationLossFn.Forward(whPred, whTarg);

    MatType confidencePred = prediction.submat(width * height * 4, prediction.n_cols - 1, width * height * 5 - 1, prediction.n_cols);
    MatType confidenceTarg = target.submat(width * height * 4, target.n_cols - 1, width * height * 5 - 1, target.n_cols - 1);
    confidenceLoss = confidenceLossFn.Forward(confidencePred, confidenceTarg);

    MatType classificationPred = prediction.submat(width * height * 5, prediction.n_cols - 1, prediction.n_rows - 1, prediction.n_cols);
    MatType classificationTarg = target.submat(width * height * 5, target.n_cols - 1, target.n_rows - 1, target.n_cols - 1);
    classificationLoss = classificationLossFn(classificationPred, classificationTarg);

    typename MatType::elem_type loss = localization * localizationLoss + 
                                       classification * classificationLoss +
                                       confidence * confidenceLoss;
    return loss;
  }

  void Backward(const MatType& prediction,
                const MatType& target,
                MatType& loss) {

  }

  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */) {
    ar(CEREAL_NVP(width));
    ar(CEREAL_NVP(height));
    ar(CEREAL_NVP(classes));
  }

 private:
  double classification;
  double localization;
  double confidence;

  BCELossType<> confidenceLossFn;
  MeanSquaredErrorType<> localizationLossFn;
  MultiLabelSoftMarginLossType<> classificationLossFn;

  size_t classes;
  size_t width;
  size_t height;
};


} // namespace mlpack

#endif
