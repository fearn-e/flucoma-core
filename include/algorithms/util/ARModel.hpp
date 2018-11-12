#pragma once

#include <Eigen/Eigen>
#include <algorithm>
#include <cmath>
#include <functional>

#include "../public/Windows.hpp"
#include "ConvolutionTools.hpp"
#include "Toeplitz.hpp"

namespace fluid {
namespace algorithm {

using Eigen::MatrixXd;
using Eigen::VectorXd;

class ARModel {

public:
  ARModel(size_t order, size_t iterations = 3, bool useWindow = true,
          double robustFactor = 3.0)
      : mParameters(VectorXd::Zero(order)), mVariance(0.0), mOrder(order),
        mIterations(iterations), mUseWindow(useWindow),
        mRobustFactor(robustFactor), mMinVariance(0.0) {}

  const double *getParameters() const { return mParameters.data(); }
  double variance() const { return mVariance; }
  size_t order() const { return mOrder; }

  void estimate(const double *input, int size) {
    if (mIterations)
      robustEstimate(input, size);
    else
      directEstimate(input, size, true);
  }

  double fowardPrediction(const double *input) {
    return modelPredict<std::negate<int>>(input);
  }

  double backwardPrediction(const double *input) {
    struct Identity {
      int operator()(int a) { return a; }
    };
    return modelPredict<Identity>(input);
  }

  double forwardError(const double *input) {
    return modelError<&ARModel::fowardPrediction>(input);
  }

  double backwardError(const double *input) {
    return modelError<&ARModel::backwardPrediction>(input);
  }

  void forwardErrorArray(double *errors, const double *input, int size) {
    modelErrorArray<&ARModel::forwardError>(errors, input, size);
  }

  void backwardErrorArray(double *errors, const double *input, int size) {
    modelErrorArray<&ARModel::backwardError>(errors, input, size);
  }

  void setMinVariance(double variance) { mMinVariance = variance; }

private:
  template <typename Op> double modelPredict(const double *input) {
    double estimate = 0.0;

    for (int i = 0; i < mOrder; i++)
      estimate += mParameters(i) * input[Op()(i + 1)];

    return estimate;
  }

  template <double (ARModel::*Method)(const double *)>
  double modelError(const double *input) {
    return input[0] - (this->*Method)(input);
  }

  template <double (ARModel::*Method)(const double *)>
  void modelErrorArray(double *errors, const double *input, int size) {
    for (int i = 0; i < size; i++)
      errors[i] = (this->*Method)(input + i);
  }

  void directEstimate(const double *input, int size, bool updateVariance) {
    std::vector<double> frame(size);

    if (mUseWindow) {
      if (mWindow.size() != size) {
        std::vector<double> newWindow =
            algorithm::windowFuncs[algorithm::WindowType::kHann](size);
        std::swap(mWindow, newWindow);
      }

      for (int i = 0; i < size; i++)
        frame[i] = input[i] * mWindow[i] * 2.0;
    } else
      std::copy(input, input + size, frame.data());

    VectorXd autocorrelation(size);
    algorithm::autocorrelateReal(autocorrelation.data(), frame.data(), size);

    // Resize to the desired order (only keep coefficients for up to the order
    // we need)

    double pN = mOrder < size ? autocorrelation(mOrder) : autocorrelation(0);
    autocorrelation.conservativeResize(mOrder);

    // Form a toeplitz matrix

    MatrixXd mat = toeplitz(autocorrelation);

    // Yule Walker

    autocorrelation(0) = pN;
    std::rotate(autocorrelation.data(), autocorrelation.data() + 1,
                autocorrelation.data() + mOrder);
    mParameters = mat.llt().solve(autocorrelation);

    if (updateVariance) {
      // Calculate variance

      double variance = mat(0, 0);

      for (int i = 0; i < mOrder - 1; i++)
        variance -= mParameters(i) * mat(0, i + 1);

      setVariance((variance - (mParameters(mOrder - 1) * pN)) / size);
    }
  }

  void robustEstimate(const double *input, int size) {
    std::vector<double> estimates(size + mOrder);

    // Calculate an intial estimate of parameters

    directEstimate(input, size, true);

    // Initialise Estimates

    for (int i = 0; i < mOrder + size; i++)
      estimates[i] = input[i - mOrder];

    // Variance

    robustVariance(estimates.data() + mOrder, input, size);

    // Iterate

    for (size_t iterations = mIterations; iterations--;)
      robustIteration(estimates.data() + mOrder, input, size);
  }

  double robustResidual(double input, double prediction, double cs) {
    return cs * psiFunction((input - prediction) / cs);
  }

  void robustVariance(double *estimates, const double *input, int size) {
    const double cs = mRobustFactor * sqrt(mVariance);
    double residualSqSum = 0.0;

    // Iterate to find new filtered input

    for (int i = 0; i < size; i++) {
      const double residual =
          robustResidual(input[i], fowardPrediction(estimates + i), cs);
      residualSqSum += residual * residual;
    }

    setVariance(residualSqSum / size);
  }

  void robustIteration(double *estimates, const double *input, int size) {
    const double cs = mRobustFactor * sqrt(mVariance);

    // Iterate to find new filtered input

    for (int i = 0; i < size; i++) {
      const double prediction = fowardPrediction(estimates);
      estimates[0] = prediction + robustResidual(input[i], prediction, cs);
    }

    // New parameters

    directEstimate(estimates, size, false);
    robustVariance(estimates, input, size);
  }

  void setVariance(double variance) {
    if (variance)
      variance = std::max(mMinVariance, variance);

    mVariance = variance;
  }

  // Huber PSI function

  double psiFunction(double x) {
    return fabs(x) > 1 ? std::copysign(1.0, x) : x;
  }

  VectorXd mParameters;
  double mVariance;

  std::vector<double> mWindow;

  bool mUseWindow;
  size_t mOrder;
  size_t mIterations;
  double mRobustFactor;
  double mMinVariance;
};

}; // namespace algorithm
}; // namespace fluid