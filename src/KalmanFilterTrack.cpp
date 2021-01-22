#include "kf_tracker/KalmanFilterTrack.h"

kf_tracking::KalmanFilterTrack::KalmanFilterTrack()
  : KF(state_dim_, measurement_dim_, ctrl_dim_, CV_32F)
{
  // initialize KF

  // state transition matrix A  // TODO this is the system equations. verify it makes sense!
  float dvx = 0.01f; // 1.0
  float dvy = 0.01f; // 1.0
  float dx = 1.0f;
  float dy = 1.0f;
  KF.transitionMatrix = (cv::Mat_<float>(4, 4) << dx, 0, 1, 0, 0, dy, 0, 1, 0, 0, dvx, 0, 0, 0, 0, dvy);

  // measurement matrix (H)  TODO verify!
  cv::setIdentity(KF.measurementMatrix);

  // Process Noise Covariance Matrix Q
  // [ Ex 0  0    0 0    0 ]
  // [ 0  Ey 0    0 0    0 ]
  // [ 0  0  Ev_x 0 0    0 ]
  // [ 0  0  0    1 Ev_y 0 ]
  //// [ 0  0  0    0 1    Ew ]
  //// [ 0  0  0    0 0    Eh ]
  float sigmaQ = 0.01;
  setIdentity(KF.processNoiseCov, cv::Scalar::all(sigmaQ));

  // measurement noise covariance matrix R
  float sigmaR = 0.1;
  cv::setIdentity(KF.measurementNoiseCov, cv::Scalar(sigmaR));
}

kf_tracking::KalmanFilterTrack::KalmanFilterTrack(Eigen::Vector4f centroid)
  : KalmanFilterTrack()
{
  // Set initial state
  KF.statePre.at<float>(0) = centroid[0];
  KF.statePre.at<float>(1) = centroid[1];
  KF.statePre.at<float>(2) = 0; // initial v_x
  KF.statePre.at<float>(3) = 0; // initial v_y
}
