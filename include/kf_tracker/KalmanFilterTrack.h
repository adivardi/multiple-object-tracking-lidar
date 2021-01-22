#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace kf_tracking
{
class KalmanFilterTrack
{
public:
  KalmanFilterTrack();

  KalmanFilterTrack(Eigen::Vector4f centroid);

  ~KalmanFilterTrack() = default;

  // vector<Vec2f> update(vector<Vec2f>);

  const cv::Mat&
  predict(const cv::Mat& control = cv::Mat())
  {
    return KF.predict(control);
  }

private:
  // params
  int state_dim_ = 4; // [x,y,v_x,v_y]
  int measurement_dim_ = 2; // [z_x,z_y,z_w,z_h]
  int ctrl_dim_ = 0;

  int id;
  cv::KalmanFilter KF;

  // TODO add ros publisher for cluster associated or and results or something??? otherwise this is a bit useless
};
} // namespace kf_tracking
