#include <vector>

#include <opencv2/core/core.hpp>

#include "kf_tracker/KalmanFilterTrack.h"

class TracksManager
{
public:
  TracksManager() {};

  ~TracksManager() = default;

  void
  add_track(Eigen::Vector4f initial_centroid)
  {
    tracks.push_back(KalmanFilterTrack(initial_centroid));
  }

  std::vector<cv::Mat>
  predict()
  {
    std::vector<cv::Mat> predictions;
    for (const auto& kf : tracks)
    {
      predictions.push_back(kf.predict());
    }
    return predictions;
  }

private:
  std::vector<KalmanFilterTrack> tracks;
};
