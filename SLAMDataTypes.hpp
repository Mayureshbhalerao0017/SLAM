#pragma once
#include <vector>

namespace slam_core {

struct Observation {
    double range;
    double bearing;
    int class_id; 
};

struct Pose2D {
    double x;
    double y;
    double yaw;
};

struct Landmark {
    double x;
    double y;
    int class_id;
    double confidence; // Number of hits or probability
};

inline double wrapToPi(double angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

} 
