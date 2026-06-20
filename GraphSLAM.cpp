#pragma once
#include "SLAMDataTypes.hpp"
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <gtsam/inference/Symbol.h>
#include <map>
#include <set>
#include <memory>

namespace slam_core {

class GraphSLAMCore {
public:
    GraphSLAMCore() {
        gtsam::ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.01; 
        isam2_ = std::make_unique<gtsam::ISAM2>(parameters);

        auto base_cone_noise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(0.05, 0.05));
        huber_noise_ = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(1.345), base_cone_noise);
        
        odom_noise_ = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(0.08, 0.08, 0.03));

        gtsam::NonlinearFactorGraph initial_graph;
        gtsam::Values initial_values;
        initial_graph.addPrior(gtsam::Symbol('x', 0), gtsam::Pose2(0, 0, 0), gtsam::noiseModel::Isotropic::Sigma(3, 1e-6));
        initial_values.insert(gtsam::Symbol('x', 0), gtsam::Pose2(0, 0, 0));
        isam2_->update(initial_graph, initial_values);

        opt_pose_ = gtsam::Pose2(0, 0, 0);
    }

    void addOdometry(const Pose2D& local_movement) {
        gtsam::Pose2 movement(local_movement.x, local_movement.y, local_movement.yaw);
        int prev_idx = pose_count_++;
        
        batch_factors_.add(gtsam::BetweenFactor<gtsam::Pose2>(
            gtsam::Symbol('x', prev_idx), gtsam::Symbol('x', pose_count_), movement, odom_noise_));
        
        gtsam::Values current_estimate;
        try { current_estimate = isam2_->calculateEstimate(); } catch (...) { }
        
        gtsam::Pose2 last_p = current_estimate.exists(gtsam::Symbol('x', prev_idx)) ? 
                              current_estimate.at<gtsam::Pose2>(gtsam::Symbol('x', prev_idx)) : opt_pose_;
        
        current_pose_ = last_p.compose(movement); 
        batch_values_.insert(gtsam::Symbol('x', pose_count_), current_pose_);
    }

    void addObservations(const std::vector<Observation>& observations, bool is_loop_closed) {
        std::set<int> matched_landmarks;
        gtsam::Values current_estimate;
        try { current_estimate = isam2_->calculateEstimate(); } catch (...) {}

        for (const auto& z : observations) {
            double dist_gate = (z.class_id == 2) ? 8.5 : (is_loop_closed ? 4.5 : 2.2); 
            int best_id = -1;
            double best_score = 1e9;
            
            gtsam::Point2 obs_global = current_pose_.transformFrom(gtsam::Point2(z.range * std::cos(z.bearing), z.range * std::sin(z.bearing)));

            auto check_values = [&](const gtsam::Values& vals) {
                for (auto const& [key, val] : vals) {
                    if (gtsam::Symbol(key).chr() == 'l' && class_id_map_[gtsam::Symbol(key).index()] == z.class_id) {
                        int l_idx = gtsam::Symbol(key).index();
                        if (matched_landmarks.count(l_idx)) continue; 

                        gtsam::Point2 landmark = vals.at<gtsam::Point2>(key);
                        double dist = (landmark - obs_global).norm();

                        if (dist < dist_gate) { 
                            gtsam::Vector2 delta = landmark - current_pose_.translation();
                            double expected_bearing = std::atan2(delta.y(), delta.x());
                            double bearing_error = std::abs(std::atan2(std::sin(expected_bearing - (current_pose_.theta() + z.bearing)), 
                                                                       std::cos(expected_bearing - (current_pose_.theta() + z.bearing))));
                            
                            double bearing_penalty = bearing_error > (M_PI / 2.0) ? 5.0 : 1.0;
                            double score = dist * bearing_penalty;

                            if (score < best_score) { best_score = score; best_id = l_idx; }
                        }
                    }
                }
            };

            check_values(current_estimate);
            check_values(batch_values_);

            if (best_id == -1) {
                if (is_loop_closed && z.class_id != 2) continue; // Lockdown on lap 2
                best_id = next_landmark_id_++;
                batch_values_.insert(gtsam::Symbol('l', best_id), current_pose_.transformFrom(gtsam::Point2(z.range * std::cos(z.bearing), z.range * std::sin(z.bearing))));
                class_id_map_[best_id] = z.class_id;
            }
            
            matched_landmarks.insert(best_id); 
            batch_factors_.add(gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Point2>(
                gtsam::Symbol('x', pose_count_), gtsam::Symbol('l', best_id), gtsam::Rot2::fromAngle(z.bearing), z.range, huber_noise_));
        }
    }

    void optimize(bool trigger_shockwave) {
        try {
            isam2_->update(batch_factors_, batch_values_);
            if (trigger_shockwave) {
                isam2_->update(); isam2_->update(); isam2_->update(); // Absorb shock
            }
            gtsam::Values current_estimate = isam2_->calculateEstimate();
            opt_pose_ = current_estimate.at<gtsam::Pose2>(gtsam::Symbol('x', pose_count_));
        } catch (...) {}

        batch_factors_.resize(0);
        batch_values_.clear();
    }

    Pose2D getLatestOptimizedPose() const {
        return {opt_pose_.x(), opt_pose_.y(), opt_pose_.theta()};
    }

private:
    std::unique_ptr<gtsam::ISAM2> isam2_;
    gtsam::NonlinearFactorGraph batch_factors_;
    gtsam::Values batch_values_;
    gtsam::Pose2 opt_pose_, current_pose_;
    gtsam::SharedNoiseModel odom_noise_, huber_noise_;
    
    int pose_count_ = 0, next_landmark_id_ = 0;
    std::map<int, int> class_id_map_;
};

} // namespace slam_core
