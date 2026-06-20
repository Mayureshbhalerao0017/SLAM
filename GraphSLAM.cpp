#include <memory>
#include <vector>
#include <cmath>
#include <algorithm>
#include <mutex>
#include <map>
#include <set>

// ROS 2 Core
#include <rclcpp/rclcpp.hpp>
#include <eufs_msgs/msg/cone_array.hpp>
#include <eufs_msgs/msg/cone_array_with_covariance.hpp>
#include <nav_msgs/msg/odometry.hpp>

// Visualization and TF
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

// GTSAM Core
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <gtsam/inference/Symbol.h>

using namespace gtsam;

class EUFSGraphSlam : public rclcpp::Node {
public:
    EUFSGraphSlam() : Node("eufs_graph_slam_node") {
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.01; 
        isam2 = std::make_unique<ISAM2>(parameters);

        auto base_cone_noise = noiseModel::Diagonal::Sigmas(Vector2(0.05, 0.05));
        huber_noise = noiseModel::Robust::Create(
            noiseModel::mEstimator::Huber::Create(1.345), base_cone_noise);
        
        odom_noise = noiseModel::Diagonal::Sigmas(Vector3(0.08, 0.08, 0.03));

        NonlinearFactorGraph initial_graph;
        Values initial_values;
        initial_graph.addPrior(Symbol('x', 0), Pose2(0, 0, 0), noiseModel::Isotropic::Sigma(3, 1e-6));
        initial_values.insert(Symbol('x', 0), Pose2(0, 0, 0));
        isam2->update(initial_graph, initial_values);

        opt_pose_ = Pose2(0, 0, 0);
        latest_predicted_pose_ = Pose2(0, 0, 0); 
        map_T_odom_ = Pose2(0, 0, 0); 

        cones_pub = create_publisher<eufs_msgs::msg::ConeArray>("/planning/cones", 10);
        marker_pub = create_publisher<visualization_msgs::msg::MarkerArray>("/slam/cone_markers", 10);
        slam_odom_pub = create_publisher<nav_msgs::msg::Odometry>("/slam/odom", 50);
        tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        odom_sub = create_subscription<nav_msgs::msg::Odometry>("/odometry/filtered", 50, 
            std::bind(&EUFSGraphSlam::odomCallback, this, std::placeholders::_1));
        cone_sub = create_subscription<eufs_msgs::msg::ConeArrayWithCovariance>("/ground_truth/cones", 10, 
            std::bind(&EUFSGraphSlam::coneCallback, this, std::placeholders::_1));
            
        RCLCPP_INFO(this->get_logger(), "GraphSLAM Online. Zero-Latency TF Architecture Active.");
    }

private:
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mtx);
        current_odom_pose_ = msg->pose.pose;
        current_odom_twist_ = msg->twist.twist;
        rclcpp::Time stamp = msg->header.stamp;
        odom_updated_ = true;

        double current_yaw = std::atan2(2.0 * (msg->pose.pose.orientation.w * msg->pose.pose.orientation.z + msg->pose.pose.orientation.x * msg->pose.pose.orientation.y), 
                                        1.0 - 2.0 * (msg->pose.pose.orientation.y * msg->pose.pose.orientation.y + msg->pose.pose.orientation.z * msg->pose.pose.orientation.z));

        Pose2 current_raw_odom(msg->pose.pose.position.x, msg->pose.pose.position.y, current_yaw);

        if (first_odom_) {
            last_raw_odom_ = current_raw_odom;
            last_graph_odom_ = current_raw_odom;
            first_odom_ = false;
            return;
        }

        
        latest_predicted_pose_ = map_T_odom_.compose(current_raw_odom);
        last_raw_odom_ = current_raw_odom;

        publishTransform(stamp, current_raw_odom);
    }

    void coneCallback(const eufs_msgs::msg::ConeArrayWithCovariance::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mtx);
        if (!odom_updated_ || first_odom_) return;

        rclcpp::Time sensor_stamp = msg->header.stamp;

        Pose2 current_raw_odom = last_raw_odom_;
        Pose2 local_movement = last_graph_odom_.between(current_raw_odom);
        total_odom_dist_ += std::hypot(local_movement.x(), local_movement.y());

        Values current_estimate;
        try { current_estimate = isam2->calculateEstimate(); } catch (...) { return; }

        if (std::abs(local_movement.x()) < 0.01 && std::abs(local_movement.y()) < 0.01 && std::abs(local_movement.theta()) < 0.005) {
            publishVisualization(sensor_stamp, current_estimate);
            return;
        }

        last_graph_odom_ = current_raw_odom;

        int prev_idx = pose_count++;
        batch_factors_.add(BetweenFactor<Pose2>(Symbol('x', prev_idx), Symbol('x', pose_count), local_movement, odom_noise));
        
        Pose2 last_p = current_estimate.exists(Symbol('x', prev_idx)) ? 
                       current_estimate.at<Pose2>(Symbol('x', prev_idx)) : opt_pose_;
        
        Pose2 current_pose = last_p.compose(local_movement); 
        batch_values_.insert(Symbol('x', pose_count), current_pose);
        
        std::set<int> matched_landmarks;
        bool loop_closure_shockwave = false;

        if (!loop_closed_ && total_odom_dist_ > 30.0) { 
            int orange_matches = 0;
            
            auto check_loop_closure = [&](const auto& cones) {
                for (const auto& c : cones) {
                    double r = std::hypot(c.point.x, c.point.y);
                    double b = std::atan2(c.point.y, c.point.x);
                    
                    int l_id = findNearestLandmarkWithBearing(r, b, 2, current_pose, matched_landmarks, current_estimate);
                    
                    if (l_id != -1 && start_line_orange_ids_.count(l_id)) {
                        orange_matches++;
                        matched_landmarks.insert(l_id); 
                    }
                }
            };
            
            check_loop_closure(msg->orange_cones);
            check_loop_closure(msg->big_orange_cones);

            if (orange_matches >= 2) {
                RCLCPP_INFO(this->get_logger(), "🏁 LOOP CLOSED! Anchoring tracking to Lap 1 map.");
                loop_closed_ = true;
                loop_closure_shockwave = true;
            }
        }

        auto process = [&](const auto& cones, int color) {
            for (const auto& c : cones) {
                if (std::isnan(c.point.x) || std::isnan(c.point.y)) continue;

                double r = std::hypot(c.point.x, c.point.y);
                double b = std::atan2(c.point.y, c.point.x);
                int l_id = findNearestLandmarkWithBearing(r, b, color, current_pose, matched_landmarks, current_estimate);

                if (l_id == -1) {
                    if (loop_closed_ && color != 2) continue; 

                    l_id = next_landmark_id++;
                    batch_values_.insert(Symbol('l', l_id), current_pose.transformFrom(Point2(c.point.x, c.point.y)));
                    cone_color_map[l_id] = color;

                    if (color == 2 && total_odom_dist_ < 15.0) {
                        start_line_orange_ids_.insert(l_id);
                    }
                }
                
                matched_landmarks.insert(l_id); 
                batch_factors_.add(BearingRangeFactor<Pose2, Point2>(Symbol('x', pose_count), Symbol('l', l_id), Rot2::fromAngle(b), r, huber_noise));
            }
        };

        process(msg->blue_cones, 0);
        process(msg->yellow_cones, 1);
        process(msg->orange_cones, 2);
        process(msg->big_orange_cones, 2); 

        batch_counter_++;
        if (batch_counter_ >= 3 || loop_closure_shockwave) {
            try {
                isam2->update(batch_factors_, batch_values_);
                
                if (loop_closure_shockwave) {
                    isam2->update();
                    isam2->update(); 
                    isam2->update();
                }
                
                current_estimate = isam2->calculateEstimate();
                opt_pose_ = current_estimate.at<Pose2>(Symbol('x', pose_count));

                
                map_T_odom_ = opt_pose_.compose(last_graph_odom_.inverse());
                
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "GTSAM Optimization Error: %s", e.what());
            }

            batch_factors_.resize(0);
            batch_values_.clear();
            batch_counter_ = 0;
        }

        publishVisualization(sensor_stamp, current_estimate);
    }

    int findNearestLandmarkWithBearing(double r, double b, int color, const Pose2& x, 
                                       const std::set<int>& matched, const Values& estimate) {
        Point2 obs_global = x.transformFrom(Point2(r * std::cos(b), r * std::sin(b)));
        int best_id = -1;
        double best_score = 1e9;
        
        double dist_gate = (color == 2) ? 8.5 : (loop_closed_ ? 4.5 : 2.2); 

        auto check_values = [&](const Values& vals) {
            for (auto const& [key, val] : vals) {
                if (Symbol(key).chr() == 'l' && cone_color_map[Symbol(key).index()] == color) {
                    int l_idx = Symbol(key).index();
                    if (matched.count(l_idx)) continue; 

                    Point2 landmark = vals.at<Point2>(key);
                    double dist = (landmark - obs_global).norm();

                    if (dist < dist_gate) { 
                        Vector2 delta = landmark - x.translation();
                        double expected_bearing = std::atan2(delta.y(), delta.x());
                        double bearing_error = std::abs(std::atan2(std::sin(expected_bearing - (x.theta() + b)), std::cos(expected_bearing - (x.theta() + b))));
                        
                        double bearing_penalty = bearing_error > (M_PI / 2.0) ? 5.0 : 1.0;
                        double score = dist * bearing_penalty;

                        if (score < best_score) { 
                            best_score = score; 
                            best_id = l_idx; 
                        }
                    }
                }
            }
        };

        check_values(estimate);
        check_values(batch_values_);

        return best_id;
    }

    void publishTransform(rclcpp::Time stamp, const Pose2& current_raw_odom) {
        // Mathematically pure 2D output
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = stamp;
        t.header.frame_id = "map";
        t.child_frame_id = "odom";
        
        t.transform.translation.x = map_T_odom_.x();
        t.transform.translation.y = map_T_odom_.y();
        t.transform.translation.z = 0.0;
        
        t.transform.rotation.x = 0.0;
        t.transform.rotation.y = 0.0;
        t.transform.rotation.z = std::sin(map_T_odom_.theta() * 0.5);
        t.transform.rotation.w = std::cos(map_T_odom_.theta() * 0.5);
        tf_broadcaster->sendTransform(t);

        nav_msgs::msg::Odometry slam_odom;
        slam_odom.header.stamp = stamp;
        slam_odom.header.frame_id = "map";
        slam_odom.child_frame_id = "base_footprint";
        
        // Exact 2D Pose
        slam_odom.pose.pose.position.x = latest_predicted_pose_.x();
        slam_odom.pose.pose.position.y = latest_predicted_pose_.y();
        slam_odom.pose.pose.position.z = 0.0; 
        
        slam_odom.pose.pose.orientation.x = 0.0;
        slam_odom.pose.pose.orientation.y = 0.0;
        slam_odom.pose.pose.orientation.z = std::sin(latest_predicted_pose_.theta() * 0.5);
        slam_odom.pose.pose.orientation.w = std::cos(latest_predicted_pose_.theta() * 0.5);
        
        slam_odom.twist.twist = current_odom_twist_; // Correctly synced Twist!
        
        slam_odom_pub->publish(slam_odom);
    }

    void publishVisualization(rclcpp::Time stamp, const Values& estimate) {
        Values visualize_estimate = estimate;
        for (auto const& [key, val] : batch_values_) {
            if (!visualize_estimate.exists(key)) visualize_estimate.insert(key, val);
        }

        eufs_msgs::msg::ConeArray cone_msg;
        visualization_msgs::msg::MarkerArray markers;
        
        cone_msg.header.frame_id = "map";
        cone_msg.header.stamp = stamp;

        // Perfectly synchronized RViz DeleteAll marker!
        visualization_msgs::msg::Marker delete_all;
        delete_all.action = visualization_msgs::msg::Marker::DELETEALL;
        markers.markers.push_back(delete_all);

        double cos_yaw = std::cos(latest_predicted_pose_.theta());
        double sin_yaw = std::sin(latest_predicted_pose_.theta());
        std::vector<Point2> all_published_cones;

        for (auto const& [key, val] : visualize_estimate) {
            if (Symbol(key).chr() == 'l') {
                Point2 p = visualize_estimate.at<Point2>(key);
                int color = cone_color_map[Symbol(key).index()];

                bool is_duplicate = false;
                for (const auto& existing_p : all_published_cones) {
                    if ((existing_p - p).norm() < 0.4) { 
                        is_duplicate = true;
                        break;
                    }
                }
                if (is_duplicate) continue; 
                
                all_published_cones.push_back(p);

                geometry_msgs::msg::Point pt;
                pt.x = p.x(); pt.y = p.y(); pt.z = 0.0;
                
                double dx = p.x() - latest_predicted_pose_.x();
                double dy = p.y() - latest_predicted_pose_.y();
                double lx = dx * cos_yaw + dy * sin_yaw; 
                
                bool send_to_planner = true;
                if (loop_closed_ && (lx < -5.0 || lx > 25.0)) send_to_planner = false;

                if (send_to_planner) {
                    if (color == 0) cone_msg.blue_cones.push_back(pt);
                    else if (color == 1) cone_msg.yellow_cones.push_back(pt);
                    else cone_msg.big_orange_cones.push_back(pt); 
                }

                visualization_msgs::msg::Marker m;
                m.header.frame_id = "map";
                m.header.stamp = stamp;
                m.ns = "cones";
                m.id = Symbol(key).index();
                m.type = visualization_msgs::msg::Marker::CYLINDER;
                m.action = visualization_msgs::msg::Marker::ADD; 
                m.scale.x = 0.2; m.scale.y = 0.2; m.scale.z = 0.3;
                m.color.a = 1.0;
                
                if (color == 0) { m.color.b = 1.0; m.color.r = 0.0; m.color.g = 0.0; } 
                else if (color == 1) { m.color.r = 1.0; m.color.g = 1.0; m.color.b = 0.0; } 
                else { m.color.r = 1.0; m.color.g = 0.5; m.color.b = 0.0; } 
                
                m.pose.position.x = pt.x; m.pose.position.y = pt.y; m.pose.position.z = 0.15;
                markers.markers.push_back(m);
            }
        }
        
        cones_pub->publish(cone_msg);
        marker_pub->publish(markers);
    }

    std::unique_ptr<ISAM2> isam2;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;
    rclcpp::Publisher<eufs_msgs::msg::ConeArray>::SharedPtr cones_pub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr slam_odom_pub;
    
    std::mutex mtx;
    int pose_count = 0, next_landmark_id = 0;
    Pose2 last_raw_odom_, last_graph_odom_, opt_pose_, latest_predicted_pose_; 
    Pose2 map_T_odom_;
    double total_odom_dist_ = 0.0;
    bool first_odom_ = true, odom_updated_ = false;
    bool loop_closed_ = false; 
    std::map<int, int> cone_color_map;
    std::set<int> start_line_orange_ids_; 
    geometry_msgs::msg::Pose current_odom_pose_;
    geometry_msgs::msg::Twist current_odom_twist_;
    SharedNoiseModel odom_noise, huber_noise;
    
    NonlinearFactorGraph batch_factors_;
    Values batch_values_;
    int batch_counter_ = 0;
    
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;
    rclcpp::Subscription<eufs_msgs::msg::ConeArrayWithCovariance>::SharedPtr cone_sub;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EUFSGraphSlam>());
    rclcpp::shutdown();
    return 0;
}