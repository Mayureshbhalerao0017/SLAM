#include <new> 
#include <memory>
#include <vector>
#include <cmath>
#include <algorithm>
#include <mutex>
#include <random> // Added for eliminating sequential bias

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>
#include "eufs_msgs/msg/cone_array.hpp"
#include "eufs_msgs/msg/cone_array_with_covariance.hpp"
#include "nav_msgs/msg/odometry.hpp"

// VISUALIZATION HEADERS
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include <Eigen/Dense> 

using std::placeholders::_1;

inline double wrapToPi(double a) {
    return std::atan2(std::sin(a), std::cos(a));
}

inline double getYaw(const geometry_msgs::msg::Quaternion& q) {
    tf2::Quaternion tf_q(q.x, q.y, q.z, q.w);
    tf2::Matrix3x3 m(tf_q);
    double r, p, y;
    m.getRPY(r, p, y);
    return y;
}

struct ConeDetection {
    double range;
    double bearing;
    int color; 
};

struct LandmarkInfo {
    int color;
    int hits;
};

class EKFSLAM : public rclcpp::Node {
public:
    EKFSLAM() : Node("ekf_slam_node") {
        x_ = Eigen::VectorXd::Zero(3);
        P_ = Eigen::MatrixXd::Identity(3, 3) * 0.01;

        R_.setZero();
        R_(0, 0) = std::pow(0.15, 2); 
        R_(1, 1) = std::pow(0.05, 2);

        cones_pub_ = create_publisher<eufs_msgs::msg::ConeArray>("/planning/cones", 10);
        slam_odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/slam/odom", 10);    
        native_marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/slam/native_cones", 10);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        
        auto latched_qos = rclcpp::QoS(rclcpp::KeepLast(1)).transient_local();
        global_map_pub_ = create_publisher<eufs_msgs::msg::ConeArray>("/slam/global_map", latched_qos);
        
        auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();
        
        cones_sub_ = create_subscription<eufs_msgs::msg::ConeArrayWithCovariance>(
            "/ground_truth/cones", qos, std::bind(&EKFSLAM::conesCallback, this, _1));
        
        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/odometry/filtered", 10, std::bind(&EKFSLAM::odomCallback, this, _1));

        RCLCPP_INFO(this->get_logger(), "EKF-SLAM initialized. Anti-Slide & Time Sync Active.");
    }

private:
    Eigen::VectorXd x_;
    Eigen::MatrixXd P_;
    Eigen::Matrix2d R_;
    std::vector<ConeDetection> z_buffer_;
    std::vector<LandmarkInfo> lm_info_;
    std::recursive_mutex slam_mutex_;
    
    bool lap_closed_ = false;
    double total_dist_ = 0.0; 
    int lap_count_ = 0;             
    
    double vx_ = 0.0, yaw_rate_ = 0.0;
    
    nav_msgs::msg::Odometry::SharedPtr current_odom_msg_;
    nav_msgs::msg::Odometry::SharedPtr last_odom_msg_;

    rclcpp::Subscription<eufs_msgs::msg::ConeArrayWithCovariance>::SharedPtr cones_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<eufs_msgs::msg::ConeArray>::SharedPtr cones_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr slam_odom_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr native_marker_pub_;
    rclcpp::Publisher<eufs_msgs::msg::ConeArray>::SharedPtr global_map_pub_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::default_random_engine rng_{std::random_device{}()}; // Seeded RNG for anti-bias

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        std::lock_guard<std::recursive_mutex> lock(slam_mutex_);
        current_odom_msg_ = msg;
        vx_ = msg->twist.twist.linear.x;
        yaw_rate_ = msg->twist.twist.angular.z;
    }

    void conesCallback(const eufs_msgs::msg::ConeArrayWithCovariance::SharedPtr msg) {
        std::lock_guard<std::recursive_mutex> lock(slam_mutex_);
        z_buffer_.clear();
        for (const auto &cone : msg->blue_cones) addConeToBuffer(cone.point, 0); 
        for (const auto &cone : msg->yellow_cones) addConeToBuffer(cone.point, 1);
        for (const auto &cone : msg->orange_cones) addConeToBuffer(cone.point, 2);
        for (const auto &cone : msg->big_orange_cones) addConeToBuffer(cone.point, 2);
        
        
        std::shuffle(z_buffer_.begin(), z_buffer_.end(), rng_);

        runSLAM();
    }

    void addConeToBuffer(const geometry_msgs::msg::Point &cone, int color) {
        ConeDetection z;
        z.range = std::hypot(cone.x, cone.y);
        z.bearing = std::atan2(cone.y, cone.x);
        z.color = color;
        if (z.range > 0.1 && z.range < 20.0) z_buffer_.push_back(z);
    }

    void predict(double dx_local, double dy_local, double dyaw) {
        double psi = x_(2);
        
        x_(0) += dx_local * std::cos(psi) - dy_local * std::sin(psi);
        x_(1) += dx_local * std::sin(psi) + dy_local * std::cos(psi);
        x_(2) = wrapToPi(x_(2) + dyaw);

        int num_landmarks = (x_.size() - 3) / 2;
        Eigen::Matrix3d G_rob = Eigen::Matrix3d::Identity();
        
        G_rob(0, 2) = -dx_local * std::sin(psi) - dy_local * std::cos(psi);
        G_rob(1, 2) =  dx_local * std::cos(psi) - dy_local * std::sin(psi);

        Eigen::MatrixXd P_rr = P_.block(0, 0, 3, 3);
        P_.block(0, 0, 3, 3) = G_rob * P_rr * G_rob.transpose();

        if (num_landmarks > 0) {
            Eigen::MatrixXd P_rm = P_.block(0, 3, 3, 2 * num_landmarks);
            P_.block(0, 3, 3, 2 * num_landmarks) = G_rob * P_rm;
            P_.block(3, 0, 2 * num_landmarks, 3) = P_.block(0, 3, 3, 2 * num_landmarks).transpose();
        }

        Eigen::Matrix3d Q = Eigen::Matrix3d::Zero();
        Q(0, 0) = 0.05 * std::abs(dx_local); 
        Q(1, 1) = 0.05 * std::abs(dx_local); 
        Q(2, 2) = 0.10 * std::abs(dyaw); 
        
        P_.block(0, 0, 3, 3) += Q;
    }

    void measurementModelSparse(int j, Eigen::Vector2d &zhat, Eigen::MatrixXd &Hr, Eigen::MatrixXd &Hm) {
        int idx = 3 + 2 * j;
        double dx = x_(idx) - x_(0), dy = x_(idx + 1) - x_(1);
        double r2 = dx * dx + dy * dy, r = std::sqrt(r2) + 1e-6;
        zhat << r, wrapToPi(std::atan2(dy, dx) - x_(2));
        
        Hr = Eigen::MatrixXd::Zero(2, 3);
        Hr(0, 0) = -dx / r;  Hr(0, 1) = -dy / r;  Hr(0, 2) = 0;
        Hr(1, 0) = dy / r2;  Hr(1, 1) = -dx / r2; Hr(1, 2) = -1.0;
        
        Hm = Eigen::MatrixXd::Zero(2, 2);
        Hm(0, 0) = dx / r; Hm(0, 1) = dy / r;
        Hm(1, 0) = -dy / r2; Hm(1, 1) = dx / r2;
    }

    void measurementModel(int j, Eigen::Vector2d &zhat, Eigen::MatrixXd &H) {
        int idx = 3 + 2 * j;
        double dx = x_(idx) - x_(0), dy = x_(idx + 1) - x_(1);
        double r2 = dx * dx + dy * dy, r = std::sqrt(r2) + 1e-6;
        zhat << r, wrapToPi(std::atan2(dy, dx) - x_(2));
        H = Eigen::MatrixXd::Zero(2, x_.size());
        H(0, 0) = -dx / r;  H(0, 1) = -dy / r;  H(0, 2) = 0;
        H(1, 0) = dy / r2;  H(1, 1) = -dx / r2; H(1, 2) = -1.0;
        H(0, idx) = dx / r; H(0, idx + 1) = dy / r;
        H(1, idx) = -dy / r2; H(1, idx + 1) = dx / r2;
    }

    void runSLAM() {
        std::lock_guard<std::recursive_mutex> lock(slam_mutex_);
        if (!current_odom_msg_) return; 

        if (!last_odom_msg_) {
            last_odom_msg_ = current_odom_msg_;
            return;
        }

        
        rclcpp::Time sync_stamp = current_odom_msg_->header.stamp;

        double x1 = last_odom_msg_->pose.pose.position.x;
        double y1 = last_odom_msg_->pose.pose.position.y;
        double yaw1 = getYaw(last_odom_msg_->pose.pose.orientation);

        double x2 = current_odom_msg_->pose.pose.position.x;
        double y2 = current_odom_msg_->pose.pose.position.y;
        double yaw2 = getYaw(current_odom_msg_->pose.pose.orientation);

        double dX = x2 - x1;
        double dY = y2 - y1;
        
        double dx_local =  dX * std::cos(yaw1) + dY * std::sin(yaw1);
        double dy_local = -dX * std::sin(yaw1) + dY * std::cos(yaw1);
        double dyaw = wrapToPi(yaw2 - yaw1);

        last_odom_msg_ = current_odom_msg_; 
        
        if (std::abs(dx_local) < 0.001 && std::abs(dy_local) < 0.001 && std::abs(dyaw) < 0.001) {
            publishVisualization(sync_stamp);
            publishTransform(sync_stamp);
            return;
        }

        predict(dx_local, dy_local, dyaw);
        total_dist_ += std::hypot(dx_local, dy_local);
        
        double dist_to_origin = std::hypot(x_(0), x_(1));

        int orange_cones_seen = 0;
        for (const auto& z : z_buffer_) {
            if (z.color == 2 && z.range < 8.0 && std::abs(z.bearing) < 1.0) {
                orange_cones_seen++;
            }
        }

        bool nearing_loop_closure = (!lap_closed_ && total_dist_ > 100.0 && dist_to_origin < 10.0);
        bool near_start = (dist_to_origin < 4.0 && std::abs(wrapToPi(x_(2))) < 0.5);
        bool crossed_orange_line = (orange_cones_seen >= 2);

        if (!lap_closed_ && total_dist_ > 100.0 && (crossed_orange_line || near_start)) {
            lap_count_++;
            lap_closed_ = true;
            RCLCPP_INFO(this->get_logger(), "🏁 LOOP CLOSURE LOCKED! Transitioning to pure localization.");

            eufs_msgs::msg::ConeArray global_msg;
            global_msg.header.stamp = sync_stamp;
            global_msg.header.frame_id = "map";

            for (size_t i = 0; i < lm_info_.size(); ++i) {
                if (lm_info_[i].hits < 5) continue;

                geometry_msgs::msg::Point p;
                p.x = x_(3 + 2 * i); p.y = x_(3 + 2 * i + 1); p.z = 0.0;

                if (lm_info_[i].color == 0) global_msg.blue_cones.push_back(p);
                else if (lm_info_[i].color == 1) global_msg.yellow_cones.push_back(p);
                else if (lm_info_[i].color == 2) global_msg.big_orange_cones.push_back(p);
            }
            global_map_pub_->publish(global_msg);
        }

        Eigen::Matrix2d R_dynamic = R_;
        if (lap_closed_) {
            R_dynamic = R_ * 20.0;
        } else if (nearing_loop_closure) {
            R_dynamic = R_ * 150.0; 
        }

        double physical_gate = (lap_closed_ || nearing_loop_closure) ? 4.0 : 1.2;
        double mahalanobis_gate = (lap_closed_ || nearing_loop_closure) ? 30.0 : 5.99;

        for (const auto &z : z_buffer_) {
            int best_j = -1; 
            double best_md = 1e9;
            double angle = wrapToPi(x_(2) + z.bearing);
            double det_x = x_(0) + z.range * std::cos(angle);
            double det_y = x_(1) + z.range * std::sin(angle);

            for (size_t j = 0; j < lm_info_.size(); ++j) {
                if (lm_info_[j].color != z.color) continue;
                
                double dx = x_(3 + 2 * j) - det_x;
                double dy = x_(3 + 2 * j + 1) - det_y;
                
                if (std::hypot(dx, dy) > physical_gate) continue;

                Eigen::Vector2d zhat; Eigen::MatrixXd Hr, Hm;
                measurementModelSparse(j, zhat, Hr, Hm);
                Eigen::Vector2d v; v << z.range - zhat(0), wrapToPi(z.bearing - zhat(1));
                
                Eigen::MatrixXd Prr = P_.block(0, 0, 3, 3);
                Eigen::MatrixXd Prm = P_.block(0, 3 + 2 * j, 3, 2);
                Eigen::MatrixXd Pmr = P_.block(3 + 2 * j, 0, 2, 3);
                Eigen::MatrixXd Pmm = P_.block(3 + 2 * j, 3 + 2 * j, 2, 2);

                Eigen::Matrix2d S = Hr * Prr * Hr.transpose() + Hr * Prm * Hm.transpose() + 
                                    Hm * Pmr * Hr.transpose() + Hm * Pmm * Hm.transpose() + R_dynamic;
                
                if (S.determinant() < 1e-6) continue;
                double md = v.transpose() * S.inverse() * v;
                
                if (md < best_md && md < mahalanobis_gate) { best_md = md; best_j = (int)j; }
            }

            if (best_j >= 0) {
                lm_info_[best_j].hits++;
                
                Eigen::Vector2d zhat; Eigen::MatrixXd H;
                measurementModel(best_j, zhat, H); 
                Eigen::Vector2d v; v << z.range - zhat(0), wrapToPi(z.bearing - zhat(1));
                
                Eigen::MatrixXd HP = H * P_;
                Eigen::Matrix2d S = HP * H.transpose() + R_dynamic; 
                Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();
                
                if (lap_closed_) {
                    K.block(3, 0, K.rows() - 3, 2).setZero(); 
                }
                
                x_ += K * v;
                x_(2) = wrapToPi(x_(2)); 
                
                P_ = P_ - K * HP; 

            } else if (!lap_closed_) {
                bool duplicate = false;
                for(size_t i=0; i < lm_info_.size(); ++i) {
                    if(lm_info_[i].color == z.color && std::hypot(x_(3+2*i)-det_x, x_(3+2*i+1)-det_y) < physical_gate) { 
                        duplicate = true; 
                        break; 
                    }
                }
                if(!duplicate && lm_info_.size() < 400) {
                    int old_size = x_.size();
                    x_.conservativeResize(old_size + 2); x_.tail<2>() << det_x, det_y;
                    Eigen::MatrixXd Gx(2, 3); Gx << 1, 0, -z.range*std::sin(angle), 0, 1, z.range*std::cos(angle);
                    Eigen::Matrix2d Gz; Gz << std::cos(angle), -z.range*std::sin(angle), std::sin(angle), z.range*std::cos(angle);
                    Eigen::MatrixXd P_new = Eigen::MatrixXd::Zero(old_size+2, old_size+2);
                    P_new.topLeftCorner(old_size, old_size) = P_;
                    Eigen::MatrixXd Pxr = P_.block(0, 0, old_size, 3) * Gx.transpose();
                    P_new.block(0, old_size, old_size, 2) = Pxr; P_new.block(old_size, 0, 2, old_size) = Pxr.transpose();
                    P_new.bottomRightCorner(2, 2) = Gx * P_.block<3, 3>(0, 0) * Gx.transpose() + Gz * R_ * Gz.transpose();
                    P_ = P_new;
                    LandmarkInfo new_lm; new_lm.color = z.color; new_lm.hits = 1;
                    lm_info_.push_back(new_lm);
                }
            }
        }
        
        P_ = 0.5 * (P_ + P_.transpose());
        for(int i = 0; i < P_.rows(); ++i) {
            P_(i, i) += 1e-9;
        }

        publishVisualization(sync_stamp);
        publishTransform(sync_stamp);
    }

    void publishVisualization(rclcpp::Time stamp) {
        auto out_msg = eufs_msgs::msg::ConeArray();
        out_msg.header.stamp = stamp; 
        out_msg.header.frame_id = "map";
        
        visualization_msgs::msg::MarkerArray marker_array;
        visualization_msgs::msg::Marker delete_all;
        delete_all.action = visualization_msgs::msg::Marker::DELETEALL;
        marker_array.markers.push_back(delete_all);

        int N = lm_info_.size();
        for (int i = 0; i < N; ++i) {
            if (lm_info_[i].hits < 5) continue;
            double cx = x_(3 + 2 * i); double cy = x_(3 + 2 * i + 1);
            
            visualization_msgs::msg::Marker m;
            m.header.stamp = stamp; 
            m.header.frame_id = "map";
            m.ns = "ekf_cones"; m.id = i; m.type = visualization_msgs::msg::Marker::CYLINDER;
            m.action = visualization_msgs::msg::Marker::ADD;
            m.pose.position.x = cx; m.pose.position.y = cy; m.pose.position.z = 0.15;
            m.scale.x = 0.3; m.scale.y = 0.3; m.scale.z = 0.3; m.color.a = 1.0;

            if (lm_info_[i].color == 0) { m.color.b = 1.0; } 
            else if (lm_info_[i].color == 1) { m.color.r = 1.0; m.color.g = 1.0; } 
            else if (lm_info_[i].color == 2) { m.color.r = 1.0; m.color.g = 0.5; } 
            marker_array.markers.push_back(m);

            double dist_to_car = std::hypot(cx - x_(0), cy - x_(1));
            if (dist_to_car < 20.0) {
                geometry_msgs::msg::Point p; p.x = cx; p.y = cy; p.z = 0.0;
                if (lm_info_[i].color == 0) out_msg.blue_cones.push_back(p);
                else if (lm_info_[i].color == 1) out_msg.yellow_cones.push_back(p);
                else if (lm_info_[i].color == 2) out_msg.big_orange_cones.push_back(p);
            }
        }
        
        cones_pub_->publish(out_msg);
        native_marker_pub_->publish(marker_array);
    }

    void publishTransform(rclcpp::Time stamp) {
        if (!current_odom_msg_) return;

        tf2::Transform map_to_base;
        map_to_base.setOrigin(tf2::Vector3(x_(0), x_(1), 0.0));
        tf2::Quaternion q_map_base;
        q_map_base.setRPY(0, 0, x_(2));
        map_to_base.setRotation(q_map_base);

        tf2::Transform odom_to_base;
        odom_to_base.setOrigin(tf2::Vector3(current_odom_msg_->pose.pose.position.x, current_odom_msg_->pose.pose.position.y, 0.0));
        tf2::Quaternion q_odom_base(
            current_odom_msg_->pose.pose.orientation.x,
            current_odom_msg_->pose.pose.orientation.y,
            current_odom_msg_->pose.pose.orientation.z,
            current_odom_msg_->pose.pose.orientation.w);
        odom_to_base.setRotation(q_odom_base);

        tf2::Transform map_to_odom = map_to_base * odom_to_base.inverse();

        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = stamp; 
        t.header.frame_id = "map";
        t.child_frame_id = "odom"; 
        t.transform.translation.x = map_to_odom.getOrigin().x();
        t.transform.translation.y = map_to_odom.getOrigin().y();
        t.transform.translation.z = 0.0;
        t.transform.rotation.x = map_to_odom.getRotation().x();
        t.transform.rotation.y = map_to_odom.getRotation().y();
        t.transform.rotation.z = map_to_odom.getRotation().z();
        t.transform.rotation.w = map_to_odom.getRotation().w();
        tf_broadcaster_->sendTransform(t);

        nav_msgs::msg::Odometry slam_odom;
        slam_odom.header.stamp = stamp; 
        slam_odom.header.frame_id = "map";
        slam_odom.child_frame_id = "base_footprint";
        slam_odom.pose.pose.position.x = x_(0); slam_odom.pose.pose.position.y = x_(1);
        slam_odom.pose.pose.orientation.z = std::sin(x_(2) * 0.5); slam_odom.pose.pose.orientation.w = std::cos(x_(2) * 0.5);
        slam_odom.twist.twist.linear.x = vx_; slam_odom.twist.twist.angular.z = yaw_rate_;
        slam_odom_pub_->publish(slam_odom);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EKFSLAM>());
    rclcpp::shutdown();
    return 0;
}