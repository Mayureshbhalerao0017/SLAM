#include <vector>
#include <cmath>
#include <algorithm>
#include <mutex>
#include <random>

// Standard ROS 2 Headers
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>
#include <nav_msgs/msg/odometry.hpp>

// EUFS Custom Messages
#include "eufs_msgs/msg/cone_array.hpp"
#include "eufs_msgs/msg/cone_array_with_covariance.hpp"

// RViz Visualization Headers
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>

// Math
#include <Eigen/Dense> 

using std::placeholders::_1;

inline double wrapToPi(double a) {
    return std::atan2(std::sin(a), std::cos(a));
}

struct ConeDetection {
    double range; 
    double bearing; 
    int color; 
};

struct Landmark {
    Eigen::Vector2d mu;
    Eigen::Matrix2d sigma;
    int color;
    double hits;
};

struct Particle {
    double weight;
    double x;
    double y;
    double yaw;
    Eigen::Matrix3d P; 
    std::vector<Landmark> map;
};

class FastSLAM2 : public rclcpp::Node {
public:
    FastSLAM2() : Node("fast_slam_node"), gen_(rd_()) {
        num_particles_ = 30; 
        
        Q_control_ << std::pow(0.2, 2), 0, 
                      0, std::pow(0.15, 2); 
                      
        R_obs_ << std::pow(0.1, 2), 0, 
                  0, std::pow(0.035, 2); 

        for (int i = 0; i < num_particles_; ++i) {
            Particle p;
            p.weight = 1.0 / num_particles_;
            p.x = 0.0; p.y = 0.0; p.yaw = 0.0;
            p.P = Eigen::Matrix3d::Identity() * 0.01;
            particles_.push_back(p);
        }

        // Publishers
        cones_pub_ = create_publisher<eufs_msgs::msg::ConeArray>("/planning/cones", 10);
        slam_odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/slam/odom", 10);    
        native_marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/slam/native_cones", 10);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        // Subscribers
        auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();
        cones_sub_ = create_subscription<eufs_msgs::msg::ConeArrayWithCovariance>(
            "/ground_truth/cones", qos, std::bind(&FastSLAM2::conesCallback, this, _1));
        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/odometry/filtered", 10, std::bind(&FastSLAM2::odomCallback, this, _1));

        timer_ = create_wall_timer(std::chrono::milliseconds(50), std::bind(&FastSLAM2::runSLAM, this));

        RCLCPP_INFO(this->get_logger(), "🔥 FastSLAM 2.0 Initialized. Racing mode active.");
    }

private:
    int num_particles_;
    std::vector<Particle> particles_;
    Eigen::Matrix2d Q_control_;
    Eigen::Matrix2d R_obs_;
    
    std::vector<ConeDetection> z_buffer_;
    std::mutex slam_mutex_; 
    std::random_device rd_;
    std::mt19937 gen_;
    
    bool lap_closed_ = false;
    double total_dist_ = 0.0; 
    int lap_count_ = 0;             
    double last_lap_dist_ = 0.0; 
    
    double vx_ = 0.0, yaw_rate_ = 0.0, dt_ = 0.0;
    rclcpp::Time last_odom_time_{0, 0, RCL_ROS_TIME};

    rclcpp::Subscription<eufs_msgs::msg::ConeArrayWithCovariance>::SharedPtr cones_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    
    rclcpp::Publisher<eufs_msgs::msg::ConeArray>::SharedPtr cones_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr slam_odom_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr native_marker_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr timer_;

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(slam_mutex_);
        vx_ = msg->twist.twist.linear.x;
        yaw_rate_ = msg->twist.twist.angular.z;
        rclcpp::Time msg_time = msg->header.stamp;
        if (last_odom_time_.nanoseconds() != 0) {
            dt_ += (msg_time - last_odom_time_).seconds();
        }
        last_odom_time_ = msg_time;
    }

    void conesCallback(const eufs_msgs::msg::ConeArrayWithCovariance::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(slam_mutex_);
        z_buffer_.clear();
        for (const auto &c : msg->blue_cones) addConeToBuffer(c.point, 0); 
        for (const auto &c : msg->yellow_cones) addConeToBuffer(c.point, 1);
        for (const auto &c : msg->orange_cones) addConeToBuffer(c.point, 2);
        for (const auto &c : msg->big_orange_cones) addConeToBuffer(c.point, 2); 
    }

    void addConeToBuffer(const geometry_msgs::msg::Point &cone, int color) {
        ConeDetection z;
        z.range = std::hypot(cone.x, cone.y);
        z.bearing = std::atan2(cone.y, cone.x);
        z.color = color;
        if (z.range > 0.1 && z.range < 30.0) z_buffer_.push_back(z);
    }

    void predictParticles(double dt, double vx, double yaw_rate) {
        double dyn_vx_noise = std::max(std::sqrt(Q_control_(0,0)), std::abs(vx) * 0.1); 
        double dyn_yaw_noise = std::max(std::sqrt(Q_control_(1,1)), std::abs(yaw_rate) * 0.15); 
        double dyn_vy_noise = std::abs(yaw_rate) * 0.25; 

        std::normal_distribution<double> noise_vx(0.0, dyn_vx_noise);
        std::normal_distribution<double> noise_vy(0.0, dyn_vy_noise);
        std::normal_distribution<double> noise_yaw(0.0, dyn_yaw_noise);

        for (auto &p : particles_) {
            double noisy_vx = vx + noise_vx(gen_);
            double noisy_vy = noise_vy(gen_); 
            double noisy_yaw_rate = yaw_rate + noise_yaw(gen_);

            p.x += (noisy_vx * std::cos(p.yaw) - noisy_vy * std::sin(p.yaw)) * dt;
            p.y += (noisy_vx * std::sin(p.yaw) + noisy_vy * std::cos(p.yaw)) * dt;
            p.yaw = wrapToPi(p.yaw + noisy_yaw_rate * dt);

            Eigen::Matrix3d G;
            G << 1.0, 0.0, -(noisy_vx * std::sin(p.yaw) + noisy_vy * std::cos(p.yaw)) * dt,
                 0.0, 1.0,  (noisy_vx * std::cos(p.yaw) - noisy_vy * std::sin(p.yaw)) * dt,
                 0.0, 0.0,  1.0;
            
            Eigen::Matrix3d Q_pose = Eigen::Matrix3d::Identity() * 0.05;
            p.P = G * p.P * G.transpose() + Q_pose;
        }
    }

    void updateParticles(const std::vector<ConeDetection> &z_buffer) {
        for (auto &p : particles_) {
            for (const auto &z : z_buffer) {
                double global_theta = wrapToPi(p.yaw + z.bearing);
                double global_zx = p.x + z.range * std::cos(global_theta);
                double global_zy = p.y + z.range * std::sin(global_theta);

                int best_idx = -1;
                double best_md = 1e9;

                for (size_t i = 0; i < p.map.size(); ++i) {
                    if (p.map[i].color != z.color) continue;
                    
                    double dx = p.map[i].mu(0) - global_zx;
                    double dy = p.map[i].mu(1) - global_zy;
                    
                    if (std::hypot(dx, dy) > 2.0) continue; 

                    double dx_r = p.map[i].mu(0) - p.x;
                    double dy_r = p.map[i].mu(1) - p.y;
                    double q = dx_r * dx_r + dy_r * dy_r;
                    double r_hat = std::sqrt(q) + 1e-6;

                    Eigen::Matrix2d Hf;
                    Hf <<  dx_r / r_hat,  dy_r / r_hat, 
                          -dy_r / q,      dx_r / q;

                    Eigen::Matrix2d S = Hf * p.map[i].sigma * Hf.transpose() + R_obs_;
                    
                    Eigen::Vector2d z_hat(r_hat, wrapToPi(std::atan2(dy_r, dx_r) - p.yaw));
                    
                    Eigen::Vector2d v(z.range - z_hat(0), wrapToPi(z.bearing - z_hat(1)));
                    double md = v.transpose() * S.inverse() * v;

                    if (md < best_md && md < 11.34) { best_md = md; best_idx = i; }
                }

                if (best_idx >= 0) {
                    Landmark &lm = p.map[best_idx];
                    
                    double dx_r = lm.mu(0) - p.x;
                    double dy_r = lm.mu(1) - p.y;
                    double q = dx_r * dx_r + dy_r * dy_r;
                    double r_hat = std::sqrt(q) + 1e-6;

                    Eigen::Matrix2d Hf;
                    Hf <<  dx_r / r_hat,  dy_r / r_hat, 
                          -dy_r / q,      dx_r / q;

                    Eigen::MatrixXd Hv(2, 3);
                    Hv << -dx_r / r_hat, -dy_r / r_hat, 0.0,
                           dy_r / q,     -dx_r / q,    -1.0;

                    Eigen::Matrix2d S = Hf * lm.sigma * Hf.transpose() + R_obs_;
                    Eigen::Matrix2d S_inv = S.inverse();
                    Eigen::Vector2d z_hat(r_hat, wrapToPi(std::atan2(dy_r, dx_r) - p.yaw));
                    Eigen::Vector2d v(z.range - z_hat(0), wrapToPi(z.bearing - z_hat(1)));

                    if (!lap_closed_) {
                        Eigen::Matrix2d K = lm.sigma * Hf.transpose() * S_inv;
                        lm.mu += K * v;
                        lm.sigma = (Eigen::Matrix2d::Identity() - K * Hf) * lm.sigma;
                    }
                    lm.hits += 1.0; 

                    Eigen::Matrix3d P_inv = p.P.inverse();
                    Eigen::Matrix3d P_proposal = (Hv.transpose() * S_inv * Hv + P_inv).inverse();
                    Eigen::Vector3d state_update = P_proposal * Hv.transpose() * S_inv * v;
                    
                    p.x += state_update(0);
                    p.y += state_update(1);
                    p.yaw = wrapToPi(p.yaw + state_update(2));
                    p.P = P_proposal;

                    double det_S = S.determinant();
                    if (det_S > 1e-6) {
                        double num = std::exp(-0.5 * best_md);
                        double den = 2.0 * M_PI * std::sqrt(det_S);
                        p.weight *= (num / den);
                    }
                } else if (!lap_closed_) {
                    double dist_to_origin = std::hypot(p.x, p.y);
                    bool finishing_lap = (total_dist_ > 30.0 && dist_to_origin < 15.0);
                    
                    if (!finishing_lap) {
                        Landmark new_lm;
                        new_lm.mu << global_zx, global_zy;
                        
                        double c = std::cos(global_theta);
                        double s = std::sin(global_theta);
                        Eigen::Matrix2d Gz; 
                        Gz << c, -z.range * s, 
                              s,  z.range * c;
                              
                        new_lm.sigma = Gz * R_obs_ * Gz.transpose();
                        new_lm.color = z.color;
                        new_lm.hits = 1.0; // FIX: Start at 1.0

                        bool too_close = false;
                        for (const auto &lm : p.map) {
                            if (std::hypot(lm.mu(0) - global_zx, lm.mu(1) - global_zy) < 1.0) {
                                too_close = true; break;
                            }
                        }
                        if (!too_close) {
                            p.map.push_back(new_lm);
                        }
                    }
                }
            }
            
            for (auto &lm_all : p.map) {
                if (lm_all.hits > 0.0) lm_all.hits -= 0.05; 
            }
        }
    }

    void resampleParticles() {
        double sum_w = 0.0;
        for (const auto &p : particles_) sum_w += p.weight;
        
        if (sum_w < 1e-9) { 
            RCLCPP_WARN(this->get_logger(), "Weight collapse detected! Resetting particles.");
            for (auto &p : particles_) {
                p.weight = 1.0 / num_particles_;
                p.P = Eigen::Matrix3d::Identity() * 0.5; 
            }
            return;
        }
        
        double w_sq_sum = 0.0;
        for (auto &p : particles_) {
            p.weight /= sum_w;
            w_sq_sum += p.weight * p.weight;
        }

        double n_eff = 1.0 / w_sq_sum;
        if (n_eff >= num_particles_ / 2.0) return; 

        std::vector<Particle> new_particles;
        std::uniform_real_distribution<double> dist(0.0, 1.0 / num_particles_);
        double r = dist(gen_);
        double c = particles_[0].weight;
        int idx = 0;

        for (int m = 0; m < num_particles_; ++m) {
            double u = r + m * (1.0 / num_particles_);
            while (u > c && idx < num_particles_ - 1) {
                idx++;
                c += particles_[idx].weight;
            }
            Particle p_copy = particles_[idx];
            p_copy.weight = 1.0 / num_particles_;
            new_particles.push_back(p_copy);
        }
        particles_ = new_particles;

        std::normal_distribution<double> jitter_pos(0.0, 0.2); 
        std::normal_distribution<double> jitter_yaw(0.0, 0.05); 
        
        for (int i = 1; i < num_particles_; ++i) {
            particles_[i].x += jitter_pos(gen_);
            particles_[i].y += jitter_pos(gen_);
            particles_[i].yaw = wrapToPi(particles_[i].yaw + jitter_yaw(gen_));
            particles_[i].P += Eigen::Matrix3d::Identity() * 1e-4; 
        }
    }

    Particle getBestParticle() {
        Particle best = particles_[0];
        for (const auto &p : particles_) {
            if (p.weight > best.weight) best = p;
        }
        return best;
    }

    void runSLAM() {
        std::vector<ConeDetection> local_z;
        double local_dt = 0, local_vx = 0, local_yaw_rate = 0;

        {
            std::lock_guard<std::mutex> lock(slam_mutex_);
            if (dt_ <= 0.0 || dt_ > 0.2) {
                Particle best_p = getBestParticle();
                geometry_msgs::msg::TransformStamped t;
                t.header.stamp = this->now(); t.header.frame_id = "map";
                t.child_frame_id = "base_footprint"; 
                t.transform.translation.x = best_p.x; t.transform.translation.y = best_p.y;
                t.transform.rotation.z = std::sin(best_p.yaw * 0.5); t.transform.rotation.w = std::cos(best_p.yaw * 0.5);
                tf_broadcaster_->sendTransform(t);
                return; 
            }
            local_z = z_buffer_; z_buffer_.clear();
            local_dt = dt_; dt_ = 0.0; 
            local_vx = vx_; local_yaw_rate = yaw_rate_;
        }

        predictParticles(local_dt, local_vx, local_yaw_rate);
        total_dist_ += std::abs(local_vx) * local_dt;

        Particle best_p = getBestParticle();

       int orange_cones_seen = 0;
        for (const auto& z : local_z) {
            
            if (z.color == 2 && z.range < 8.0 && std::abs(z.bearing) < 1.0) orange_cones_seen++;
        }

        double dist_to_origin = std::hypot(best_p.x, best_p.y);
        
        
        bool near_start = (dist_to_origin < 5.0 && std::abs(wrapToPi(best_p.yaw)) < 0.5);
        
        
        bool crossed_orange_line = (orange_cones_seen >= 2);

        
        if (!lap_closed_ && total_dist_ > 40.0 && (crossed_orange_line || near_start)) {
            lap_closed_ = true;
            RCLCPP_INFO(this->get_logger(), "🏁 LAP 1 COMPLETE! Loop Closure Triggered. Freezing Map.");
            
            
            for (auto &p : particles_) {
                p.x = best_p.x;
                p.y = best_p.y;
                p.yaw = best_p.yaw;
                p.weight = 1.0 / num_particles_;
            }
        }

        if (!local_z.empty()) {
            updateParticles(local_z);
            resampleParticles();
            best_p = getBestParticle(); 
        }

        auto out_msg = eufs_msgs::msg::ConeArray();
        out_msg.header.stamp = this->now();
        out_msg.header.frame_id = "map";
        
        int N = best_p.map.size(); 
        
        for (int i = 0; i < N; ++i) {
            if (best_p.map[i].hits < 3.0) continue; 
            
            geometry_msgs::msg::Point pt; 
            pt.x = best_p.map[i].mu(0); pt.y = best_p.map[i].mu(1); pt.z = 0.0;
            if (best_p.map[i].color == 0) out_msg.blue_cones.push_back(pt);
            else if (best_p.map[i].color == 1) out_msg.yellow_cones.push_back(pt);
            else if (best_p.map[i].color == 2) out_msg.big_orange_cones.push_back(pt);
        }
        cones_pub_->publish(out_msg);

        nav_msgs::msg::Odometry slam_odom;
        slam_odom.header.stamp = this->now();
        slam_odom.header.frame_id = "map";
        slam_odom.pose.pose.position.x = best_p.x; 
        slam_odom.pose.pose.position.y = best_p.y;
        slam_odom.pose.pose.orientation.z = std::sin(best_p.yaw * 0.5); 
        slam_odom.pose.pose.orientation.w = std::cos(best_p.yaw * 0.5);
        slam_odom.twist.twist.linear.x = local_vx; 
        slam_odom.twist.twist.angular.z = local_yaw_rate;
        slam_odom_pub_->publish(slam_odom);

        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = this->now(); t.header.frame_id = "map"; t.child_frame_id = "base_footprint"; 
        t.transform.translation.x = best_p.x; t.transform.translation.y = best_p.y;
        t.transform.rotation.z = std::sin(best_p.yaw * 0.5); t.transform.rotation.w = std::cos(best_p.yaw * 0.5);
        tf_broadcaster_->sendTransform(t);

        visualization_msgs::msg::MarkerArray marker_array;
        visualization_msgs::msg::Marker delete_all;
        delete_all.action = visualization_msgs::msg::Marker::DELETEALL;
        marker_array.markers.push_back(delete_all);

        for (int i = 0; i < N; ++i) {
            if (best_p.map[i].hits < 3.0) continue; 
            
            visualization_msgs::msg::Marker m;
            m.header.stamp = this->now(); m.header.frame_id = "map"; m.ns = "track_cones"; m.id = i; 
            m.type = visualization_msgs::msg::Marker::CYLINDER; m.action = visualization_msgs::msg::Marker::ADD;
            m.pose.position.x = best_p.map[i].mu(0); m.pose.position.y = best_p.map[i].mu(1); m.pose.position.z = 0.15; 
            m.scale.x = 0.3; m.scale.y = 0.3; m.scale.z = 0.3; m.color.a = 1.0; 
            if (best_p.map[i].color == 0) { m.color.r = 0.0; m.color.g = 0.0; m.color.b = 1.0; }
            else if (best_p.map[i].color == 1) { m.color.r = 1.0; m.color.g = 1.0; m.color.b = 0.0; }
            else if (best_p.map[i].color == 2) { m.color.r = 1.0; m.color.g = 0.5; m.color.b = 0.0; }
            marker_array.markers.push_back(m);
        }
        native_marker_pub_->publish(marker_array);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FastSLAM2>());
    rclcpp::shutdown();
    return 0;
}