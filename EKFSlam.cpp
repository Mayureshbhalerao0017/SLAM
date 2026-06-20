#pragma once
#include "SLAMDataTypes.hpp"
#include <eigen3/Eigen/Dense>
#include <vector>
#include <cmath>

namespace slam_core {

class EKFSLAMCore {
public:
    EKFSLAMCore(double r_range = 0.15, double r_bearing = 0.05) {
        x_ = Eigen::VectorXd::Zero(3);
        P_ = Eigen::MatrixXd::Identity(3, 3) * 0.01;
        R_ = Eigen::Matrix2d::Zero();
        R_(0, 0) = std::pow(r_range, 2);
        R_(1, 1) = std::pow(r_bearing, 2);
    }

    void predict(double dx, double dy, double dyaw) {
        double psi = x_(2);
        
        x_(0) += dx * std::cos(psi) - dy * std::sin(psi);
        x_(1) += dx * std::sin(psi) + dy * std::cos(psi);
        x_(2) = wrapToPi(x_(2) + dyaw);

        int num_landmarks = (x_.size() - 3) / 2;
        Eigen::Matrix3d G_rob = Eigen::Matrix3d::Identity();
        G_rob(0, 2) = -dx * std::sin(psi) - dy * std::cos(psi);
        G_rob(1, 2) =  dx * std::cos(psi) - dy * std::sin(psi);

        Eigen::MatrixXd P_rr = P_.block(0, 0, 3, 3);
        P_.block(0, 0, 3, 3) = G_rob * P_rr * G_rob.transpose();

        if (num_landmarks > 0) {
            Eigen::MatrixXd P_rm = P_.block(0, 3, 3, 2 * num_landmarks);
            P_.block(0, 3, 3, 2 * num_landmarks) = G_rob * P_rm;
            P_.block(3, 0, 2 * num_landmarks, 3) = P_.block(0, 3, 3, 2 * num_landmarks).transpose();
        }

        Eigen::Matrix3d Q = Eigen::Matrix3d::Zero();
        Q(0, 0) = 0.05 * std::abs(dx); 
        Q(1, 1) = 0.05 * std::abs(dx); 
        Q(2, 2) = 0.10 * std::abs(dyaw); 
        P_.block(0, 0, 3, 3) += Q;
    }

    void update(const std::vector<Observation>& observations, bool is_loop_closed) {
        double physical_gate = is_loop_closed ? 4.0 : 1.2;
        double mahalanobis_gate = is_loop_closed ? 30.0 : 5.99;
        Eigen::Matrix2d R_dynamic = is_loop_closed ? (R_ * 20.0) : R_;

        for (const auto &z : observations) {
            int best_j = -1; 
            double best_md = 1e9;
            double angle = wrapToPi(x_(2) + z.bearing);
            double det_x = x_(0) + z.range * std::cos(angle);
            double det_y = x_(1) + z.range * std::sin(angle);

            for (size_t j = 0; j < class_ids_.size(); ++j) {
                if (class_ids_[j] != z.class_id) continue;
                
                double dx = x_(3 + 2 * j) - det_x;
                double dy = x_(3 + 2 * j + 1) - det_y;
                if (std::hypot(dx, dy) > physical_gate) continue;

                Eigen::Vector2d zhat; Eigen::MatrixXd Hr, Hm;
                measurementModelSparse(j, zhat, Hr, Hm);
                Eigen::Vector2d v(z.range - zhat(0), wrapToPi(z.bearing - zhat(1)));
                
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
                hit_counts_[best_j]++;
                Eigen::Vector2d zhat; Eigen::MatrixXd H;
                measurementModel(best_j, zhat, H); 
                Eigen::Vector2d v(z.range - zhat(0), wrapToPi(z.bearing - zhat(1)));
                
                Eigen::MatrixXd HP = H * P_;
                Eigen::Matrix2d S = HP * H.transpose() + R_dynamic; 
                Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();
                
                if (is_loop_closed) { K.block(3, 0, K.rows() - 3, 2).setZero(); }
                
                x_ += K * v;
                x_(2) = wrapToPi(x_(2)); 
                P_ = P_ - K * HP; 

            } else if (!is_loop_closed && class_ids_.size() < 400) {
                int old_size = x_.size();
                x_.conservativeResize(old_size + 2); 
                x_.tail<2>() << det_x, det_y;
                
                Eigen::MatrixXd Gx(2, 3); Gx << 1, 0, -z.range*std::sin(angle), 0, 1, z.range*std::cos(angle);
                Eigen::Matrix2d Gz; Gz << std::cos(angle), -z.range*std::sin(angle), std::sin(angle), z.range*std::cos(angle);
                
                Eigen::MatrixXd P_new = Eigen::MatrixXd::Zero(old_size+2, old_size+2);
                P_new.topLeftCorner(old_size, old_size) = P_;
                Eigen::MatrixXd Pxr = P_.block(0, 0, old_size, 3) * Gx.transpose();
                P_new.block(0, old_size, old_size, 2) = Pxr; 
                P_new.block(old_size, 0, 2, old_size) = Pxr.transpose();
                P_new.bottomRightCorner(2, 2) = Gx * P_.block<3, 3>(0, 0) * Gx.transpose() + Gz * R_ * Gz.transpose();
                
                P_ = P_new;
                class_ids_.push_back(z.class_id);
                hit_counts_.push_back(1.0);
            }
        }
        
        P_ = 0.5 * (P_ + P_.transpose()); // Guarantee symmetry
        for(int i = 0; i < P_.rows(); ++i) P_(i, i) += 1e-9;
    }

    Pose2D getVehiclePose() const { return {x_(0), x_(1), x_(2)}; }

    std::vector<Landmark> getMap() const {
        std::vector<Landmark> map;
        for (size_t i = 0; i < class_ids_.size(); ++i) {
            map.push_back({x_(3 + 2 * i), x_(3 + 2 * i + 1), class_ids_[i], hit_counts_[i]});
        }
        return map;
    }

private:
    void measurementModelSparse(int j, Eigen::Vector2d &zhat, Eigen::MatrixXd &Hr, Eigen::MatrixXd &Hm) {
        int idx = 3 + 2 * j;
        double dx = x_(idx) - x_(0), dy = x_(idx + 1) - x_(1);
        double r2 = dx * dx + dy * dy, r = std::sqrt(r2) + 1e-6;
        zhat << r, wrapToPi(std::atan2(dy, dx) - x_(2));
        
        Hr = Eigen::MatrixXd::Zero(2, 3);
        Hr << -dx/r, -dy/r, 0,  dy/r2, -dx/r2, -1.0;
        Hm = Eigen::MatrixXd::Zero(2, 2);
        Hm << dx/r, dy/r, -dy/r2, dx/r2;
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

    Eigen::VectorXd x_;
    Eigen::MatrixXd P_;
    Eigen::Matrix2d R_;
    std::vector<int> class_ids_;
    std::vector<double> hit_counts_;
};

} // namespace slam_core
