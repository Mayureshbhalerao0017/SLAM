#pragma once
#include "SLAMDataTypes.hpp"
#include <eigen3/Eigen/Dense>
#include <vector>
#include <random>

namespace slam_core {

struct ParticleLandmark {
    Eigen::Vector2d mu;
    Eigen::Matrix2d sigma;
    int class_id;
    double hits;
};

struct Particle {
    double weight;
    Pose2D pose;
    Eigen::Matrix3d P; 
    std::vector<ParticleLandmark> map;
};

class FastSLAMCore {
public:
    FastSLAMCore(int num_particles = 30) : num_particles_(num_particles), gen_(rd_()) {
        Q_control_ << std::pow(0.2, 2), 0, 0, std::pow(0.15, 2); 
        R_obs_ << std::pow(0.1, 2), 0, 0, std::pow(0.035, 2); 

        for (int i = 0; i < num_particles_; ++i) {
            Particle p;
            p.weight = 1.0 / num_particles_;
            p.pose = {0.0, 0.0, 0.0};
            p.P = Eigen::Matrix3d::Identity() * 0.01;
            particles_.push_back(p);
        }
    }

    void predict(double dt, double vx, double yaw_rate) {
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

            p.pose.x += (noisy_vx * std::cos(p.pose.yaw) - noisy_vy * std::sin(p.pose.yaw)) * dt;
            p.pose.y += (noisy_vx * std::sin(p.pose.yaw) + noisy_vy * std::cos(p.pose.yaw)) * dt;
            p.pose.yaw = wrapToPi(p.pose.yaw + noisy_yaw_rate * dt);

            Eigen::Matrix3d G;
            G << 1.0, 0.0, -(noisy_vx * std::sin(p.pose.yaw) + noisy_vy * std::cos(p.pose.yaw)) * dt,
                 0.0, 1.0,  (noisy_vx * std::cos(p.pose.yaw) - noisy_vy * std::sin(p.pose.yaw)) * dt,
                 0.0, 0.0,  1.0;
            
            p.P = G * p.P * G.transpose() + (Eigen::Matrix3d::Identity() * 0.05);
        }
    }

    void update(const std::vector<Observation>& observations, bool is_loop_closed) {
        for (auto &p : particles_) {
            for (const auto &z : observations) {
                double global_theta = wrapToPi(p.pose.yaw + z.bearing);
                double global_zx = p.pose.x + z.range * std::cos(global_theta);
                double global_zy = p.pose.y + z.range * std::sin(global_theta);

                int best_idx = -1;
                double best_md = 1e9;

                for (size_t i = 0; i < p.map.size(); ++i) {
                    if (p.map[i].class_id != z.class_id) continue;
                    
                    double dx = p.map[i].mu(0) - global_zx;
                    double dy = p.map[i].mu(1) - global_zy;
                    if (std::hypot(dx, dy) > 2.0) continue; 

                    double dx_r = p.map[i].mu(0) - p.pose.x;
                    double dy_r = p.map[i].mu(1) - p.pose.y;
                    double q = dx_r * dx_r + dy_r * dy_r;
                    double r_hat = std::sqrt(q) + 1e-6;

                    Eigen::Matrix2d Hf;
                    Hf <<  dx_r / r_hat,  dy_r / r_hat, 
                          -dy_r / q,      dx_r / q;

                    Eigen::Matrix2d S = Hf * p.map[i].sigma * Hf.transpose() + R_obs_;
                    Eigen::Vector2d z_hat(r_hat, wrapToPi(std::atan2(dy_r, dx_r) - p.pose.yaw));
                    Eigen::Vector2d v(z.range - z_hat(0), wrapToPi(z.bearing - z_hat(1)));
                    
                    double md = v.transpose() * S.inverse() * v;
                    if (md < best_md && md < 11.34) { best_md = md; best_idx = i; }
                }

                if (best_idx >= 0) {
                    ParticleLandmark &lm = p.map[best_idx];
                    double dx_r = lm.mu(0) - p.pose.x, dy_r = lm.mu(1) - p.pose.y;
                    double q = dx_r * dx_r + dy_r * dy_r, r_hat = std::sqrt(q) + 1e-6;

                    Eigen::Matrix2d Hf; Hf << dx_r/r_hat, dy_r/r_hat, -dy_r/q, dx_r/q;
                    Eigen::MatrixXd Hv(2, 3); Hv << -dx_r/r_hat, -dy_r/r_hat, 0.0, dy_r/q, -dx_r/q, -1.0;

                    Eigen::Matrix2d S = Hf * lm.sigma * Hf.transpose() + R_obs_;
                    Eigen::Matrix2d S_inv = S.inverse();
                    Eigen::Vector2d z_hat(r_hat, wrapToPi(std::atan2(dy_r, dx_r) - p.pose.yaw));
                    Eigen::Vector2d v(z.range - z_hat(0), wrapToPi(z.bearing - z_hat(1)));

                    if (!is_loop_closed) {
                        Eigen::Matrix2d K = lm.sigma * Hf.transpose() * S_inv;
                        lm.mu += K * v;
                        lm.sigma = (Eigen::Matrix2d::Identity() - K * Hf) * lm.sigma;
                    }
                    lm.hits += 1.0; 

                    Eigen::Matrix3d P_proposal = (Hv.transpose() * S_inv * Hv + p.P.inverse()).inverse();
                    Eigen::Vector3d state_update = P_proposal * Hv.transpose() * S_inv * v;
                    
                    p.pose.x += state_update(0);
                    p.pose.y += state_update(1);
                    p.pose.yaw = wrapToPi(p.pose.yaw + state_update(2));
                    p.P = P_proposal;

                    double det_S = S.determinant();
                    if (det_S > 1e-6) p.weight *= (std::exp(-0.5 * best_md) / (2.0 * M_PI * std::sqrt(det_S)));

                } else if (!is_loop_closed) {
                    ParticleLandmark new_lm;
                    new_lm.mu << global_zx, global_zy;
                    double c = std::cos(global_theta), s = std::sin(global_theta);
                    Eigen::Matrix2d Gz; Gz << c, -z.range*s, s, z.range*c;
                          
                    new_lm.sigma = Gz * R_obs_ * Gz.transpose();
                    new_lm.class_id = z.class_id;
                    new_lm.hits = 1.0; 
                    p.map.push_back(new_lm);
                }
            }
            for (auto &lm : p.map) { if (lm.hits > 0.0) lm.hits -= 0.05; } // Decay
        }
        resample();
    }

    Particle getBestParticle() const {
        Particle best = particles_[0];
        for (const auto &p : particles_) { if (p.weight > best.weight) best = p; }
        return best;
    }

private:
    void resample() {
        double sum_w = 0.0;
        for (const auto &p : particles_) sum_w += p.weight;
        
        if (sum_w < 1e-9) { 
            for (auto &p : particles_) { p.weight = 1.0 / num_particles_; p.P = Eigen::Matrix3d::Identity() * 0.5; }
            return;
        }
        
        double w_sq_sum = 0.0;
        for (auto &p : particles_) {
            p.weight /= sum_w;
            w_sq_sum += p.weight * p.weight;
        }

        if ((1.0 / w_sq_sum) >= num_particles_ / 2.0) return; 

        std::vector<Particle> new_particles;
        std::uniform_real_distribution<double> dist(0.0, 1.0 / num_particles_);
        double r = dist(gen_);
        double c = particles_[0].weight;
        int idx = 0;

        for (int m = 0; m < num_particles_; ++m) {
            double u = r + m * (1.0 / num_particles_);
            while (u > c && idx < num_particles_ - 1) {
                idx++; c += particles_[idx].weight;
            }
            Particle p_copy = particles_[idx];
            p_copy.weight = 1.0 / num_particles_;
            new_particles.push_back(p_copy);
        }
        particles_ = new_particles;

        std::normal_distribution<double> jitter_pos(0.0, 0.2); 
        std::normal_distribution<double> jitter_yaw(0.0, 0.05); 
        for (int i = 1; i < num_particles_; ++i) {
            particles_[i].pose.x += jitter_pos(gen_);
            particles_[i].pose.y += jitter_pos(gen_);
            particles_[i].pose.yaw = wrapToPi(particles_[i].pose.yaw + jitter_yaw(gen_));
            particles_[i].P += Eigen::Matrix3d::Identity() * 1e-4; 
        }
    }

    int num_particles_;
    std::vector<Particle> particles_;
    Eigen::Matrix2d Q_control_, R_obs_;
    std::random_device rd_;
    std::mt19937 gen_;
};

} // namespace slam_core
