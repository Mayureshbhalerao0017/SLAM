clear; clc; close all;

%% 1. Simulation Setup
T_max = 20;           % Simulation duration [s]
dt = 0.1;             % Time step [s]
steps = T_max / dt;

% Ground Truth Landmark Map (Randomly scattered)
n_landmarks = 15;
map_size = 40;        % meters
GT_Landmarks = (rand(2, n_landmarks) - 0.5) * map_size;

% Sensor Parameters
max_range = 15;       % Max LiDAR range [m]
fov = deg2rad(120);   % Field of View [rad]

% Noise Parameters (Standard Deviations)
sigma_range = 0.1;    % Measurement noise [m]
sigma_bearing = 0.05; % Measurement noise [rad]
sigma_v = 0.5;        % Velocity noise [m/s]
sigma_yaw = 0.1;      % Yaw rate noise [rad/s]

% Matrices for SLAM
R_meas = diag([sigma_range^2, sigma_bearing^2]);
Q_process = diag([0.1, 0.1, 0.05]).^2;

%% 2. Initialization
% Ground Truth Robot State [x, y, theta]
pose_gt = [0; 0; 0]; 

% SLAM Inputs (Start empty)
x_est = []; 
P_est = []; 
landmark_ids = [];

% History for plotting
h_gt = [];
h_est = [];

figure(1); set(gcf, 'Color', 'w');
hold on; axis equal; grid on;
xlabel('X [m]'); ylabel('Y [m]'); title('EKF SLAM Validation');
% Plot GT Landmarks
plot(GT_Landmarks(1,:), GT_Landmarks(2,:), 'k*', 'MarkerSize', 8, 'DisplayName', 'True Landmarks');

%% 3. Simulation Loop
for k = 1:steps
    %% A. Move Robot (Ground Truth Kinematics)
    % Driving in a circle: const velocity, const yaw rate
    vx_gt = 5.0; 
    vy_gt = 0.0;
    yaw_rate_gt = 0.4; % rad/s
    
    % Update GT Pose
    c = cos(pose_gt(3)); s = sin(pose_gt(3));
    pose_gt(1) = pose_gt(1) + (vx_gt*c - vy_gt*s) * dt;
    pose_gt(2) = pose_gt(2) + (vx_gt*s + vy_gt*c) * dt;
    pose_gt(3) = pose_gt(3) + yaw_rate_gt * dt;
    
    %% B. Generate Noisy Sensor Data
    % 1. Noisy Odometry (Inputs to EKF)
    vx_noise = vx_gt + randn * sigma_v;
    vy_noise = vy_gt + randn * sigma_v;
    yaw_rate_noise = yaw_rate_gt + randn * sigma_yaw;
    
    % 2. Generate Noisy Measurements (z_list)
    z_list = [];
    
    % Check every landmark to see if it is visible
    for i = 1:n_landmarks
        dx = GT_Landmarks(1,i) - pose_gt(1);
        dy = GT_Landmarks(2,i) - pose_gt(2);
        
        range_true = sqrt(dx^2 + dy^2);
        bearing_true = atan2(dy, dx) - pose_gt(3);
        bearing_true = wrapToPi(bearing_true); % Normalize
        
        % Check Sensor Constraints
        if range_true < max_range && abs(bearing_true) < fov/2
            % Add Measurement Noise
            r_meas = range_true + randn * sigma_range;
            b_meas = bearing_true + randn * sigma_bearing;
            
            % Append to z_list
            z_list = [z_list, [r_meas; b_meas]];
        end
    end
    
    %% C. Run Your EKF SLAM
    % We assume mappingPhase = true for this test
    [x_est, P_est, landmark_ids, assoc] = ekf_slam(x_est, P_est, z_list, ...
                                                   R_meas, vx_noise, vy_noise, ...
                                                   yaw_rate_noise, dt, ...
                                                   Q_process, true, landmark_ids);
    
    %% D. Visualization
    h_gt(:,end+1) = pose_gt;
    h_est(:,end+1) = x_est(1:3);
    
    % Real-time Plotting (every 10 frames for speed)
    if mod(k, 10) == 0
        cla;
        % Re-plot static ground truth
        plot(GT_Landmarks(1,:), GT_Landmarks(2,:), 'g*', 'MarkerSize', 8); 
        hold on; grid on; axis equal;
        
        % Plot Paths
        plot(h_gt(1,:), h_gt(2,:), 'b--', 'LineWidth', 1.5); 
        plot(h_est(1,:), h_est(2,:), 'r-', 'LineWidth', 2);
        
        % Plot Estimated Landmarks
        if length(x_est) > 3
            lm_x = x_est(4:2:end);
            lm_y = x_est(5:2:end);
            plot(lm_x, lm_y, 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);
            
            % Draw covariances (simple uncertainty radii)
            for m = 1:length(lm_x)
                idx = 3 + (m-1)*2 + 1;
                sigma_x = sqrt(P_est(idx, idx));
                viscircles([lm_x(m), lm_y(m)], sigma_x*2, 'Color', 'r', 'LineWidth', 0.5);
            end
        end
        
        legend('True Landmarks', 'True Path', 'SLAM Est', 'Est Landmarks');
        title(sprintf('Step %d/%d | Landmarks: %d', k, steps, (length(x_est)-3)/2));
        drawnow;
    end
end