function slam_comparison_driver()
    % SLAM_COMPARISON_DRIVER
    % Runs EKF, FastSLAM, and GraphSLAM on identical synthetic data.
    % Metric: RMSE (Accuracy) vs Computation Time.

    clc; clear; close all;

    %% 1. Configuration & Common Data Generation
    fprintf('--- 1. Generating Shared Synthetic Data ---\n');
    DT = 0.1;
    SIM_TIME = 50.0;
    
    % True Noise Parameters (Used for generation)
    Q_true = diag([0.1, 0.0, deg2rad(1.0)]).^2; % [vx, vy, yaw_rate]
    R_true = diag([0.5, deg2rad(5.0)]).^2;      % [range, bearing]
    
    % Solver Noise Parameters (Passed to algorithms)
    Q_solver = diag([0.1, 0.1, deg2rad(1.0)]).^2; 
    R_solver = R_true;
    
    % Landmarks [x, y]
    lm_true = [ 10, 0; 20, 5; 20, -5; 10, -10; 0, -10; -10, -5; -10, 5; 0, 10 ]'; 
    
    % Generate Data
    [time_vec, hx_true, data_log] = generate_dataset(DT, SIM_TIME, lm_true, Q_true, R_true);
    fprintf('Data generated: %d steps.\n', length(time_vec));

    %% 2. Run EKF SLAM
    fprintf('\n--- 2. Running EKF SLAM ---\n');
    x_ekf = []; P_ekf = []; lm_ids = [];
    hx_ekf = zeros(3, length(time_vec));
    
    tic_ekf = tic;
    for k = 1:length(time_vec)
        u = data_log(k).u;      % [vx; vy; yaw_rate]
        z = data_log(k).z_ekf;  % [range; bearing]
        
        % Call user's EKF function
        [x_ekf, P_ekf, lm_ids, ~] = ekf_slam( ...
            x_ekf, P_ekf, z, R_solver, ...
            u(1), u(2), u(3), DT, ...
            Q_solver, true, lm_ids ...
        );
        hx_ekf(:, k) = x_ekf(1:3);
    end
    t_ekf = toc(tic_ekf);
    fprintf('EKF Finished. Time: %.4fs\n', t_ekf);

    %% 3. Run FastSLAM 1.0
    fprintf('\n--- 3. Running FastSLAM 1.0 ---\n');
    % Init Particles
    N_PARTICLE = 150;
    particles = init_particles(N_PARTICLE, size(lm_true, 2));
    hx_fast = zeros(3, length(time_vec));
    
    tic_fast = tic;
    for k = 1:length(time_vec)
        u_fast = [data_log(k).u(1); data_log(k).u(3)]; % [v; w]
        z_fast = data_log(k).z_fast; % [range; bearing; id]
        
        % Call user's FastSLAM core function
        % Note: We assume 'fast_slam_core' is the function extracted from your previous file
        particles = fast_slam_core(particles, u_fast, z_fast);
        
        % Compute mean state
        hx_fast(:, k) = calc_particle_mean(particles);
    end
    t_fast = toc(tic_fast);
    fprintf('FastSLAM Finished. Time: %.4fs\n', t_fast);

    %% 4. Run Graph SLAM
    fprintf('\n--- 4. Running Graph SLAM ---\n');
    % Graph SLAM requires a Dead Reckoning trajectory (hxDR) and history of observations (hz)
    % We must build these first (Prep time usually not counted in solver time, but we can include it)
    
    % Prep Data formats
    hxDR = zeros(3, length(time_vec));
    hz_graph = cell(1, length(time_vec));
    
    % Build Dead Reckoning Path & Observation Cell Array
    x_dr = [0;0;0];
    for k = 1:length(time_vec)
        u = data_log(k).u;
        x_dr = motion_model(x_dr, [u(1); u(3)], DT);
        hxDR(:, k) = x_dr;
        
        % Graph SLAM z format: [range, angle, phi(unused), id]
        % data_log.z_fast has [range; angle; id]
        z_f = data_log(k).z_fast;
        if ~isempty(z_f)
            z_g = zeros(size(z_f, 2), 4);
            z_g(:, 1) = z_f(1, :)'; % r
            z_g(:, 2) = z_f(2, :)'; % angle
            z_g(:, 3) = 0;          % phi (global) - unused in your code
            z_g(:, 4) = z_f(3, :)'; % id
            hz_graph{k} = z_g;
        else
            hz_graph{k} = [];
        end
    end
    
    tic_graph = tic;
    % Call user's Graph SLAM core function
    % Params: x_init, hz, MAX_ITR, s1, s2, s3
    C_SIGMA = [0.1, 0.1, deg2rad(1.0)];
    x_graph = graph_slam_core(hxDR, hz_graph, 10, C_SIGMA(1), C_SIGMA(2), C_SIGMA(3));
    
    t_graph = toc(tic_graph);
    fprintf('Graph SLAM Finished. Time: %.4fs\n', t_graph);
    
    % Align Graph output (which might be poses) to history
    hx_graph = x_graph(1:3, :);

    %% 5. Analysis & Plotting
    rmse_ekf   = calc_rmse(hx_true, hx_ekf);
    rmse_fast  = calc_rmse(hx_true, hx_fast);
    rmse_graph = calc_rmse(hx_true, hx_graph);
    
    % --- Plot 1: Trajectories ---
    figure('Name', 'SLAM Comparison', 'Color', 'w', 'Position', [100, 100, 1400, 600]);
    
    subplot(1, 2, 1); hold on; grid on; axis equal;
    plot(lm_true(1,:), lm_true(2,:), 'pk', 'MarkerFaceColor','y', 'MarkerSize', 10, 'DisplayName', 'Landmarks');
    plot(hx_true(1,:), hx_true(2,:), 'y--', 'LineWidth', 2, 'DisplayName', 'Ground Truth');
    plot(hx_ekf(1,:), hx_ekf(2,:), 'b-', 'LineWidth', 1.5, 'DisplayName', 'EKF SLAM');
    plot(hx_fast(1,:), hx_fast(2,:), 'g-', 'LineWidth', 1.5, 'DisplayName', 'FastSLAM');
    plot(hx_graph(1,:), hx_graph(2,:), 'r-.', 'LineWidth', 1.5, 'DisplayName', 'Graph SLAM');
    
    xlabel('X [m]'); ylabel('Y [m]'); title('Map & Trajectory Comparison');
    legend('Location', 'best');
    
    % --- Plot 2: Accuracy vs Computation Matrix ---
    subplot(1, 2, 2);
    
    % Prepare Bar Data
    % Normalize for visualization if scales are wildly different, 
    % but raw values are usually better for engineering analysis.
    metrics = [t_ekf, rmse_ekf; t_fast, rmse_fast; t_graph, rmse_graph];
    names = {'EKF', 'FastSLAM', 'Graph'};
    
    yyaxis left
    b1 = bar(1:3, metrics(:,1), 0.4, 'FaceColor', 'b');
    ylabel('Computation Time (s)');
    
    yyaxis right
    hold on;
    p1 = plot(1:3, metrics(:,2), 'ro-', 'LineWidth', 2, 'MarkerFaceColor', 'r');
    ylabel('RMSE (m)');
    
    xticks(1:3); xticklabels(names);
    title('Performance Matrix: Speed vs Accuracy');
    grid on;
    
    % Print Table
    fprintf('\n=== PERFORMANCE MATRIX ===\n');
    fprintf('%-10s | %-10s | %-10s\n', 'Method', 'Time(s)', 'RMSE(m)');
    fprintf('------------------------------------\n');
    fprintf('%-10s | %10.4f | %10.4f\n', 'EKF', t_ekf, rmse_ekf);
    fprintf('%-10s | %10.4f | %10.4f\n', 'FastSLAM', t_fast, rmse_fast);
    fprintf('%-10s | %10.4f | %10.4f\n', 'Graph', t_graph, rmse_graph);
end

%% --- HELPER FUNCTIONS (Driver Logic) ---

function [time_vec, hx_true, data_log] = generate_dataset(dt, T, lm, Q, R)
    time_vec = 0:dt:T;
    steps = length(time_vec);
    hx_true = zeros(3, steps);
    x = [0;0;0];
    hx_true(:,1) = x;
    
    data_log = struct('u', {}, 'z_ekf', {}, 'z_fast', {});
    
    for k = 1:steps
        t = time_vec(k);
        % Control (Figure 8)
        v = 2.0; w = 0.3 * cos(0.2 * t);
        
        % Move Ground Truth
        x = motion_model(x, [v;w], dt);
        if k < steps, hx_true(:, k+1) = x; end
        
        % Noisy Input
        vn = v + randn*sqrt(Q(1,1));
        wn = w + randn*sqrt(Q(3,3));
        
        % Observations
        z_ekf = [];
        z_fast = [];
        for i = 1:size(lm, 2)
            dx = lm(1,i) - x(1);
            dy = lm(2,i) - x(2);
            d = hypot(dx, dy);
            phi = wrapToPi_local(atan2(dy, dx) - x(3));
            
            if d < 15.0 % Max Range
                dn = d + randn*sqrt(R(1,1));
                bn = phi + randn*sqrt(R(2,2));
                
                % EKF expects: [range; bearing] (Association is internal)
                % But for fairness, we usually pass ID if available. 
                % Your EKF code does internal NN association.
                z_ekf = [z_ekf, [dn; bn]]; %#ok<AGROW>
                
                % FastSLAM expects: [range; bearing; id]
                % GraphSLAM expects: [range; bearing; id]
                z_fast = [z_fast, [dn; bn; i]]; %#ok<AGROW>
            end
        end
        
        data_log(k).u = [vn; 0; wn]; % vx, vy, yaw_rate
        data_log(k).z_ekf = z_ekf;
        data_log(k).z_fast = z_fast;
    end
end

function x = motion_model(x, u, dt)
    x(1) = x(1) + u(1)*cos(x(3))*dt;
    x(2) = x(2) + u(1)*sin(x(3))*dt;
    x(3) = wrapToPi_local(x(3) + u(2)*dt);
end

function particles = init_particles(N, n_lm)
    % Helper to initialize struct array matching FastSLAM requirements
    dummy_lm = struct('x', 0, 'y', 0, 'P', zeros(2), 'init', false);
    % Note: Adjust 'lm' size to n_lm if your code uses fixed array
    p.w = 1/N; p.x = 0; p.y = 0; p.yaw = 0;
    % Assuming your fast_slam_core handles dynamic or fixed size, 
    % we initialize empty or fixed.
    % Based on previous code:
    p.lm = zeros(n_lm, 2); % Position
    p.lmP = zeros(n_lm*2, 2); % Stacked Covariance
    particles = repmat(p, N, 1);
end

function x_mean = calc_particle_mean(particles)
    x = 0; y = 0; cx = 0; cy = 0;
    for i = 1:length(particles)
        w = particles(i).w;
        x = x + w * particles(i).x;
        y = y + w * particles(i).y;
        cx = cx + w * cos(particles(i).yaw);
        cy = cy + w * sin(particles(i).yaw);
    end
    x_mean = [x; y; atan2(cy, cx)];
end

function rmse = calc_rmse(traj_true, traj_est)
    % Ensure lengths match
    n = min(size(traj_true, 2), size(traj_est, 2));
    diff = traj_true(1:2, 1:n) - traj_est(1:2, 1:n);
    sq_err = sum(diff.^2, 1);
    rmse = sqrt(mean(sq_err));
end

function ang = wrapToPi_local(ang)
    ang = mod(ang + pi, 2*pi) - pi;
end