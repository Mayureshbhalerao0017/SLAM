function fast_slam1_main()
    % FastSLAM 1.0 example
    % Converted from Python to MATLAB
    % Original Author: Atsushi Sakai
    
    clear; clc; close all;

    % --- Configuration Parameters ---
    global Q R Q_SIM R_SIM DT MAX_RANGE STATE_SIZE LM_SIZE N_PARTICLE NTH OFFSET_YAW_RATE_NOISE
    
    % Fast SLAM covariance
    Q = diag([3.0, deg2rad(10.0)]).^2;
    R = diag([1.0, deg2rad(20.0)]).^2;

    % Simulation parameter
    Q_SIM = diag([0.3, deg2rad(2.0)]).^2;
    R_SIM = diag([0.5, deg2rad(10.0)]).^2;
    OFFSET_YAW_RATE_NOISE = 0.01;

    DT = 0.1;           % time tick [s]
    SIM_TIME = 50.0;    % simulation time [s]
    MAX_RANGE = 20.0;   % maximum observation range
    STATE_SIZE = 3;     % State size [x,y,yaw]
    LM_SIZE = 2;        % LM state size [x,y]
    N_PARTICLE = 100;   % number of particles
    NTH = N_PARTICLE / 1.5; % Threshold for resampling

    show_animation = true;

    % --- Initialization ---
    disp('FastSLAM1 start!!');
    time = 0.0;

    % RFID positions [x, y]
    rfid = [10.0, -2.0;
            15.0, 10.0;
            15.0, 15.0;
            10.0, 20.0;
             3.0, 15.0;
            -5.0, 20.0;
            -5.0,  5.0;
           -10.0, 15.0];
       
    n_landmark = size(rfid, 1);

    % State Vector [x; y; yaw]
    x_est = zeros(STATE_SIZE, 1);  % SLAM estimation
    x_true = zeros(STATE_SIZE, 1); % True state
    x_dr = zeros(STATE_SIZE, 1);   % Dead reckoning

    % History
    hx_est = x_est;
    hx_true = x_true;
    hx_dr = x_dr;

    % Initialize Particles
    % We use a struct array to represent particles
    particles(N_PARTICLE) = struct();
    for i = 1:N_PARTICLE
        particles(i).w = 1.0 / N_PARTICLE;
        particles(i).x = 0.0;
        particles(i).y = 0.0;
        particles(i).yaw = 0.0;
        % landmark x-y positions (N_LANDMARK x 2)
        particles(i).lm = zeros(n_landmark, LM_SIZE);
        % landmark position covariance (2*N_LANDMARK x 2)
        % Stacking 2x2 covariances vertically
        particles(i).lmP = zeros(n_landmark * LM_SIZE, LM_SIZE);
    end

    % Prepare Figure
    if show_animation
        figure('Name', 'FastSLAM 1.0', 'Color', 'w');
    end

    % --- Main Loop ---
    while time <= SIM_TIME
        time = time + DT;
        u = calc_input(time);

        [x_true, z, x_dr, ud] = observation(x_true, x_dr, u, rfid);

        particles = fast_slam1(particles, ud, z);

        x_est = calc_final_state(particles);

        % Store history
        hx_est = [hx_est, x_est];       %#ok<AGROW>
        hx_dr  = [hx_dr, x_dr];         %#ok<AGROW>
        hx_true = [hx_true, x_true];    %#ok<AGROW>

        if show_animation
            clf; hold on; grid on; axis equal;
            
            % Plot Landmarks
            plot(rfid(:, 1), rfid(:, 2), '*k', 'DisplayName', 'True Landmarks');
            
            % Plot Particles (downsample for speed if needed)
            px = [particles.x];
            py = [particles.y];
            plot(px, py, '.r', 'DisplayName', 'Particles');
            
            % Plot Estimated Landmarks from the first particle (for visualization)
            lm_vis = particles(1).lm;
            % Filter only initialized landmarks (not 0,0)
            valid_lm = abs(lm_vis(:,1)) > 0.01; 
            if any(valid_lm)
                plot(lm_vis(valid_lm, 1), lm_vis(valid_lm, 2), 'xb', 'DisplayName', 'Est Landmarks');
            end

            % Plot Trajectories
            plot(hx_true(1, :), hx_true(2, :), '-b', 'LineWidth', 1.5, 'DisplayName', 'True Path');
            plot(hx_dr(1, :), hx_dr(2, :), '-k', 'LineWidth', 1, 'DisplayName', 'DR Path');
            plot(hx_est(1, :), hx_est(2, :), '-r', 'LineWidth', 1.5, 'DisplayName', 'SLAM Path');
            
            title(['Time: ' num2str(time, '%.2f') 's']);
            legend('show', 'Location', 'best');
            drawnow;
        end
    end
end

%% --- SLAM Core Functions ---

function particles = fast_slam1(particles, u, z)
    particles = predict_particles(particles, u);
    particles = update_with_observation(particles, z);
    particles = resampling(particles);
end

function particles = predict_particles(particles, u)
    global R N_PARTICLE STATE_SIZE
    for i = 1:N_PARTICLE
        px = [particles(i).x; particles(i).y; particles(i).yaw];
        
        % Add noise to input
        % randn(1,2) * sqrt(R) -> (1x2) vector
        noise = (randn(1, 2) * sqrt(R))'; 
        ud = u + noise;
        
        px = motion_model(px, ud);
        
        particles(i).x = px(1);
        particles(i).y = px(2);
        particles(i).yaw = px(3);
    end
end

function particles = update_with_observation(particles, z)
    global N_PARTICLE Q
    % z is 3 x N_obs (range, angle, id)
    
    if isempty(z)
        return;
    end
    
    for iz = 1:size(z, 2)
        l_id = floor(z(3, iz)); % Landmark ID (1-based)
        
        for ip = 1:N_PARTICLE
            % Check if landmark is new (using 0 initialization check)
            % In Python code: abs(lm[id, 0]) <= 0.01
            if abs(particles(ip).lm(l_id, 1)) <= 0.01
                particles(ip) = add_new_landmark(particles(ip), z(:, iz), Q, l_id);
            else
                w = compute_weight(particles(ip), z(:, iz), Q, l_id);
                particles(ip).w = particles(ip).w * w;
                particles(ip) = update_landmark(particles(ip), z(:, iz), Q, l_id);
            end
        end
    end
end

function particle = add_new_landmark(particle, z, Q_cov, l_id)
    r = z(1);
    b = z(2);
    % l_id passed as argument

    s = sin(pi_2_pi(particle.yaw + b));
    c = cos(pi_2_pi(particle.yaw + b));

    particle.lm(l_id, 1) = particle.x + r * c;
    particle.lm(l_id, 2) = particle.y + r * s;

    % Covariance
    dx = r * c;
    dy = r * s;
    d2 = dx^2 + dy^2;
    d = sqrt(d2);
    
    Gz = [dx / d, dy / d;
         -dy / d2, dx / d2];
     
    % Compute inverse Jacobian once
    Gz_inv = inv(Gz);
    
    % Store covariance in the stacked array
    % Indices: (2*l_id - 1) to (2*l_id)
    idx_start = 2*l_id - 1;
    particle.lmP(idx_start:idx_start+1, :) = Gz_inv * Q_cov * Gz_inv';
    
end

function w = compute_weight(particle, z, Q_cov, l_id)
    xf = particle.lm(l_id, :)';
    idx_start = 2*l_id - 1;
    Pf = particle.lmP(idx_start:idx_start+1, :);

    [zp, ~, ~, Sf] = compute_jacobians(particle, xf, Pf, Q_cov);

    dx = z(1:2) - zp;
    dx(2) = pi_2_pi(dx(2));

    try
        % Calculate Gaussian likelihood
        % w = 1/sqrt(|2pi*S|) * exp(-0.5 * dx' * inv(S) * dx)
        invS = inv(Sf);
        num = exp(-0.5 * dx' * invS * dx);
        den = 2.0 * pi * sqrt(det(Sf));
        w = num / den;
    catch
        % Singularity fallback
        w = 1.0;
    end
end

function particle = update_landmark(particle, z, Q_cov, l_id)
    xf = particle.lm(l_id, :)';
    idx_start = 2*l_id - 1;
    Pf = particle.lmP(idx_start:idx_start+1, :);

    [zp, ~, Hf, ~] = compute_jacobians(particle, xf, Pf, Q_cov);

    dz = z(1:2) - zp;
    dz(2) = pi_2_pi(dz(2));

    [xf, Pf] = update_kf_with_cholesky(xf, Pf, dz, Q_cov, Hf);

    particle.lm(l_id, :) = xf';
    particle.lmP(idx_start:idx_start+1, :) = Pf;
end

function [zp, Hv, Hf, Sf] = compute_jacobians(particle, xf, Pf, Q_cov)
    dx = xf(1) - particle.x;
    dy = xf(2) - particle.y;
    d2 = dx^2 + dy^2;
    d = sqrt(d2);

    zp = [d; pi_2_pi(atan2(dy, dx) - particle.yaw)];

    % Jacobian w.r.t Vehicle state (unused in standard FastSLAM1 weight but good for ref)
    Hv = [-dx / d, -dy / d, 0.0;
           dy / d2, -dx / d2, -1.0];

    % Jacobian w.r.t Landmark state
    Hf = [dx / d, dy / d;
         -dy / d2, dx / d2];

    Sf = Hf * Pf * Hf' + Q_cov;
end

function [x, P] = update_kf_with_cholesky(xf, Pf, v, Q_cov, Hf)
    % Efficient KF Update using Cholesky Decomposition
    PHt = Pf * Hf';
    S = Hf * PHt + Q_cov;
    
    % Ensure Symmetry
    S = (S + S') * 0.5;
    
    % MATLAB chol() returns Upper Triangular by default (U' * U = S)
    % Python's np.linalg.cholesky returns Lower (L * L' = S)
    % The Python code uses s_chol = np...cholesky(S).T which is UPPER.
    
    try
        s_chol = chol(S); % Upper triangular
        s_chol_inv = inv(s_chol);
        W1 = PHt * s_chol_inv;
        W = W1 * s_chol_inv'; % Kalman Gain
        
        x = xf + W * v;
        P = Pf - W1 * W1';
    catch
        % Fallback if Cholesky fails (non-positive definite)
        K = PHt / S;
        x = xf + K * v;
        P = (eye(2) - K * Hf) * Pf;
    end
end

function particles = resampling(particles)
    global N_PARTICLE NTH
    
    % Normalize weights
    particles = normalize_weight(particles);
    
    pw = [particles.w];
    n_eff = 1.0 / (pw * pw'); % Effective particle number
    
    if n_eff < NTH
        % Low Variance Resampling
        w_cum = cumsum(pw);
        base = cumsum(ones(1, N_PARTICLE) / N_PARTICLE) - 1/N_PARTICLE;
        resample_id = base + rand(1, N_PARTICLE) / N_PARTICLE;
        
        keep_inds = zeros(1, N_PARTICLE);
        ind = 1;
        for i = 1:N_PARTICLE
            while ind < N_PARTICLE && resample_id(i) > w_cum(ind)
                ind = ind + 1;
            end
            keep_inds(i) = ind;
        end
        
        % Create new particles array based on indices
        new_particles = particles(keep_inds);
        
        % Reset weights
        for i = 1:N_PARTICLE
            new_particles(i).w = 1.0 / N_PARTICLE;
        end
        
        particles = new_particles;
    end
end

function particles = normalize_weight(particles)
    sum_w = sum([particles.w]);
    if sum_w ~= 0
        for i = 1:length(particles)
            particles(i).w = particles(i).w / sum_w;
        end
    else
        for i = 1:length(particles)
            particles(i).w = 1.0 / length(particles);
        end
    end
end

function x_est = calc_final_state(particles)
    global STATE_SIZE N_PARTICLE
    
    particles = normalize_weight(particles);
    
    x_est = zeros(STATE_SIZE, 1);
    for i = 1:N_PARTICLE
        x_est(1) = x_est(1) + particles(i).w * particles(i).x;
        x_est(2) = x_est(2) + particles(i).w * particles(i).y;
        x_est(3) = x_est(3) + particles(i).w * particles(i).yaw;
    end
    x_est(3) = pi_2_pi(x_est(3));
end

%% --- Simulation Helpers ---

function u = calc_input(time)
    if time <= 3.0
        v = 0.0;
        yaw_rate = 0.0;
    else
        v = 1.0; % [m/s]
        yaw_rate = 0.1; % [rad/s]
    end
    u = [v; yaw_rate];
end

function x = motion_model(x, u)
    global DT
    F = eye(3);
    B = [DT * cos(x(3)), 0;
         DT * sin(x(3)), 0;
         0,             DT];
     
    x = F * x + B * u;
    x(3) = pi_2_pi(x(3));
end

function [x_true, z, x_dr, ud] = observation(x_true, x_dr, u, rfid)
    global Q_SIM R_SIM MAX_RANGE OFFSET_YAW_RATE_NOISE
    
    x_true = motion_model(x_true, u);
    
    z = zeros(3, 0); % [range; angle; id]
    
    for i = 1:size(rfid, 1)
        dx = rfid(i, 1) - x_true(1);
        dy = rfid(i, 2) - x_true(2);
        d = hypot(dx, dy);
        angle = pi_2_pi(atan2(dy, dx) - x_true(3));
        
        if d <= MAX_RANGE
            dn = d + randn() * sqrt(Q_SIM(1, 1));
            angle_noise = angle + randn() * sqrt(Q_SIM(2, 2));
            
            % ID is i (1-based index)
            zi = [dn; pi_2_pi(angle_noise); i];
            z = [z, zi]; %#ok<AGROW>
        end
    end
    
    % Add noise to input
    ud1 = u(1) + randn() * sqrt(R_SIM(1, 1));
    ud2 = u(2) + randn() * sqrt(R_SIM(2, 2)) + OFFSET_YAW_RATE_NOISE;
    ud = [ud1; ud2];
    
    x_dr = motion_model(x_dr, ud);
end

function angle = pi_2_pi(angle)
    angle = mod(angle + pi, 2 * pi) - pi;
end