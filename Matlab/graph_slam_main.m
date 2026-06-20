function graph_based_slam()
    % Graph based SLAM example
    % Converted from Python to MATLAB
    %
    % Original Author: Atsushi Sakai
    % Reference: A Tutorial on Graph-Based SLAM (Grisetti et al.)

    clear; clc; close all;

    % --- Simulation Parameters ---
    % Q_sim: covariance of sensor noise [range, bearing]
    Q_sim = diag([0.2, deg2rad(1.0)]).^2;
    % R_sim: covariance of process noise [velocity, yaw_rate]
    R_sim = diag([0.1, deg2rad(10.0)]).^2;

    DT = 2.0;               % time tick [s]
    SIM_TIME = 100.0;       % simulation time [s]
    MAX_RANGE = 30.0;       % maximum observation range
    STATE_SIZE = 3;         % State size [x,y,yaw]

    % --- Graph SLAM Covariance Parameters ---
    C_SIGMA1 = 0.1;
    C_SIGMA2 = 0.1;
    C_SIGMA3 = deg2rad(1.0);

    MAX_ITR = 20;           % Maximum iteration

    show_graph_d_time = 20.0; % [s]
    show_animation = true;

    % --- Main Loop Setup ---
    time = 0.0;

    % RFID positions [x, y, yaw] (Yaw is unused for point landmarks usually)
    RFID = [10.0, -2.0, 0.0;
            15.0, 10.0, 0.0;
             3.0, 15.0, 0.0;
            -5.0, 20.0, 0.0;
            -5.0,  5.0, 0.0];

    % State Vector [x; y; yaw]
    xTrue = zeros(STATE_SIZE, 1);
    xDR = zeros(STATE_SIZE, 1);  % Dead reckoning

    % History
    hxTrue = [];
    hxDR = [];
    hz = {}; % Cell array to store observations per time step
    
    d_time = 0.0;
    is_init = false;

    % Initialize figure
    if show_animation
        figure('Name', 'Graph Based SLAM', 'Color', 'w');
    end

    while SIM_TIME >= time
        if ~is_init
            hxTrue = xTrue;
            hxDR = xDR;
            is_init = true;
        else
            hxTrue = [hxTrue, xTrue]; %#ok<AGROW>
            hxDR = [hxDR, xDR];       %#ok<AGROW>
        end

        time = time + DT;
        d_time = d_time + DT;
        
        u = calc_input();

        [xTrue, z, xDR, ~] = observation(xTrue, xDR, u, RFID, MAX_RANGE, DT, Q_sim, R_sim);

        % Store observations (z is N x 4)
        hz{end+1} = z; %#ok<AGROW>

        if d_time >= show_graph_d_time
            x_opt = run_graph_based_slam(hxDR, hz, MAX_ITR, C_SIGMA1, C_SIGMA2, C_SIGMA3);
            d_time = 0.0;

            if show_animation
                clf;
                hold on; grid on; axis equal;
                
                plot(RFID(:, 1), RFID(:, 2), '*k', 'MarkerSize', 8);
                plot(hxTrue(1, :), hxTrue(2, :), '-b', 'LineWidth', 1.5, 'DisplayName', 'True');
                plot(hxDR(1, :), hxDR(2, :), '-k', 'LineWidth', 1.5, 'DisplayName', 'DR');
                plot(x_opt(1, :), x_opt(2, :), '-r', 'LineWidth', 1.5, 'DisplayName', 'Optimized');
                
                legend('show');
                title(['Time: ', num2str(time, '%.2f')]);
                xlabel('X [m]'); ylabel('Y [m]');
                drawnow;
                pause(0.1);
            end
        end
    end
    disp('Simulation Finished.');
end

%% --- SLAM Logic Functions ---

function x_opt = run_graph_based_slam(x_init, hz, MAX_ITR, s1, s2, s3)
    disp('start graph based slam');
    
    x_opt = x_init;
    nt = size(x_opt, 2);
    STATE_SIZE = 3;
    n = nt * STATE_SIZE;

    for itr = 1:MAX_ITR
        edges = calc_edges(x_opt, hz, s1, s2, s3);

        H = zeros(n, n);
        b = zeros(n, 1);

        for i = 1:length(edges)
            edge = edges(i);
            [H, b] = fill_H_and_b(H, b, edge);
        end

        % Fix origin (Constraints on the first pose to remove gauge freedom)
        H(1:STATE_SIZE, 1:STATE_SIZE) = H(1:STATE_SIZE, 1:STATE_SIZE) + eye(STATE_SIZE);

        % Solve linear system
        dx = - (H \ b);

        % Update state
        for i = 1:nt
            idx = (i-1)*3;
            x_opt(1:3, i) = x_opt(1:3, i) + dx(idx+1:idx+3);
        end

        diff_val = dx' * dx;
        fprintf('iteration: %d, diff: %f\n', itr, diff_val);
        
        if diff_val < 1.0e-5
            break;
        end
    end
end

function edges = calc_edges(x_list, z_list, s1, s2, s3)
    edges = []; % Struct array
    cost = 0.0;
    
    num_steps = length(z_list);
    
    % Use combinations of time steps (t1 < t2)
    % Note: t1, t2 are 1-based indices here
    for t1 = 1:num_steps
        for t2 = (t1+1):num_steps
            
            x1 = x_list(1, t1); y1 = x_list(2, t1); yaw1 = x_list(3, t1);
            x2 = x_list(1, t2); y2 = x_list(2, t2); yaw2 = x_list(3, t2);

            z1 = z_list{t1};
            z2 = z_list{t2};
            
            if isempty(z1) || isempty(z2)
                continue;
            end
            
            % Check for shared landmark IDs
            % z matrix columns: [range, angle, phi, id]
            for iz1 = 1:size(z1, 1)
                id1 = z1(iz1, 4);
                for iz2 = 1:size(z2, 1)
                    id2 = z2(iz2, 4);
                    
                    if id1 == id2
                        % Found same landmark
                        d1 = z1(iz1, 1);
                        angle1 = z1(iz1, 2);
                        
                        d2 = z2(iz2, 1);
                        angle2 = z2(iz2, 2);
                        
                        edge = calc_edge(x1, y1, yaw1, x2, y2, yaw2, ...
                                         d1, angle1, d2, angle2, ...
                                         t1, t2, s1, s2, s3);
                        
                        if isempty(edges)
                            edges = edge;
                        else
                            edges(end+1) = edge; %#ok<AGROW>
                        end
                        
                        cost = cost + (edge.e' * edge.omega * edge.e);
                    end
                end
            end
        end
    end
    fprintf('cost: %f, n_edge: %d\n', cost, length(edges));
end

function edge = calc_edge(x1, y1, yaw1, x2, y2, yaw2, d1, angle1, d2, angle2, t1, t2, s1, s2, s3)
    tangle1 = pi_2_pi(yaw1 + angle1);
    tangle2 = pi_2_pi(yaw2 + angle2);
    
    tmp1 = d1 * cos(tangle1);
    tmp2 = d2 * cos(tangle2);
    tmp3 = d1 * sin(tangle1);
    tmp4 = d2 * sin(tangle2);

    edge.e = zeros(3, 1);
    edge.e(1) = x2 - x1 - tmp1 + tmp2;
    edge.e(2) = y2 - y1 - tmp3 + tmp4;
    edge.e(3) = 0; % Yaw error is treated as 0 in this specific derivation or ignored for point features in this code

    Rt1 = calc_3d_rotational_matrix(tangle1);
    Rt2 = calc_3d_rotational_matrix(tangle2);

    sig = diag([s1^2, s2^2, s3^2]); % Using provided C_SIGMA params

    % Information matrix
    edge.omega = inv(Rt1 * sig * Rt1' + Rt2 * sig * Rt2');

    edge.d1 = d1;
    edge.d2 = d2;
    edge.yaw1 = yaw1;
    edge.yaw2 = yaw2;
    edge.angle1 = angle1;
    edge.angle2 = angle2;
    edge.id1 = t1;
    edge.id2 = t2;
end

function [H, b] = fill_H_and_b(H, b, edge)
    [A, B] = calc_jacobian(edge);
    
    STATE_SIZE = 3;
    % Convert 1-based time index to 1-based matrix index
    % t1 is 1..N. id1 in struct is the time index.
    % Matrix indices: (id-1)*3 + 1
    
    idx1 = (edge.id1 - 1) * STATE_SIZE + (1:STATE_SIZE);
    idx2 = (edge.id2 - 1) * STATE_SIZE + (1:STATE_SIZE);

    H(idx1, idx1) = H(idx1, idx1) + A' * edge.omega * A;
    H(idx1, idx2) = H(idx1, idx2) + A' * edge.omega * B;
    H(idx2, idx1) = H(idx2, idx1) + B' * edge.omega * A;
    H(idx2, idx2) = H(idx2, idx2) + B' * edge.omega * B;

    b(idx1) = b(idx1) + (A' * edge.omega * edge.e);
    b(idx2) = b(idx2) + (B' * edge.omega * edge.e);
end

function [A, B] = calc_jacobian(edge)
    t1 = edge.yaw1 + edge.angle1;
    A = [-1.0, 0, edge.d1 * sin(t1);
          0, -1.0, -edge.d1 * cos(t1);
          0,    0,    0];

    t2 = edge.yaw2 + edge.angle2;
    B = [1.0, 0, -edge.d2 * sin(t2);
         0, 1.0,  edge.d2 * cos(t2);
         0,   0,    0];
end

%% --- Helper Functions ---

function u = calc_input()
    v = 1.0;        % [m/s]
    yaw_rate = 0.1; % [rad/s]
    u = [v; yaw_rate];
end

function R = calc_3d_rotational_matrix(angle)
    % Z-axis rotation matrix
    c = cos(angle);
    s = sin(angle);
    R = [c, -s, 0;
         s,  c, 0;
         0,  0, 1];
end

function angle = pi_2_pi(angle)
    angle = mod(angle + pi, 2 * pi) - pi;
end

function x = motion_model(x, u, dt)
    F = eye(3);
    B = [dt * cos(x(3)), 0;
         dt * sin(x(3)), 0;
         0,             dt];
    x = F * x + B * u;
end

function [xTrue, z, xDR, ud] = observation(xTrue, xDR, u, RFID, MAX_RANGE, dt, Q_sim, R_sim)
    xTrue = motion_model(xTrue, u, dt);

    z = zeros(0, 4); % [range, angle, phi, id]

    for i = 1:size(RFID, 1)
        dx = RFID(i, 1) - xTrue(1);
        dy = RFID(i, 2) - xTrue(2);
        d = hypot(dx, dy);
        angle = pi_2_pi(atan2(dy, dx) - xTrue(3));
        phi = pi_2_pi(atan2(dy, dx)); % Global angle to feature

        if d <= MAX_RANGE
            dn = d + randn() * sqrt(Q_sim(1, 1));
            angle_noise = randn() * sqrt(Q_sim(2, 2));
            angle_n = angle + angle_noise;
            phi_n = phi + angle_noise;
            
            % Note: Storing 'i' as ID. In Python code, i is 0-indexed index.
            % Here i is 1-indexed. We store 'i' directly.
            zi = [dn, angle_n, phi_n, i];
            z = [z; zi]; %#ok<AGROW>
        end
    end

    % Add noise to input
    ud1 = u(1) + randn() * sqrt(R_sim(1, 1));
    ud2 = u(2) + randn() * sqrt(R_sim(2, 2));
    ud = [ud1; ud2];

    xDR = motion_model(xDR, ud, dt);
end