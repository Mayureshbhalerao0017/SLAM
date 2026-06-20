function x_opt = graph_slam_core(x_init, hz, MAX_ITR, s1, s2, s3)
    % GRAPH_SLAM_CORE (Hybrid High-Accuracy Version)
    % - Uses "First + Previous" connections for maximum robustness.
    % - fast O(N) complexity.
    
    STATE_SIZE = 3;
    x_opt = x_init;
    nt = size(x_opt, 2);
    n = nt * STATE_SIZE;

    % Tuning: Trust Odometry a bit more to smooth out noise
    odom_sigma = [0.1; 0.1; deg2rad(2.0)]; 

    for itr = 1:MAX_ITR
        edges = [];
        
        % --- 1. Add Odometry Edges ---
        edges_odom = repmat(create_empty_edge(), 1, nt-1);
        for t = 1:(nt-1)
            edge = create_empty_edge();
            edge.type = 'odom';
            edge.id1 = t;
            edge.id2 = t+1;
            
            x_curr = x_opt(:, t);
            x_next = x_opt(:, t+1);

            
            yaw = x_curr(3);
            dx = x_next(1) - x_curr(1);
            dy = x_next(2) - x_curr(2);
            d_local_x =  cos(yaw)*dx + sin(yaw)*dy;
            d_local_y = -sin(yaw)*dx + cos(yaw)*dy;
            d_local_yaw = pi_2_pi(x_next(3) - x_curr(3));
            
            edge.measure = [d_local_x; d_local_y; d_local_yaw];
            edge.omega = inv(diag(odom_sigma.^2)); 
            edges_odom(t) = edge;
        end
        
        % --- 2. Add Landmark Edges (Hybrid Strategy) ---
        lm_edges = calc_landmark_edges_hybrid(x_opt, hz, s1, s2, s3);
        
        if isempty(lm_edges)
            edges = edges_odom;
        else
            edges = [edges_odom, lm_edges];
        end
        
        % --- 3. Build System ---
        H = sparse(n, n);
        b = zeros(n, 1);

        for i = 1:length(edges)
            edge = edges(i);
            [H, b] = fill_H_and_b(H, b, edge, x_opt);
        end

        % Anchor Start
        H(1:3, 1:3) = H(1:3, 1:3) + eye(3) * 1e6;

        % --- 4. Solve ---
        dx = - (H \ b);
        for i = 1:nt
            idx = (i-1)*3;
            x_opt(1:3, i) = x_opt(1:3, i) + dx(idx+1:idx+3);
            x_opt(3, i) = pi_2_pi(x_opt(3, i));
        end

        if (dx' * dx) < 1.0e-5, break; end
    end
end

function edges = calc_landmark_edges_hybrid(x_list, z_list, s1, s2, s3)
    edges = [];
    
    % Maps to store history:
    % 1. first_sightings: ID -> [Time, Index] (Global Anchor)
    % 2. last_sightings:  ID -> [Time, Index] (Local Chain)
    first_sightings = containers.Map('KeyType','double','ValueType','any');
    last_sightings  = containers.Map('KeyType','double','ValueType','any');
    
    for t = 1:length(z_list)
        z = z_list{t};
        if isempty(z), continue; end
        
        for k = 1:size(z,1)
            id = z(k,4);
            
            % --- Connection 1: Sequential (Chain to Previous) ---
            if isKey(last_sightings, id)
                last_s = last_sightings(id);
                t_prev = last_s(1); idx_prev = last_s(2);
                
                % Create edge to previous sighting
                if t ~= t_prev
                    edges = add_lm_edge(edges, t_prev, idx_prev, t, k, z_list, s1, s2, s3);
                end
            end
            
            % --- Connection 2: Global (Chain to First) ---
            if isKey(first_sightings, id)
                first_s = first_sightings(id);
                t_first = first_s(1); idx_first = first_s(2);
                
                % Add edge to first sighting ONLY if it's different from "previous"
                % (Avoid duplicate edges)
                if isKey(last_sightings, id)
                     last_s = last_sightings(id);
                     if last_s(1) ~= t_first && t ~= t_first
                         edges = add_lm_edge(edges, t_first, idx_first, t, k, z_list, s1, s2, s3);
                     end
                elseif t ~= t_first
                     edges = add_lm_edge(edges, t_first, idx_first, t, k, z_list, s1, s2, s3);
                end
            else
                % This is the first time seeing this landmark
                first_sightings(id) = [t, k];
            end
            
            % Update "Last Seen" to current
            last_sightings(id) = [t, k];
        end
    end
end

function edges = add_lm_edge(edges, t_old, idx_old, t_curr, idx_curr, z_list, s1, s2, s3)
    z_old  = z_list{t_old}(idx_old, :);
    z_curr = z_list{t_curr}(idx_curr, :);

    edge = create_empty_edge();
    edge.type = 'land';
    edge.id1 = t_old;
    edge.id2 = t_curr;
    edge.d1 = z_old(1); edge.angle1 = z_old(2);
    edge.d2 = z_curr(1); edge.angle2 = z_curr(2);
    edge.s = [s1, s2, s3];
    
    if isempty(edges), edges = edge; else, edges(end+1) = edge; end
end

function edge = create_empty_edge()
    edge = struct('type', '', 'id1', 0, 'id2', 0, ...
                  'measure', zeros(3,1), 'omega', zeros(3,3), 'e', zeros(3,1), ...
                  'd1', 0, 'angle1', 0, 'd2', 0, 'angle2', 0, 's', zeros(1,3));
end

function [H, b] = fill_H_and_b(H, b, edge, x_state)
    STATE_SIZE = 3;
    idx1 = (edge.id1 - 1) * STATE_SIZE + (1:STATE_SIZE);
    idx2 = (edge.id2 - 1) * STATE_SIZE + (1:STATE_SIZE);
    x1 = x_state(:, edge.id1); x2 = x_state(:, edge.id2);

    if strcmp(edge.type, 'odom')
        yaw = x1(3);
        dx = x2(1) - x1(1); dy = x2(2) - x1(2);
        pred = [cos(yaw)*dx + sin(yaw)*dy; -sin(yaw)*dx + cos(yaw)*dy; pi_2_pi(x2(3)-x1(3))];
        e = pred - edge.measure; e(3) = pi_2_pi(e(3));
        A = [-1 0 0; 0 -1 0; 0 0 -1]; B = -A;
        omega = edge.omega;
    else
        t1 = pi_2_pi(x1(3) + edge.angle1); t2 = pi_2_pi(x2(3) + edge.angle2);
        e = [x2(1) - x1(1) - edge.d1*cos(t1) + edge.d2*cos(t2);
             x2(2) - x1(2) - edge.d1*sin(t1) + edge.d2*sin(t2); 0];
        
        Rt1 = [cos(t1) -sin(t1) 0; sin(t1) cos(t1) 0; 0 0 1];
        Rt2 = [cos(t2) -sin(t2) 0; sin(t2) cos(t2) 0; 0 0 1];
        omega = inv(Rt1*diag(edge.s.^2)*Rt1' + Rt2*diag(edge.s.^2)*Rt2');
        
        A = [-1, 0, edge.d1*sin(x1(3)+edge.angle1); 0, -1, -edge.d1*cos(x1(3)+edge.angle1); 0 0 0];
        B = [ 1, 0, -edge.d2*sin(x2(3)+edge.angle2); 0,  1,  edge.d2*cos(x2(3)+edge.angle2); 0 0 0];
    end

    H(idx1, idx1) = H(idx1, idx1) + A' * omega * A;
    H(idx1, idx2) = H(idx1, idx2) + A' * omega * B;
    H(idx2, idx1) = H(idx2, idx1) + B' * omega * A;
    H(idx2, idx2) = H(idx2, idx2) + B' * omega * B;
    b(idx1) = b(idx1) + (A' * omega * e);
    b(idx2) = b(idx2) + (B' * omega * e);
end

function angle = pi_2_pi(angle)
    angle = mod(angle + pi, 2 * pi) - pi;
end