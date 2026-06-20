function [x,P,landmark_ids,assoc] = ekf_slam(x,P,z_list,R_meas,vx,vy,yaw_rate,dt,Q_process,mappingPhase,landmark_ids)
% State definition:
%   x = [ x_car; y_car; psi_car; l1x; l1y; l2x; l2y; ... ]   (world frame)
%
% Inputs:
%   x, P          : previous SLAM state and covariance (can be [] on first call)
%   z_list (2xM)  : range-bearing measurements in *vehicle frame*
%                   z(:,i) = [range_i; bearing_i]
%   R_meas (2x2)  : measurement noise covariance (same for all cones)
%   vx, vy        : body-frame velocities [m/s] from your EKF
%   yaw_rate      : yaw rate r [rad/s] from your EKF
%   dt            : timestep [s]
%   Q_process     : 3x3 process noise for [vx;vy;yaw_rate] (tuning parameter)
%   mappingPhase  : true  -> first lap: mapping + localization
%                   false -> later laps: localization only (no new landmarks)
%   landmark_ids  : integer ID per landmark (vector), can be [] at start
%
% Outputs:
%   x, P          : updated SLAM state and covariance
%   landmark_ids  : updated landmark ID list
%   assoc         : association for each measurement:
%                   assoc(i) = landmark index (1..N), 0=new landmark, -1=ignored

% Intialisation of landmark
if nargin <11 || isempty(landmark_ids)
    landmark_ids = [];
end

% Initialisation of states and covariance
if isempty(x)
    x = [0;0;0];
    P = eye(size(x,1)) * .0;   

    %P = diag([1,1,5]);
end

n_pose = 3; % x,y,orientation
n_state = length(x); % length of state vector 
n_landmark = (n_state - n_pose)/2 ; %number of landmarks

% prediction
[x,P] = slam_predict_step(x,P,vx,vy,yaw_rate,dt,Q_process);

% update
M = size(z_list,2);
assoc = -1*ones(1,M);
 
if M==0
    % no measurement do nothing
    return;
end

euclid_gate = 1;

% looping over measurement

for i = 1:M
    z=z_list(:,i);

    best_d = inf;
    best_j = 0;

    for j = 1:n_landmark
        [z_hat,~] =slam_measurement_model(x,j);
        v = wrapInnovation(z-z_hat);

        d=norm(v);
        if d<best_d
            best_d=d;
            best_j=j;
        end
    end
    % create a new landmark or associate

    if n_landmark>0 && best_d < euclid_gate
        j = best_j;
        [z_hat,H]=slam_measurement_model(x,j);
        v = wrapInnovation(z-z_hat);

        S = H*P*H.' + R_meas;
        K= P*H.' / S ;
        dx = K*v;
        if ~mappingPhase
            dx(n_pose+1:end) = 0;
        end
        x = x + dx;
        x(3) = wrapToPi_local(x(3));

        I = eye(size(P));
        P = (I - K*H) * P * (I - K*H).' + K * R_meas * K.';
        
        assoc(i) = j;
    else
            if mappingPhase
             % new landmark
                [x,P,landmark_ids] = slam_add_landmark(x,P,z,R_meas,landmark_ids);
                n_state = length(x);
                n_landmark = (n_state - n_pose)/2;
                assoc(i) = n_landmark;
            else
                assoc(i)= -1; % to ignore in localization phase
            end
    end
end
end

function [x_pred,P_pred] = slam_predict_step(x,P,vx,vy,yaw_rate,dt,Q_process)

n_state = length(x);
%n_pose = 3;

X = x(1);
Y = x(2);
psi = x(3);

c = cos(psi);
s = sin(psi);

dx_world = (vx*c - vy*s)*dt;
dy_world = (vx*s + vy*c)*dt;
dpsi = yaw_rate*dt;

X_pred = X+dx_world;
Y_pred = Y+dy_world;
psi_pred = wrapToPi_local(psi + dpsi);

x_pred = x;                    % start from old state
x_pred(1) = X_pred;
x_pred(2) = Y_pred;
x_pred(3) = psi_pred;          % keep landmarks unchanged


% Jacobian F wrt state
F = eye(n_state);
F(1,3) = (-vx * s - vy * c) * dt;
F(2,3) = ( vx * c - vy * s) * dt;

% Process noise mapped by G
G = zeros(3,3);
G(1,1) = c * dt;   % dX/dvx
G(1,2) = -s * dt;  % dX/dvy
G(2,1) = s * dt;   % dY/dvx
G(2,2) = c * dt;   % dY/dvy
G(3,3) = dt;       % dpsi/dyaw_rate

Q_pose = G*Q_process*G.';

Q_big = zeros(n_state);
Q_big(1:3,1:3) = Q_pose;

P_pred = F*P*F.' + Q_big;


end

function [z_hat,H] = slam_measurement_model(x,landmark_index)

n_state = length(x);
n_pose = 3;

X = x(1);
Y = x(2);
psi = x(3);

lm_start = n_pose + 2*(landmark_index-1)+1;
lx = x(lm_start);
ly = x(lm_start+1);

dx = lx - X;
dy = ly - Y;
q = dx^2 + dy^2;
sqrt_q = sqrt(q);

range_hat = sqrt_q;
bearing_hat = wrapToPi_local(atan2(dy,dx)-psi);
z_hat = [range_hat;bearing_hat];

H = zeros(2,n_state);

H_pose = [-dx/sqrt_q,-dy/sqrt_q,0;
    dy/q,-dx/q,-1];

% Derivatives wrt landmark [lx, ly]
H_lm = [  dx / sqrt_q,       dy / sqrt_q;
         -dy / q,            dx / q ];
H(:,1:3) = H_pose;
H(:,lm_start:lm_start+1) = H_lm;
end

function [x_new,P_new,landmark_ids]= slam_add_landmark(x, P, z, R_meas, landmark_ids)
n_pose = 3;
range   = z(1);
bearing = z(2);

X   = x(1);
Y   = x(2);
psi = x(3);

% Global angle to landmark
phi_global = psi + bearing;

% Landmark position in world frame
lx = X + range * cos(phi_global);
ly = Y + range * sin(phi_global);

% Augment state
x_new = [x; lx; ly];

% Jacobians of landmark position wrt pose and measurement (for covariance)
% l = f(x_pose, z)
mx_x   = 1;  mx_y   = 0;  mx_psi = -range * sin(phi_global);
my_x   = 0;  my_y   = 1;  my_psi =  range * cos(phi_global);

Gx = [mx_x, mx_y, mx_psi;
      my_x, my_y, my_psi];

mx_r   = cos(phi_global);    mx_b = -range * sin(phi_global);
my_r   = sin(phi_global);    my_b =  range * cos(phi_global);

Gz = [mx_r, mx_b;
      my_r, my_b];

n_old = length(x);
P_ll  = Gx * P(1:n_pose,1:n_pose) * Gx.' + Gz * R_meas * Gz.';  % 2x2

P_xl  = P(:,1:n_pose) * Gx.';  % n_old x 2
P_lx  = P_xl.';

% Augment covariance
P_new = zeros(n_old + 2);
P_new(1:n_old, 1:n_old) = P;
P_new(1:n_old, n_old+1:n_old+2) = P_xl;
P_new(n_old+1:n_old+2, 1:n_old) = P_lx;
P_new(n_old+1:n_old+2, n_old+1:n_old+2) = P_ll;

% Assign a new landmark ID (simple increment)
if isempty(landmark_ids)
    next_id = 1;
else
    next_id = max(landmark_ids) + 1;
end
landmark_ids = [landmark_ids, next_id];
end

function ang = wrapToPi_local(ang)
% Wrap angle to [-pi, pi]
ang = mod(ang + pi, 2*pi) - pi;
end

function v_wrapped = wrapInnovation(v)
% Wrap the bearing part of innovation to [-pi, pi]
v_wrapped = v;
v_wrapped(2) = wrapToPi_local(v(2));
end
