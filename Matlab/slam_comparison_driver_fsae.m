function slam_comparison_driver_fsae()
% SLAM_COMPARISON_DRIVER
% EKF SLAM vs FastSLAM vs GraphSLAM on FSAE DV-style trackdrive data

clc; clear; close all;

%% ================= CONFIG =================
DT = 0.1;
SIM_TIME = 50.0;

Q_true = diag([0.08, 0.3, deg2rad(1.0)]).^2;
Q_solver = Q_true;

R_true = diag([0.6, deg2rad(6.5)]).^2;
R_solver = R_true;

%% ========== GENERATE FSAE DATA ==========
fprintf('--- Generating FSAE Trackdrive Data ---\n');
[time_vec, hx_true, data_log, lm_true] = ...
    generate_fsae_dataset(DT, SIM_TIME, Q_true, R_true);

fprintf('Steps: %d | Cones: %d\n', length(time_vec), size(lm_true,2));

%% ================= EKF SLAM =================
fprintf('\n--- Running EKF SLAM ---\n');
x_ekf = []; P_ekf = []; lm_ids = [];
hx_ekf = zeros(3,length(time_vec));

tic;
for k = 1:length(time_vec)
    u = data_log(k).u;
    z = data_log(k).z_ekf;

    [x_ekf, P_ekf, lm_ids, ~] = ekf_slam( ...
        x_ekf, P_ekf, z, R_solver, ...
        u(1), u(2), u(3), DT, ...
        Q_solver, true, lm_ids);

    hx_ekf(:,k) = x_ekf(1:3);
end
t_ekf = toc;

%% ================= FASTSLAM =================
fprintf('\n--- Running FastSLAM ---\n');
N_PARTICLE = 200;
particles = init_particles(N_PARTICLE);
hx_fast = zeros(3,length(time_vec));

tic;
for k = 1:length(time_vec)
    u_fast = [data_log(k).u(1); data_log(k).u(3)];
    z_fast = data_log(k).z_fast;
    particles = fast_slam_core(particles, u_fast, z_fast);
    hx_fast(:,k) = calc_particle_mean(particles);
end
t_fast = toc;

%% ================= GRAPH SLAM =================
fprintf('\n--- Running Graph SLAM ---\n');
hxDR = zeros(3,length(time_vec));
x = [0;0;0];

for k = 1:length(time_vec)
    u = data_log(k).u;
    x = motion_model(x,[u(1);u(3)],DT);
    hxDR(:,k) = x;
end

hz_graph = cell(1,length(time_vec));
for k = 1:length(time_vec)
    z = data_log(k).z_fast;
    if isempty(z)
        hz_graph{k} = [];
    else
        hz_graph{k} = [z(1,:)', z(2,:)', zeros(size(z,2),1), z(3,:)'];
    end
end

tic;
C_SIGMA = [2 4 deg2rad(8)];
x_graph = graph_slam_core(hxDR, hz_graph, 15, ...
    C_SIGMA(1), C_SIGMA(2), C_SIGMA(3));
t_graph = toc;

hx_graph = x_graph(1:3,:);

%% ================= METRICS =================
rmse_ekf   = calc_rmse(hx_true, hx_ekf);
rmse_fast  = calc_rmse(hx_true, hx_fast);
rmse_graph = calc_rmse(hx_true, hx_graph);

%% ================= PLOTS =================
figure('Color','w','Position',[100 100 1400 600])

subplot(1,2,1); hold on; grid on; axis equal;
plot(lm_true(1,:),lm_true(2,:),'y*','DisplayName','Cones');
plot(hx_true(1,:),hx_true(2,:),'w--','LineWidth',2,'DisplayName','Ground Truth');
plot(hx_ekf(1,:),hx_ekf(2,:),'b','DisplayName','EKF');
plot(hx_fast(1,:),hx_fast(2,:),'g','DisplayName','FastSLAM');
plot(hx_graph(1,:),hx_graph(2,:),'r','DisplayName','Graph SLAM');
legend; title('FSAE DV Trackdrive SLAM Comparison');

subplot(1,2,2)
bar([t_ekf t_fast t_graph]); hold on
plot([rmse_ekf rmse_fast rmse_graph],'ro-','LineWidth',2)
set(gca,'XTickLabel',{'EKF','FastSLAM','Graph'})
ylabel('Time / RMSE'); grid on

fprintf('\nMETHOD      TIME(s)    RMSE(m)\n');
fprintf('EKF        %.3f     %.3f\n',t_ekf,rmse_ekf);
fprintf('FastSLAM   %.3f     %.3f\n',t_fast,rmse_fast);
fprintf('GraphSLAM  %.3f     %.3f\n',t_graph,rmse_graph);

end

%% ==========================================================
%% ================= HELPER FUNCTIONS =======================
%% ==========================================================

function [time_vec, hx_true, data_log, lm] = generate_fsae_dataset(dt, T, Q, R)
% GUARANTEED: Ground truth lies exactly between cones

%% ===== TRACK PARAMETERS =====
track_width  = 3.0;     % meters
cone_spacing = 2.0;     % meters
v_nom        = 7.0;     % m/s

%% ===== BUILD CENTERLINE =====
s = linspace(0, 220, round(220 / (v_nom*dt)));
cx = zeros(size(s));
cy = zeros(size(s));
psi = zeros(size(s));

for i = 2:length(s)

    if s(i) < 60
        kappa = 0;
    elseif s(i) < 120
        kappa = 1/18;
    elseif s(i) < 160
        kappa = -1/12;
    else
        kappa = 0;
    end

    % CORRECT heading integration
    psi(i) = psi(i-1) + v_nom * kappa * dt;

    % Position update
    cx(i) = cx(i-1) + v_nom * cos(psi(i)) * dt;
    cy(i) = cy(i-1) + v_nom * sin(psi(i)) * dt;
end



%% ===== GENERATE CONES FROM SAME CENTERLINE =====
lm = [];
for i = 1:round(cone_spacing/(v_nom*dt)):length(cx)
    n = [-sin(psi(i)); cos(psi(i))];

    left  = [cx(i); cy(i)] + n * track_width/2;
    right = [cx(i); cy(i)] - n * track_width/2;

    lm = [lm, left, right];
end

assignin('base','N_LANDMARKS', size(lm,2));

%% ===== TIME VECTOR =====
N = length(cx);
time_vec = (0:N-1) * dt;

%% ===== GROUND TRUTH (EXACT CENTERLINE) =====
hx_true = [cx; cy; psi];

%% ===== MEASUREMENTS =====
data_log = struct('u', {}, 'z_ekf', {}, 'z_fast', {});

for k = 1:N

    % True motion (for reference only)
    v = v_nom;
    if k < N
        w = (psi(k+1) - psi(k)) / dt;
    else
        w = 0;
    end

    % Noisy odometry
    vn = v + randn * sqrt(Q(1,1));
    wn = w + randn * sqrt(Q(3,3));

    z_ekf  = [];
    z_fast = [];

    for i = 1:size(lm,2)
        dx = lm(1,i) - cx(k);
        dy = lm(2,i) - cy(k);
        r  = hypot(dx, dy);
        b  = wrapToPi_local(atan2(dy,dx) - psi(k));

        if r < 12 && abs(b) < deg2rad(60)
            rn = r + randn * sqrt(R(1,1));
            bn = b + randn * sqrt(R(2,2));

            z_ekf  = [z_ekf,  [rn; bn]];
            z_fast = [z_fast, [rn; bn; i]];
        end
    end

    data_log(k).u      = [vn; 0; wn];
    data_log(k).z_ekf  = z_ekf;
    data_log(k).z_fast = z_fast;
end
end


function x = motion_model(x,u,dt)
x(1)=x(1)+u(1)*cos(x(3))*dt;
x(2)=x(2)+u(1)*sin(x(3))*dt;
x(3)=wrapToPi_local(x(3)+u(2)*dt);
end

function particles = init_particles(N)
N_LM = evalin('base','N_LANDMARKS');
p.w=1/N; p.x=0; p.y=0; p.yaw=0;
p.lm=zeros(N_LM,2);
p.lmP=zeros(2*N_LM,2);
particles=repmat(p,N,1);
end

function x_mean = calc_particle_mean(particles)
x=0; y=0; cx=0; cy=0;
for i=1:length(particles)
    w=particles(i).w;
    x=x+w*particles(i).x;
    y=y+w*particles(i).y;
    cx=cx+w*cos(particles(i).yaw);
    cy=cy+w*sin(particles(i).yaw);
end
x_mean=[x;y;atan2(cy,cx)];
end

function rmse = calc_rmse(x_true,x_est)
n=min(size(x_true,2),size(x_est,2));
e=x_true(1:2,1:n)-x_est(1:2,1:n);
rmse=sqrt(mean(sum(e.^2,1)));
end

function a = wrapToPi_local(a)
a = mod(a+pi,2*pi)-pi;
end
