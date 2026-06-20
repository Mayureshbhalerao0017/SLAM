function particles = fast_slam_core(particles, u, z)
% FAST_SLAM_CORE
% RBPF FastSLAM with EKF landmarks + ESS resampling
%
% Inputs:
%   particles : struct array
%   u         : [v; yaw_rate]
%   z         : [range; bearing; landmark_id] (3xM)
%
% Output:
%   particles : updated particles

dt = 0.1;              % MUST match driver DT
N  = length(particles);

% ---------- Noise parameters (TUNED) ----------
SIGMA_V = 0.25;                 % m/s
SIGMA_W = deg2rad(2.5);          % rad/s
R_meas  = diag([0.7, deg2rad(7)]).^2;

% =========================================================
% 1. MOTION PREDICTION
% =========================================================
for i = 1:N

    v = u(1) + randn * SIGMA_V;
    w = u(2) + randn * SIGMA_W;

    particles(i).x = particles(i).x + v * cos(particles(i).yaw) * dt;
    particles(i).y = particles(i).y + v * sin(particles(i).yaw) * dt;
    particles(i).yaw = wrapToPi_local(particles(i).yaw + w * dt);
end

% =========================================================
% 2. MEASUREMENT UPDATE (LANDMARK EKF + WEIGHTS)
% =========================================================
for i = 1:N

    for k = 1:size(z,2)

        r  = z(1,k);
        b  = z(2,k);
        id = z(3,k);

        % --- Landmark never seen before ---
        if all(particles(i).lm(id,:) == 0)

            lx = particles(i).x + r * cos(b + particles(i).yaw);
            ly = particles(i).y + r * sin(b + particles(i).yaw);

            particles(i).lm(id,:) = [lx ly];
            particles(i).lmP(2*id-1:2*id,:) = diag([2.5 2.5]).^2;
            continue;
        end

        % --- EKF update ---
        mu = particles(i).lm(id,:)';
        dx = mu(1) - particles(i).x;
        dy = mu(2) - particles(i).y;
        q  = dx^2 + dy^2;

        z_hat = [
            sqrt(q);
            wrapToPi_local(atan2(dy,dx) - particles(i).yaw)
        ];

        H = [
            dx/sqrt(q), dy/sqrt(q);
           -dy/q,        dx/q
        ];

        P = particles(i).lmP(2*id-1:2*id,:);
        S = H*P*H' + R_meas;
        K = P*H'/S;

        innov = [r; b] - z_hat;
        innov(2) = wrapToPi_local(innov(2));

        mu = mu + K*innov;
        P  = (eye(2) - K*H)*P;

        particles(i).lm(id,:) = mu';
        particles(i).lmP(2*id-1:2*id,:) = P;

        % --- Weight update ---
        w = exp(-0.5 * innov' / S * innov) / sqrt(det(2*pi*S));
        particles(i).w = particles(i).w * max(w,1e-12);
    end
end

% =========================================================
% 3. WEIGHT NORMALIZATION
% =========================================================
w = [particles.w];
w = w / sum(w);
for i = 1:N
    particles(i).w = w(i);
end

% =========================================================
% 4. ESS-BASED RESAMPLING  <<< CRITICAL BLOCK
% =========================================================
ESS = 1 / sum(w.^2);

if ESS < 0.6 * N
    particles = systematic_resample(particles);
end

end

% =========================================================
% SYSTEMATIC RESAMPLING
% =========================================================
function particles = systematic_resample(particles)
% SYSTEMATIC_RESAMPLE
% Safe systematic resampling (no index overflow)

N = length(particles);
w = [particles.w];
w = w / sum(w);   % safety

c = cumsum(w);
c(end) = 1;       % force exact 1

u0 = rand / N;
u  = u0 + (0:N-1)/N;

idx = zeros(1,N);
i = 1;
j = 1;

while i <= N
    if u(i) <= c(j)
        idx(i) = j;
        i = i + 1;
    else
        j = j + 1;
    end
end

particles = particles(idx);

for i = 1:N
    particles(i).w = 1/N;
end
end


% =========================================================
% WRAP TO PI
% =========================================================
function a = wrapToPi_local(a)
a = mod(a + pi, 2*pi) - pi;
end
