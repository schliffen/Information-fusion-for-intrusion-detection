function [err, xhat, G, P, Ge, Pe] = kalman(z, ze, A, C, R, Q, B, E1, da, w)
% Simple Kalman Filter (linear) with optimal gain, no control signal
%
% z Measurement signal               m observations X # of observations
% A State transition model          n X n, n = # of state values
% C Observation model                m X n
% R Covariance of observation noise m X m
% Q Covariance of process noise     n X n
%
% Based on http://en.wikipedia.org/wiki/Kalman_filter, but matrices named
% A, C, G instead of F, H, K.
%
% See http://home.wlu.edu/~levys/kalman_tutorial for background
%
% Copyright (C) 2014 Simon D. Levy
%
% This code is free software: you can redistribute it and/or modify
% it under the terms of the GNU Lesser General Public License as
% published by the Free Software Foundation, either version 3 of the
% License, or (at your option) any later version.
%
% This code is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU Lesser General Public License
% along with this code. If not, see <http:#www.gnu.org/licenses/>.
%
% Initializtion =====================================================
%
% Number of sensors
m = size(C, 1);
%
% Number of state values
n = size(C, 2);
%
% Number of observations
numobs = size(z, 2);
%
% Use linear least squares to estimate initial state from initial
% observation
xhat = zeros(n, numobs);
xhat(:,1) = C \ z(:,1);
% Initialize P, I
P = ones(size(A));
I = eye(size(A));
% errors
err = zeros(n, numobs);
err(:,1) = C \ ze(:,1);
% Initialize P, I
Pe = ones(size(A));
Ie = eye(size(A));
% Kalman Filter =====================================================
%
for k = 2:numobs
    % Predict state
    xhat(:,k) = A * xhat(:,k-1) +  B*da(:,k-1) + E1*w(:,k-1);
    P         = A * P * A' + Q;
    % predict error
    err(:,k) = A*err(:,k-1) + .3*B*da(:,k-1) +  .3*E1*w(:,k-1);
    Pe = A* Pe * A';
    % Update state
    G         = P  * C' / (C * P * C' + R);
    P         = (I - G * C) * P;
    xhat(:,k) = xhat(:,k) + G * (z(:,k) - C * xhat(:,k));
    % Update error
    Ge         = Pe  * C' / (C * Pe * C' + R);
    Pe         = (Ie - Ge * C) * Pe;
    err(:,k) = err(:,k) + Ge * (ze(:,k) - C * err(:,k));
end

