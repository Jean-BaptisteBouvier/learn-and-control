%%%% Inverted pendulum dynamics
% MATLAB code to compute the norm bound on the trajectory distance

g = 9.81;
A = [0, 1; 3*g/2, 0];
B = [0; 15*3];

K = 0.7*[1, 0.5];
A_tilde = A-B*K;
% eig(A_tilde)

Q = eye(2);
P = lyap(A_tilde', Q);
% exponential coefficient for the Lyapunov differential inequality
gamma = min( eig(Q))/( 2 * max(eig(P)))
    
% matrix 2-norm is the max singular values + 3% of safety margin for the sin(theta) linearization
L = 1.03*max( svd(A_tilde) )

times = 0:0.01:10;

L1 = ((1 - exp(-gamma*times)) * L/gamma);
L2 = exp(-gamma*times);
min_bound = min( [L1; L2] );
x0 = [0.1; 0.1];

norm_bound = 2*sqrt( x0'*P*x0) * min_bound / sqrt(min(eig(P)));


figure(1)
hold on
grid on
plot(times, norm_bound)
