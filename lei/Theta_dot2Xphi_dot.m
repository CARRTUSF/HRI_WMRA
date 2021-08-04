function y = Theta_dot2Xphi_dot(theta_dot)
    [L1,L2,L3,L4,L5] = powerbot_read_parameters();
    
    J_Dtheta2Dphi = [
    L5/2,  L5/2;
    -L5/L1, L5/L1
    ];

    y = J_Dtheta2Dphi * theta_dot;
end