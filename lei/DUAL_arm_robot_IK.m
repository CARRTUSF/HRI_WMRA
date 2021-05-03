function [J_arms_G, q_dot] = DUAL_arm_robot_IK(phi, Q_arm, Xe_dot, X0_dot, frame)
Q_left=Q_arm(1:7);
Q_right=Q_arm(8:14);
%[J_le, J_l07, R_l] = DUAL_arm_robot_Jacobian(phi, Q_left, 'left');
[J_re, J_r07, R_r] = DUAL_arm_robot_Jacobian(phi, Q_right, 'right');

J_arms_G = J_r07;
R_r6 = [transpose(R_r) zeros(3);zeros(3) transpose(R_r)];%6x6
J_arms_E = R_r6*J_r07;
INV_J_arms_G=pinv(J_arms_G);
INV_J_arms_E=pinv(J_arms_E);

Ux=R_r*[1;0;0];
Vy=cross([0;0;1],Ux);
Uy=Vy./norm(Vy);
Uz=cross(Ux,Uy);
R_IG=[[1 0 0]*Ux [0 1 0]*Ux [0 0 1]*Ux;
      [1 0 0]*Uy [0 1 0]*Uy [0 0 1]*Uy;
      [1 0 0]*Uz [0 1 0]*Uz [0 0 1]*Uz];
R_IG6=[R_IG zeros(3);zeros(3) R_IG];


if(frame=="Ground")
    q_dotarms = INV_J_arms_G*[Xe_dot(7);Xe_dot(8);Xe_dot(9);Xe_dot(10);Xe_dot(11);Xe_dot(12)];
elseif(frame=="End-effector")
    q_dotarms = INV_J_arms_E*[Xe_dot(7);Xe_dot(8);Xe_dot(9);Xe_dot(10);Xe_dot(11);Xe_dot(12)];
elseif(frame=="Intuitive")
    INV_J_arms_I=pinv(R_IG6*J_r07);
    q_dotarms = INV_J_arms_I*[Xe_dot(7);Xe_dot(8);Xe_dot(9);Xe_dot(10);Xe_dot(11);Xe_dot(12)];
elseif(frame=="Hybrid")
    INV_J_arms_I=pinv(R_IG6*J_r07);
    q_dotarms = INV_J_arms_G*[0;0;Xe_dot(9);0;0;Xe_dot(12)]+INV_J_arms_I*[Xe_dot(7);Xe_dot(8);0;Xe_dot(10);Xe_dot(11);0];
else
    disp("error frame???")
    while(1)
    end
end
q_dot = [0;0;0;0;0;0;0;
    0;0;
    q_dotarms];
end