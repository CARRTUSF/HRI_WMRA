function [Link_Origin_Positions, Mobile_Platform_frames] = Get_frames_position (phi_old, DXphi_dot, Wheel_axis_center_position_old, Q_arm)

[L1,L2,L3,L4,L5] = powerbot_read_parameters();

P0x = Wheel_axis_center_position_old(1);
P0y = Wheel_axis_center_position_old(2);

DX=DXphi_dot(1);
phi_new=DXphi_dot(2);
R=DX/phi_new;

theta1=Q_arm(1);
theta2=Q_arm(2);
theta3=Q_arm(3);
theta4=Q_arm(4);
theta5=Q_arm(5);
theta6=Q_arm(6);
theta7=Q_arm(7);

thetaB1=Q_arm(8);
thetaB2=Q_arm(9);
thetaB3=Q_arm(10);
thetaB4=Q_arm(11);
thetaB5=Q_arm(12);
thetaB6=Q_arm(13);
thetaB7=Q_arm(14);

if(abs(phi_new)<=0.001)
    TGW1 =[ %if Dphi==0
        cos(phi_old), -sin(phi_old), 0, P0x + DX*cos(phi_old);
        sin(phi_old),  cos(phi_old), 0, P0y + DX*sin(phi_old);
        0,             0, 1,                    L5;
        0,             0, 0,                     1
        ];
    
else
    TGW1 =[ %if Dphi!=0
        cos(phi_old + phi_new), -sin(phi_old + phi_new), 0, P0x + R*sin(phi_old + phi_new) - R*sin(phi_old);
        sin(phi_old + phi_new),  cos(phi_old + phi_new), 0, P0y - R*cos(phi_old + phi_new) + R*cos(phi_old);
        0,                 0, 1,                                                          L5;
        0,                 0, 0,                                                           1
        ];
end

% Get link origion positions
TW1toCenter = [
    eye(3) [L2;0;0];
    0 0 0 1
    ];

TW1toA = [
    eye(3) [L2;L3;L4];
    0 0 0 1
    ];
TW1toA2 = [
    eye(3) [L2;-L3;L4];
    0 0 0 1
    ];

TGC = TGW1 * TW1toCenter;
TGA = TGW1 * TW1toA;
TGA2= TGW1 * TW1toA2;

T01= Trans(  0, 0 , 270.35, theta1);
T12= Trans(-90, 69, 0     , theta2);
T23= Trans(+90, 0 , 364.35, theta3);
T34= Trans(-90, 69, 0     , theta4);
T45= Trans(+90, 0 , 374.29, theta5);
T56= Trans(-90, 10, 0     , theta6);
T67= Trans(+90, 0 , 280   , theta7);

TG1= TGA*T01;
TG2= TG1*T12;
TG3= TG2*T23;
TG4= TG3*T34;
TG5= TG4*T45;
TG6= TG5*T56;
TG7= TG6*T67;

TB01= Trans(  0, 0 , 270.35, thetaB1);
TB12= Trans(-90, 69, 0     , thetaB2);
TB23= Trans(+90, 0 , 364.35, thetaB3);
TB34= Trans(-90, 69, 0     , thetaB4);
TB45= Trans(+90, 0 , 374.29, thetaB5);
TB56= Trans(-90, 10, 0     , thetaB6);
TB67= Trans(+90, 0 , 280   , thetaB7);

TGB1= TGA2*TB01;
TGB2= TGB1*TB12;
TGB3= TGB2*TB23;
TGB4= TGB3*TB34;
TGB5= TGB4*TB45;
TGB6= TGB5*TB56;
TGB7= TGB6*TB67;
Link_Origin_Positions = [TGW1(1:3,4), TGC(1:3,4), TGA(1:3,4), TG1(1:3,4), TG2(1:3,4), TG3(1:3,4), TG4(1:3,4), TG5(1:3,4), TG6(1:3,4), TG7(1:3,4), TGA2(1:3,4), TGB1(1:3,4), TGB2(1:3,4), TGB3(1:3,4), TGB4(1:3,4), TGB5(1:3,4), TGB6(1:3,4), TGB7(1:3,4)];

% Get mobile platform frame positions
TW1toBack_left = [
    eye(3) [0;L1/2;0];
    0 0 0 1
    ];
TW1toBack_right = [
    eye(3) [0;-L1/2;0];
    0 0 0 1
    ];
TW1toFront_left = [
    eye(3) [L2;L3;0];
    0 0 0 1
    ];
TW1toFront_right = [
    eye(3) [L2;-L3;0];
    0 0 0 1
    ];
TGtobl=TGW1*TW1toBack_left;
TGtobr=TGW1*TW1toBack_right;
TGtofl=TGW1*TW1toFront_left;
TGtofr=TGW1*TW1toFront_right;
Mobile_Platform_frames = [TGtobl(1:3,4) TGtobr(1:3,4) TGtofr(1:3,4) TGtofl(1:3,4)];

end
