close all
clear all
clc

syms theta1 theta2 theta3 theta4 theta5 theta6 theta7
syms phi

T01= Trans(  0, 0 , 270.35 , theta1);
T12= Trans(-90, 69, 0      , theta2);
T23= Trans(+90, 0 , 364.35, theta3);
T34= Trans(-90, 69, 0      , theta4);
T45= Trans(+90, 0 , 374.29, theta5);
T56= Trans(-90, 10, 0      , theta6);
T67= Trans(+90, 0 , 280    , theta7);

% T01= Trans(-90, 69, 270.35 , theta1);
% T12= Trans( 90, 0 , 0      , theta2);
% T23= Trans(-90, 69, 364.35 , theta3);
% T34= Trans( 90, 0 , 0      , theta4);
% T45= Trans(-90, 10, 374.29 , theta5);
% T56= Trans( 90, 10, 0      , theta6);
% T67= Trans(  0, 10, 280.00 , theta7);

T02= T01*T12;
T03= T02*T23;
T04= T03*T34;
T05= T04*T45;
T06= T05*T56;
T07= T06*T67;
R07=T07(1:3,1:3)
XpG=T07(1:3,4)
XpE= R07'*XpG

Xp=XpG;

Z1= T01(1:3,3);
Z2= T02(1:3,3);
Z3= T03(1:3,3);
Z4= T04(1:3,3);
Z5= T05(1:3,3);
Z6= T06(1:3,3);
Z7= T07(1:3,3);

q = [theta1;theta2;theta3;theta4;theta5;theta6;theta7];
Jv= simplify(jacobian(Xp,q))
Jw= simplify([Z1,Z2,Z3,Z4,Z5,Z6,Z7])
J07=[Jv;Jw]

Pxy=simplify(T07(1:2,4));
J0E=simplify([
    1 0 0 0 0 -(Pxy(1)*sin(phi)+Pxy(2)*cos(phi));
    0 1 0 0 0   Pxy(1)*cos(phi)-Pxy(2)*sin(phi);
    0 0 1 0 0 0;
    0 0 0 1 0 0;
    0 0 0 0 1 0;
    0 0 0 0 0 1])

