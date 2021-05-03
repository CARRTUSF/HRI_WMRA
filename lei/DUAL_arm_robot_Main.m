% debug:
% make sure iphone, baxter and PC are in same network.
% make sure iphone set correct IP and Port. ip is PC's ip, port
% is what you set in matlab.
clear all
close all
clc
% use which devices, 1:use  0:not use
powerBot=0;
Baxter=1;
% use which input device, only one device can be set to 1.  1:use  0:not use
iPhone=0;
Omni=0;
XboxController=1;
SpaceMouse=0;
% use which kinematics, only one can be set to 1.   1:use  0:not use
%frame="Ground";
frame="End-effector";
%frame="Intuitive";
%frame="Hybrid";
gripper=1; % 1:open -1:close
Xe_dot=[0;0;0;0;0;0;
    0;0;0;0;0;0];
Initial_Q_arm = [
    deg2rad(45);%45
    deg2rad(90);%90
    deg2rad(0);
    deg2rad(0);
    deg2rad(0);
    deg2rad(0);
    deg2rad(0);
    
    deg2rad(-45);%-45
    deg2rad(90);%90
    deg2rad(0);
    deg2rad(0);
    deg2rad(0);
    deg2rad(0);
    deg2rad(0)];

X0_dot=[0;0];  %[x;y] for wheelchair, unused variable here

q_dot=[0;0;0;0;0;0;0  % joint 1-7
    0;0; % wheel left;right
    0;0;0;0;0;0;0]; % arm_2

dt = 0;
phi= deg2rad(0);
Wheel_axis_center_position_old = [0,0,0];
Old_wheel_theta = [0;0];
Wheel_Theta_dot = q_dot(8:9);
Xphi_dot = Theta_dot2Xphi_dot(Wheel_Theta_dot);
DXphi_dot = Xphi_dot * dt;

Current_Q_arm = [
    deg2rad(0);
    deg2rad(-60);%-60
    deg2rad(0);
    deg2rad(90);%90
    deg2rad(0);
    deg2rad(60);%60
    deg2rad(0);
    
    deg2rad(0);
    deg2rad(-60);%-60
    deg2rad(0);
    deg2rad(90);%90
    deg2rad(0);
    deg2rad(0);
    deg2rad(0)];
Q_arm = Initial_Q_arm + Current_Q_arm;

% ******************  Socket Initialization ***********************
if(powerBot)
    t1 = tcpip('0.0.0.0', 5005, 'NetworkRole', 'server');
    set(t1, 'InputBufferSize', 1024);
    fopen(t1);
    disp('connected to PowerBot');
end
if(iPhone)
    t3 = tcpip('0.0.0.0', 5002, 'NetworkRole', 'server');
    disp(t3)
    set(t3, 'InputBufferSize', 2048);
    disp('Trying to connect to iPhone')
    fopen(t3);
    disp('connected to iPhone');
end
if(Omni)
    t4 = tcpip('0.0.0.0', 5007, 'NetworkRole', 'server');
    disp(t4)
    set(t4, 'InputBufferSize', 2048);
    disp('Trying to connect to Omni')
    fopen(t4);
    disp('connected to Omni');
end
if(Baxter)
    disp('Trying to connect to Baxter')
    t2 = tcpip('0.0.0.0', 5006, 'NetworkRole', 'server');
    set(t2, 'InputBufferSize', 1024);
    fopen(t2);
    disp('connected to Baxter');
    while (get(t2, 'BytesAvailable') ==0)
    end
    frombaxter = fscanf(t2);
    Q_arm=transpose(str2num(frombaxter))+Initial_Q_arm;
end
if(SpaceMouse)
    SpaceM=vrspacemouse('USB1')
    SpaceM.DominantMode=true
end
if(XboxController)
    joy = vrjoystick(2)
    % joy = vrjoystick(2) maybe it is 1 or 2 or 3, different device number
    % on different computers
end
% *************************** END *********************************

[L_F, P_F] = Get_frames_position (phi, DXphi_dot, Wheel_axis_center_position_old, Q_arm);
figure(2)
% Plots of the Arm and platform
frame1=plot3([P_F(1,1), P_F(1,2)], [P_F(2,1),P_F(2,2)], [P_F(3,1),P_F(3,2)],'-b','LineWidth',3);
hold on;
frame2=plot3([P_F(1,2), P_F(1,3)], [P_F(2,2),P_F(2,3)], [P_F(3,2),P_F(3,3)],'-b','LineWidth',3);
frame3=plot3([P_F(1,3), P_F(1,4)], [P_F(2,3),P_F(2,4)], [P_F(3,3),P_F(3,4)],'-b','LineWidth',3);
frame4=plot3([P_F(1,4), P_F(1,1)], [P_F(2,4),P_F(2,1)], [P_F(3,4),P_F(3,1)],'-b','LineWidth',3);

W1_to_center = plot3([L_F(1,1), L_F(1,2)], [L_F(2,1),L_F(2,2)], [L_F(3,1),L_F(3,2)],'-b','LineWidth',3);
Center_to_A  = plot3([L_F(1,2), L_F(1,3)], [L_F(2,2),L_F(2,3)], [L_F(3,2),L_F(3,3)],'-b','LineWidth',3);
link1        = plot3([L_F(1,3), L_F(1,4)], [L_F(2,3),L_F(2,4)], [L_F(3,3),L_F(3,4)],'-r','LineWidth',9);
link2        = plot3([L_F(1,4), L_F(1,5)], [L_F(2,4),L_F(2,5)], [L_F(3,4),L_F(3,5)],'-b','LineWidth',9);
link3        = plot3([L_F(1,5), L_F(1,6)], [L_F(2,5),L_F(2,6)], [L_F(3,5),L_F(3,6)],'-g','LineWidth',7);
link4        = plot3([L_F(1,6), L_F(1,7)], [L_F(2,6),L_F(2,7)], [L_F(3,6),L_F(3,7)],'-c','LineWidth',3);
link5        = plot3([L_F(1,7), L_F(1,8)], [L_F(2,7),L_F(2,8)], [L_F(3,7),L_F(3,8)],'-y','LineWidth',3);
link6        = plot3([L_F(1,8), L_F(1,9)], [L_F(2,8),L_F(2,9)], [L_F(3,8),L_F(3,9)],'-r','LineWidth',3);
link7        = plot3([L_F(1,9), L_F(1,10)], [L_F(2,9),L_F(2,10)], [L_F(3,9),L_F(3,10)],'-k','LineWidth',3);
Center_to_B  = plot3([L_F(1,2), L_F(1,11)], [L_F(2,2),L_F(2,11)], [L_F(3,2),L_F(3,11)],'-b','LineWidth',3);
linkB1       = plot3([L_F(1,11), L_F(1,12)], [L_F(2,11),L_F(2,12)], [L_F(3,11),L_F(3,12)],'-r','LineWidth',9);
linkB2       = plot3([L_F(1,12), L_F(1,13)], [L_F(2,12),L_F(2,13)], [L_F(3,12),L_F(3,13)],'-b','LineWidth',9);
linkB3       = plot3([L_F(1,13), L_F(1,14)], [L_F(2,13),L_F(2,14)], [L_F(3,13),L_F(3,14)],'-g','LineWidth',7);
linkB4       = plot3([L_F(1,14), L_F(1,15)], [L_F(2,14),L_F(2,15)], [L_F(3,14),L_F(3,15)],'-c','LineWidth',3);
linkB5       = plot3([L_F(1,15), L_F(1,16)], [L_F(2,15),L_F(2,16)], [L_F(3,15),L_F(3,16)],'-y','LineWidth',3);
linkB6       = plot3([L_F(1,16), L_F(1,17)], [L_F(2,16),L_F(2,17)], [L_F(3,16),L_F(3,17)],'-r','LineWidth',3);
linkB7       = plot3([L_F(1,17), L_F(1,18)], [L_F(2,17),L_F(2,18)], [L_F(3,17),L_F(3,18)],'-k','LineWidth',3);
plot3(-600, -1000, 0);
plot3(2000, 1000, 600);
Trajectory_L = animatedline('Color','b','LineWidth',2);
Trajectory_R = animatedline('Color','r','LineWidth',2);
grid on;
axis equal;
title('BaxterBot Animation'); xlabel('x, (mm)'); ylabel('y (mm)'); zlabel('z (mm)');

% ***************** Recorder  ***********************
t(1)=0;
Q_dot_Recorder(:,1)=q_dot;
Manipulability(1)=0;
loop_counter = 1;
% ***************************************************

tic
while(1)
    dt=Get_dt()
    % ******************************  Send/Receive data to/from clients. ****************
    if(powerBot)
        Xphi_dot_send=transpose(Theta_dot2Xphi_dot(q_dot(8:9)));% In [ mm/s, radius/s] need to be transfer to deg/s
        Xphi_dot_send(2)=rad2deg(Xphi_dot_send(2));
        fprintf(t1, num2str(Xphi_dot_send))    % fprintf, write data to text file. fwrite, write data to binary file
        while (get(t1, 'BytesAvailable') ==0)  % block the program until bytesavailable
        end
        powerBot_encoders = transpose(str2num(fscanf(t1)))
        New_wheel_theta = powerbotEncoder2rad(powerBot_encoders);
        Wheel_Theta_dot = New_wheel_theta - Old_wheel_theta;
        Old_wheel_theta = New_wheel_theta;
        Xphi_dot = Theta_dot2Xphi_dot(Wheel_Theta_dot);
        DXphi_dot = Xphi_dot; % Not like simulation, DXphi_dot = Xphi_dot*dt;
    else
        Wheel_Theta_dot = q_dot(8:9);
        Xphi_dot = Theta_dot2Xphi_dot(Wheel_Theta_dot);
        DXphi_dot = Xphi_dot*dt;
    end
    if(Baxter)
        %sendtobaxter=num2str([transpose(Q_arm + q_dot([1:7 10:16])*1 - Initial_Q_arm) gripper]);%position mode
        sendtobaxter=num2str([transpose(q_dot([1:7 10:16])) gripper]) %velocity mode
        fprintf(t2, sendtobaxter);
        while (get(t2, 'BytesAvailable') ==0)
        end
        frombaxter = fscanf(t2);
        Q_arm=transpose(str2num(frombaxter))+Initial_Q_arm;
    else
        Q_arm = Q_arm + q_dot([1:7 10:16])*dt;
    end
    if(Omni)
        sendtoOmni="Hello from Matlab";
        fprintf(t4, sendtoOmni);
        while (get(t4, 'BytesAvailable') ==0)
        end
        fromOmni = fscanf(t4);
        [Xe_dot, gripper]=OmniControl(fromOmni, frame);
    end
    if(iPhone)
        while (get(t3, 'BytesAvailable') ==0)
        end
        rawdata=fscanf(t3)
        while(get(t3, 'BytesAvailable'))
            rawdata= fscanf(t3);
        end
        iphone_str = strsplit(rawdata);
        [Xe_dot, gripper]=iPhoneControl(iphone_str);
    end
    if(SpaceMouse)
        [Xe_dot,gripper]=SpaceMouseControl(SpaceM, gripper);
    end
    if(XboxController)
        [Xe_dot,gripper]=XboxControl(joy,gripper)
    end
    [J, q_dot] = DUAL_arm_robot_IK(phi, Q_arm, Xe_dot, X0_dot, frame); % inverse kinematics
    phi = phi + DXphi_dot(2);
    [L_F, P_F] = Get_frames_position (phi, DXphi_dot, Wheel_axis_center_position_old, Q_arm);  % link frames and platform frames
    Wheel_axis_center_position_old = L_F(1:3,1);
    % Update the simulation plot using matlab function "set"  **********
    %L_F: Link_Frames;  P_F: Platform_Frames
    set(W1_to_center,'XData',[L_F(1,1), L_F(1,2)],'YData',[L_F(2,1), L_F(2,2)],'ZData',[L_F(3,1), L_F(3,2)]);
    set(Center_to_A, 'XData',[L_F(1,2), L_F(1,3)],'YData',[L_F(2,2), L_F(2,3)],'ZData',[L_F(3,2), L_F(3,3)]);
    set(link1,'XData',[L_F(1,3), L_F(1,4)],   'YData',[L_F(2,3), L_F(2,4)],    'ZData',[L_F(3,3), L_F(3,4)]);
    set(link2,'XData',[L_F(1,4), L_F(1,5)],   'YData',[L_F(2,4), L_F(2,5)],    'ZData',[L_F(3,4), L_F(3,5)]);
    set(link3,'XData',[L_F(1,5), L_F(1,6)],   'YData',[L_F(2,5), L_F(2,6)],    'ZData',[L_F(3,5), L_F(3,6)]);
    set(link4,'XData',[L_F(1,6), L_F(1,7)],   'YData',[L_F(2,6), L_F(2,7)],    'ZData',[L_F(3,6), L_F(3,7)]);
    set(link5,'XData',[L_F(1,7), L_F(1,8)],   'YData',[L_F(2,7), L_F(2,8)],    'ZData',[L_F(3,7), L_F(3,8)]);
    set(link6,'XData',[L_F(1,8), L_F(1,9)],   'YData',[L_F(2,8), L_F(2,9)],    'ZData',[L_F(3,8), L_F(3,9)]);
    set(link7,'XData',[L_F(1,9), L_F(1,10)],  'YData',[L_F(2,9), L_F(2,10)],   'ZData',[L_F(3,9), L_F(3,10)]);
    
    set(Center_to_B,'XData',[L_F(1,2), L_F(1,11)],'YData',[L_F(2,2), L_F(2,11)],'ZData',[L_F(3,2), L_F(3,11)]);
    set(linkB1,'XData',[L_F(1,11), L_F(1,12)],   'YData',[L_F(2,11), L_F(2,12)],    'ZData',[L_F(3,11), L_F(3,12)]);
    set(linkB2,'XData',[L_F(1,12), L_F(1,13)],   'YData',[L_F(2,12), L_F(2,13)],    'ZData',[L_F(3,12), L_F(3,13)]);
    set(linkB3,'XData',[L_F(1,13), L_F(1,14)],   'YData',[L_F(2,13), L_F(2,14)],    'ZData',[L_F(3,13), L_F(3,14)]);
    set(linkB4,'XData',[L_F(1,14), L_F(1,15)],   'YData',[L_F(2,14), L_F(2,15)],    'ZData',[L_F(3,14), L_F(3,15)]);
    set(linkB5,'XData',[L_F(1,15), L_F(1,16)],   'YData',[L_F(2,15), L_F(2,16)],    'ZData',[L_F(3,15), L_F(3,16)]);
    set(linkB6,'XData',[L_F(1,16), L_F(1,17)],   'YData',[L_F(2,16), L_F(2,17)],    'ZData',[L_F(3,16), L_F(3,17)]);
    set(linkB7,'XData',[L_F(1,17), L_F(1,18)],   'YData',[L_F(2,17), L_F(2,18)],    'ZData',[L_F(3,17), L_F(3,18)]);
    
    set(frame1,'XData',[P_F(1,1), P_F(1,2)],'YData',[P_F(2,1),P_F(2,2)],'ZData',[P_F(3,1),P_F(3,2)]);
    set(frame2,'XData',[P_F(1,2), P_F(1,3)],'YData',[P_F(2,2),P_F(2,3)],'ZData',[P_F(3,2),P_F(3,3)]);
    set(frame3,'XData',[P_F(1,3), P_F(1,4)],'YData',[P_F(2,3),P_F(2,4)],'ZData',[P_F(3,3),P_F(3,4)]);
    set(frame4,'XData',[P_F(1,4), P_F(1,1)],'YData',[P_F(2,4),P_F(2,1)],'ZData',[P_F(3,4),P_F(3,1)]);
    addpoints(Trajectory_L,L_F(1,10),L_F(2,10),L_F(3,10));
    addpoints(Trajectory_R,L_F(1,18),L_F(2,18),L_F(3,18));
    drawnow;
    [left_distance, right_distance] = DistanceTraveled(L_F(:,10),L_F(:,18))
    %pause(.42);
    
    % ****************************************************************
    % update recorder *************************************
    t(loop_counter+1) = t(loop_counter) + dt;
    time=t(loop_counter)
    if(get(gcf,'CurrentCharacter')=='g')
        frame="Ground"
    end
    if(get(gcf,'CurrentCharacter')=='e')
        frame="End-effector"
    end
    if(get(gcf,'CurrentCharacter')=='i')
        frame="Intuitive"
    end
    if(get(gcf,'CurrentCharacter')=='h')
        frame="Hybrid"
    end
    %Q_dot_Recorder(:,loop_counter)=q_dot;
    %Manipulability(loop_counter) = sqrt(det(J*transpose(J)));
    loop_counter = loop_counter + 1;
    % *****************************************************
end

figure()
plot(t(1:end-1),Manipulability);
title('Manipulability Measure');
if(using_hardware)
    if(powerBot)
        fclose(t1);
    end
    if(Baxter)
        fclose(t2);
    end
end


