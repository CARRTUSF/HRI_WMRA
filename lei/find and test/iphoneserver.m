clear all
close all
clc


disp('Trying to connect to iPhone')
t3 = tcpip('0.0.0.0', 5008, 'NetworkRole', 'server');
set(t3, 'InputBufferSize', 2048);
fopen(t3);
disp('connected to iPhone');

while(1)
    while (get(t3, 'BytesAvailable') ==0)
    end
    fromiphone = str2double(strsplit(fscanf(t3)));
    QuaternionXYZW = [fromiphone(2) fromiphone(4) fromiphone(6) fromiphone(8)];
    [roll, pitch, yaw] = quat2angle(QuaternionXYZW);
    roll=rad2deg(roll)
    pitch=rad2deg(pitch)
    yaw=rad2deg(yaw)
    XYZbutton=fromiphone(32)
    RPYbutton=fromiphone(34)
end

fclose(t3)