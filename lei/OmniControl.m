function [Xe_dot, gripper]=OmniControl(fromOmni, frame)
Xe_dot=[0;0;0;0;0;0;
    0;0;0;0;0;0]; %have to keep this line, it's like a initialization
omnidata=str2num(fromOmni);
xdot=-omnidata(3);
ydot=-omnidata(1);
zdot=omnidata(2);
xrotation=-omnidata(6);
yrotation=-omnidata(5);
zrotation=-omnidata(4);
Xe_dot(1+6)=xdot*2;
Xe_dot(2+6)=ydot*2;
Xe_dot(3+6)=zdot*2;
Xe_dot(4+6)=xrotation*0.6;
Xe_dot(5+6)=yrotation*0.6;
Xe_dot(6+6)=zrotation*0.6;
gripper=omnidata(7);
end
