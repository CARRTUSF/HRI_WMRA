function [Xe_dot, gripper] = SpaceMouseControl(SpaceM,gripper)
% SpaceM data: Left-/right+, up+/down-, forward-/backward+, pitch
% down-/pitch up+, yaw left+/yaw right-, roll left+/roll right-
Xe_dot=[0;0;0;0;0;0;
    0;0;0;0;0;0]; %have to keep this line, it's like a initialization
i=0;
while(i<10)
    i=i+1;
    M_speed=SpaceM.speed;
end

Xe_dot(7)=-M_speed(3)*50;
Xe_dot(8)=-M_speed(1)*50;
Xe_dot(9)=M_speed(2)*50;
Xe_dot(10)=-M_speed(6)*0.2;
Xe_dot(11)=-M_speed(4)*0.2;
Xe_dot(12)=M_speed(5)*0.2;
if(button(SpaceM,1))
    gripper=-1; %close
end
if(button(SpaceM,2))
    gripper=1; %open
end
end