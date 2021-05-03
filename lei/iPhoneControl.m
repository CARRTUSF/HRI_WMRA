function [Xe_dot, gripper] = iPhoneControl(iphonestr)
Xe_dot=[0;0;0;0;0;0;
    0;0;0;0;0;0]; %have to keep this line, it's like a initialization
mode=iphonestr(2);
fromiphone=str2double(iphonestr);
gripper=fromiphone(18);

%%
if(mode=="OneTouch")
    disp("one touch mode")
    pitch=fromiphone(4);
    roll=fromiphone(6);
    yaw=fromiphone(8);
    screentouched=fromiphone(10);
    vector_x=fromiphone(12);%phone left/right
    vector_y=fromiphone(14);%phone forward/backward
    Zmode=fromiphone(16);
    if(screentouched) % =1:pressed   =0:up
        if(Zmode)
            Xe_dot(3+6)=vector_y*-0.25;
            Xe_dot(2+6)=vector_x*-0.3;
        else
            Xe_dot(1+6)=vector_y*-0.25;
            Xe_dot(2+6)=vector_x*-0.3;
            Xe_dot(4+6)=roll*0.003;
            if(pitch>0)  %tilt up
                Xe_dot(5+6)=pitch*-0.0025;
            else  % tilt down, higher gain due to not easy to tilt down
                Xe_dot(5+6)=pitch*-0.003;
            end
            Xe_dot(6+6)=yaw*0.003;
        end
    end
end
end
