function [left_distance, right_distance] = DistanceTraveled(left_position, right_position)
persistent iteration;
if isempty(iteration)
    iteration=0;
end
iteration=iteration+1;

persistent L_d;
if isempty(L_d) %initialize this variable
    L_d = 0;
end

persistent R_d;
if isempty(R_d) %initialize this variable
    R_d = 0;
end

persistent left_pre_position;
if isempty(left_pre_position) %initialize this variable
    left_pre_position = left_position;
end

persistent right_pre_position;
if isempty(right_pre_position) %initialize this variable
    right_pre_position = right_position;
end



if(iteration>20)
    iteration=0;
    L_d=L_d+norm(left_position-left_pre_position);
    R_d=R_d+norm(right_position-right_pre_position);
    left_pre_position=left_position;
    right_pre_position=right_position;
end
left_distance=L_d;
right_distance=R_d;
end
