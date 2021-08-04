function return_value = Trans(alpha,a,d,theta)
return_value = [cos(theta)            -sin(theta)             0           a; 
                sin(theta)*cosd(alpha)  cos(theta)*cosd(alpha) -sind(alpha) -sind(alpha)*d;
                sin(theta)*sind(alpha)  cos(theta)*sind(alpha)  cosd(alpha)  cosd(alpha)*d; 
                0                      0                      0           1];
end