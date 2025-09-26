program triangle_third_side
    implicit none
    real :: a, b, angle_deg, angle_rad, c
    print *, '请输入两边长度a, b和夹角(度):'
    read *, a, b, angle_deg
    angle_rad = angle_deg * 3.14159265358979323846 / 180.0
    c = sqrt(a**2 + b**2 - 2.0*a*b*cos(angle_rad))
    print *, '第三边长度为:', c
end program triangle_third_side
