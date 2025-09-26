program Task11
    implicit none
    real(kind = 8) :: x, y
    integer :: x_int
    print *, "Enter X Value:"
    read *, x
    x_int = int(x)          ! 受限于编译器无法对real变量进行区间判断
    select case (x_int)
    case (-15:-1)           ! x > -15 且 x < 0
        y = cos(x + 1d0)
    case (0:9)              ! x >= 0 且 x <= 10
        y = log(x**2 + 1d0)
    case (15:19)            ! x >= 15 且 x <= 20
        y = x**(1d0/3d0)
    case default
        y = x**2
    end select
    print *, "Y Value is:", y
end program Task11