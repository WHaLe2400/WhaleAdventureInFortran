program Task11
    implicit none
    real(kind = 8) :: x, y
    print *, "Enter X Value:"
    read *, x
    if (x < 0d0 .and. x > -15d0) then
        y = cos(x + 1d0)
    elseif (x >= 0d0 .and. x <= 10d0) then
        y = log(x**2 + 1d0)
    elseif (x > 15d0 .and. x < 20d0) then
        y = x**(1d0/3d0)
    else
        y = x**2
    end if
    print *, "Y Value is:", y
end program Task11