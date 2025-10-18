program Test
    implicit none
    integer :: iostat
    character(len=200) :: line

    open(101, file='TEST_ForInput.in', status='old', action='read', form='formatted', encoding='UTF-8', iostat=iostat)
    
    if (iostat /= 0) then
        print *, "Cannot open input file TEST_ForInput.in (iostat=", iostat, ")."
        stop
    end if

    do
        read(101, '(A)', iostat=iostat) line
        if (iostat /= 0) exit
        print *, "Read line: ", trim(line)
    end do

    close(101)

end program Test