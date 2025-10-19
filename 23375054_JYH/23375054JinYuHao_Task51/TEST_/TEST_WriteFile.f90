program Test_Write
    implicit none
    integer :: iostat, i, j, k
    character(len=200) :: ToWrite(0:9)

    do i=0,9
        write(ToWrite(i), '(A,I3,1X,A)') '这是第', i+1, '行内容。'
    end do


    open(102, file='TEST_ForOutput.out', status='unknown', action='write', form='formatted', encoding='UTF-8', iostat=iostat)
    
    if (iostat /= 0) then
        print *, "Cannot open output file TEST_ForOutput.out (iostat=", iostat, ")."
        stop
    end if

    do
        do j=0,9
            write(102, '(A)') trim(ToWrite(j))
        end do
        exit
    end do

    close(102)

end program Test_Write