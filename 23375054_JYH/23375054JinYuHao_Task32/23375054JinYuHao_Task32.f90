program Task32
    implicit none
    real(8) :: f(5, 10), x(5), y(10), s=0.0_8
    integer :: i, j
    !初始化x与y
    do i = 1, 5
        x(i) = 2.0_8*i-1.0_8
    end do
    do i = 1, 10
        y(i) = 2.0_8 + i*0.1_8
    end do

    !计算f矩阵
    do i = 1, 5
        do j = 1, 10
            f(i, j) = sin(x(i) + y(j)) / (1.0_8 + x(i)*y(j))
            s = s + f(i, j)
        end do
    end do

    write(*,'(A,ES20.10)') "Sum of all elements in f: ", s

end program Task32
