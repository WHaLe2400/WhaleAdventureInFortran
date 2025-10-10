program Task31_1
    implicit none
    character(len=100) :: str
    integer(8) :: i, ans, n
    real(16) :: approx_ans
    print *, "Enter a Number: "
    print *, "Enter 'exit' to quit."
    do 
        read '(A)', str
        if (trim(adjustl(str)) == 'exit') then !处理退出程序的情况
            print *,''
            print *, "Exiting the program."
            print *,''
            exit
        end if

        if (verify(str, '0123456789. ') /= 0) then !对于存在字符的情况报错
            print *, "Invalid input. Please enter a valid number."
            cycle
        elseif (count([(str(i:i), i=1,len_trim(str))] == '.') > 1) then !对于存在小数点的情况报错
            print *, "Invalid input. Please enter a valid number."
            cycle
        elseif (count([(str(i:i), i=1,len_trim(str))] == '.') > 0) then !对于小数输入的处理
            print *, "Theoretically, factorial of a non-integer is defined, "
            print *, "But it's too long to fit in the output."
            print *, "(The program didn't implement this part)"
            cycle
        else
            read(str, *) n !获得最终要进行处理的数据
        end if

        if (n < 0) then
            print *, "Factorial is not defined for negative numbers."
        elseif (n <= 20) then !没错，当n>20之后，整形变量就撑不住了
            ans = 1
            do i = 2, n
            ans = ans * i
            end do
            write(*,'(A,I3,A,I20)') "Factorial of ", n, " is ", ans
        else
            approx_ans = sqrt(2.0_16 * 3.1415926535897932384626433832795_16 * real(n, kind=16)) &
                * (real(n, kind=16) / 2.7182818284590452353602874713527_16)**real(n, kind=16)
            print *, "n is too large, using Stirling's approximation."
            write(*,'(A,I4,A)') "Approximate factorial of ", n, " is: "
            write(*,*) approx_ans
        end if
    end do

end program Task31_1