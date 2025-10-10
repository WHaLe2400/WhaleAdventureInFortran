program Task31_1
    implicit none
    character(len=100) :: str
    integer(8) :: i, ans, n
    real(16) :: approx_ans
    character(len=800) :: PROGRAM_READY
    PROGRAM_READY = &
    "______                                                                    _        " // new_line('a') // &
    "| ___ \                                                                  | |       " // new_line('a') // &
    "| |_/ /_ __  ___    __ _  _ __  __ _  _ __ ___      _ __  ___   __ _   __| | _   _ " // new_line('a') // &
    "|  __/| '__|/ _ \  / _` || '__|/ _` || '_ ` _ \    | '__|/ _ \ / _` | / _` || | | |" // new_line('a') // &
    "| |   | |  | (_) || (_| || |  | (_| || | | | | |   | |  |  __/| (_| || (_| || |_| |" // new_line('a') // &
    "|_|   |_|   \___/  \__, ||_|   \__,_||_| |_| |_|   |_|   \___| \__,_| \__,_| \__, |" // new_line('a') // &
    "                    __/ |                                                     __/ |" // new_line('a') // &
    "                   |___/                                                     |___/ "
    print *, PROGRAM_READY
    do 
        print *,''
        print *,'____________________________________'
        print *, "Enter a Number: (Enter 'exit' to quit.)"
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
            write(*,'(A,I3,A,/,/,20x,I20)') "Factorial of ", n, " is: ", ans
        elseif(n <= 1754) then !当n>1754之后，16字节的实数也撑不住了（除非我们还能有新的方式获得更多的位数
             ! 1754! 大约是 2.59e5731
            approx_ans = sqrt(2.0_16 * 3.1415926535897932384626433832795_16 * real(n, kind=16)) &
                * (real(n, kind=16) / 2.7182818284590452353602874713527_16)**real(n, kind=16)
            print *, "n is too large, using Stirling's approximation."
            write(*,'(A,I4,A,/)') "Approximate factorial of ", n, " is: "
            write(*,*) approx_ans
        else
            print *, "n is too large, even Stirling's approximation overflows."
        end if
    end do

end program Task31_1