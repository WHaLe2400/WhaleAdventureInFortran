program Task31_1
    implicit none
    character(len=100) :: str
    integer(8) :: i, ans, n
    integer :: iostat_1, iostat_2
    real(16) :: approx_ans
    character(len=800) :: PROGRAM_RUNNING
    PROGRAM_RUNNING = &
    "  ______              __                               "// new_line('a') // &
    "  / ____/  ____   ____/ /  ___                          "// new_line('a') // &
    " / /      / __ \ / __  /  / _ \                         "// new_line('a') // &
    "/ /___   / /_/ // /_/ /  /  __/                         "// new_line('a') // &
    "\____/__ \____/ \__,_/   \___/       _                  "// new_line('a') // &
    "   / __ \  __  __   ____    ____    (_)   ____    ____ _"// new_line('a') // &
    "  / /_/ / / / / /  / __ \  / __ \  / /   / __ \  / __ `/"// new_line('a') // &
    " / _, _/ / /_/ /  / / / / / / / / / /   / / / / / /_/ / "// new_line('a') // &
    "/_/ |_|  \__,_/  /_/ /_/ /_/ /_/ /_/   /_/ /_/  \__, /  "// new_line('a') // &
    "                                               /____/   "
    print *, PROGRAM_RUNNING

    open(101, file='FORTRAN_FileIn.in', status='old', action='read', form='formatted', encoding='UTF-8', iostat=iostat_1)
        if (iostat_1 /= 0) then
            print *, "Cannot open input file FORTRAN_FileIn.in (iostat=", iostat_1, ")."
            stop
        end if

    open(201, file='FORTRAN_FileOut.out', status='replace', action='write', form='formatted', encoding='UTF-8', iostat=iostat_2)
        if (iostat_2 /= 0) then
            print *, "Cannot open output file FORTRAN_FileOut.out (iostat=", iostat_2, ")."
            stop
        end if
    

    do  ! 数据的读入更改为从文件中读取
        read (101,'(A)', iostat=iostat_1) str
        if (iostat_1 /= 0) then
            print *, "Error occurred during read (iostat=", iostat_1, ")."
            exit
        end if
        print *, "Read input:", trim(str)


        if (verify(str, '0123456789. ') /= 0) then !对于存在字符的情况报错
            write(201, *) "Invalid input"
            cycle
        elseif (count([(str(i:i), i=1,len_trim(str))] == '.') > 1) then             !对于存在小数点的情况报错
            write(201, *) "Invalid input"   
            cycle
        elseif (count([(str(i:i), i=1,len_trim(str))] == '.') > 0) then             !对于小数输入的处理
            write(201, *) "Invalid input"
            cycle
        else
            read(str, *) n                                                          !获得最终要进行处理的数据
        end if


        if (n < 0) then
            write(201, *) "Invalid input"
        elseif (n <= 20) then                                                       !没错，当n>20之后，整形变量就撑不住了
            ans = 1
            do i = 2, n
            ans = ans * i
            end do
            write(201,*)ans
        elseif(n <= 1754) then                                                      !当n>1754之后，16字节的实数也撑不住了（除非我们还能有新的方式获得更多的位数
                                   
                                                                                    ! 1754! 大约是 2.59e5731
            approx_ans = sqrt(2.0_16 * 3.1415926535897932384626433832795_16 * real(n, kind=16)) &
                * (real(n, kind=16) / 2.7182818284590452353602874713527_16)**real(n, kind=16)
            write(201,*) approx_ans
        else
            write(201, *) "Oversize input"
        end if

    end do

    close(101)
    close(201)

end program Task31_1