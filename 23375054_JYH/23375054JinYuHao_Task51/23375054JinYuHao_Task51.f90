program Task31_1
    implicit none
    character(len = 100) :: INPUT_PATH = 'FORTRAN_FileIn.in'
    character(len = 100) :: OUTPUT_PATH = 'FORTRAN_FileOut.out'


    character(len=100) :: str
    integer(8) :: i, ans, n
    integer :: iostat_1, iostat_2, iostat_3, All_Fighure, Dot_Count, E_Count
    real(16) :: approx_ans, tmp_real
    real(16), parameter :: EPS = 1.0e-12_16  ! 允许的数值误差范围
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


    open(101, file=INPUT_PATH, status='old', action='read', form='formatted', encoding='UTF-8', iostat=iostat_1)
        if (iostat_1 /= 0) then
            print *, "Cannot open input file (iostat=", iostat_1, ")."
            stop
        end if

    open(201, file=OUTPUT_PATH, status='replace', action='write', form='formatted', encoding='UTF-8', iostat=iostat_2)
        if (iostat_2 /= 0) then
            print *, "Cannot open output file (iostat=", iostat_2, ")."
            stop
        end if
    

    do  ! 数据的读入更改为从文件中读取
        read (101,'(A)', iostat=iostat_1) str
        if (iostat_1 /= 0) then
            print *, "Error occurred during read (iostat=", iostat_1, ")."
            exit
        end if
        print *, "Read input:    ", trim(str)


        All_Fighure = verify(str, '0123456789. eE+-')           ! 允许 e/E 和 + - 与空格
        Dot_Count = count([(str(i:i), i=1,len_trim(str))] == '.')
        E_Count   = count([(str(i:i), i=1,len_trim(str))] == 'e') &
                  + count([(str(i:i), i=1,len_trim(str))] == 'E')

        ! 先做字符合法性与简单结构检查
        if (All_Fighure /= 0) then
            write(201, *) "Invalid input"
            cycle
        end if
        if (Dot_Count > 1 .or. E_Count > 1) then
            write(201, *) "Invalid input"
            cycle
        end if

        ! 以实数读取并校验
        read(str, *, iostat=iostat_3) tmp_real
        if (iostat_3 /= 0) then
            write(201, *) "Invalid input"
            cycle
        end if
        if (tmp_real < 0.0_16) then
            write(201, *) "Invalid input"
            cycle
        end if

        ! 必须是整数值（允许微小数值误差）
        if (abs(tmp_real - nint(tmp_real)) > EPS) then
            write(201, *) "Invalid input"
            cycle
        end if
        ! 在转换为 integer(8) 前检测是否会溢出
        if (tmp_real > real(huge(1_8), kind=16)) then
            write(201, *) "Oversize input"
            cycle
        end if

        n = int(nint(tmp_real), kind=8)


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