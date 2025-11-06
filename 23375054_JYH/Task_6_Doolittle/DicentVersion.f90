module matrix_ops_dicent
    implicit none
contains

    ! 子程序：打印矩阵
    ! 功能：将一个二维实数数组（矩阵）以格式化的方式输出到控制台。
    ! 参数：
    !   A (输入): 要打印的二维实数数组。
    subroutine PrintMatrix(A)
        implicit none
        real, intent(in) :: A(:,:)  ! 输入的矩阵 A
        integer :: num_rows, num_cols, i, j ! 变量用于存储矩阵维度和循环

        num_rows = size(A, 1)  ! 获取矩阵的行数
        num_cols = size(A, 2)  ! 获取矩阵的列数

        ! 外部循环遍历矩阵的每一行
        do i = 1, num_rows
            write(*,'(A)',advance='no') '  ' ! 在每行开头打印两个空格用于对齐
            ! 内部循环遍历当前行的每一列
            do j = 1, num_cols
                ! 以固定的浮点数格式打印每个元素，并保留6位小数，然后跟一个空格
                write(*,'(F12.6,1X)',advance='no') A(i,j)
            end do
            write(*,*) ! 当前行所有元素打印完毕后，换行
        end do
    end subroutine PrintMatrix


    ! 子程序：计算矩阵的 LU 分解 (Doolittle 方法)
    ! 功能：将输入矩阵 A 分解为一个下三角矩阵 L 和一个上三角矩阵 U，
    !       并将它们合并存储在输出矩阵 LU 中。
    !       其中 L 的对角线元素为 1（不存储），U 存储在 LU 的上三角部分（包括对角线）。
    ! 参数：
    !   A (输入): 原始的 n x n 矩阵。
    !   LU (输出): 存储分解结果的 n x n 矩阵。
    !   n (输入): 矩阵的维度。
    subroutine GetLUMatrix(A, L, U)
        implicit none
        real, intent(in) :: A(:,:)      ! 输入矩阵 A
        real, intent(out) :: L(:,:), U(:,:) ! 输出的 L 和 U 矩阵
        integer :: n, i, j, k           ! 循环变量
        real :: sum                     ! 用于累加计算

        n = size(A, 1) ! 获取矩阵维度

        ! 初始化 L 为单位矩阵, U 为零矩阵
        U = 0.0
        do i = 1, n
            L(i, i) = 1.0
            do j = 1, i - 1
                L(i, j) = 0.0
            end do
            do j = i + 1, n
                L(i, j) = 0.0
            end do
        end do

        ! Doolittle 分解的核心循环
        do k = 1, n
            ! 第 1 步：计算上三角矩阵 U 的第 k 行
            do j = k, n
                sum = 0.0
                ! 计算 U(k,j) 的求和部分: sum = L(k, 1:k-1) * U(1:k-1, j)
                if (k > 1) then
                    sum = dot_product(L(k, 1:k-1), U(1:k-1, j))
                end if
                ! U(k,j) = A(k,j) - sum
                U(k, j) = A(k, j) - sum
            end do

            ! 第 2 步：计算下三角矩阵 L 的第 k 列
            do i = k + 1, n
                sum = 0.0
                ! 计算 L(i,k) 的求和部分: sum = L(i, 1:k-1) * U(1:k-1, k)
                if (k > 1) then
                    sum = dot_product(L(i, 1:k-1), U(1:k-1, k))
                end if
                ! 检查除数是否为零
                if (U(k, k) == 0.0) then
                    print *, "错误：除数为零。无法进行 LU 分解。"
                    stop ! 终止程序
                end if
                ! L(i,k) = (A(i,k) - sum) / U(k,k)
                L(i, k) = (A(i, k) - sum) / U(k, k)
            end do
        end do
    end subroutine GetLUMatrix

    subroutine GetYMatrix(L, B, Y)
        implicit none
        real, intent(in) :: L(:,:), B(:,:)  ! 输入矩阵 L 和 B
        real, intent(out) :: Y(:,:)         ! 输出结果矩阵 Y
        integer :: n                        ! 矩阵维度变量
        integer :: i, j                     ! 循环变量

        n = size(L, 1) ! 获取矩阵维度

        ! 前向替换法求解 LY = B
        do i = 1, n
            Y(i, 1) = B(i, 1)
            do j = 1, i - 1
                Y(i, 1) = Y(i, 1) - L(i, j) * Y(j, 1)
            end do
            ! L 的对角线元素为 1，无需除以 L(i,i)
        end do

    end subroutine GetYMatrix


    subroutine GetXMatrix(U, Y, X)
        implicit none
        real, intent(in) :: U(:,:), Y(:,:)  ! 输入矩阵 U 和 Y
        real, intent(out) :: X(:,:)         ! 输出结果矩阵 X
        integer :: n                        ! 矩阵维度变量
        integer :: i, j                     ! 循环变量

        n = size(U, 1) ! 获取矩阵维度

        ! 后向替换法求解 UX = Y
        do i = n, 1, -1
            X(i, 1) = Y(i, 1)
            do j = i + 1, n
                X(i, 1) = X(i, 1) - U(i, j) * X(j, 1)
            end do
            ! 除以 U 的对角线元素
            if (U(i, i) == 0.0) then
                print *, "错误：除数为零。无法求解方程组。"
                stop ! 终止程序
            end if
            X(i, 1) = X(i, 1) / U(i, i)
        end do

    end subroutine GetXMatrix


    ! 子程序：矩阵乘法
    ! 功能：计算两个 n x n 矩阵 A 和 B 的乘积，结果存入 C。
    ! 参数：
    !   A, B (输入): 要相乘的两个 n x n 矩阵。
    !   C (输出): 存储乘法结果的 n x n 矩阵。
    !   n (输入): 矩阵的维度。
    subroutine MultiplyMatrices(A, B, C)
        implicit none
        real, intent(in) :: A(:,:), B(:,:)  ! 输入矩阵 A 和 B
        real, intent(out) :: C(:,:)         ! 输出结果矩阵 C
        integer :: m, p, n                  ! 矩阵维度变量
        integer :: i, j, k                  ! 循环变量

        ! 从输入矩阵获取维度
        m = size(A, 1) ! A 的行数
        p = size(A, 2) ! A 的列数 (也必须是 B 的行数)
        n = size(B, 2) ! B 的列数

        ! 检查矩阵维度是否兼容乘法
        if (p /= size(B, 1)) then
            print *, "错误: 矩阵 A 的列数必须等于矩阵 B 的行数。"
            stop
        end if

        ! 检查输出矩阵 C 的维度是否正确
        if (size(C, 1) /= m .or. size(C, 2) /= n) then
            print *, "错误: 输出矩阵 C 的维度不正确。"
            stop
        end if

        ! 使用 Fortran 内置的 matmul 函数进行矩阵乘法，更高效简洁
        C = matmul(A, B)

    end subroutine MultiplyMatrices


    subroutine SolveLinearSystem(A, B, X)
        implicit none
        real, intent(in) :: A(:,:), B(:,:)  ! 输入矩阵 A 和 B
        real, intent(out) :: X(:,:)         ! 输出结果矩阵 X
        integer :: n                        ! 矩阵维度变量
        real, allocatable :: L(:,:), U(:,:), Y(:,:) ! 分解矩阵和中间结果

        n = size(A, 1) ! 获取矩阵维度

        ! 分配内存
        allocate(L(n,n), U(n,n), Y(n,1))

        ! 进行 LU 分解
        call GetLUMatrix(A, L, U)

        ! 求解 LY = B
        call GetYMatrix(L, B, Y)

        ! 求解 UX = Y
        call GetXMatrix(U, Y, X)

        ! 释放内存
        deallocate(L, U, Y)

    end subroutine SolveLinearSystem

end module matrix_ops_dicent


program test_lu_decomposition
    use matrix_ops_dicent  ! 引用包含矩阵操作的模块
    implicit none
    character(len = 100) :: INPUT_PATH, OUTPUT_PATH
    integer(8) :: iostat_1, iostat_2
    real, allocatable :: A(:,:), B(:,:), X(:,:)
    integer :: n, i, j, k

    print *, "请输入输入文件路径："
    read(*,'(A)') INPUT_PATH
    print *, "请输入输出文件路径："
    read(*,'(A)') OUTPUT_PATH

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

    do
        read(101, *, iostat=iostat_1) n
        ! 先检查 iostat：EOF 或错误优先处理，避免使用未初始化或旧的 n
        print *, "读取矩阵维度 n =", n
        print *, "读取到的 n 值检查 iostat:", iostat_1
        if (n == -1) then
            print *, "END"  ! 如果读取到-1，表示结束
            exit
        end if
        
        if (iostat_1 < 0) then
            print *, "Reached end of file."
            exit  ! 文件结束，退出循环
        end if
        if (iostat_1 > 0) then
            write(*,*) "读取维度 n 时出错 (iostat=", iostat_1, ")"
            stop
        end if




        allocate(A(n,n), B(n,1), X(n,1))

        ! 逐元素读取矩阵 A，读取后立即检查 iostat
        read(101, *, iostat=iostat_1) A
        if (iostat_1 < 0) then
            print *, "在读取矩阵 A 时遇到 EOF。"
            deallocate(A, B, X)
            exit
        else if (iostat_1 > 0) then
            write(*,*) "读取矩阵 A 时出错 (iostat=", iostat_1, ")"
            stop
        end if

        ! 逐元素读取向量 B，读取后立即检查 iostat
        read(101, *, iostat=iostat_1) B
        if (iostat_1 < 0) then
            print *, "在读取向量 B 时遇到 EOF。"
            deallocate(A, B, X)
            exit
        else if (iostat_1 > 0) then
            write(*,*) "读取向量 B 时出错 (iostat=", iostat_1, ")"
            stop
        end if

        write(201, *) "求解方程组 AX = B"
        write(201, *) "矩阵 A:"
        do i = 1, n
            write(201,'(100(F12.6,1X))') (A(i,j), j=1,n)
        end do

        write(201, *) "向量 B:"
        do i = 1, n
            write(201,'(F12.6)') B(i,1)
        end do

        call SolveLinearSystem(A, B, X)
        print *, "方程组求解完成，解向量 X 如下："
        call PrintMatrix(X)

        write(201, *) "解 X:"
        do i = 1, n
            write(201,'(F12.6)') X(i,1)
        end do
        write(201, *) "------------------------------------"

        deallocate(A, B, X)
    end do

    close(101)
    close(201)

    print *, "处理完成，结果已写入输出文件。"


end program test_lu_decomposition


! INPUT FILE EXAMPLE 
! /root/0_FoRemote/WhaleAdventureInFortran/23375054_JYH/Task_6_Doolittle/TEST_IN_FILE.txt
! OUTPUT FILE EXAMPLE
! /root/0_FoRemote/WhaleAdventureInFortran/23375054_JYH/Task_6_Doolittle/TEST_OUT_FILE.txt

! gfortran -std=f2008 -g DicentVersion.f90 -o DicentVersion