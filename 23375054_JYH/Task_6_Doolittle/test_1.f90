! 定义一个模块，包含矩阵操作相关的子程序
module matrix_ops
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

end module matrix_ops


! 主程序：用于测试 LU 分解功能
program test
    use matrix_ops  ! 引用包含矩阵操作的模块
    implicit none
    real :: A(3,3), L(3,3), U(3,3), B(3, 1), Y(3, 1) ,X(3, 1)! 声明原始矩阵 A 和用于存储 LU 分解结果的矩阵 LU
    integer :: n            ! 矩阵维度
    integer :: i, j         ! 循环变量（在此程序中未使用，但保留是良好习惯）

    ! 初始化一个 3x3 的矩阵 A
    ! Fortran 是列主序，所以 reshape 的顺序是按列填充
    A = reshape([0.5, 5.0, 2.0, &
                 1.1, .96, 4.5, &
                 3.1, 6.5, .36], [3,3]) ! 初始化A矩阵（注意这里需要将矩阵对称后输入
    B = reshape([6.0, .96, .02], [3,1]) ! 初始化矩阵 B
    n = 3 ! 设置矩阵维度

    ! 打印原始矩阵 A
    print *, '原始矩阵 A:'
    call PrintMatrix(A)
    print *, '矩阵 B:'
    call PrintMatrix(B)

    ! 调用子程序进行 LU 分解
    ! 注意：调用前需要确保 L 和 U 矩阵已经被分配了与 A 匹配的尺寸
    call GetLUMatrix(A, L, U)

    ! 计算Y矩阵
    call GetYMatrix(L, B, Y)

    ! 计算X矩阵
    call GetXMatrix(U, Y, X)

    ! 打印结果矩阵 X
    print *, '方程组的解 X:'
    call PrintMatrix(X)

end program test
