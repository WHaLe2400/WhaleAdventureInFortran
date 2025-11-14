program PReluFuncTest
    use PReluFunc_mod
    ! 推荐使用 iso_fortran_env 来定义精度，这更具可移植性
    use iso_fortran_env, only: dp => real64
    implicit none

    ! --- 主程序 ---
    write(*, '(/, A, /)') "================ PReLU 4D Test ================"
    call test_4d_version()

    write(*, '(/, A, /)') "================ PReLU 2D Test ================"
    call test_2d_version()

contains

    ! --- 4D 版本测试 ---
    subroutine test_4d_version()
        implicit none
        real(dp), allocatable :: x(:,:,:,:), y(:,:,:,:), dout(:,:,:,:), dx(:,:,:,:)
        real(dp), allocatable :: a(:)
        integer :: N, C, H, W

        ! 参数示例
        N = 1; C = 2; H = 3; W = 3

        ! 分配并初始化
        allocate(x(N, C, H, W), a(C))

        ! 初始化输入 x，使其包含正负值
        x(1, 1, :, :) = reshape([ -1.0_dp, -2.0_dp, -3.0_dp, &
                                   4.0_dp,  5.0_dp,  6.0_dp, &
                                   0.0_dp, -8.0_dp,  9.0_dp ], [3, 3])
        x(1, 2, :, :) = x(1, 1, :, :) * (-0.5_dp)

        ! 初始化 PReLU 的可学习参数 a
        a = [0.25_dp, 0.5_dp]

        write(*, '(A)') "Input x (channel 1):"
        call print_matrix(x(1,1,:,:))
        write(*, '(A, F5.2)') "PReLU param a(1): ", a(1)

        ! --- 前向传播 ---
        y = prelu_forward(x, a)
        write(*, '(A)') "Forward output y (channel 1):"
        call print_matrix(y(1,1,:,:))
        write(*,*) "--> 验证: 负数应乘以 0.25, 正数和零不变"

        ! --- 反向传播 ---
        allocate(dout, source=y)
        dout = 1.0_dp ! 假设上游梯度全部为 1

        dx = prelu_backward(dout, x, a)
        write(*, '(A)') "Backward output dx (channel 1):"
        call print_matrix(dx(1,1,:,:))
        write(*,*) "--> 验证: 对应 x>0 的位置梯度为 1.0, 否则为 0.25"

        deallocate(x, y, a, dout, dx)
    end subroutine test_4d_version

    ! --- 2D 版本测试 ---
    subroutine test_2d_version()
        implicit none
        real(dp), allocatable :: x(:,:), y(:,:), dout(:,:), dx(:,:)
        real(dp), allocatable :: a(:)
        integer :: N, L

        ! 参数示例
        N = 2; L = 5

        ! 分配并初始化
        allocate(x(N, L), a(L))
        x(1, :) = [ 1.0_dp, -2.0_dp, 3.0_dp, 0.0_dp, -5.0_dp ]
        x(2, :) = x(1, :) * (-1.0_dp)
        a       = [ 0.1_dp,  0.2_dp,  0.3_dp, 0.4_dp,  0.5_dp ]

        write(*, '(A)') "Input x (batch 1):"
        write(*, '(5(F6.2, 1X))') x(1, :)
        write(*, '(A)') "PReLU params a:"
        write(*, '(5(F6.2, 1X))') a(:)

        ! --- 前向传播 ---
        y = prelu_forward(x, a)
        write(*, '(A)') "Forward output y (batch 1):"
        write(*, '(5(F6.2, 1X))') y(1, :)
        write(*,*) "--> 验证: y(i) = x(i) if x(i)>0 else a(i)*x(i)"

        ! --- 反向传播 ---
        allocate(dout, source=y)
        dout = 1.0_dp

        dx = prelu_backward(dout, x, a)
        write(*, '(A)') "Backward output dx (batch 1):"
        write(*, '(5(F6.2, 1X))') dx(1, :)
        write(*,*) "--> 验证: dx(i) = dout(i) if x(i)>0 else a(i)*dout(i)"

        deallocate(x, y, a, dout, dx)
    end subroutine test_2d_version

    ! 辅助函数，用于打印 2D 矩阵
    subroutine print_matrix(mat)
        real(dp), intent(in) :: mat(:,:)
        integer :: i
        do i = 1, size(mat, 1)
            write(*, '(100(F8.2, 1X))') mat(i, :)
        end do
    end subroutine print_matrix

end program PReluFuncTest
