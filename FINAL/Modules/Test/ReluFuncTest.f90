program ReluFuncTest
    use ReluFunc_mod
    use iso_fortran_env, only: dp => real64
    implicit none

    ! 1. 修正数据维度以匹配 ReLU 函数（从 1D 改为 4D）
    real(dp), allocatable :: x(:,:,:,:), y(:,:,:,:), dout(:,:,:,:), dx(:,:,:,:)
    integer :: N, C, H, W

    ! 参数示例
    N = 2; C = 3; H = 4; W = 4

    ! 分配并初始化输入 (N, C, H, W)
    allocate(x(N, C, H, W))
    call random_seed()
    call random_number(x)
    x = x * 2.0_dp - 1.0_dp ! 将数据范围从 [0, 1) 调整到 [-1, 1) 以测试负值

    ! 2. 移除面向对象的代码，直接调用模块函数
    ! ReLU 是一个无状态的函数，不需要实例化
    write(*,*) "输入 x (部分):", x(1, 1, 1:2, 1:2)

    ! 前向传播
    y = relu_forward(x)
    write(*,*) "forward 输出 y (部分):", y(1, 1, 1:2, 1:2)

    ! 假设从后续层得到的上游梯度 dout 形状与 y 相同
    allocate(dout, source=y) ! 创建与 y 相同形状的 dout
    call random_number(dout)

    ! 3. 修正反向传播的调用
    ! backward 函数需要原始输入 x 来判断梯度是否通过
    dx = relu_backward(dout, x)
    write(*,*) "backward 返回 dx 维度:", shape(dx)
    write(*,*) "backward 输出 dx (部分):", dx(1, 1, 1:2, 1:2)

    ! 4. ReLU 没有参数，不需要更新步骤

    ! 清理
    deallocate(x, y, dout, dx)

end program ReluFuncTest