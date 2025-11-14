program use_conv_example
    use Conv_mod
    use iso_fortran_env, only: dp => real64
    implicit none

    type(ConvLayer) :: conv
    real(dp), allocatable :: x(:,:,:,:), y(:,:,:,:), dout(:,:,:,:), dx(:,:,:,:)
    integer :: N, C_in, H, W, C_out, kH, kW, stride, pad
    real(dp) :: lr

    ! 参数示例
    N = 2; C_in = 3; H = 8; W = 8
    C_out = 4; kH = 3; kW = 3; stride = 1; pad = 1
    lr = 0.01_dp

    ! 分配并初始化输入 (N, C_in, H, W)
    allocate(x(N, C_in, H, W))
    call random_seed()           ! 可选：设置随机种子
    call random_number(x)        ! [0,1) 随机数示例

    ! 实例化并初始化卷积层
    call conv%init(C_in, C_out, kH, kW, stride, pad)

    ! 前向传播
    y = conv%forward(x)
    write(*,*) "forward 输出维度:", shape(y)

    ! 假设从后续层得到的上游梯度 dout 形状与 y 相同
    allocate(dout(size(y,1), size(y,2), size(y,3), size(y,4)))
    call random_number(dout)

    ! 反向传播，得到对输入的梯度 dx
    dx = conv%backward(dout)
    write(*,*) "backward 返回 dx 维度:", shape(dx)

    ! 更新参数（使用 conv 中已累积的 dW, db）
    call conv%update(lr)

    ! 清理
    deallocate(x, y, dout, dx)
end program use_conv_example