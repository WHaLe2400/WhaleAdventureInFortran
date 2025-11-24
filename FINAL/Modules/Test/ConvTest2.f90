program use_conv_example
    use Conv_mod
    use iso_fortran_env, only: dp => real64
    implicit none

    type(ConvLayer) :: conv_1, conv_2
    real(dp), allocatable :: x(:,:,:,:), mid(:,:,:,:), y(:,:,:,:), dout(:,:,:,:), dmid(:,:,:,:), dx(:,:,:,:)
    integer :: N,  H, W, C_in, C_out_1, kH_1, kW_1, stride_1, pad_1
    integer :: C_out_2, kH_2, kW_2, stride_2, pad_2
    real(dp) :: lr

    ! 参数示例
    N = 2; C_in = 3; H = 8; W = 8
    C_out_1 = 4; kH_1 = 3; kW_1 = 3; stride_1 = 1; pad_1 = 1
    C_out_2 = 16; kH_2 = 3; kW_2 = 3; stride_2 = 1; pad_2 = 1
    lr = 0.01_dp

    ! 分配并初始化输入 (N, C_in, H, W)
    allocate(x(N, C_in, H, W))
    call random_seed()           ! 可选：设置随机种子
    call random_number(x)        ! [0,1) 随机数示例

    ! 实例化并初始化卷积层
    call conv_1%init(C_in, C_out_1, kH_1, kW_1, stride_1, pad_1)
    call conv_2%init(C_out_1, C_out_2, kH_2, kW_2, stride_2, pad_2)
    ! 前向传播
    mid = conv_1%forward(x)
    y = conv_2%forward(mid)
    write(*,*) "forward 输出维度:", shape(mid)
    write(*,*) "forward 输出维度:", shape(y)

    ! 假设从后续层得到的上游梯度 dout 形状与 y 相同
    allocate(dout(size(y,1), size(y,2), size(y,3), size(y,4)))
    call random_number(dout)

    ! 反向传播，得到对输入的梯度 dx
    dmid = conv_2%backward(dout)
    dx = conv_1%backward(dmid)
    write(*,*) "backward 返回 dx 维度:", shape(dmid)
    write(*,*) "backward 返回 dx 维度:", shape(dx)


    ! 更新参数（使用 conv 中已累积的 dW, db）
    call conv_2%update(lr)

    ! 清理
    deallocate(x, y, dout, dx)
end program use_conv_example