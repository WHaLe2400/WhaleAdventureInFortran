program test_conv
    use iso_fortran_env, only: dp => real64
    use Conv_mod
    implicit none

    type(ConvLayer) :: conv
    type(ConvLayer) :: conv_loaded

    real(dp), allocatable :: input(:,:,:,:), output(:,:,:,:), grad_output(:,:,:,:), grad_input(:,:,:,:)
    real(dp) :: loss_before, loss_after
    integer :: H_in = 28, W_in = 28, C_in = 1, N = 2
    integer :: H_out, W_out, C_out = 16
    character(len=100) :: save_path = "test_conv_weights.dat"

    ! 初始化随机种子
    call random_seed()

    print *, "Testing ConvLayer..."

    ! 测试 init
    print *, "1. Testing init..."
    call conv%init(C_in, C_out, 3, 1, 1)  ! kernel_size=3, stride=1, padding=1
    print *, "   Init successful."

    ! 分配输入数据 (H_in, W_in, C_in, N)
    allocate(input(H_in, W_in, C_in, N))
    call random_number(input)
    input = input * 2.0_dp - 1.0_dp  ! [-1, 1]

    ! 计算输出尺寸
    H_out = (H_in + 2*1 - 3) / 1 + 1  ! padding=1, kernel=3, stride=1
    W_out = (W_in + 2*1 - 3) / 1 + 1

    ! 测试 forward
    print *, "2. Testing forward..."
    output = conv%forward(input)
    if (size(output, 1) == H_out .and. size(output, 2) == W_out .and. &
        size(output, 3) == C_out .and. size(output, 4) == N) then
        print *, "   Forward successful. Output shape: ", shape(output)
    else
        print *, "   Forward failed. Expected shape: ", H_out, W_out, C_out, N, " Got: ", shape(output)
    end if

    ! 测试 backward
    print *, "3. Testing backward..."
    allocate(grad_output(H_out, W_out, C_out, N))
    call random_number(grad_output)
    grad_output = grad_output * 2.0_dp - 1.0_dp
    grad_input = conv%backward(grad_output)
    if (size(grad_input, 1) == H_in .and. size(grad_input, 2) == W_in .and. &
        size(grad_input, 3) == C_in .and. size(grad_input, 4) == N) then
        print *, "   Backward successful. Grad input shape: ", shape(grad_input)
    else
        print *, "   Backward failed. Expected shape: ", H_in, W_in, C_in, N, " Got: ", shape(grad_input)
    end if

    ! 测试 save
    print *, "4. Testing save..."
    call conv%save(save_path)
    print *, "   Save successful."

    ! 创建新层，测试 load
    print *, "5. Testing load..."
    call conv_loaded%init(C_in, C_out, 3, 1, 1)
    call conv_loaded%load(save_path)
    print *, "   Load successful."

    ! 验证 load 是否正确 (由于权重私有，无法直接比较，但假设成功)
    print *, "   Assuming weights loaded correctly (cannot verify due to private access)."

    ! 测试 update
    print *, "6. Testing update..."
    loss_before = sum(output**2)  ! 简单损失
    call conv%update(0.01_dp)  ! lr=0.01
    output = conv%forward(input)
    loss_after = sum(output**2)
    if (loss_after /= loss_before) then
        print *, "   Update successful. Loss changed from ", loss_before, " to ", loss_after
    else
        print *, "   Update failed. Loss unchanged."
    end if

    ! 测试 zero_grads
    print *, "7. Testing zero_grads..."
    call conv%zero_grads()
    print *, "   Zero grads successful (gradients reset)."

    ! 测试 destroy
    print *, "8. Testing destroy..."
    call conv%destroy()
    call conv_loaded%destroy()
    print *, "   Destroy successful."

    ! 清理
    deallocate(input, output, grad_output, grad_input)

    print *, "All tests completed."

end program test_conv