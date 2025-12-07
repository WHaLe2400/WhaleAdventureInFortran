program test_fullconnect
    use iso_fortran_env, only: dp => real64
    use FullConnect_mod
    implicit none

    type(FullConnectLayer) :: fc
    type(FullConnectLayer) :: fc_loaded
    real(dp), allocatable :: input_batch(:,:), output_batch(:,:), grad_output_batch(:,:), grad_input_batch(:,:)
    real(dp), allocatable :: weights_before(:,:), biases_before(:), grad_weights(:,:), grad_biases(:)
    real(dp) :: loss_before, loss_after
    integer :: input_size = 784, output_size = 128, batch_size = 32
    character(len=100) :: save_path = "test_fc_weights.dat"

    ! 初始化随机种子
    call random_seed()

    print *, "Testing FullConnectLayer..."

    ! 测试 init
    print *, "1. Testing init..."
    call fc%init(input_size, output_size)
    print *, "   Init successful. Input size: ", fc%get_input_size(), " Output size: ", fc%get_output_size()

    ! 分配输入数据 (input_size, batch_size)
    allocate(input_batch(input_size, batch_size))
    call random_number(input_batch)
    input_batch = input_batch * 2.0_dp - 1.0_dp  ! [-1, 1]

    ! 测试 forward
    print *, "2. Testing forward..."
    output_batch = fc%forward(input_batch)
    if (size(output_batch, 1) == output_size .and. size(output_batch, 2) == batch_size) then
        print *, "   Forward successful. Output shape: ", shape(output_batch)
    else
        print *, "   Forward failed. Expected shape: ", output_size, batch_size, " Got: ", shape(output_batch)
    end if

    ! 测试 backward
    print *, "3. Testing backward..."
    allocate(grad_output_batch(output_size, batch_size))
    call random_number(grad_output_batch)
    grad_output_batch = grad_output_batch * 2.0_dp - 1.0_dp
    grad_input_batch = fc%backward(grad_output_batch)
    if (size(grad_input_batch, 1) == input_size .and. size(grad_input_batch, 2) == batch_size) then
        print *, "   Backward successful. Grad input shape: ", shape(grad_input_batch)
    else
        print *, "   Backward failed. Expected shape: ", input_size, batch_size, " Got: ", shape(grad_input_batch)
    end if

    ! 测试 save
    print *, "4. Testing save..."
    call fc%save(save_path)
    print *, "   Save successful."

    ! 获取当前权重用于比较
    weights_before = fc%get_weights()
    biases_before = fc%get_biases()

    ! 创建新层，测试 load
    print *, "5. Testing load..."
    call fc_loaded%init(input_size, output_size)
    call fc_loaded%load(save_path)
    print *, "   Load successful."

    ! 验证 load 是否正确 (比较权重)
    if (all(abs(weights_before - fc_loaded%get_weights()) < 1e-10) .and. &
        all(abs(biases_before - fc_loaded%get_biases()) < 1e-10)) then
        print *, "   Weights match after load."
    else
        print *, "   Weights do not match after load."
    end if

    ! 测试 update
    print *, "6. Testing update..."
    loss_before = sum(output_batch**2)  ! 简单损失
    call fc%update(0.01_dp)  ! lr=0.01
    output_batch = fc%forward(input_batch)
    loss_after = sum(output_batch**2)
    if (loss_after /= loss_before) then
        print *, "   Update successful. Loss changed from ", loss_before, " to ", loss_after
    else
        print *, "   Update failed. Loss unchanged."
    end if

    ! 测试 zero_grads
    print *, "7. Testing zero_grads..."
    call fc%zero_grads()
    grad_weights = fc%get_grad_weights()
    grad_biases = fc%get_grad_biases()
    if (all(grad_weights == 0.0_dp) .and. all(grad_biases == 0.0_dp)) then
        print *, "   Zero grads successful."
    else
        print *, "   Zero grads failed."
    end if

    ! 测试 destroy
    print *, "8. Testing destroy..."
    call fc%destroy()
    call fc_loaded%destroy()
    print *, "   Destroy successful."

    ! 清理
    deallocate(input_batch, output_batch, grad_output_batch, grad_input_batch, &
               weights_before, biases_before, grad_weights, grad_biases)

    print *, "All tests completed."

end program test_fullconnect
