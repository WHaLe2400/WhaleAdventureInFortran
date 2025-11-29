program test_loss_func
    use LossFunc_mod
    use iso_fortran_env, only: dp => real64
    implicit none

    ! --- 测试参数 ---
    integer, parameter :: batch_size = 4
    integer, parameter :: num_classes = 5
    real(dp), parameter :: tolerance = 1e-6_dp

    ! --- 测试变量 ---
    type(LossFunc) :: loss_func
    real(dp), allocatable :: logits(:,:), grad_logits(:,:)
    integer, allocatable :: labels(:)
    real(dp) :: loss, expected_loss
    integer :: i
    logical :: test_failed = .false.

    print *, "========================================"
    print *, "         Testing LossFunc_mod"
    print *, "========================================"

    ! --- 1. 测试 "自信" 的预测 (损失应该很低) ---
    print *, "1. Testing with 'confident' predictions..."
    allocate(logits(batch_size, num_classes))
    allocate(labels(batch_size))

    ! 创建 logits，其中正确类别的分数很高
    logits = 0.0_dp
    labels = [0, 1, 2, 3] ! 标签从 0 开始
    do i = 1, batch_size
        logits(i, labels(i) + 1) = 10.0_dp ! 正确类别的 logit 值很高
    end do

    loss = loss_func%forward(logits, labels)
    print *, "   Calculated loss (confident): ", loss
    if (loss < 0.1_dp .and. loss >= 0.0_dp) then
        print *, "   SUCCESS: Loss is small and positive as expected."
    else
        print *, "   ERROR: Loss value is not in the expected range."
        test_failed = .true.
    end if
    print *, "----------------------------------------"

    ! --- 2. 测试 "不确定" 的预测 (均匀分布) ---
    print *, "2. Testing with 'uncertain' predictions..."
    ! 创建 logits，所有值都相同，模拟均匀概率
    logits = 0.0_dp
    labels = [0, 1, 2, 3]

    ! 理论损失应该是 log(类别数)
    expected_loss = log(real(num_classes, dp))
    loss = loss_func%forward(logits, labels)

    print *, "   Calculated loss (uncertain): ", loss
    print *, "   Theoretical loss:            ", expected_loss
    if (abs(loss - expected_loss) < tolerance) then
        print *, "   SUCCESS: Calculated loss matches theoretical loss."
    else
        print *, "   ERROR: Calculated loss does not match theoretical loss."
        test_failed = .true.
    end if
    print *, "----------------------------------------"

    ! --- 3. 测试反向传播 (backward) ---
    print *, "3. Testing backward pass..."
    grad_logits = loss_func%backward()

    ! 检查梯度形状
    if (allocated(grad_logits)) then
        print *, "   Gradient shape: (", shape(grad_logits), ")"
        print *, "   Expected shape: (", batch_size, ",", num_classes, ")"
        if (any(shape(grad_logits) /= [batch_size, num_classes])) then
            print *, "   ERROR: Gradient shape mismatch."
            test_failed = .true.
        else
            print *, "   SUCCESS: Gradient shape is correct."
            ! 检查梯度总和属性
            do i = 1, batch_size
                if (abs(sum(grad_logits(i, :))) > tolerance) then
                    print *, "   ERROR: Sum of gradients for sample", i, " is not zero."
                    test_failed = .true.
                    exit
                end if
            end do
            if (.not. test_failed) then
                print *, "   SUCCESS: Sum of gradients for all samples is zero."
            end if
        end if
    else
        print *, "   ERROR: Backward pass failed to allocate gradient."
        test_failed = .true.
    end if
    print *, "----------------------------------------"

    ! --- 4. 清理 ---
    print *, "4. Cleaning up..."
    call loss_func%destroy()
    deallocate(logits, labels)
    if (allocated(grad_logits)) deallocate(grad_logits)
    print *, "   Done."
    print *, "========================================"

    ! --- 最终结果 ---
    if (test_failed) then
        print *, "        TEST FAILED"
    else
        print *, "        ALL TESTS PASSED"
    end if
    print *, "========================================"

end program test_loss_func