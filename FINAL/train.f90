program Train
    use iso_fortran_env, only: dp => real64
    use ModelCombine_mod
    use LoadData_mod
    use LoadLabel_mod
    use LossFunc_mod

    implicit none

    ! --- 变量声明 ---
    ! 1. 模型、损失函数和数据加载器
    type(Model) :: my_model
    type(LossFunc) :: loss_func
    type(Data_Loader) :: train_data_loader, test_data_loader
    type(Label_Loader) :: train_label_loader, test_label_loader

    ! 2. 文件路径 (使用 parameter 来定义常量)
    character(len=*), parameter :: file_root = "/root/0_FoRemote/WhaleAdventureInFortran/FINAL/1_DATA_Reread/"
    character(len=*), parameter :: train_data_path = file_root // "train-images3-.bin"
    character(len=*), parameter :: train_label_path = file_root // "train-labels1-.bin"
    character(len=*), parameter :: test_data_path = file_root // "t10k-images3-.bin"
    character(len=*), parameter :: test_label_path = file_root // "t10k-labels1-.bin"
    character(len=*), parameter :: model_save_path = "/root/0_FoRemote/WhaleAdventureInFortran/FINAL/RESULTS/Models"
    character(len=*), parameter :: Torch_weights_path = "/root/0_FoRemote/WhaleAdventureInFortran/FINAL/RESULTS/Models/" &
                                     // "config_fromTroch"

    ! 3. 训练超参数 (每个类型分开声明)
    integer :: epoch = 16
    integer :: batch_size = 64
    real(dp) :: learning_rate = 0.1_dp

    real(dp), allocatable :: input(:,:,:,:), output(:,:), grad_output(:,:)

    real(dp) :: loss, accuracy

    ! --- 程序逻辑开始 ---
    print *, ""
    print *, "#==========================================================================#"
    print *, "|               Whale Adventure In Fortran - Training Script               |"
    print *, "#==========================================================================#"
    print *,    "Training Configuration:      "
    print *,    "   Epochs: ", epoch
    print *,    "   Batch Size: ", batch_size
    print *,    "   Learning Rate: ", learning_rate
    print *,    "   Using Data: ", trim(file_root)
    print *,    "   Saving Model To: ", trim(model_save_path)
    print *,    "#==========================================================================#"

    ! 初始化
    call init()

    ! call load_model(Torch_weights_path)

    call train_process()

    call destroy()

    print *, ""
    print *, "#==========================================================================#"
    print *, "|                        Training Process Completed                        |"
    print *, "#==========================================================================#"
    print *, "Model saved to: ", trim(model_save_path)
    print *, "                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
    print *, ""

contains

    subroutine init()
        ! 作为内部过程，这里可以访问主程序的所有变量
        call my_model%init()
        call loss_func%destroy() ! 清空loss的缓存
        call train_data_loader%init(train_data_path, batch_size, 6000, 28, 28, 1)
        call train_label_loader%init(train_label_path, batch_size, 6000)
        call test_data_loader%init(test_data_path, batch_size, 1000, 28, 28, 1)
        call test_label_loader%init(test_label_path, batch_size, 1000)
    end subroutine init


    subroutine load_model(base_path)
        character(len=*), intent(in) :: base_path
        call my_model%load(base_path)
    end subroutine load_model


    subroutine save_model(base_path)
        character(len=*), intent(in) :: base_path
        call my_model%save(base_path)
    end subroutine save_model


    function train_one_epoch() result(avg_loss)
        integer :: num_batches, i, j
        real(dp), allocatable :: input(:,:,:,:), labels(:,:), labels_onehot(:,:), output(:,:), grad_output(:,:)
        real(dp) :: current_loss, total_loss
        real(dp) :: avg_loss

        num_batches = train_data_loader%get_len()
        total_loss = 0.0_dp

        ! 切换到训练模式 (启用 Dropout)
        call my_model%train()

        write(*, '(A)', advance='no') "Training: "
        do i = 1, num_batches
            if (mod(i, 1) == 0) write(*, '(A)', advance='no') "*"
            ! 在计算新梯度前，清零所有层的梯度
            call my_model%zero_grads()

            ! 获取当前批次的数据和标签
            call train_data_loader%get_batch(i, input)
            call train_label_loader%get_batch(i, labels)
            
            ! 将标签转换为 one-hot 编码 (batch_size, 10)
            allocate(labels_onehot(batch_size, 10))
            labels_onehot = 0.0_dp
            do j = 1, batch_size
                if (int(labels(j, 1)) >= 0 .and. int(labels(j, 1)) <= 9) then
                    labels_onehot(j, int(labels(j, 1)) + 1) = 1.0_dp
                end if
                ! print *, " -> One-hot: ", labels_onehot(j, :)
            end do
            
            ! 前向传播
            output = my_model%forward(input)
            ! 计算损失和梯度
            current_loss = loss_func%forward(output, labels_onehot)
            total_loss = total_loss + current_loss
            grad_output = loss_func%backward()
            ! 反向传播 (此过程会累积梯度)
            call my_model%backward(grad_output)
            ! 更新参数
            call my_model%update(learning_rate)
            ! 释放内存
            deallocate(input, labels, labels_onehot, output, grad_output)
        end do
        write(*, '(A)') ""  ! 换行
        
        avg_loss = total_loss / real(num_batches, dp)
        
    end function train_one_epoch


    subroutine evaluate_model(accuracy)
        integer :: num_batches, i, j, correct_count, total_count
        ! 修改：使用固定大小数组，避免循环内频繁分配/释放内存
        integer :: predicted_loc(1) 
        real(dp), allocatable :: input(:,:,:,:), labels(:,:), output(:,:)
        real(dp) :: accuracy

        num_batches = test_data_loader%get_len()
        correct_count = 0
        total_count = 0

        ! 切换到评估模式 (禁用 Dropout)
        call my_model%eval()

        write(*, '(A)', advance='no') "Evaluating: "
        do i = 1, num_batches
            if (mod(i, 1) == 0) write(*, '(A)', advance='no') "*"
            ! 获取当前批次的数据和标签
            call test_data_loader%get_batch(i, input)
            call test_label_loader%get_batch(i, labels)
            
            ! 前向传播
            output = my_model%forward(input)
            
            ! 计算正确预测
            do j = 1, size(output, 1)  ! 遍历批次中的每个样本
                ! maxloc 返回一个单元素数组，直接赋值给固定大小数组
                predicted_loc = maxloc(output(j, :), dim=1)
                
                ! 检查预测类别是否与真实标签匹配
                ! predicted_loc(1) 获取索引值
                if ((predicted_loc(1) - 1) == int(labels(j, 1))) then
                    correct_count = correct_count + 1
                end if
                total_count = total_count + 1
            end do
            
            ! 在循环内释放内存以防止泄漏
            if (allocated(input)) deallocate(input)
            if (allocated(labels)) deallocate(labels)
            if (allocated(output)) deallocate(output)
        end do
        write(*, '(A)') ""  ! 换行

        if (total_count > 0) then
            accuracy = real(correct_count, dp) / real(total_count, dp)
        else
            accuracy = 0.0_dp
        end if
    end subroutine evaluate_model


    subroutine train_process()
        integer :: epoch_idx
        real(dp) :: loss, accuracy
        character(len=10) :: epoch_str
        integer :: start_time, end_time, clock_rate
        real(dp) :: elapsed_time

        ! 获取时钟频率
        call system_clock(count_rate=clock_rate)

        do epoch_idx = 1, epoch
            ! 开始计时
            call system_clock(count=start_time)

            loss = train_one_epoch()
            ! loss = 0

            call evaluate_model(accuracy)

            ! 结束计时
            call system_clock(count=end_time)
            elapsed_time = real(end_time - start_time, dp) / real(clock_rate, dp)

            print *, ""
            write(*, '(A, I0, A, I0, A, F8.4, A, F6.2, A, F8.2, A)') &
            &"Epoch ", epoch_idx, "/", epoch, " completed. Training Loss: ", loss, "      Test Accuracy: ", accuracy * 100.0_dp, "%", " Time: ", elapsed_time, "s"
            if (mod(epoch_idx, 2) == 0) then
                write(epoch_str, '(I0)') epoch_idx
                call save_model(trim(model_save_path) // "/epoch_" // trim(epoch_str))
                print *, "Model saved at epoch ", trim(epoch_str)
            end if
        end do
    end subroutine train_process

    subroutine destroy()
        call my_model%destroy()
        call loss_func%destroy()
        call train_data_loader%destroy()
        call train_label_loader%destroy()
        call test_data_loader%destroy()
        call test_label_loader%destroy()
    end subroutine destroy

end program Train