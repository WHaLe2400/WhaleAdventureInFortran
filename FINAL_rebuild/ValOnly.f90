program Validation
    use iso_fortran_env, only: dp => real64
    use ModelCombine_mod
    use LoadData_mod
    use LoadLabel_mod

    implicit none

    ! --- 变量声明 ---
    type(Model) :: my_model
    type(Data_Loader) :: test_data_loader
    type(Label_Loader) :: test_label_loader

    ! --- 文件路径和参数 ---
    character(len=*), parameter :: file_root = "/root/0_FoRemote/WhaleAdventureInFortran/FINAL_rebuild/1_DATA_Reread/"
    character(len=*), parameter :: test_data_path = file_root // "t10k-images3-.bin"
    character(len=*), parameter :: test_label_path = file_root // "t10k-labels1-.bin"
    character(len=*), parameter :: model_load_path = "/root/0_FoRemote/WhaleAdventureInFortran/FINAL_rebuild/RESULTS/Models/epoch_2"

    integer, parameter :: batch_size = 100
    integer, parameter :: test_item_num = 10000

    ! --- 评估变量 ---
    real(dp), allocatable :: input(:,:,:,:), labels(:,:), output(:,:)
    integer :: num_batches, i, j, correct_count, total_count
    integer :: predicted_loc(1)
    real(dp) :: accuracy
    real(dp) :: start_time, end_time

    ! --- 程序开始 ---
    print *, "#==========================================================================#"
    print *, "|              Whale Adventure In Fortran - Validation Script              |"
    print *, "#==========================================================================#"
    print *, "Loading model from: ", trim(model_load_path)
    print *, "Using test data: ", trim(file_root)
    print *, ""

    call cpu_time(start_time)

    ! 1. 初始化
    call my_model%init()
    call test_data_loader%init(test_data_path, batch_size, test_item_num, 28, 28, 1)
    call test_label_loader%init(test_label_path, batch_size, test_item_num)

    ! 2. 加载模型权重
    call my_model%load(model_load_path)

    ! 3. 执行评估
    num_batches = test_data_loader%get_len()
    correct_count = 0
    total_count = 0

    ! 切换到评估模式 (禁用 Dropout)
    call my_model%eval()

    write(*, '(A)', advance='no') "Evaluating on test set: "
    do i = 1, num_batches
        if (mod(i, 10) == 0) write(*, '(A)', advance='no') "."

        ! 获取当前批次的数据和标签
        call test_data_loader%get_batch(i, input)
        call test_label_loader%get_batch(i, labels)
        
        ! 前向传播
        output = my_model%forward(input)
        
        ! 计算正确预测的数量
        do j = 1, batch_size
            predicted_loc = maxloc(output(:, j))
            if ((predicted_loc(1) - 1) == int(labels(1, j))) then
                correct_count = correct_count + 1
            end if
            total_count = total_count + 1
        end do
        
        deallocate(input, labels, output)
    end do
    write(*, '(A)') " Done."
    
    call cpu_time(end_time)

    ! 4. 计算并打印最终准确率
    if (total_count > 0) then
        accuracy = real(correct_count, dp) / real(total_count, dp)
    else
        accuracy = 0.0_dp
    end if

    print *, ""
    print *, "#=========================== RESULT =======================================#"
    write(*, '("Total samples: ", I0, ", Correct predictions: ", I0)') total_count, correct_count
    write(*, '("Final Accuracy: ", F6.2, "%")') accuracy * 100.0_dp
    write(*, '("Total evaluation time: ", F8.3, " seconds")') end_time - start_time
    print *, "#==========================================================================#"
    print *, ""

    ! 5. 销毁并释放内存
    call my_model%destroy()
    call test_data_loader%destroy()
    call test_label_loader%destroy()

end program Validation
