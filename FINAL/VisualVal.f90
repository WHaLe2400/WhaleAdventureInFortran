program VisualVal
    use iso_fortran_env, only: dp => real64
    use ModelCombine_mod
    use LoadData_mod
    use LoadLabel_mod
    implicit none

    ! --- 变量声明 ---
    type(Model) :: my_model
    type(Data_Loader) :: test_data_loader
    type(Label_Loader) :: test_label_loader
    
    real(dp), allocatable :: input_batch(:,:,:,:)
    real(dp), allocatable :: label_batch(:,:)
    real(dp), allocatable :: output_batch(:,:)
    
    integer :: i, j, img_idx
    integer :: batch_size = 4 ! 可视化 4 张图片
    integer :: predicted_label, true_label
    real(dp) :: pixel_val
    integer :: predicted_loc(1)
    real(dp) :: exp_logits(10), sum_exp, probabilities(10)
    
    ! 路径配置
    character(len=*), parameter :: data_root = "/root/0_FoRemote/WhaleAdventureInFortran/FINAL_rebuild/1_DATA_Reread/"
    ! 使用 epoch_12 的权重，你可以根据需要修改为其他 epoch
    character(len=*), parameter :: model_path = "/root/0_FoRemote/WhaleAdventureInFortran/FINAL_rebuild/config_fromTorch"
    
    print *, "========================================"
    print *, "      Model Visualization Validation    "
    print *, "========================================"

    ! 1. 初始化模型
    print *, "Initializing Model..."
    call my_model%init()
    
    ! 2. 加载权重
    print *, "Loading Weights from: ", model_path
    call my_model%load(model_path)
    
    ! 3. 切换到评估模式
    call my_model%eval()
    
    ! 4. 初始化数据加载器
    print *, "Initializing Data Loaders..."
    call test_data_loader%init(data_root // "t10k-images3-.bin", batch_size, 100, 28, 28, 1)
    call test_label_loader%init(data_root // "t10k-labels1-.bin", batch_size, 100)
    
    ! 5. 获取一个批次的数据
    print *, "Loading a batch of test data..."
    call test_data_loader%get_batch(3, input_batch)
    call test_label_loader%get_batch(3, label_batch)
    
    ! 6. 前向传播
    print *, "Running Forward Pass..."
    output_batch = my_model%forward(input_batch)
    
    ! 7. 可视化结果
    do img_idx = 1, batch_size
        ! 获取预测结果 (argmax)
        ! output_batch 形状: (10, batch_size), 我们需要对第 img_idx 个样本的10个类别值找最大
        predicted_loc = maxloc(output_batch(:, img_idx))
        predicted_label = predicted_loc(1) - 1
        
        ! 获取真实标签
        ! label_batch 形状: (1, batch_size)
        true_label = int(label_batch(1, img_idx))
        
        print *, ""
        print *, "#==============================================================================#"
        print *, "Image Index: ", img_idx
        ! 计算预测概率 (softmax)
        ! 对第 img_idx 个样本的 logits 计算 softmax
        exp_logits = exp(output_batch(:, img_idx))
        sum_exp = sum(exp_logits)
        probabilities = exp_logits / sum_exp
        
        print *, "Probabilities:"

        write(*, "(A)", advance = "no") "|  "
        do j = 0, 9
            write(*, '(I1, 7X)', advance = "no")  j
        end do
        write(*, *) "|"
        
        write(*, "(A)", advance = "no") "|  "
        do j = 1, 10
            write(*, '(F6.4, 2X)', advance = "no")  probabilities(j)
        end do
        write(*, *) "|"
        print *, "Predicted: ", predicted_label
        print *, "True Label: ", true_label
        
        
        print *, "Visualization:"
        print *, "+----------------------------------------------------------+"
        
        do i = 1, 28
            write(*, '(A, A)', advance='no') " | "
            do j = 1, 28
                ! input_batch 维度: (H, W, C, N) -> (i, j, 1, img_idx)
                pixel_val = input_batch(i, j, 1, img_idx)

                ! -0.2 作为阈值来显示数字轮廓
                if (pixel_val > -0.3_dp) then
                    write(*, '(A)', advance='no') "##"
                else
                    write(*, '(A)', advance='no') "  "
                end if
            end do
            write(*, '(A)') " |"
        end do
        print *, "+----------------------------------------------------------+"
    end do
    
    ! 8. 清理资源
    call my_model%destroy()
    call test_data_loader%destroy()
    call test_label_loader%destroy()
    if (allocated(input_batch)) deallocate(input_batch)
    if (allocated(label_batch)) deallocate(label_batch)
    if (allocated(output_batch)) deallocate(output_batch)

end program VisualVal
