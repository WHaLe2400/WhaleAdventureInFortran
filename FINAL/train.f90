program Train
    use iso_fortran_env, only: dp => real64
    use ModelCombine_mod
    use LoadData_mod
    use LossFunc_mod

    implicit none

    ! --- 变量声明 ---
    ! 1. 模型、损失函数和数据加载器
    type(Model) :: my_model
    type(LossFunc) :: loss_func
    type(Data_Loader) :: train_data_loader, train_label_loader, test_data_loader, test_label_loader

    ! 2. 文件路径 (使用 parameter 来定义常量)
    character(len=*), parameter :: file_root = "/root/0_FoRemote/WhaleAdventureInFortran/FINAL/1_DATA_Reread"
    character(len=*), parameter :: train_data_path = file_root // "/train-images3-.bin"
    character(len=*), parameter :: train_label_path = file_root // "/train-labels1-.bin"
    character(len=*), parameter :: test_data_path = file_root // "/t10k-images3-.bin"
    character(len=*), parameter :: test_label_path = file_root // "/t10k-labels1-.bin"

    ! 3. 训练超参数 (每个类型分开声明)
    integer :: epoch = 16
    integer :: batch_size = 64
    real(dp) :: learning_rate = 0.01_dp

    real(dp), allocatable :: input(:,:,:,:), output(:,:), grad_output(:,:)

    ! --- 程序逻辑开始 ---
    print *, "Epochs: ", epoch
    print *, "Batch Size: ", batch_size
    print *, "Learning Rate: ", learning_rate

    ! 初始化
    call init()

    ! ... 后续的训练逻辑 ...
    call train_data_loader%get_batch(1, input)  ! 示例调用，实际使用时应传递正确的数组

    print *, "Test_data_shape: ", shape(input)
contains

    subroutine init()
        ! 作为内部过程，这里可以访问主程序的所有变量
        call my_model%init()
        call loss_func%destroy() ! 清空loss的缓存
        call train_data_loader%init(train_data_path, batch_size, 60000, 28, 28, 1)
        call train_label_loader%init(train_label_path, batch_size, 60000, 1, 1, 1)
        call test_data_loader%init(test_data_path, batch_size, 10000, 28, 28, 1)
        call test_label_loader%init(test_label_path, batch_size, 10000, 1, 1, 1)
    end subroutine init

end program Train