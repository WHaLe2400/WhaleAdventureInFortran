program loadDataTest_2
    use iso_fortran_env, only: dp => real64
    use LoadData_mod
    implicit none

    ! --- 变量声明 ---
    type(Data_Loader) :: loader
    real(dp), allocatable :: batch_data(:,:,:,:) ! (N, C, H, W)
    integer :: i, j, img_idx
    integer :: batch_size, h, w
    real(dp) :: pixel_val
    
    ! --- 配置路径 (请根据实际情况修改) ---
    character(len=*), parameter :: data_path = "/root/0_FoRemote/WhaleAdventureInFortran/FINAL/1_DATA_Reread/t10k-images3-.bin"
    
    ! --- 初始化参数 ---
    batch_size = 4
    h = 28
    w = 28
    
    print *, "========================================"
    print *, "      Data Loader Visualization Test    "
    print *, "========================================"
    print *, "Reading from: ", data_path


    ! 1. 初始化加载器
    call loader%init(data_path, batch_size, 100, h, w, 1)

    ! 2. 获取第一个批次
    print *, "Loading batch 1..."
    call loader%get_batch(1, batch_data)

    ! 3. 验证数据形状
    print *, "Batch Shape: (", size(batch_data, 1), ", ", size(batch_data, 2), ", ", &
             size(batch_data, 3), ", ", size(batch_data, 4), ")"

    ! 4. 绘制前几张图片
    do img_idx = 1, min(batch_size, 3) ! 只画前3张
        print *, ""
        print *, "--- Image ", img_idx, " ---"
        print *, "   +----------------------------+"
        
        do i = 1, h
            write(*, '(A, A)', advance='no') "   | "
            do j = 1, w
                ! 获取像素值 (N, C, H, W) -> (img_idx, 1, i, j)
                pixel_val = batch_data(img_idx, 1, i, j)
                
                ! 根据像素亮度选择字符
                ! 0.1 是阈值，大于0.1显示为 ## (墨水/前景)，小于显示为 -- (背景)
                if (pixel_val > 2.0_dp) then
                    write(*, '(A)', advance='no') "##"
                else
                    write(*, '(A)', advance='no') "--"
                end if
            end do
            write(*, '(A)') " |" ! 行尾换行
        end do
        
        print *, "   +----------------------------+"
    end do

    ! 5. 清理
    deallocate(batch_data)
    call loader%destroy()
    
    print *, ""
    print *, "Test Completed."

end program loadDataTest_2