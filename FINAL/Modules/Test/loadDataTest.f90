program test_loaddata
    use iso_fortran_env, only: dp => real64
    use LoadData_mod
    implicit none

    type(Data_Loader) :: test_data_loader
    real(dp), allocatable :: batch_data(:,:,:,:)
    integer :: num_batches, batch_idx
    character(len=200) :: data_root
    integer :: batch_size = 4, item_num = 20  ! 小批量用于测试
    integer :: data_h = 28, data_w = 28, data_c = 1

    print *, ""
    print *, "Testing Data_Loader..."

    ! 设置数据路径 (假设有测试数据文件)
    data_root = "/root/0_FoRemote/WhaleAdventureInFortran/FINAL_rebuild/1_DATA_Reread/train-images3-.bin"

    ! 测试 init
    print *, "1. Testing init..."
    call test_data_loader%init(data_root, batch_size, item_num, data_h, data_w, data_c)
    print *, "   Init successful."
    print *, ""

    ! 测试 getter 函数
    print *, "2. Testing getters..."
    print *, "   Data root: ", trim(test_data_loader%get_Data_Root())
    print *, "   Batch size: ", test_data_loader%get_batch_size()
    print *, "   Item num: ", test_data_loader%get_item_num()
    print *, ""

    ! 测试 get_len
    print *, "3. Testing get_len..."
    num_batches = test_data_loader%get_len()
    print *, "   Number of batches: ", num_batches
    print *, ""

    ! 测试 get_batch
    print *, "4. Testing get_batch..."
    do batch_idx = 1, min(2, num_batches)  ! 测试前几个批次
        call test_data_loader%get_batch(batch_idx, batch_data)
        if (size(batch_data, 1) == data_h .and. size(batch_data, 2) == data_w .and. &
            size(batch_data, 3) == data_c .and. size(batch_data, 4) == batch_size) then
            print *, "   Batch ", batch_idx, " shape correct: ", shape(batch_data)
            print *, "   Data range: ", minval(batch_data), " to ", maxval(batch_data)
        else
            print *, "   Batch ", batch_idx, " shape incorrect. Expected: ", data_h, data_w, data_c, batch_size, &
                     " Got: ", shape(batch_data)
        end if
        deallocate(batch_data)
    end do
    print *, ""

    ! 测试 destroy
    print *, "5. Testing destroy..."
    call test_data_loader%destroy()
    print *, "   Destroy successful."
    print *, ""

    print *, "All tests completed."

end program test_loaddata
