program test_load_data
    use LoadData_mod
    use iso_fortran_env, only: dp => real64
    implicit none

    ! --- 测试参数 ---
    character(len=*), parameter :: data_file = '/root/0_FoRemote/WhaleAdventureInFortran/FINAL/1_DATA_Reread/train-images3-.bin'
    integer, parameter :: total_items = 60000
    integer, parameter :: test_batch_size = 32
    integer, parameter :: img_h = 28, img_w = 28, img_c = 1

    ! --- 测试变量 ---
    type(Data_Loader) :: loader
    real(dp), allocatable :: batch_data(:,:,:,:)
    integer :: num_batches, i
    logical :: test_failed

    test_failed = .false.

    print *, "========================================"
    print *, "        Testing LoadData_mod"
    print *, "========================================"

    ! 1. 测试初始化
    print *, "1. Initializing Data_Loader..."
    call loader%init(data_file, test_batch_size, total_items, img_h, img_w, img_c)

    ! 使用 getter 验证参数是否设置正确
    if (loader%get_batch_size() /= test_batch_size) then
        print *, "   ERROR: Batch size mismatch. Expected:", test_batch_size, "Got:", loader%get_batch_size()
        test_failed = .true.
    end if
    if (loader%get_item_num() /= total_items) then
        print *, "   ERROR: Item number mismatch. Expected:", total_items, "Got:", loader%get_item_num()
        test_failed = .true.
    end if
    if (trim(loader%get_Data_Root()) /= trim(data_file)) then
        print *, "   ERROR: Data root mismatch."
        test_failed = .true.
    end if

    if (.not. test_failed) then
        print *, "   SUCCESS: Initialization and getters seem correct."
    end if
    print *, "----------------------------------------"

    ! 2. 测试 get_len()
    print *, "2. Testing get_len()..."
    num_batches = loader%get_len()
    if (num_batches == total_items / test_batch_size) then
        print *, "   SUCCESS: Correct number of batches calculated:", num_batches
    else
        print *, "   ERROR: Incorrect number of batches. Expected:", total_items / test_batch_size, "Got:", num_batches
        test_failed = .true.
    end if
    print *, "----------------------------------------"

    ! 3. 测试 get_batch()
    print *, "3. Testing get_batch()..."
    if (num_batches > 0) then
        ! 测试获取第一个批次 (batch_idx = 128)
        print *, "   Fetching first batch (index 128)..."
        call loader%get_batch(128, batch_data)

        if (allocated(batch_data)) then
            print *, "   SUCCESS: First batch data array allocated."
            print *, "   Shape of returned data: (", shape(batch_data), ")"
            print *, "   Expected shape:         (", test_batch_size, img_h, img_w, img_c, ")"
            if (any(shape(batch_data) /= [test_batch_size, img_h, img_w, img_c])) then
                print *, "   ERROR: Data shape mismatch for first batch."
                test_failed = .true.
            else
                ! 检查单个值以查看读取是否合理
                ! 虚拟数据不全为零，因此最大值应 > 0
                if (maxval(batch_data) > 0.0_dp .or. minval(batch_data) < 0.0_dp) then
                    print *, "   SUCCESS: Data values seem plausible (normalized, > 0)."
                else
                    print *, "   ERROR: Data values are unexpected (all zeros or negative)."
                    test_failed = .true.
                end if
            end if
            deallocate(batch_data)
        else
            print *, "   ERROR: get_batch(1) failed to allocate data array."
            test_failed = .true.
        end if
    else
        print *, "   SKIPPED: No batches to test."
    end if
    print *, "----------------------------------------"

    ! 4. 测试 destroy()
    print *, "4. Testing destroy()..."
    call loader%destroy()
    if (loader%get_item_num() == 0 .and. loader%get_batch_size() == 0) then
        print *, "   SUCCESS: Loader parameters reset after destroy."
    else
        print *, "   ERROR: Loader parameters not reset after destroy."
        test_failed = .true.
    end if
    print *, "----------------------------------------"

    ! 最终结果
    print *, "========================================"
    if (test_failed) then
        print *, "        TEST FAILED"
    else
        print *, "        ALL TESTS PASSED"
    end if
    print *, "========================================"

end program test_load_data