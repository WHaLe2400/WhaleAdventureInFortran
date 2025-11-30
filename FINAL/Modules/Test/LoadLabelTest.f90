program LoadLabelTest
    use LoadLabel_mod
    use iso_fortran_env, only: dp => real64
    implicit none

    type(Label_Loader) :: test_label_loader
    real(dp), allocatable :: labels(:,:)
    integer :: batch_size = 10, item_num = 10000  ! 测试用小批量
    character(len=200) :: label_path = "/root/0_FoRemote/WhaleAdventureInFortran/FINAL/1_DATA_Reread/train-labels1-.bin"
    integer :: i

    ! 1. 初始化标签加载器
    print *, "--- 1. Initializing Label Loader ---"
    call test_label_loader%init(label_path, batch_size, item_num)
    print *, "Label loader initialized."
    print *, "Batch size: ", test_label_loader%get_batch_size()
    print *, "Item num: ", test_label_loader%get_item_num()
    print *, "Label root: ", trim(test_label_loader%get_Label_Root())
    print *, "-----------------------------"
    print *, ""

    ! 2. 获取第一批标签
    print *, "--- 2. Getting First Batch of Labels ---"
    call test_label_loader%get_batch(1, labels)
    print *, "First batch labels shape: ", shape(labels)
    print *, "First 10 labels:"
    do i = 1, min(10, batch_size)
        print *, "Label ", i, ": ", labels(i, 1)
    end do
    print *, "-----------------------------"
    print *, ""

    ! 3. 获取第二批标签（可选）
    print *, "--- 3. Getting Second Batch of Labels ---"
    call test_label_loader%get_batch(2, labels)
    print *, "Second batch first 5 labels:"
    do i = 1, min(5, batch_size)
        print *, "Label ", i, ": ", labels(i, 1)
    end do
    print *, "-----------------------------"
    print *, ""

    ! 4. 清理
    print *, "--- 4. Cleaning Up ---"
    call test_label_loader%destroy()
    if (allocated(labels)) deallocate(labels)
    print *, "Cleanup complete."
    print *, "-----------------------------"

end program LoadLabelTest