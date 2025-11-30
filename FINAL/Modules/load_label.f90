module LoadLabel_mod
    use iso_fortran_env, only: dp => real64
    implicit none
    private
    public :: Label_Loader

    type, public :: Label_Loader
    private
        character(len=200) :: LABEL_ROOT
        integer :: batch_size = 32, item_num = 0
        integer :: file_unit = -1 ! 用于存储文件单元号，-1表示未打开
        
    contains
        procedure, public :: init => LL_init
        procedure, public :: get_len => LL_get_len
        procedure, public :: get_batch => LL_get_batch
        procedure, public :: destroy => LL_destroy
        ! ---get funcs---
        procedure, public :: get_Label_Root => LL_get_Label_Root
        procedure, public :: get_batch_size => LL_get_batch_size
        procedure, public :: get_item_num => LL_get_item_num
        
    end type Label_Loader

contains

    subroutine LL_init(self, label_root, batch_size, item_num)
        class(Label_Loader), intent(inout) :: self
        character(len=*), intent(in) :: label_root
        integer, intent(in) :: batch_size, item_num
        integer :: iostat
        character(len=:), allocatable :: filename

        self%LABEL_ROOT = label_root
        self%batch_size = batch_size
        self%item_num = item_num

        filename = trim(self%LABEL_ROOT)

        ! 使用 newunit 获取一个可用的文件单元号并存储在对象中
        open(newunit=self%file_unit, file=filename, form='unformatted', access='stream', status='old', iostat=iostat)
        if (iostat /= 0) then
            print *, "Error opening label file: ", trim(filename), ", iostat=", iostat
            self%file_unit = -1 ! 确保出错时 file_unit 是无效值
            stop
        end if
    end subroutine LL_init

    function LL_get_len(self) result(len)
        class(Label_Loader), intent(in) :: self
        integer :: len
        len = self%item_num / self%batch_size
    end function LL_get_len

    subroutine LL_get_batch(self, batch_idx, labels)
        class(Label_Loader), intent(inout) :: self
        integer, intent(in) :: batch_idx
        real(dp), allocatable, intent(out) :: labels(:,:)  ! 形状 (batch_size, 1)
        integer :: start_pos
        integer :: iostat
        integer(kind=1), allocatable :: temp_labels(:)  ! 用于读取 kind=1 的整数

        ! 检查文件是否已成功打开
        if (self%file_unit == -1) then
            print *, "Error: Label file not open. Call init first."
            stop
        end if

        ! 计算读取位置 (跳过魔数和数量，标签从第9字节开始)
        start_pos = 8 + (batch_idx - 1) * self%batch_size + 1  ! 魔数4字节 + 数量4字节 = 8字节

        ! 分配输出数组和临时数组
        allocate(labels(self%batch_size, 1))
        allocate(temp_labels(self%batch_size))

        ! 从对象中存储的单元号读取文件到临时整数数组
        read(unit=self%file_unit, pos=start_pos, iostat=iostat) temp_labels
        
        if (iostat /= 0) then
            print *, "Error reading from label file, batch_idx=", batch_idx, ", iostat=", iostat
            deallocate(temp_labels)
            stop
        end if

        ! 将整数数据转换为 real(dp) 类型
        labels(:, 1) = real(temp_labels, kind=dp)

        ! 释放临时数组
        deallocate(temp_labels)

    end subroutine LL_get_batch

    subroutine LL_destroy(self)
        class(Label_Loader), intent(inout) :: self
        
        ! 如果文件单元有效（即文件已打开），则关闭它
        if (self%file_unit >= 0) then
            close(self%file_unit)
            self%file_unit = -1 ! 重置为无效值
        end if

        ! 清理其他成员
        self%item_num = 0
        self%batch_size = 0
    end subroutine LL_destroy

    ! --- Getter Functions ---
    function LL_get_Label_Root(self) result(val)
        class(Label_Loader), intent(in) :: self
        character(len=200) :: val
        val = self%LABEL_ROOT
    end function LL_get_Label_Root

    function LL_get_batch_size(self) result(val)
        class(Label_Loader), intent(in) :: self
        integer :: val
        val = self%batch_size
    end function LL_get_batch_size

    function LL_get_item_num(self) result(val)
        class(Label_Loader), intent(in) :: self
        integer :: val
        val = self%item_num
    end function LL_get_item_num

end module LoadLabel_mod