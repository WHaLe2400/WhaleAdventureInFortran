Module LoadData_mod
    use iso_fortran_env, only: dp => real64
    implicit none
    private
    public :: Data_Loader

    type, public :: Data_Loader
    private
        character(len=200) :: DATA_ROOT
        integer :: batch_size=32, item_num = 0
        integer :: data_h = 28, data_w = 28, data_c = 1
        integer :: file_unit = -1 ! 用于存储文件单元号，-1表示未打开
        
    contains
        procedure, public :: init => DL_init
        procedure, public :: get_len => DL_get_len
        procedure, public :: get_batch => DL_get_batch
        procedure, public :: destroy => DL_destroy
        ! ---get funcs---
        procedure, public :: get_Data_Root => DL_get_Data_Root
        procedure, public :: get_batch_size => DL_get_batch_size
        procedure, public :: get_item_num => DL_get_item_num
        
    end type Data_Loader
contains

    subroutine DL_init(self, data_root, batch_size, item_num, data_h, data_w, data_c)
        class(Data_Loader), intent(inout) :: self
        character(len=*), intent(in) :: data_root
        integer, intent(in) :: batch_size, item_num
        integer, intent(in) :: data_h, data_w, data_c
        integer :: iostat
        character(len=:), allocatable :: filename

        self%DATA_ROOT = data_root
        self%batch_size = batch_size
        self%item_num = item_num
        self%data_h = data_h
        self%data_w = data_w
        self%data_c = data_c

        filename = trim(self%DATA_ROOT)

        ! 使用 newunit 获取一个可用的文件单元号并存储在对象中
        open(newunit=self%file_unit, file=filename, form='unformatted', access='stream', status='old', iostat=iostat)
        ! print *, "Opened file: ", trim(filename), " with unit: ", self%file_unit
        if (iostat /= 0) then
            print *, "Error opening file: ", trim(filename), ", iostat=", iostat
            self%file_unit = -1 ! 确保出错时 file_unit 是无效值
            stop
        end if
    end subroutine DL_init

    function DL_get_len(self) result(len)
        class(Data_Loader), intent(in) :: self
        integer :: len
        len = self%item_num / self%batch_size
    end function DL_get_len

    subroutine DL_get_batch(self, batch_idx, data)
        class(Data_Loader), intent(inout) :: self
        integer, intent(in) :: batch_idx
        real(dp), allocatable, intent(out) :: data(:,:,:,:) ! Fortran-friendly: (H, W, C, N)
        integer :: start_pos
        integer :: iostat
        ! 仍然按 (W, H, C, N) 读取以匹配文件布局
        integer(kind=1), allocatable :: temp_data(:,:,:,:) 
        integer :: n, c, h, w
        integer :: pixel_val

        if (self%file_unit == -1) then
            print *, "Error: File not open. Call init first."
            stop
        end if

        start_pos =(batch_idx - 1) * self%batch_size * self%data_h * self%data_w * self%data_c + 1

        ! 分配为 Fortran 友好的 (H, W, C, N) 格式
        allocate(data(self%data_h, self%data_w, self%data_c, self%batch_size))
        
        ! 临时数组仍然匹配文件中的物理布局 (W, H, C, N)
        allocate(temp_data(self%data_w, self%data_h, self%data_c, self%batch_size))

        read(unit=self%file_unit, pos=start_pos, iostat=iostat) temp_data
        
        if (iostat /= 0) then
            print *, "Error reading from file, batch_idx=", batch_idx, ", iostat=", iostat
            deallocate(temp_data)
            stop
        end if

        ! 从 (W, H, C, N) 转置为 (H, W, C, N) 并归一化
        do n = 1, self%batch_size
            do c = 1, self%data_c
                do h = 1, self%data_h
                    do w = 1, self%data_w
                        pixel_val = int(temp_data(w, h, c, n))
                        if (pixel_val < 0) pixel_val = pixel_val + 256
                        
                        data(h, w, c, n) = (real(pixel_val, kind=dp) / 255.0_dp) - 0.5_dp
                    end do
                end do
            end do
        end do

        deallocate(temp_data)

    end subroutine DL_get_batch

    subroutine DL_destroy(self)
        class(Data_Loader), intent(inout) :: self
        
        ! 如果文件单元有效（即文件已打开），则关闭它
        if (self%file_unit >= 0) then
            close(self%file_unit)
            self%file_unit = -1 ! 重置为无效值
        end if

        ! 清理其他成员
        self%item_num = 0
        self%batch_size = 0
    end subroutine DL_destroy

    ! --- Getter Functions ---
    function DL_get_Data_Root(self) result(val)
        class(Data_Loader), intent(in) :: self
        character(len=200) :: val
        val = self%DATA_ROOT
    end function DL_get_Data_Root

    function DL_get_batch_size(self) result(val)
        class(Data_Loader), intent(in) :: self
        integer :: val
        val = self%batch_size
    end function DL_get_batch_size

    function DL_get_item_num(self) result(val)
        class(Data_Loader), intent(in) :: self
        integer :: val
        val = self%item_num
    end function DL_get_item_num

end Module LoadData_mod