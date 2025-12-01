module PReluFunc_mod
    use iso_fortran_env, only: dp => real64
    implicit none
    private

    public :: PReluLayer

    type, public :: PReluLayer
        private
        real(dp), allocatable :: a(:)
        real(dp), allocatable :: grad_a(:)
        ! 用于反向传播的缓存
        real(dp), allocatable :: x_cache_4d(:,:,:,:)
        real(dp), allocatable :: x_cache_2d(:,:)
        integer :: input_channels = 0
    contains
        procedure, public :: init => prelu_init
        procedure, public :: destroy => prelu_destroy
        procedure, public :: update => prelu_update
        procedure, public :: save => prelu_save
        procedure, public :: load => prelu_load 
        procedure, public :: zero_grads => prelu_zero_grads       
        ! 具体过程是私有的
        procedure, private :: prelu_forward_4d
        procedure, private :: prelu_forward_2d
        procedure, private :: prelu_backward_4d
        procedure, private :: prelu_backward_2d

        ! 通用接口是公共的
        generic, public :: forward => prelu_forward_4d, prelu_forward_2d
        generic, public :: backward => prelu_backward_4d, prelu_backward_2d
    end type PReluLayer

contains

    subroutine prelu_init(self, num_channels)
        class(PReluLayer), intent(inout) :: self
        integer, intent(in) :: num_channels
        
        self%input_channels = num_channels
        if (allocated(self%a)) deallocate(self%a)
        if (allocated(self%grad_a)) deallocate(self%grad_a)
        
        allocate(self%a(num_channels))
        allocate(self%grad_a(num_channels))
        
        ! 用一个小的正值初始化 'a'
        self%a = 0.01_dp
        self%grad_a = 0.0_dp
    end subroutine prelu_init

    subroutine prelu_save(self, filename)
        class(PReluLayer), intent(in) :: self
        character(len=*), intent(in) :: filename
        integer :: unit
        open(newunit=unit, file=filename, status='replace', action='write', form='formatted')
        write(unit, *) self%input_channels
        write(unit, *) self%a
        close(unit)
    end subroutine prelu_save

    subroutine prelu_load(self, filename)
        class(PReluLayer), intent(inout) :: self
        character(len=*), intent(in) :: filename
        integer :: unit, channels_read
        open(newunit=unit, file=filename, status='old', action='read', form='formatted')
        read(unit, *) channels_read
        if (channels_read /= self%input_channels) then
            print *, "PReLU Load Error: Channels mismatch. Expected ", self%input_channels, ", got ", channels_read
            close(unit)
            return
        end if
        read(unit, *) self%a
        close(unit)
    end subroutine prelu_load

    subroutine prelu_destroy(self)
        class(PReluLayer), intent(inout) :: self
        if (allocated(self%a)) deallocate(self%a)
        if (allocated(self%grad_a)) deallocate(self%grad_a)
        if (allocated(self%x_cache_4d)) deallocate(self%x_cache_4d)
        if (allocated(self%x_cache_2d)) deallocate(self%x_cache_2d)
        self%input_channels = 0
    end subroutine prelu_destroy

    subroutine prelu_update(self, learning_rate)
        class(PReluLayer), intent(inout) :: self
        real(dp), intent(in) :: learning_rate
        if (.not. allocated(self%a)) return
        self%a = self%a - learning_rate * self%grad_a
    end subroutine prelu_update

    subroutine prelu_zero_grads(self)
        class(PReluLayer), intent(inout) :: self
        if (allocated(self%grad_a)) then
            self%grad_a = 0.0_dp
        end if
    end subroutine prelu_zero_grads

    ! --- 4D (N, C, H, W) 版本 ---
    function prelu_forward_4d(self, x) result(out)
        class(PReluLayer), intent(inout) :: self
        real(dp), intent(in) :: x(:,:,:,:)
        real(dp), allocatable :: out(:,:,:,:)
        integer :: c_idx
        
        ! (N, C, H, W) layout: channel is the 2nd dimension
        if (size(x, 2) /= self%input_channels) then
            print *, "PReLU Error: Input channels mismatch in forward_4d. Expected ", &
                     self%input_channels, ", got ", size(x, 2)
            allocate(out(0,0,0,0))
            return
        end if

        ! 缓存输入以用于反向传播
        if (allocated(self%x_cache_4d)) deallocate(self%x_cache_4d)
        self%x_cache_4d = x
        
        allocate(out, source=x)
        do c_idx = 1, self%input_channels
            where (x(:,c_idx,:,:) <= 0.0_dp)
                out(:,c_idx,:,:) = self%a(c_idx) * x(:,c_idx,:,:)
            end where
        end do
    end function prelu_forward_4d

    function prelu_backward_4d(self, dout) result(dx)
        class(PReluLayer), intent(inout) :: self
        real(dp), intent(in) :: dout(:,:,:,:)
        real(dp), allocatable :: dx(:,:,:,:)
        integer :: c_idx
        real(dp), allocatable :: da_sum(:)

        if (.not. allocated(self%x_cache_4d)) then
            print *, "PReLU Error: Must call forward before backward."
            allocate(dx(0,0,0,0))
            return
        end if

        allocate(dx, source=dout)
        allocate(da_sum(self%input_channels))
        da_sum = 0.0_dp

        do c_idx = 1, self%input_channels
            ! 计算 dx for (N, C, H, W) layout
            where (self%x_cache_4d(:,c_idx,:,:) > 0.0_dp)
                dx(:,c_idx,:,:) = dout(:,c_idx,:,:)
            elsewhere
                dx(:,c_idx,:,:) = self%a(c_idx) * dout(:,c_idx,:,:)
            end where
            ! 计算负数部分的 'a' 的梯度
            da_sum(c_idx) = sum(self%x_cache_4d(:,c_idx,:,:) * dout(:,c_idx,:,:), &
                                mask=self%x_cache_4d(:,c_idx,:,:) <= 0.0_dp)
        end do
        self%grad_a = da_sum
    end function prelu_backward_4d

    ! --- 2D (B*L) 版本 ---
    function prelu_forward_2d(self, x) result(out)
        class(PReluLayer), intent(inout) :: self
        real(dp), intent(in) :: x(:,:)
        real(dp), allocatable :: out(:,:)
        integer :: c_idx

        if (size(x, 2) /= self%input_channels) then
            print *, "PReLU Error: Input features mismatch in forward_2d. Expected ", &
                     self%input_channels, ", got ", size(x, 2)
            allocate(out(0,0))
            return
        end if

        if (allocated(self%x_cache_2d)) deallocate(self%x_cache_2d)
        self%x_cache_2d = x
        
        allocate(out, source=x)
        do c_idx = 1, self%input_channels
            where (x(:,c_idx) <= 0.0_dp)
                out(:,c_idx) = self%a(c_idx) * x(:,c_idx)
            end where
        end do
    end function prelu_forward_2d

    function prelu_backward_2d(self, dout) result(dx)
        class(PReluLayer), intent(inout) :: self
        real(dp), intent(in) :: dout(:,:)
        real(dp), allocatable :: dx(:,:)
        integer :: c_idx
        real(dp), allocatable :: da_sum(:)

        if (.not. allocated(self%x_cache_2d)) then
            print *, "PReLU Error: Must call forward before backward."
            allocate(dx(0,0))
            return
        end if

        allocate(dx, source=dout)
        allocate(da_sum(self%input_channels))
        da_sum = 0.0_dp

        do c_idx = 1, self%input_channels
            ! 计算 dx
            where (self%x_cache_2d(:,c_idx) > 0.0_dp)
                dx(:,c_idx) = dout(:,c_idx)
            elsewhere
                dx(:,c_idx) = self%a(c_idx) * dout(:,c_idx)
            end where
            ! 计算负数部分的 'a' 的梯度
            da_sum(c_idx) = sum(self%x_cache_2d(:,c_idx) * dout(:,c_idx), &
                                mask=self%x_cache_2d(:,c_idx) <= 0.0_dp)
        end do
        self%grad_a = da_sum
    end function prelu_backward_2d

end module PReluFunc_mod
