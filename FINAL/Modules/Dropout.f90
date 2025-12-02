module Dropout_mod
    use iso_fortran_env, only: dp => real64
    implicit none
    private

    public :: DropoutLayer

    type :: DropoutLayer
        private
        real(dp) :: p = 0.5_dp        ! 丢弃概率 (0.0 - 1.0)
        logical :: is_training = .false. ! 模式标志
        
        ! 缓存掩码 (Mask)，用于反向传播
        ! 1.0 表示保留，0.0 表示丢弃
        real(dp), allocatable :: mask_2d(:,:)
        real(dp), allocatable :: mask_4d(:,:,:,:)
    contains
        procedure, public :: init => dropout_init
        procedure, public :: train => dropout_train
        procedure, public :: eval => dropout_eval
        procedure, public :: destroy => dropout_destroy
        
        ! 私有实现
        procedure, private :: dropout_forward_2d
        procedure, private :: dropout_forward_4d
        procedure, private :: dropout_backward_2d
        procedure, private :: dropout_backward_4d
        
        ! 公共泛型接口
        generic, public :: forward => dropout_forward_2d, dropout_forward_4d
        generic, public :: backward => dropout_backward_2d, dropout_backward_4d
    end type DropoutLayer

contains

    ! 初始化
    subroutine dropout_init(self, p)
        class(DropoutLayer), intent(inout) :: self
        real(dp), intent(in) :: p
        self%p = p
        self%is_training = .false. ! 默认为评估模式
    end subroutine dropout_init

    ! 切换到训练模式
    subroutine dropout_train(self)
        class(DropoutLayer), intent(inout) :: self
        self%is_training = .true.
    end subroutine dropout_train

    ! 切换到评估模式
    subroutine dropout_eval(self)
        class(DropoutLayer), intent(inout) :: self
        self%is_training = .false.
    end subroutine dropout_eval

    ! -------------------------------------------------------------------------
    ! 2D Forward (通常用于全连接层后: Batch, Features)
    ! -------------------------------------------------------------------------
    function dropout_forward_2d(self, x) result(y)
        class(DropoutLayer), intent(inout) :: self
        real(dp), intent(in) :: x(:,:)
        real(dp), allocatable :: y(:,:)
        real(dp) :: scale

        allocate(y, source=x)

        if (.not. self%is_training .or. self%p <= 0.0_dp) then
            ! 推理模式或 p=0：直接通过
            y = x
            return
        end if

        ! 重新分配 mask
        if (allocated(self%mask_2d)) deallocate(self%mask_2d)
        allocate(self%mask_2d, source=x)

        ! 生成随机数 [0, 1]
        call random_number(self%mask_2d)

        ! 生成掩码: > p 保留(1), <= p 丢弃(0)
        where (self%mask_2d > self%p)
            self%mask_2d = 1.0_dp
        elsewhere
            self%mask_2d = 0.0_dp
        end where

        ! Inverted Dropout: 缩放因子
        scale = 1.0_dp / (1.0_dp - self%p)

        ! 应用掩码和缩放
        y = x * self%mask_2d * scale

    end function dropout_forward_2d

    ! -------------------------------------------------------------------------
    ! 2D Backward
    ! -------------------------------------------------------------------------
    function dropout_backward_2d(self, grad_output) result(grad_input)
        class(DropoutLayer), intent(in) :: self
        real(dp), intent(in) :: grad_output(:,:)
        real(dp), allocatable :: grad_input(:,:)
        real(dp) :: scale

        allocate(grad_input, source=grad_output)

        if (.not. self%is_training .or. self%p <= 0.0_dp) then
            grad_input = grad_output
            return
        end if

        if (.not. allocated(self%mask_2d)) then
            print *, "Error: Dropout backward called without forward mask."
            stop
        end if

        scale = 1.0_dp / (1.0_dp - self%p)
        grad_input = grad_output * self%mask_2d * scale

    end function dropout_backward_2d

    ! -------------------------------------------------------------------------
    ! 4D Forward (通常用于卷积层后: N, C, H, W)
    ! -------------------------------------------------------------------------
    function dropout_forward_4d(self, x) result(y)
        class(DropoutLayer), intent(inout) :: self
        real(dp), intent(in) :: x(:,:,:,:)
        real(dp), allocatable :: y(:,:,:,:)
        real(dp) :: scale

        allocate(y, source=x)

        if (.not. self%is_training .or. self%p <= 0.0_dp) then
            y = x
            return
        end if

        if (allocated(self%mask_4d)) deallocate(self%mask_4d)
        allocate(self%mask_4d, source=x)

        call random_number(self%mask_4d)

        where (self%mask_4d > self%p)
            self%mask_4d = 1.0_dp
        elsewhere
            self%mask_4d = 0.0_dp
        end where

        scale = 1.0_dp / (1.0_dp - self%p)
        y = x * self%mask_4d * scale

    end function dropout_forward_4d

    ! -------------------------------------------------------------------------
    ! 4D Backward
    ! -------------------------------------------------------------------------
    function dropout_backward_4d(self, grad_output) result(grad_input)
        class(DropoutLayer), intent(in) :: self
        real(dp), intent(in) :: grad_output(:,:,:,:)
        real(dp), allocatable :: grad_input(:,:,:,:)
        real(dp) :: scale

        allocate(grad_input, source=grad_output)

        if (.not. self%is_training .or. self%p <= 0.0_dp) then
            grad_input = grad_output
            return
        end if

        if (.not. allocated(self%mask_4d)) then
            print *, "Error: Dropout backward called without forward mask."
            stop
        end if

        scale = 1.0_dp / (1.0_dp - self%p)
        grad_input = grad_output * self%mask_4d * scale

    end function dropout_backward_4d

    ! 销毁
    subroutine dropout_destroy(self)
        class(DropoutLayer), intent(inout) :: self
        if (allocated(self%mask_2d)) deallocate(self%mask_2d)
        if (allocated(self%mask_4d)) deallocate(self%mask_4d)
    end subroutine dropout_destroy

end module Dropout_mod
