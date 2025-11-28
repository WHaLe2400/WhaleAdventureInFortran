module ReluFunc_mod
    implicit none
    private

    ! 公开泛型接口
    public :: relu_forward, relu_backward

    ! 为 relu_forward 定义泛型接口
    interface relu_forward
        module procedure relu_forward_4d
        module procedure relu_forward_2d
    end interface relu_forward

    ! 为 relu_backward 定义泛型接口
    interface relu_backward
        module procedure relu_backward_4d
        module procedure relu_backward_2d
    end interface relu_backward

    ! 假定 dp 在别处定义，例如: integer, parameter :: dp = kind(1.0d0)
    integer, parameter :: dp = kind(1.0d0)

contains

    ! 4D (BCWH) 版本的前向传播
    function relu_forward_4d(x) result(out)
        real(dp), intent(in) :: x(:,:,:,:)
        real(dp), allocatable :: out(:,:,:,:)

        allocate(out, source=x) ! 使用 source=x 更简洁地分配和初始化

        where (x <= 0.0_dp)
            out = 0.0_dp
        end where
    end function relu_forward_4d

    ! 2D (BL) 版本的前向传播
    function relu_forward_2d(x) result(out)
        real(dp), intent(in) :: x(:,:)
        real(dp), allocatable :: out(:,:)

        allocate(out, source=x)

        where (x <= 0.0_dp)
            out = 0.0_dp
        end where
    end function relu_forward_2d

    ! 4D (BCWH) 版本的反向传播
    function relu_backward_4d(dout, x) result(dx)
        real(dp), intent(in) :: dout(:,:,:,:), x(:,:,:,:)
        real(dp), allocatable :: dx(:,:,:,:)

        allocate(dx, shape=shape(x))

        where (x > 0.0_dp)
            dx = dout
        elsewhere
            dx = 0.0_dp
        end where
    end function relu_backward_4d

    ! 2D (BL) 版本的反向传播
    function relu_backward_2d(dout, x) result(dx)
        real(dp), intent(in) :: dout(:,:), x(:,:)
        real(dp), allocatable :: dx(:,:)

        allocate(dx, shape=shape(x))

        where (x > 0.0_dp)
            dx = dout
        elsewhere
            dx = 0.0_dp
        end where
    end function relu_backward_2d

end module ReluFunc_mod