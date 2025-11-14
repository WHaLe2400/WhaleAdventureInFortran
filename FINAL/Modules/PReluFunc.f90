module PReluFunc_mod
    implicit none
    private

    ! 定义泛型接口，使其可以根据输入参数的秩（维度）选择正确的函数
    public :: prelu_forward, prelu_backward
    interface prelu_forward
        module procedure prelu_forward_4d
        module procedure prelu_forward_2d
    end interface prelu_forward

    interface prelu_backward
        module procedure prelu_backward_4d
        module procedure prelu_backward_2d
    end interface prelu_backward

    ! 假定 dp 在别处定义，例如: integer, parameter :: dp = kind(1.0d0)
    integer, parameter :: dp = kind(1.0d0)

contains

    ! -------------------------------------------------------------------------
    ! 4D (BCHW) 版本
    ! -------------------------------------------------------------------------
    function prelu_forward_4d(x, a) result(out)
        !> PReLU forward pass for 4D (BCHW) tensor.
        implicit none
        real(dp), intent(in) :: x(:,:,:,:), a(:)
        real(dp), allocatable :: out(:,:,:,:)
        integer :: N, C, H, W

        N = size(x, 1)
        C = size(x, 2)
        H = size(x, 3)
        W = size(x, 4)

        allocate(out(N, C, H, W))

        do C = 1, size(x, 2) ! 使用大写 C 以保持一致性
            where (x(:,C,:,:) > 0.0_dp)
                out(:,C,:,:) = x(:,C,:,:)
            elsewhere
                out(:,C,:,:) = a(C) * x(:,C,:,:)
            end where
        end do

    end function prelu_forward_4d

    function prelu_backward_4d(dout, x, a) result(dx)
        !> PReLU backward pass for 4D (BCHW) tensor.
        implicit none
        real(dp), intent(in) :: dout(:,:,:,:), x(:,:,:,:), a(:)
        real(dp), allocatable :: dx(:,:,:,:)
        integer :: N, C, H, W

        N = size(x, 1)
        C = size(x, 2)
        H = size(x, 3)
        W = size(x, 4)

        allocate(dx(N, C, H, W))

        do C = 1, size(x, 2) ! 使用大写 C 以保持一致性
            where (x(:,C,:,:) > 0.0_dp)
                dx(:,C,:,:) = dout(:,C,:,:)
            elsewhere
                dx(:,C,:,:) = a(C) * dout(:,C,:,:)
            end where
        end do

    end function prelu_backward_4d

    ! -------------------------------------------------------------------------
    ! 2D (BL) 版本
    ! -------------------------------------------------------------------------
    function prelu_forward_2d(x, a) result(out)
        !> PReLU forward pass for 2D (BL) tensor.
        implicit none
        real(dp), intent(in) :: x(:,:), a(:)
        real(dp), allocatable :: out(:,:)
        integer :: N, C

        N = size(x, 1)
        C = size(x, 2)

        allocate(out(N, C))

        do C = 1, size(x, 2) ! 使用大写 C 以保持一致性
            where (x(:,C) > 0.0_dp)
                out(:,C) = x(:,C)
            elsewhere
                out(:,C) = a(C) * x(:,C)
            end where
        end do

    end function prelu_forward_2d

    function prelu_backward_2d(dout, x, a) result(dx)
        !> PReLU backward pass for 2D (BL) tensor.
        implicit none
        real(dp), intent(in) :: dout(:,:), x(:,:), a(:)
        real(dp), allocatable :: dx(:,:)
        integer :: N, C

        N = size(x, 1)
        C = size(x, 2)

        allocate(dx(N, C))

        do C = 1, size(x, 2) ! 使用大写 C 以保持一致性
            where (x(:,C) > 0.0_dp)
                dx(:,C) = dout(:,C)
            elsewhere
                dx(:,C) = a(C) * dout(:,C)
            end where
        end do

    end function prelu_backward_2d

end module PReluFunc_mod