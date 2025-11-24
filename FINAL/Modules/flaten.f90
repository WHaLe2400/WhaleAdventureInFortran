module Flaten_mod

    use iso_fortran_env, only: dp => real64
    implicit none
    private
    public :: FlatenLayer

    type, public :: FlatenLayer
        private
        integer :: input_C = 0
        integer :: input_H = 0
        integer :: input_W = 0
        integer :: output_size = 0
    contains
        procedure, public :: init => flaten_init
        procedure, public :: forward => flaten_forward
        procedure, public :: backward => flaten_backward
        procedure, public :: destroy => flaten_destroy
        ! Getter functions
        procedure, public :: get_input_shape => flaten_get_input_shape
        procedure, public :: get_output_size => flaten_get_output_size
        
    end type FlatenLayer

contains
    subroutine flaten_init(self, C, H, W)
        class(FlatenLayer), intent(inout) :: self
        integer, intent(in) :: C, H, W

        self%input_C = C
        self%input_H = H
        self%input_W = W
        self%output_size = C * H * W
    end subroutine flaten_init

    function flaten_forward(self, x) result(y)
        class(FlatenLayer), intent(inout) :: self
        real(dp), intent(in) :: x(:,:,:,:) ! Shape: (N, C, H, W)
        real(dp), allocatable :: y(:,:)       ! Shape: (N, C*H*W)
        integer :: batch_size

        batch_size = size(x, 1)

        ! Reshape the entire batch in one operation
        y = reshape(x, [batch_size, self%output_size])
        
    end function flaten_forward

    function flaten_backward(self, dout) result(dx)
        class(FlatenLayer), intent(inout) :: self
        real(dp), intent(in) :: dout(:,:)      ! Shape: (N, C*H*W)
        real(dp), allocatable :: dx(:,:,:,:) ! Shape: (N, C, H, W)
        integer :: batch_size

        batch_size = size(dout, 1)

        ! Reshape the entire batch back to its original 4D shape
        dx = reshape(dout, [batch_size, self%input_C, self%input_H, self%input_W])

    end function flaten_backward

    subroutine flaten_destroy(self)
        class(FlatenLayer), intent(inout) :: self
        self%input_C = 0
        self%input_H = 0
        self%input_W = 0
        self%output_size = 0
    end subroutine flaten_destroy

    subroutine flaten_get_input_shape(self, C, H, W)
        class(FlatenLayer), intent(in) :: self
        integer, intent(out) :: C, H, W

        C = self%input_C
        H = self%input_H
        W = self%input_W
    end subroutine flaten_get_input_shape

    function flaten_get_output_size(self) result(size)
        class(FlatenLayer), intent(in) :: self
        integer :: size

        size = self%output_size
    end function flaten_get_output_size

end module Flaten_mod