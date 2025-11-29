module Flaten_mod

    use iso_fortran_env, only: dp => real64
    implicit none
    private
    public :: FlatenLayer

    type, public :: FlatenLayer
    private
        integer :: input_c = 0, input_h = 0, input_w = 0
    contains
        procedure, public :: init => flaten_init
        procedure, public :: forward => flaten_forward
        procedure, public :: backward => flaten_backward
        procedure, public :: destroy => flaten_destroy
    end type FlatenLayer

contains

    subroutine flaten_init(self, c, h, w)
        class(FlatenLayer), intent(inout) :: self
        integer, intent(in) :: c, h, w
        self%input_c = c
        self%input_h = h
        self%input_w = w
    end subroutine flaten_init

    function flaten_forward(self, input_data) result(output_data)
        class(FlatenLayer), intent(inout) :: self
        real(dp), intent(in) :: input_data(:,:,:,:)  ! Input layout: (N, C, H, W)
        real(dp), allocatable :: output_data(:,:)
        integer :: n, c, h, w, idx
        integer :: N_dim, C_dim, H_dim, W_dim

        N_dim = size(input_data, 1)
        C_dim = size(input_data, 2)
        H_dim = size(input_data, 3)
        W_dim = size(input_data, 4)

        allocate(output_data(N_dim, C_dim * H_dim * W_dim))

        do n = 1, N_dim
            idx = 1
            do c = 1, C_dim
                do h = 1, H_dim
                    do w = 1, W_dim
                        output_data(n, idx) = input_data(n, c, h, w)
                        idx = idx + 1
                    end do
                end do
            end do
        end do
    end function flaten_forward

    function flaten_backward(self, grad_output) result(grad_input)
        class(FlatenLayer), intent(in) :: self
        real(dp), intent(in) :: grad_output(:,:)
        real(dp), allocatable :: grad_input(:,:,:,:)
        integer :: n, c, h, w, idx
        integer :: N_dim

        N_dim = size(grad_output, 1)
        
        allocate(grad_input(N_dim, self%input_c, self%input_h, self%input_w))

        do n = 1, N_dim
            idx = 1
            do c = 1, self%input_c
                do h = 1, self%input_h
                    do w = 1, self%input_w
                        grad_input(n, c, h, w) = grad_output(n, idx)
                        idx = idx + 1
                    end do
                end do
            end do
        end do
    end function flaten_backward

    subroutine flaten_destroy(self)
        class(FlatenLayer), intent(inout) :: self
        self%input_c = 0
        self%input_h = 0
        self%input_w = 0
    end subroutine flaten_destroy

end module Flaten_mod