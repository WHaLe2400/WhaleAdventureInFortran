module ModelCombine_mod
    use iso_fortran_env, only: dp => real64
    use Conv_mod, only: ConvLayer
    use Flaten_mod, only: FlatenLayer
    use FullConnect_mod, only: FullConnectLayer
    use PReluFunc_mod, only: PReluLayer
    implicit none

    private
    public :: Model

    type, public :: Model
        ! --- Public Parameters ---
        integer :: BatchSize = 1
        integer, dimension(3) :: H = [28, 14, 7], W = [28, 14, 7], C = [1, 8, 16]
        integer :: FC_in = 784, FC_hidden = 128, FC_out = 10
        integer :: Conv1_kernel = 5, Conv1_stride = 2, Conv1_padding = 2
        integer :: Conv2_kernel = 5, Conv2_stride = 2, Conv2_padding = 2
        ! --- Private Layer Components ---
        type(ConvLayer) :: Conv1, Conv2
        type(PReluLayer) :: PReLU1, PReLU2, PReLU3
        type(FlatenLayer) :: Flaten
        type(FullConnectLayer) :: FC1, FC2
    contains
        procedure, public :: init => model_init
        procedure, public :: forward => model_forward
        procedure, public :: backward => model_backward
        procedure, public :: update => model_update
        procedure, public :: destroy => model_destroy
    end type Model

contains

    subroutine model_init(self)
        class(Model), intent(inout) :: self
        
        call self%Conv1%init(self%C(1), self%C(2), self%Conv1_kernel, self%Conv1_stride, self%Conv1_padding)

        call self%PReLU1%init(self%C(2)) ! Channels for PReLU1 is output of Conv1

        call self%Conv2%init(self%C(2), self%C(3), self%Conv2_kernel, self%Conv2_stride, self%Conv2_padding)

        call self%PReLU2%init(self%C(3)) ! Channels for PReLU2 is output of Conv2

        call self%Flaten%init(self%C(3), self%H(3), self%W(3))

        call self%FC1%init(self%FC_in, self%FC_hidden)

        call self%PReLU3%init(self%FC_hidden) ! Features for PReLU3 is output of FC1
        
        call self%FC2%init(self%FC_hidden, self%FC_out)

    end subroutine model_init

    function model_forward(self, input_data) result(output_data)
        class(Model), intent(inout) :: self
        real(dp), intent(in) :: input_data(:, :, :, :)
        real(dp), allocatable :: output_data(:, :)

        ! Intermediate results
        real(dp), allocatable :: conv1_out(:, :, :, :), prelu1_out(:, :, :, :)
        real(dp), allocatable :: conv2_out(:, :, :, :), prelu2_out(:, :, :, :)
        real(dp), allocatable :: flaten_out(:, :), fc1_out(:, :), prelu3_out(:, :)

        ! Forward pass
        write (*,*) "DATA_shape:", shape(input_data)

        conv1_out = self%Conv1%forward(input_data)
        write (*,*) "DATA_shape_after_conv1:", shape(conv1_out)

        prelu1_out = self%PReLU1%forward(conv1_out)
        write (*,*) "DATA_shape_after_PReLU1:", shape(prelu1_out)
        conv2_out = self%Conv2%forward(prelu1_out)
        write (*,*) "DATA_shape_after_conv2:", shape(conv2_out)
        prelu2_out = self%PReLU2%forward(conv2_out)
        write (*,*) "DATA_shape_after_PReLU2:", shape(prelu2_out)
        flaten_out = self%Flaten%forward(prelu2_out)
        write (*,*) "DATA_shape_after_Flaten:", shape(flaten_out)
        fc1_out = self%FC1%forward(flaten_out)
        write (*,*) "DATA_shape_after_FC1:", shape(fc1_out)
        prelu3_out = self%PReLU3%forward(fc1_out)
        write (*,*) "DATA_shape_after_PReLU3:", shape(prelu3_out)
        output_data = self%FC2%forward(prelu3_out)
        write (*,*) "DATA_shape_after_FC2:", shape(output_data)

        ! Deallocate intermediate arrays
    end function model_forward

    subroutine model_backward(self, grad_output)
        class(Model), intent(inout) :: self
        real(dp), intent(in) :: grad_output(:, :)

        ! Intermediate gradients
        real(dp), allocatable :: grad_prelu3(:,:), grad_fc1(:, :), grad_flaten(:, :)
        real(dp), allocatable :: grad_prelu2(:, :, :, :), grad_conv2(:, :, :, :)
        real(dp), allocatable :: grad_prelu1(:, :, :, :), grad_conv1(:, :, :, :)
        real(dp), allocatable :: dummy(:,:,:,:)

        ! Backward pass
        grad_prelu3 = self%FC2%backward(grad_output)
        grad_fc1 = self%PReLU3%backward(grad_prelu3)
        grad_flaten = self%FC1%backward(grad_fc1)
        grad_prelu2 = self%Flaten%backward(grad_flaten)
        grad_conv2 = self%PReLU2%backward(grad_prelu2)
        grad_prelu1 = self%Conv2%backward(grad_conv2)
        grad_conv1 = self%PReLU1%backward(grad_prelu1)
        dummy = self%Conv1%backward(grad_conv1)

        ! Deallocate intermediate arrays
    end subroutine model_backward

    subroutine model_update(self, learning_rate)
        class(Model), intent(inout) :: self
        real(dp), intent(in) :: learning_rate

        call self%FC2%update(learning_rate)
        call self%PReLU3%update(learning_rate)
        call self%FC1%update(learning_rate)
        call self%PReLU2%update(learning_rate)
        call self%Conv2%update(learning_rate)
        call self%PReLU1%update(learning_rate)
        call self%Conv1%update(learning_rate)
    end subroutine model_update

    subroutine model_destroy(self)
        class(Model), intent(inout) :: self
        call self%Conv1%destroy()
        call self%PReLU1%destroy()
        call self%Conv2%destroy()
        call self%PReLU2%destroy()
        call self%Flaten%destroy()
        call self%FC1%destroy()
        call self%PReLU3%destroy()
        call self%FC2%destroy()
    end subroutine model_destroy

end module ModelCombine_mod