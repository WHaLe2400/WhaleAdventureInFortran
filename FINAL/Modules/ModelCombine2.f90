module ModelCombine2_mod
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
        integer, dimension(4) :: H = [28, 14, 7, 4], W = [28, 14, 7, 4], C = [1, 8, 16, 64]
        integer :: FC_in = 4*4*64, FC_hidden = 128, FC_out = 10
        integer :: Conv1_kernel = 5, Conv1_stride = 2, Conv1_padding = 2
        integer :: Conv2_kernel = 5, Conv2_stride = 2, Conv2_padding = 2
        ! --- Private Layer Components ---
        type(ConvLayer) :: Conv1, Conv2, Conv3
        type(PReluLayer) :: PReLU1, PReLU2, PReLU3, PReLU4
        type(FlatenLayer) :: Flaten
        type(FullConnectLayer) :: FC1, FC2

        ! --- Intermediate results for backpropagation ---
        real(dp), allocatable :: conv1_out(:, :, :, :), prelu1_out(:, :, :, :)
        real(dp), allocatable :: conv2_out(:, :, :, :), prelu2_out(:, :, :, :)
        real(dp), allocatable :: conv3_out(:, :, :, :), prelu3_out(:, :, :, :)
        real(dp), allocatable :: flaten_out(:, :), fc1_out(:, :), prelu4_out(:, :)

    contains
        procedure, public :: init => model_init
        procedure, public :: load => model_load
        procedure, public :: save => model_save
        procedure, public :: forward => model_forward
        procedure, public :: backward => model_backward
        procedure, public :: update => model_update
        procedure, public :: destroy => model_destroy
        procedure, public :: zero_grads => model_zero_grads
    end type Model

contains

    subroutine model_init(self)
        class(Model), intent(inout) :: self
        
        call self%Conv1%init(self%C(1), self%C(2), self%Conv1_kernel, self%Conv1_stride, self%Conv1_padding)

        call self%PReLU1%init(self%C(2)) ! Channels for PReLU1 is output of Conv1

        call self%Conv2%init(self%C(2), self%C(3), self%Conv2_kernel, self%Conv2_stride, self%Conv2_padding)

        call self%PReLU2%init(self%C(3)) ! Channels for PReLU2 is output of Conv2

        call self%Conv3%init(self%C(3), self%C(4), self%Conv2_kernel, self%Conv2_stride, self%Conv2_padding)

        call self%PReLU3%init(self%C(4)) ! Channels for PReLU3 is output of Conv3

        call self%Flaten%init(self%C(4), self%H(4), self%W(4))

        call self%FC1%init(self%FC_in, self%FC_hidden)

        call self%PReLU4%init(self%FC_hidden) ! Features for PReLU4 is output of FC1
        
        call self%FC2%init(self%FC_hidden, self%FC_out)

    end subroutine model_init


    subroutine model_load(self, base_path)
        class(Model), intent(inout) :: self
        character(len=*), intent(in) :: base_path
        logical :: exists
        ! 检查目录是否存在（修正语法）
        inquire(file=trim(base_path)//'/.', exist=exists)
        if (.not. exists) then
            print *, "Directory ", trim(base_path), " does not exist."
            return  ! Or handle error as needed
        end if

        print *, ""
        print *, "Loading model from ", trim(base_path)

        call self%Conv1%load(trim(base_path) // "/_Conv1.dat")
        print*, "Conv_1 loaded."
        call self%PReLU1%load(trim(base_path) // "/_PReLU1.dat")  ! 新增
        print*, "PReLU_1 loaded."
        call self%Conv2%load(trim(base_path) // "/_Conv2.dat")
        print*, "Conv_2 loaded."
        call self%PReLU2%load(trim(base_path) // "/_PReLU2.dat")  ! 新增
        print*, "PReLU_2 loaded."
        call self%Conv3%load(trim(base_path) // "/_Conv3.dat")
        print*, "Conv_3 loaded."
        call self%PReLU3%load(trim(base_path) // "/_PReLU3.dat")  ! 新增
        print*, "PReLU_3 loaded."
        call self%FC1%load(trim(base_path) // "/_FC1.dat")
        print*, "FC_1 loaded."
        call self%PReLU4%load(trim(base_path) // "/_PReLU4.dat")  ! 新增
        print*, "PReLU_4 loaded."
        call self%FC2%load(trim(base_path) // "/_FC2.dat")
        print*, "FC_2 loaded."

        print *, "Model loaded successfully."
    end subroutine model_load


    subroutine model_save(self, base_path)
        class(Model), intent(in) :: self
        character(len=*), intent(in) :: base_path
        logical :: exists
        ! 检查目录是否存在（修正语法）
        inquire(file=trim(base_path)//'/.', exist=exists)
        if (.not. exists) then
            call execute_command_line('mkdir -p ' // trim(base_path))
        end if

        print *, ""
        print *, "Saving model to ", trim(base_path)

        call self%Conv1%save(trim(base_path) // "/_Conv1.dat")
        call self%PReLU1%save(trim(base_path) // "/_PReLU1.dat")  ! 新增
        call self%Conv2%save(trim(base_path) // "/_Conv2.dat")
        call self%PReLU2%save(trim(base_path) // "/_PReLU2.dat")  ! 新增
        call self%Conv3%save(trim(base_path) // "/_Conv3.dat")
        call self%PReLU3%save(trim(base_path) // "/_PReLU3.dat")  ! 新增
        call self%FC1%save(trim(base_path) // "/_FC1.dat")
        call self%PReLU4%save(trim(base_path) // "/_PReLU4.dat")  ! 新增
        call self%FC2%save(trim(base_path) // "/_FC2.dat")
        print *, "Model saved successfully."
    end subroutine model_save


    function model_forward(self, input_data) result(output_data)
        class(Model), intent(inout) :: self
        real(dp), intent(in) :: input_data(:, :, :, :)
        real(dp), allocatable :: output_data(:, :)

        ! Forward pass, storing intermediate results in self
        self%conv1_out = self%Conv1%forward(input_data)
        self%prelu1_out = self%PReLU1%forward(self%conv1_out)

        self%conv2_out = self%Conv2%forward(self%prelu1_out)
        self%prelu2_out = self%PReLU2%forward(self%conv2_out)

        self%conv3_out = self%Conv3%forward(self%prelu2_out)
        self%prelu3_out = self%PReLU3%forward(self%conv3_out)

        ! 修正: Flaten 的输入应为 prelu3_out
        self%flaten_out = self%Flaten%forward(self%prelu3_out)

        self%fc1_out = self%FC1%forward(self%flaten_out)
        
        ! 修正: 应为 PReLU4，且输出到 prelu4_out
        self%prelu4_out = self%PReLU4%forward(self%fc1_out)

        ! 修正: FC2 的输入应为 prelu4_out
        output_data = self%FC2%forward(self%prelu4_out)

    end function model_forward

    subroutine model_backward(self, grad_output)
        class(Model), intent(inout) :: self
        real(dp), intent(in) :: grad_output(:, :)

        ! Intermediate gradients
        real(dp), allocatable :: grad_prelu4(:,:), grad_fc1(:, :), grad_flaten(:, :)
        real(dp), allocatable :: grad_prelu3(:, :, :, :), grad_conv3(:, :, :, :)
        real(dp), allocatable :: grad_prelu2(:, :, :, :), grad_conv2(:, :, :, :)
        real(dp), allocatable :: grad_prelu1(:, :, :, :), grad_conv1(:, :, :, :)
        real(dp), allocatable :: dummy(:,:,:,:)

        ! --- 修正: 严格按照前向传播的逆序进行反向传播 ---
        ! 10. FC2 backward
        grad_prelu4 = self%FC2%backward(grad_output)
        ! 9. PReLU4 backward
        grad_fc1 = self%PReLU4%backward(grad_prelu4)
        ! 8. FC1 backward
        grad_flaten = self%FC1%backward(grad_fc1)
        ! 7. Flaten backward
        grad_prelu3 = self%Flaten%backward(grad_flaten)
        ! 6. PReLU3 backward
        grad_conv3 = self%PReLU3%backward(grad_prelu3)
        ! 5. Conv3 backward
        grad_prelu2 = self%Conv3%backward(grad_conv3)
        ! 4. PReLU2 backward
        grad_conv2 = self%PReLU2%backward(grad_prelu2)
        ! 3. Conv2 backward
        grad_prelu1 = self%Conv2%backward(grad_conv2)
        ! 2. PReLU1 backward
        grad_conv1 = self%PReLU1%backward(grad_prelu1)
        ! 1. Conv1 backward (the final gradient is not used further)
        dummy = self%Conv1%backward(grad_conv1)

        ! Deallocate intermediate gradient arrays
        deallocate(grad_prelu4, grad_fc1, grad_flaten, grad_prelu3, grad_conv3, &
                   grad_prelu2, grad_conv2, grad_prelu1, grad_conv1, dummy)

    end subroutine model_backward

    subroutine model_zero_grads(self)
        class(Model), intent(inout) :: self
        call self%Conv1%zero_grads()
        call self%Conv2%zero_grads()
        call self%Conv3%zero_grads()
        call self%FC1%zero_grads()
        call self%FC2%zero_grads()
        ! PReLU 层的梯度在 update 时已清零，但保持一致性也无妨
        call self%PReLU1%zero_grads()
        call self%PReLU2%zero_grads()
        call self%PReLU3%zero_grads()
        call self%PReLU4%zero_grads()
    end subroutine model_zero_grads

    subroutine model_update(self, learning_rate)
        class(Model), intent(inout) :: self
        real(dp), intent(in) :: learning_rate

        call self%FC2%update(learning_rate)
        call self%PReLU4%update(learning_rate)
        call self%FC1%update(learning_rate)
        call self%PReLU3%update(learning_rate)
        call self%Conv3%update(learning_rate)
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
        call self%Conv3%destroy()
        call self%PReLU3%destroy()
        call self%Flaten%destroy()
        call self%FC1%destroy()
        call self%PReLU4%destroy()
        call self%FC2%destroy()

        ! Deallocate intermediate results
        if (allocated(self%conv1_out)) deallocate(self%conv1_out)
        if (allocated(self%prelu1_out)) deallocate(self%prelu1_out)
        if (allocated(self%conv2_out)) deallocate(self%conv2_out)
        if (allocated(self%prelu2_out)) deallocate(self%prelu2_out)
        if (allocated(self%conv3_out)) deallocate(self%conv3_out)
        if (allocated(self%prelu3_out)) deallocate(self%prelu3_out)
        if (allocated(self%flaten_out)) deallocate(self%flaten_out)
        if (allocated(self%fc1_out)) deallocate(self%fc1_out)
        if (allocated(self%prelu4_out)) deallocate(self%prelu4_out)
    end subroutine model_destroy

end module ModelCombine2_mod