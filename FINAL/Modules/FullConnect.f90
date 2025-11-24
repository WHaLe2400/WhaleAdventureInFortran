module FullConnect_mod
    use iso_fortran_env, only: dp => real64
    implicit none
    private

    ! Public interface
    public :: FullConnectLayer

    ! Type definition for the fully connected layer
    type, public :: FullConnectLayer
        private
        integer :: input_size = 0
        integer :: output_size = 0
        real(dp), allocatable :: weights(:, :)
        real(dp), allocatable :: biases(:)
        ! For backpropagation with batches
        real(dp), allocatable :: input_cache(:, :)
        real(dp), allocatable :: grad_weights(:, :)
        real(dp), allocatable :: grad_biases(:)
    contains
        procedure, public :: init => fc_init
        procedure, public :: forward => fc_forward
        procedure, public :: backward => fc_backward
        procedure, public :: update => fc_update
        procedure, public :: destroy => fc_destroy
        ! Getter functions
        procedure, public :: get_input_size => fc_get_input_size
        procedure, public :: get_output_size => fc_get_output_size
        procedure, public :: get_weights => fc_get_weights
        procedure, public :: get_biases => fc_get_biases
        procedure, public :: get_grad_weights => fc_get_grad_weights
        procedure, public :: get_grad_biases => fc_get_grad_biases
    end type FullConnectLayer

contains

    ! Subroutine to initialize the fully connected layer
    subroutine fc_init(self, input_dim, output_dim)
        class(FullConnectLayer), intent(inout) :: self
        integer, intent(in) :: input_dim
        integer, intent(in) :: output_dim

        self%input_size = input_dim
        self%output_size = output_dim

        ! Deallocate if already allocated
        call self%destroy()

        ! Allocate weights and biases
        allocate(self%weights(self%output_size, self%input_size))
        allocate(self%biases(self%output_size))

        ! Allocate storage for gradients
        allocate(self%grad_weights(self%output_size, self%input_size))
        allocate(self%grad_biases(self%output_size))
        ! input_cache is allocated dynamically in forward pass

        ! Initialize with random numbers and zero gradients
        call random_number(self%weights)
        call random_number(self%biases)
        self%grad_weights = 0.0_dp
        self%grad_biases = 0.0_dp
    end subroutine fc_init

    ! Function to perform the forward pass for a batch
    function fc_forward(self, input_batch) result(output_batch)
        class(FullConnectLayer), intent(inout) :: self
        real(dp), intent(in) :: input_batch(:, :) ! Shape: (batch_size, input_size)
        real(dp), allocatable :: output_batch(:, :)   ! Shape: (batch_size, output_size)

        integer :: batch_size

        ! Check for dimension mismatch
        if (size(input_batch, 2) /= self%input_size) then
            print *, "Error: Input batch feature size (", size(input_batch, 2), &
                     ") does not match layer input size (", self%input_size, ")."
            allocate(output_batch(0, 0))
            return
        end if

        batch_size = size(input_batch, 1)

        ! Cache the input for backpropagation
        if (allocated(self%input_cache)) deallocate(self%input_cache)
        allocate(self%input_cache(batch_size, self%input_size))
        self%input_cache = input_batch

        ! Allocate output
        allocate(output_batch(batch_size, self%output_size))

        ! Perform the matrix-matrix multiplication and add biases
        ! output = input * weights' + biases
        output_batch = matmul(input_batch, transpose(self%weights))
        output_batch = output_batch + spread(self%biases, dim=1, ncopies=batch_size)
    end function fc_forward

    ! Function to perform the backward pass for a batch
    function fc_backward(self, grad_output_batch) result(grad_input_batch)
        class(FullConnectLayer), intent(inout) :: self
        real(dp), intent(in) :: grad_output_batch(:, :) ! Grad of loss w.r.t. layer's output. Shape: (batch_size, output_size)
        real(dp), allocatable :: grad_input_batch(:, :)   ! Grad of loss w.r.t. layer's input. Shape: (batch_size, input_size)

        integer :: batch_size

        ! Check for dimension mismatch
        if (size(grad_output_batch, 2) /= self%output_size) then
            print *, "Error: Gradient output batch feature size (", size(grad_output_batch, 2), &
                     ") does not match layer output size (", self%output_size, ")."
            allocate(grad_input_batch(0, 0))
            return
        end if
        
        batch_size = size(grad_output_batch, 1)
        if (size(self%input_cache, 1) /= batch_size) then
            print *, "Error: Gradient output batch size (", batch_size, &
                     ") does not match cached input batch size (", size(self%input_cache, 1), ")."
            allocate(grad_input_batch(0, 0))
            return
        end if

        ! 1. Calculate gradient w.r.t. weights (dLoss/dW)
        ! dLoss/dW = dLoss/dOut^T * In = In^T * dLoss/dOut (transposed)
        ! grad_output_batch: (batch, out_size), input_cache: (batch, in_size)
        ! grad_weights: (out_size, in_size)
        self%grad_weights = matmul(transpose(grad_output_batch), self%input_cache)

        ! 2. Calculate gradient w.r.t. biases (dLoss/dB)
        ! Sum gradients across the batch dimension
        self%grad_biases = sum(grad_output_batch, dim=1)

        ! 3. Calculate gradient w.r.t. input (dLoss/dIn)
        ! dLoss/dIn = dLoss/dOut * W
        allocate(grad_input_batch(batch_size, self%input_size))
        grad_input_batch = matmul(grad_output_batch, self%weights)

    end function fc_backward

    ! Subroutine to update weights and biases
    subroutine fc_update(self, learning_rate)
        class(FullConnectLayer), intent(inout) :: self
        real(dp), intent(in) :: learning_rate

        ! Update weights and biases using gradient descent
        self%weights = self%weights - learning_rate * self%grad_weights
        self%biases = self%biases - learning_rate * self%grad_biases
    end subroutine fc_update

    ! Subroutine to deallocate memory
    subroutine fc_destroy(self)
        class(FullConnectLayer), intent(inout) :: self
        if (allocated(self%weights)) deallocate(self%weights)
        if (allocated(self%biases)) deallocate(self%biases)
        if (allocated(self%input_cache)) deallocate(self%input_cache)
        if (allocated(self%grad_weights)) deallocate(self%grad_weights)
        if (allocated(self%grad_biases)) deallocate(self%grad_biases)
        self%input_size = 0
        self%output_size = 0
    end subroutine fc_destroy

    ! --- Getter Functions ---

    function fc_get_input_size(self) result(val)
        class(FullConnectLayer), intent(in) :: self
        integer :: val
        val = self%input_size
    end function fc_get_input_size

    function fc_get_output_size(self) result(val)
        class(FullConnectLayer), intent(in) :: self
        integer :: val
        val = self%output_size
    end function fc_get_output_size

    function fc_get_weights(self) result(arr)
        class(FullConnectLayer), intent(in) :: self
        real(dp), allocatable :: arr(:,:)
        if (allocated(self%weights)) then
            arr = self%weights
        end if
    end function fc_get_weights

    function fc_get_biases(self) result(arr)
        class(FullConnectLayer), intent(in) :: self
        real(dp), allocatable :: arr(:)
        if (allocated(self%biases)) then
            arr = self%biases
        end if
    end function fc_get_biases

    function fc_get_grad_weights(self) result(arr)
        class(FullConnectLayer), intent(in) :: self
        real(dp), allocatable :: arr(:,:)
        if (allocated(self%grad_weights)) then
            arr = self%grad_weights
        end if
    end function fc_get_grad_weights

    function fc_get_grad_biases(self) result(arr)
        class(FullConnectLayer), intent(in) :: self
        real(dp), allocatable :: arr(:)
        if (allocated(self%grad_biases)) then
            arr = self%grad_biases
        end if
    end function fc_get_grad_biases

end module FullConnect_mod
