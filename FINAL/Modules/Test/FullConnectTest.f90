program test_fullconnect
    use FullConnect_mod
    use iso_fortran_env, only: dp => real64
    implicit none

    ! Test parameters
    integer, parameter :: batch_size = 4
    integer, parameter :: input_dim = 3
    integer, parameter :: output_dim = 2
    real(dp), parameter :: learning_rate = 0.1

    ! Layer instance
    type(FullConnectLayer) :: fc_layer

    ! Test data (Batch versions)
    real(dp), allocatable :: input_batch(:, :)      ! Shape: (batch_size, input_dim)
    real(dp), allocatable :: output_batch(:, :)     ! Shape: (batch_size, output_dim)
    real(dp), allocatable :: grad_output_batch(:, :)! Shape: (batch_size, output_dim)
    real(dp), allocatable :: grad_input_batch(:, :) ! Shape: (batch_size, input_dim)
    
    ! Variables to hold data from getters
    real(dp), allocatable :: initial_weights(:,:), initial_biases(:)
    real(dp), allocatable :: updated_weights(:,:), updated_biases(:)
    real(dp), allocatable :: grad_weights(:,:), grad_biases(:)

    integer :: i, j

    print *, "========================================"
    print *, "   Testing FullConnectLayer (Batch Mode)"
    print *, "========================================"

    ! 1. Initialize the layer
    print *, "1. Initializing FullConnectLayer..."
    call fc_layer%init(input_dim, output_dim)
    print *, "   Input size: ", fc_layer%get_input_size()
    print *, "   Output size: ", fc_layer%get_output_size()
    
    initial_weights = fc_layer%get_weights()
    initial_biases = fc_layer%get_biases()

    print *, "   Initial Weights shape: (", size(initial_weights, 1), ",", size(initial_weights, 2), ")"
    print *, "   Initial Biases shape:  (", size(initial_biases), ")"
    print *, "---------------------------------"

    ! 2. Prepare input data (Batch)
    allocate(input_batch(batch_size, input_dim))
    ! Fill with some dummy data
    do i = 1, batch_size
        do j = 1, input_dim
            input_batch(i, j) = real(i + j, dp)
        end do
    end do

    print *, "2. Forward Pass..."
    print *, "   Input batch shape: (", size(input_batch, 1), ",", size(input_batch, 2), ")"
    
    ! 3. Perform forward pass
    output_batch = fc_layer%forward(input_batch)
    
    if (.not. allocated(output_batch)) then
        print *, "   ERROR: Forward pass returned unallocated array."
        stop
    end if
    
    print *, "   Output batch shape: (", size(output_batch, 1), ",", size(output_batch, 2), ")"
    print *, "   Expected shape:     (", batch_size, ",", output_dim, ")"
    
    if (size(output_batch, 1) /= batch_size .or. size(output_batch, 2) /= output_dim) then
        print *, "   ERROR: Output shape mismatch!"
        stop
    else
        print *, "   SUCCESS: Forward pass shape correct."
    end if
    print *, "---------------------------------"

    ! 4. Prepare gradient from the next layer (dummy gradient)
    allocate(grad_output_batch(batch_size, output_dim))
    grad_output_batch = 1.0_dp ! Simple gradient of all 1s

    print *, "3. Backward Pass..."
    print *, "   Gradient from next layer shape: (", size(grad_output_batch, 1), ",", size(grad_output_batch, 2), ")"

    ! 5. Perform backward pass
    grad_input_batch = fc_layer%backward(grad_output_batch)
    
    if (.not. allocated(grad_input_batch)) then
        print *, "   ERROR: Backward pass returned unallocated array."
        stop
    end if

    print *, "   Calculated grad_input shape: (", size(grad_input_batch, 1), ",", size(grad_input_batch, 2), ")"
    print *, "   Expected shape:              (", batch_size, ",", input_dim, ")"

    if (size(grad_input_batch, 1) /= batch_size .or. size(grad_input_batch, 2) /= input_dim) then
        print *, "   ERROR: Gradient input shape mismatch!"
        stop
    else
        print *, "   SUCCESS: Backward pass shape correct."
    end if

    grad_weights = fc_layer%get_grad_weights()
    grad_biases = fc_layer%get_grad_biases()

    print *, "   Grad Weights shape: (", size(grad_weights, 1), ",", size(grad_weights, 2), ")"
    print *, "   Grad Biases shape:  (", size(grad_biases), ")"
    print *, "---------------------------------"

    ! 6. Update weights
    print *, "4. Updating weights and biases..."
    print *, "   Learning rate: ", learning_rate
    call fc_layer%update(learning_rate)

    updated_weights = fc_layer%get_weights()
    updated_biases = fc_layer%get_biases()

    ! Simple check to see if weights changed
    if (any(updated_weights /= initial_weights)) then
        print *, "   SUCCESS: Weights have been updated."
    else
        print *, "   WARNING: Weights did not change (gradient might be zero)."
    end if

    if (any(updated_biases /= initial_biases)) then
        print *, "   SUCCESS: Biases have been updated."
    else
        print *, "   WARNING: Biases did not change."
    end if
    print *, "---------------------------------"

    ! 7. Clean up
    print *, "5. Cleaning up..."
    call fc_layer%destroy()
    if (allocated(input_batch)) deallocate(input_batch)
    if (allocated(output_batch)) deallocate(output_batch)
    if (allocated(grad_output_batch)) deallocate(grad_output_batch)
    if (allocated(grad_input_batch)) deallocate(grad_input_batch)
    if (allocated(initial_weights)) deallocate(initial_weights)
    if (allocated(initial_biases)) deallocate(initial_biases)
    if (allocated(updated_weights)) deallocate(updated_weights)
    if (allocated(updated_biases)) deallocate(updated_biases)
    if (allocated(grad_weights)) deallocate(grad_weights)
    if (allocated(grad_biases)) deallocate(grad_biases)

    print *, "========================================"
    print *, "   Test completed."
    print *, "========================================"

end program test_fullconnect

!gfortran -std=f2008 -o FullConnectTest FullConnectTest.f90 ../FullConnect.f90