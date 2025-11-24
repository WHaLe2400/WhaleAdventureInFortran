program test_fullconnect
    use FullConnect_mod
    implicit none

    ! Test parameters
    integer, parameter :: input_dim = 3
    integer, parameter :: output_dim = 2
    real, parameter :: learning_rate = 0.1

    ! Layer instance
    type(FullConnectLayer) :: fc_layer

    ! Test data
    real :: input_vector(input_dim)
    real, allocatable :: output_vector(:)
    real :: grad_output(output_dim)
    real, allocatable :: grad_input(:)
    
    ! Variables to hold data from getters
    real, allocatable :: initial_weights(:,:), initial_biases(:)
    real, allocatable :: updated_weights(:,:), updated_biases(:)
    real, allocatable :: grad_weights(:,:), grad_biases(:)

    integer :: i

    ! 1. Initialize the layer
    print *, "1. Initializing FullConnectLayer..."
    call fc_layer%init(input_dim, output_dim)
    print *, "   Input size: ", fc_layer%get_input_size()
    print *, "   Output size: ", fc_layer%get_output_size()
    
    initial_weights = fc_layer%get_weights()
    initial_biases = fc_layer%get_biases()

    print *, "   Initial Weights:"
    do i = 1, fc_layer%get_output_size()
        print '(100F8.4)', initial_weights(i, :)
    end do
    print *, "   Initial Biases:"
    print '(100F8.4)', initial_biases(:)
    print *, "---------------------------------"

    ! 2. Prepare input data
    input_vector = [1.0, 2.0, 3.0]
    print *, "2. Forward Pass..."
    print *, "   Input vector: ", input_vector
    
    ! 3. Perform forward pass
    output_vector = fc_layer%forward(input_vector)
    if (.not. allocated(output_vector) .or. size(output_vector) == 0) then
        print *, "   Forward pass failed."
        call fc_layer%destroy()
        stop
    end if
    print *, "   Output vector: ", output_vector
    print *, "---------------------------------"

    ! 4. Prepare gradient from the next layer (dummy gradient)
    grad_output = [0.5, -0.5]
    print *, "3. Backward Pass..."
    print *, "   Gradient from next layer (grad_output): ", grad_output

    ! 5. Perform backward pass
    grad_input = fc_layer%backward(grad_output)
    if (.not. allocated(grad_input) .or. size(grad_input) == 0) then
        print *, "   Backward pass failed."
        call fc_layer%destroy()
        if (allocated(output_vector)) deallocate(output_vector)
        stop
    end if

    grad_weights = fc_layer%get_grad_weights()
    grad_biases = fc_layer%get_grad_biases()

    print *, "   Calculated grad_input: ", grad_input
    print *, "   Calculated grad_weights:"
    do i = 1, fc_layer%get_output_size()
        print '(100F8.4)', grad_weights(i, :)
    end do
    print *, "   Calculated grad_biases: ", grad_biases(:)
    print *, "---------------------------------"

    ! 6. Update weights
    print *, "4. Updating weights and biases..."
    print *, "   Learning rate: ", learning_rate
    call fc_layer%update(learning_rate)

    updated_weights = fc_layer%get_weights()
    updated_biases = fc_layer%get_biases()

    print *, "   Original Weights:"
    do i = 1, fc_layer%get_output_size()
        print '(100F8.4)', initial_weights(i, :)
    end do
    print *, "   Updated Weights:"
    do i = 1, fc_layer%get_output_size()
        print '(100F8.4)', updated_weights(i, :)
    end do
    
    print *, "   Original Biases:"
    print '(100F8.4)', initial_biases(:)
    print *, "   Updated Biases:"
    print '(100F8.4)', updated_biases(:)
    print *, "---------------------------------"

    ! 7. Clean up
    print *, "5. Cleaning up..."
    call fc_layer%destroy()
    if (allocated(output_vector)) deallocate(output_vector)
    if (allocated(grad_input)) deallocate(grad_input)
    if (allocated(initial_weights)) deallocate(initial_weights)
    if (allocated(initial_biases)) deallocate(initial_biases)
    if (allocated(updated_weights)) deallocate(updated_weights)
    if (allocated(updated_biases)) deallocate(updated_biases)
    if (allocated(grad_weights)) deallocate(grad_weights)
    if (allocated(grad_biases)) deallocate(grad_biases)
    print *, "   Done."

end program test_fullconnect