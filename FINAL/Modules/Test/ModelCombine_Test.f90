program ModelCombine_Test
    use ModelCombine2_mod
    use iso_fortran_env, only: dp => real64
    implicit none

    type(Model) :: my_model, loaded_model
    integer :: i
    integer :: batch_size=64, input_height=28, input_width=28, input_channels=1, output_size=10
    real(dp), allocatable :: input(:,:,:,:), output_before_update(:,:), output_after_update(:,:)
    real(dp), allocatable :: grad_output(:,:), output_loaded(:,:)
    real(dp) :: learning_rate = 0.01_dp
    character(len=100) :: temp_base_path = "RESULT"

    ! 1. Initialize the model
    print *, "--- 1. Initializing Model ---"
    call my_model%init()
    print *, "Model initialized successfully."
    print *, "Configuration:"
    print *, "  Conv1: kernel=", my_model%Conv1_kernel, ", stride=", my_model%Conv1_stride
    print *, "  Conv2: kernel=", my_model%Conv2_kernel, ", stride=", my_model%Conv2_stride
    print *, "  FC Layers: FC_in=", my_model%FC_in, ", FC_hidden=", my_model%FC_hidden, ", FC_out=", my_model%FC_out
    print *, "-----------------------------"
    print *, ""

    ! 2. Prepare Input Data
    print *, "--- 2. Preparing Input Data ---"
    allocate(input(batch_size, input_channels, input_height, input_width))
    call random_number(input)
    print *, "Input data created with shape: ", shape(input)
    print *, "-----------------------------"
    print *, ""

    ! 3. Perform Forward Pass and Check Intermediate Shapes
    print *, "--- 3. Performing Forward Pass (Before Update) ---"
    output_before_update = my_model%forward(input)
    print *, "Forward pass complete."
    print *, "Verifying intermediate shapes:"
    print *, "  - Conv1 output shape:    ", shape(my_model%conv1_out)
    print *, "  - PReLU1 output shape:   ", shape(my_model%prelu1_out)
    print *, "  - Conv2 output shape:    ", shape(my_model%conv2_out)
    print *, "  - PReLU2 output shape:   ", shape(my_model%prelu2_out)
    print *, "  - Conv3 output shape:    ", shape(my_model%conv3_out)
    print *, "  - PReLU3 output shape:   ", shape(my_model%prelu3_out)
    print *, "  - Flaten output shape:   ", shape(my_model%flaten_out)
    print *, "  - FC1 output shape:      ", shape(my_model%fc1_out)
    print *, "  - PReLU4 output shape:   ", shape(my_model%prelu4_out)
    print *, "  - Final output shape:    ", shape(output_before_update)
    print *, "Output before update (first batch):"
    print *, output_before_update(1, :)
    print *, "-----------------------------"
    print *, ""

    ! 4. Prepare Gradient for Backward Pass
    print *, "--- 4. Preparing for Backward Pass ---"
    allocate(grad_output(batch_size, output_size))
    call random_number(grad_output)
    print *, "Dummy output gradient created with shape: ", shape(grad_output)
    print *, "-----------------------------"
    print *, ""

    ! 5. Perform Backward Pass
    print *, "--- 5. Performing Backward Pass ---"
    call my_model%backward(grad_output)
    print *, "Backward pass complete. Gradients have been computed internally."
    print *, "-----------------------------"
    print *, ""

    ! 6. Update Model Parameters and Verify Change
    print *, "--- 6. Updating Model and Verifying Change ---"
    print *, "Updating model parameters with learning rate: ", learning_rate
    call my_model%update(learning_rate)
    print *, "Model update complete."
    
    ! Perform a second forward pass with the same input
    output_after_update = my_model%forward(input)
    print *, "Output after update (first batch):"
    print *, output_after_update(1, :)

    ! Compare outputs. They should be different if weights were updated.
    if (any(output_before_update /= output_after_update)) then
        print *, "SUCCESS: Model output has changed, indicating weights were updated."
    else
        print *, "FAILURE: Model output is identical. Weights were not updated."
    end if
    print *, "-----------------------------"
    print *, ""

    ! 7. Test Save and Load Functionality
    print *, "--- 8. Testing Save and Load Functionality ---"
    ! Save the model
    call my_model%save(temp_base_path)
    print *, "Model saved to files with base path: ", trim(temp_base_path)

    ! Initialize a new model and load
    call loaded_model%init()
    call loaded_model%load(temp_base_path)
    print *, "Model loaded from files with base path: ", trim(temp_base_path)

    ! Perform forward pass with loaded model
    output_loaded = loaded_model%forward(input)
    print *, "Output from loaded model (first batch):"
    print *, output_loaded(1, :)

    ! Compare outputs from original and loaded model
    if (all(abs(output_after_update - output_loaded) < 1.0e-10_dp)) then
        print *, "SUCCESS: Loaded model produces identical output to the original model."
    else
        print *, "FAILURE: Loaded model output differs from the original model."
    end if
    print *, "-----------------------------"
    print *, ""

    ! 8. Clean up
    print *, "--- 7. Cleaning Up ---"
    call my_model%destroy()
    call loaded_model%destroy()
    if (.not. allocated(my_model%fc1_out) .and. .not. allocated(loaded_model%fc1_out)) then
        print *, "SUCCESS: Intermediate arrays deallocated."
    else
        print *, "FAILURE: Intermediate arrays not deallocated."
    end if
    print *, "Cleanup complete."
    print *, "-----------------------------"

end program ModelCombine_Test

!gfortran -std=f2008 -o ModelCombine_Test ModelCombine_Test.f90 ../*.f90