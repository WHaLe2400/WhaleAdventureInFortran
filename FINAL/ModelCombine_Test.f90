program ModelCombine_Test
    use ModelCombine_mod
    use iso_fortran_env, only: dp => real64
    implicit none

    type(Model) :: my_model
    integer :: i
    integer :: batch_size=2, input_height=28, input_width=28, input_channels=1, output_size=10
    real(dp), allocatable :: input(:,:,:,:), output(:,:)
    
    ! Initialize the model
    call my_model%init()

    ! Allocate input and output arrays
    allocate(input(batch_size, input_height, input_width, input_channels))
    allocate(output(batch_size, output_size))
    ! Create dummy input data
    call random_number(input)
    ! Perform forward pass
    output = my_model%forward(input)
    print *, "Output from the model:"
    do i = 1, batch_size
        print *, output(i, :)
    end do

    ! Clean up
    call my_model%destroy()
end program ModelCombine_Test