program ModelCombine_Test
    use ModelCombine_mod
    use iso_fortran_env, only: dp => real64
    implicit none

    type(Model) :: my_model
    integer :: i
    integer :: batch_size=2, input_height=28, input_width=28, input_channels=1, output_size=10
    integer :: tmp1, tmp2
    real(dp), allocatable :: input(:,:,:,:), output(:,:)
    
    ! Initialize the model
    call my_model%init()
    tmp1 = my_model%FC1%get_input_size()  ! Just to avoid unused variable warning
    write(*,*) "FC1 input size: ", tmp1
    tmp2 = my_model%FC_in
    write(*,*) "Model FC_in parameter: ", tmp2
    

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

!gfortran -std=f2008 -o ModelCombine_Test ModelCombine_Test.f90 ModelCombine.f90 Modules/Conv.f90 Modules/flaten.f90 Modules/FullConnect.f90 Modules/PReluFunc.f90 
!gfortran -std=f2008 -o ModelCombine_Test ModelCombine_Test.f90 ModelCombine.f90 Modules/*.f90