module getting_command_args
    implicit none
    character(len=512)allocatable :: args()
    type, public :: CommandArgs
        character(len=512), allocatable :: args(:)
    contains
        procedure :: read_from_command_line => ca_read
        procedure :: print => ca_print
        final :: ca_final
    end type CommandArgs

    contains

    ! 构造器（可选）：创建并读取命令行参数
    function new_command_args() result(obj)
        type(CommandArgs) :: obj
        call obj%read_from_command_line()
    end function new_command_args

    ! Type-bound: 从命令行读取参数
    subroutine ca_read(this)
        class(CommandArgs), intent(inout) :: this
        integer :: i, n
        n = command_argument_count()
        if (allocated(this%args)) then
            deallocate(this%args)
        end if
        if (n > 0) then
            allocate(this%args(n))
            do i = 1, n
                call get_command_argument(i, this%args(i))
            end do
        else
            allocate(this%args(0))
        end if
    end subroutine ca_read

    ! Type-bound: 打印参数
    subroutine ca_print(this)
        class(CommandArgs), intent(in) :: this
        integer :: i, n
        n = size(this%args)
        print *, "传递了 ", n, " 个参数:"
        do i = 1, n
            print *, "  参数 ", i, ": ", trim(this%args(i))
        end do
    end subroutine ca_print

    ! finalizer：清理内存
    subroutine ca_final(this)
        type(CommandArgs), intent(inout) :: this
        if (allocated(this%args)) deallocate(this%args)
    end subroutine ca_final
contains

    subroutine cout_args()
        integer :: i, arg_count
        arg_count = command_argument_count()
        allocate(args(arg_count))
        do i = 1, arg_count
            call get_command_argument(i, args(i))
        end do
    end subroutine get_command_args

    subroutine print_command_args()
        integer :: i, arg_count
        arg_count = size(args)
        print *, "传递了 ", arg_count, " 个参数:"
        do i = 1, arg_count
            print *, "  参数 ", i, ": ", trim(args(i))
        end do
    end subroutine print_command_args

end module getting_command_args
