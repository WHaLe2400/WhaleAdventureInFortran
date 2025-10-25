program test_conv
    use iso_fortran_env, only: dp => real64
    use Conv_mod
    implicit none

    type(ConvLayer) :: conv
    real(dp), allocatable :: x(:,:,:,:), y(:,:,:,:), dout(:,:,:,:), dx(:,:,:,:)
    integer, allocatable :: seed(:)
    integer :: N, in_ch, out_ch, H, W, kH, kW, stride, pad
    real(dp) :: lr
    integer :: i, j

    ! 配置小型测试
    N = 1; in_ch = 1; out_ch = 1
    H = 5; W = 5
    kH = 3; kW = 3
    stride = 1; pad = 1
    lr = 0.1_dp

    allocate(seed(8))
    do i = 1, size(seed)
        seed(i) = i * 137    ! 固定种子以保证可重复
    end do

    call conv%init(in_ch, out_ch, kH, kW, stride, pad, seed)

    allocate(x(N, in_ch, H, W))
    x = 0.0_dp
    ! 填入简单可辨识的输入模式
    do i = 1, H
        do j = 1, W
            x(1,1,i,j) = real(i*10 + j, dp)
        end do
    end do

    y = conv%forward(x)

    write(*,*) 'Forward output y (N,out_ch,H_out,W_out):'
    do i = 1, N
        do j = 1, out_ch
            write(*,'(A,I0,A,I0)') 'Sample N=', i, ' out_ch=', j
            do integer :: r = 1, c = 1, rr = 1  ! dummy to satisfy f77-style do; replaced below
            end do
        end do
    end do

    ! 逐元素打印 y（简洁行输出）
    do i = 1, N
        do j = 1, out_ch
            do integer :: r = 1, r_end = size(y,3)
                do integer :: c = 1, c_end = size(y,4)
                end do
            end do
        end do
    end do

    ! 更直接且可读的打印实现
    do i = 1, N
        do j = 1, out_ch
            do i = 1, size(y,3)
                write(*,'(A,I0,A,I0)') 'y(',i,',',j,') row ', i
                write(*,'(100(F8.3,1X))') (y(i,j,i,k), k=1,size(y,4))
            end do
        end do
    end do

    ! 反向并更新
    allocate(dout(N, out_ch, size(y,3), size(y,4)))
    dout = 1.0_dp

    dx = conv%backward(dout)

    write(*,*) 'db (bias gradients):'
    write(*,'(100(F10.5,1X))') conv%db

    write(*,*) 'dW Frobenius norm: ', sqrt(sum(conv%dW**2))

    call conv%update(lr)

    write(*,*) 'After update: W(1,1,1,1) = ', conv%W(1,1,1,1)

contains
    ! 为兼容各种编译器和保持简短，将重复打印逻辑放在内部子程序中
    subroutine quiet_exit()
    end subroutine quiet_exit

end program test_conv