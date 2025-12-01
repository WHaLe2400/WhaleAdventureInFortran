module Conv_mod
    ! 卷积层 (CNN)，面向对象 Fortran 实现
    ! 数据形状: (N, C, H, W) - 通道在前
    use iso_fortran_env, only: dp => real64! 规定实型变量的精度为双精度
    implicit none

    type :: ConvLayer
        ! 持久状态（权重和梯度）
        private
        integer :: in_ch = 0, out_ch = 0, kH = 0, kW = 0
        integer :: stride = 1, pad = 0
        ! W(out_ch, in_ch, kH, kW), b(out_ch)
        real(dp), allocatable :: W(:,:,:,:), b(:)
        real(dp), allocatable :: dW(:,:,:,:), db(:)    ! 梯度

        ! 后向传播时缓存的输入 (N, in_ch, H, W)
        real(dp), allocatable :: x_cache(:,:,:,:)

    contains
        procedure, public :: init => conv_init
        procedure, public :: load => conv_load
        procedure, public :: save => conv_save
        procedure, public :: forward => conv_forward
        procedure, public :: backward => conv_backward
        procedure, public :: update => conv_update
        procedure, public :: zero_grads => conv_zero_grads
        procedure, public :: destroy => conv_destroy
        ! Getter functions
        procedure, public :: get_weights => conv_get_weights
        procedure, public :: get_biases => conv_get_biases
        procedure, public :: get_grad_weights => conv_get_grad_weights
        procedure, public :: get_grad_biases => conv_get_grad_biases
    end type ConvLayer

contains

    subroutine conv_init(self, in_ch_, out_ch_, kernel_, stride_, pad_, seed)
        ! 构建卷积层，初始化权重和偏置
        implicit none
        class(ConvLayer), intent(inout) :: self
        integer, intent(in) :: in_ch_, out_ch_, kernel_, stride_, pad_
        integer, intent(in), optional :: seed(:)
        real(dp) :: scale

        self%in_ch = in_ch_; self%out_ch = out_ch_
        self%kH = kernel_; self%kW = kernel_
        self%stride = stride_; self%pad = pad_

        allocate(self%W(self%out_ch, self%in_ch, self%kH, self%kW))
        allocate(self%b(self%out_ch))
        allocate(self%dW(self%out_ch, self%in_ch, self%kH, self%kW))
        allocate(self%db(self%out_ch))

        if (present(seed)) then
            call random_seed(put=seed)
        else
            call random_seed()
        end if

        scale = sqrt(2.0_dp / real(self%in_ch * self%kH * self%kW, dp))
        call random_number(self%W)
        self%W = scale * (2.0_dp * self%W - 1.0_dp)

        self%b = 0.0_dp
        call self%zero_grads()
    end subroutine conv_init


    subroutine conv_save(self, filename)
        class(ConvLayer), intent(in) :: self
        character(len=*), intent(in) :: filename
        integer :: unit, i, j, k, l
        open(newunit=unit, file=filename, status='replace', action='write', form='formatted')
        ! 写出尺寸信息
        write(unit, *) self%out_ch, self%in_ch, self%kH, self%kW
        ! 写出权重 (out_ch, in_ch, kH, kW)
        do i = 1, self%out_ch
            do j = 1, self%in_ch
                do k = 1, self%kH
                    write(unit, *) self%W(i, j, k, :)
                end do
            end do
        end do
        ! 写出偏置 (out_ch)
        write(unit, *) self%b
        close(unit)
    end subroutine conv_save

    subroutine conv_load(self, filename)
        class(ConvLayer), intent(inout) :: self
        character(len=*), intent(in) :: filename
        integer :: unit, i, j, k, l
        integer :: out_ch_read, in_ch_read, kH_read, kW_read
        open(newunit=unit, file=filename, status='old', action='read', form='formatted')
        ! 读取尺寸信息
        read(unit, *) out_ch_read, in_ch_read, kH_read, kW_read
        ! 检查尺寸是否匹配（可选，但推荐）
        if (out_ch_read /= self%out_ch .or. in_ch_read /= self%in_ch .or. &
            kH_read /= self%kH .or. kW_read /= self%kW) then
            print *, "Error: Dimensions in file do not match layer dimensions."
            close(unit)
            return
        end if
        ! 读取权重
        do i = 1, self%out_ch
            do j = 1, self%in_ch
                do k = 1, self%kH
                    read(unit, *) self%W(i, j, k, :)
                end do
            end do
        end do
        ! 读取偏置
        read(unit, *) self%b
        close(unit)
        ! 清零梯度
        call self%zero_grads()
    end subroutine conv_load


    function conv_forward(self, x_in) result(y)
        ! 前向传播
        implicit none
        class(ConvLayer), intent(inout) :: self
        real(dp), intent(in) :: x_in(:,:,:,:)   ! (N, C_in, H, W)
        real(dp), allocatable :: y(:,:,:,:)     ! (N, C_out, H_out, W_out)

        ! 本函数的局部变量
        integer :: batch, C_in, H, W_in
        integer :: H_out, W_out
        integer :: n, c_out_idx, c_in_idx, i_out, j_out, kh, kw
        integer :: i_start, j_start, i_in, j_in
        real(dp), allocatable :: xpad(:,:,:,:)

        ! 缓存输入
        if (allocated(self%x_cache)) deallocate(self%x_cache)
        allocate(self%x_cache, source=x_in)

        batch = size(x_in, 1)
        C_in = size(x_in, 2)
        H = size(x_in, 3)
        W_in = size(x_in, 4)

        H_out = (H + 2*self%pad - self%kH) / self%stride + 1
        W_out = (W_in + 2*self%pad - self%kW) / self%stride + 1

        allocate(y(batch, self%out_ch, H_out, W_out))
        y = 0.0_dp

        allocate(xpad(batch, C_in, H + 2*self%pad, W_in + 2*self%pad))
        xpad = 0.0_dp
        if (self%pad > 0) then
            xpad(:, :, self%pad+1:self%pad+H, self%pad+1:self%pad+W_in) = x_in
        else
            ! Even if pad is 0, xpad needs to be a copy of x_in
            xpad = x_in
        end if

        do n = 1, batch
            do c_out_idx = 1, self%out_ch
                do i_out = 1, H_out
                    i_start = (i_out-1)*self%stride + 1
                    do j_out = 1, W_out
                        j_start = (j_out-1)*self%stride + 1
                        do c_in_idx = 1, C_in
                            do kh = 1, self%kH
                                i_in = i_start + kh - 1
                                do kw = 1, self%kW
                                    j_in = j_start + kw - 1
                                    y(n, c_out_idx, i_out, j_out) = y(n, c_out_idx, i_out, j_out) + &
                                         self%W(c_out_idx, c_in_idx, kh, kw) * xpad(n, c_in_idx, i_in, j_in)
                                end do
                            end do
                        end do
                        y(n, c_out_idx, i_out, j_out) = y(n, c_out_idx, i_out, j_out) + self%b(c_out_idx)
                    end do
                end do
            end do
        end do

        deallocate(xpad)
    end function conv_forward

    function conv_backward(self, dout) result(dx)
        ! 误差反相传播
        class(ConvLayer), intent(inout) :: self
        real(dp), intent(in) :: dout(:,:,:,:)   ! (N, C_out, H_out, W_out)
        real(dp), allocatable :: dx(:,:,:,:)    ! (N, C_in, H, W)

        ! 本函数的局部变量
        integer :: batch, C_in, H, W_in
        integer :: H_out, W_out, C_out
        integer :: n, c_out_idx, c_in_idx, i_out, j_out, kh, kw
        integer :: i_start, j_start, i_in, j_in
        real(dp), allocatable :: xpad(:,:,:,:), dxpad(:,:,:,:)

        if (.not. allocated(self%x_cache)) stop "conv_backward: no cached input"

        batch = size(self%x_cache, 1)
        C_in = size(self%x_cache, 2)
        H = size(self%x_cache, 3)
        W_in = size(self%x_cache, 4)

        C_out = size(dout, 2)
        H_out = size(dout, 3)
        W_out = size(dout, 4)

        ! Gradients are zeroed by model%zero_grads() before this call.
        ! self%dW = 0.0_dp
        ! self%db = 0.0_dp

        ! Vectorized gradient calculation for biases
        self%db = self%db + sum(sum(sum(dout, dim=4), dim=3), dim=1)

        allocate(xpad(batch, C_in, H + 2*self%pad, W_in + 2*self%pad))
        xpad = 0.0_dp
        if (self%pad > 0) then
            xpad(:, :, self%pad+1:self%pad+H, self%pad+1:self%pad+W_in) = self%x_cache
        else
            ! Even if pad is 0, xpad needs to be a copy of x_cache
            xpad = self%x_cache
        end if

        allocate(dxpad(batch, C_in, H + 2*self%pad, W_in + 2*self%pad))
        dxpad = 0.0_dp

        do n = 1, batch
            do c_out_idx = 1, C_out
                do i_out = 1, H_out
                    i_start = (i_out-1)*self%stride + 1
                    do j_out = 1, W_out
                        j_start = (j_out-1)*self%stride + 1
                        ! self%db(c_out_idx) = self%db(c_out_idx) + dout(n, c_out_idx, i_out, j_out) ! Replaced by vectorized version
                        do c_in_idx = 1, C_in
                            do kh = 1, self%kH
                                i_in = i_start + kh - 1
                                do kw = 1, self%kW
                                    j_in = j_start + kw - 1
                                    self%dW(c_out_idx, c_in_idx, kh, kw) = self%dW(c_out_idx, c_in_idx, kh, kw) + &
                                         xpad(n, c_in_idx, i_in, j_in) * dout(n, c_out_idx, i_out, j_out)
                                    dxpad(n, c_in_idx, i_in, j_in) = dxpad(n, c_in_idx, i_in, j_in) + &
                                         self%W(c_out_idx, c_in_idx, kh, kw) * dout(n, c_out_idx, i_out, j_out)
                                end do
                            end do
                        end do
                    end do
                end do
            end do
        end do

        allocate(dx(batch, C_in, H, W_in))
        if (self%pad > 0) then
            dx = dxpad(:, :, self%pad+1:self%pad+H, self%pad+1:self%pad+W_in)
        else
            dx = dxpad
        end if

        deallocate(xpad, dxpad)
    end function conv_backward

    subroutine conv_update(self, lr)
        implicit none
        class(ConvLayer), intent(inout) :: self
        real(dp), intent(in) :: lr

        if (.not. allocated(self%dW)) stop "conv_update: gradients not computed"
        self%W = self%W - lr * self%dW
        self%b = self%b - lr * self%db
    end subroutine conv_update

    subroutine conv_zero_grads(self)
        implicit none
        class(ConvLayer), intent(inout) :: self

        if (allocated(self%dW)) self%dW = 0.0_dp
        if (allocated(self%db)) self%db = 0.0_dp
    end subroutine conv_zero_grads

    subroutine conv_destroy(self)
        implicit none
        class(ConvLayer), intent(inout) :: self

        if (allocated(self%W)) deallocate(self%W)
        if (allocated(self%b)) deallocate(self%b)
        if (allocated(self%dW)) deallocate(self%dW)
        if (allocated(self%db)) deallocate(self%db)
        if (allocated(self%x_cache)) deallocate(self%x_cache)

        self%in_ch = 0; self%out_ch = 0; self%kH = 0; self%kW = 0
        self%stride = 1; self%pad = 0
    end subroutine conv_destroy

    function conv_get_weights(self) result(W_out)
        implicit none
        class(ConvLayer), intent(in) :: self
        real(dp), allocatable :: W_out(:,:,:,:)

        if (allocated(self%W)) then
            allocate(W_out(size(self%W,1), size(self%W,2), size(self%W,3), size(self%W,4)))
            W_out = self%W
        else
            allocate(W_out(0,0,0,0))
        end if
    end function conv_get_weights

    function conv_get_biases(self) result(b_out)
        implicit none
        class(ConvLayer), intent(in) :: self
        real(dp), allocatable :: b_out(:)

        if (allocated(self%b)) then
            allocate(b_out(size(self%b)))
            b_out = self%b
        else
            allocate(b_out(0))
        end if
    end function conv_get_biases

    function conv_get_grad_weights(self) result(dW_out)
        implicit none
        class(ConvLayer), intent(in) :: self
        real(dp), allocatable :: dW_out(:,:,:,:)

        if (allocated(self%dW)) then
            allocate(dW_out(size(self%dW,1), size(self%dW,2), size(self%dW,3), size(self%dW,4)))
            dW_out = self%dW
        else
            allocate(dW_out(0,0,0,0))
        end if
    end function conv_get_grad_weights

    function conv_get_grad_biases(self) result(db_out)
        implicit none
        class(ConvLayer), intent(in) :: self
        real(dp), allocatable :: db_out(:)

        if (allocated(self%db)) then
            allocate(db_out(size(self%db)))
            db_out = self%db
        else
            allocate(db_out(0))
        end if
    end function conv_get_grad_biases

end module Conv_mod


! gfortran  -std=f2008 -o ModelCombine_test ModelCombine.f90 Modules/Conv.f90 Modules/flaten.f90 Modules/FullConnect.f90 Modules/PReluFunc.f90
