module Conv_mod
    ! 卷积层 (CNN)，面向对象 Fortran 实现
    ! 数据形状: (N, H, W, C) - 通道在后
    use iso_fortran_env, only: dp => real64! 规定实型变量的精度为双精度
    implicit none

    type :: ConvLayer
        ! 持久状态（权重和梯度）
        private
        integer :: in_ch = 0, out_ch = 0, kH = 0, kW = 0
        integer :: stride = 1, pad = 0
        ! W(kH, kW, in_ch, out_ch), b(out_ch)
        real(dp), allocatable :: W(:,:,:,:), b(:)
        real(dp), allocatable :: dW(:,:,:,:), db(:)    ! 梯度

        ! 后向传播时缓存的输入 (N, H, W, in_ch)
        real(dp), allocatable :: x_cache(:,:,:,:)

    contains
        procedure, public :: init => conv_init
        procedure, public :: load => conv_load
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

        allocate(self%W(self%kH, self%kW, self%in_ch, self%out_ch))
        allocate(self%b(self%out_ch))
        allocate(self%dW(self%kH, self%kW, self%in_ch, self%out_ch))
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


    subroutine conv_load(self, W_in, b_in)
        ! 加载预训练的权重和偏置
        implicit none
        class(ConvLayer), intent(inout) :: self
        real(dp), intent(in) :: W_in(:,:,:,:), b_in(:)

        if (size(W_in, 1) /= self%kH .or. size(W_in, 2) /= self%kW .or. &
            size(W_in, 3) /= self%in_ch .or. size(W_in, 4) /= self%out_ch) then
            stop "conv_load: weight dimensions do not match"
        end if
        if (size(b_in) /= self%out_ch) then
            stop "conv_load: bias dimensions do not match"
        end if

        self%W = W_in
        self%b = b_in
    end subroutine conv_load


    function conv_forward(self, x_in) result(y)
        ! 前向传播
        implicit none
        class(ConvLayer), intent(inout) :: self
        real(dp), intent(in) :: x_in(:,:,:,:)   ! (N, H, W, 输入通道)
        real(dp), allocatable :: y(:,:,:,:)     ! (N, H_out, W_out, 输出通道)

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
        H = size(x_in, 2)
        W_in = size(x_in, 3)
        C_in = size(x_in, 4)

        H_out = (H + 2*self%pad - self%kH) / self%stride + 1
        W_out = (W_in + 2*self%pad - self%kW) / self%stride + 1

        allocate(y(batch, H_out, W_out, self%out_ch))
        y = 0.0_dp

        allocate(xpad(batch, H + 2*self%pad, W_in + 2*self%pad, C_in))
        xpad = 0.0_dp
        if (self%pad > 0) then
            xpad(:, self%pad+1:self%pad+H, self%pad+1:self%pad+W_in, :) = x_in
        else
            xpad = x_in
        end if

        do n = 1, batch
            do i_out = 1, H_out
                i_start = (i_out-1)*self%stride + 1
                do j_out = 1, W_out
                    j_start = (j_out-1)*self%stride + 1
                    do c_out_idx = 1, self%out_ch
                        do kh = 1, self%kH
                            i_in = i_start + kh - 1
                            do kw = 1, self%kW
                                j_in = j_start + kw - 1
                                do c_in_idx = 1, C_in
                                    y(n, i_out, j_out, c_out_idx) = y(n, i_out, j_out, c_out_idx) + &
                                         self%W(kh, kw, c_in_idx, c_out_idx) * xpad(n, i_in, j_in, c_in_idx)
                                end do
                            end do
                        end do
                        y(n, i_out, j_out, c_out_idx) = y(n, i_out, j_out, c_out_idx) + self%b(c_out_idx)
                    end do
                end do
            end do
        end do

        deallocate(xpad)
    end function conv_forward

    function conv_backward(self, dout) result(dx)
        ! 误差反相传播
        class(ConvLayer), intent(inout) :: self
        real(dp), intent(in) :: dout(:,:,:,:)   ! (N, H_out, W_out, 输出通道)
        real(dp), allocatable :: dx(:,:,:,:)    ! (N, H, W, 输入通道)

        ! 本函数的局部变量
        integer :: batch, C_in, H, W_in
        integer :: H_out, W_out
        integer :: n, c_out_idx, c_in_idx, i_out, j_out, kh, kw
        integer :: i_start, j_start, i_in, j_in
        real(dp), allocatable :: xpad(:,:,:,:), dxpad(:,:,:,:)

        if (.not. allocated(self%x_cache)) stop "conv_backward: no cached input"

        batch = size(self%x_cache, 1)
        H = size(self%x_cache, 2)
        W_in = size(self%x_cache, 3)
        C_in = size(self%x_cache, 4)

        H_out = size(dout, 2)
        W_out = size(dout, 3)

        self%dW = 0.0_dp
        self%db = 0.0_dp

        allocate(xpad(batch, H + 2*self%pad, W_in + 2*self%pad, C_in))
        xpad = 0.0_dp
        if (self%pad > 0) then
            xpad(:, self%pad+1:self%pad+H, self%pad+1:self%pad+W_in, :) = self%x_cache
        else
            xpad = self%x_cache
        end if

        allocate(dxpad(batch, H + 2*self%pad, W_in + 2*self%pad, C_in))
        dxpad = 0.0_dp

        do n = 1, batch
            do i_out = 1, H_out
                i_start = (i_out-1)*self%stride + 1
                do j_out = 1, W_out
                    j_start = (j_out-1)*self%stride + 1
                    do c_out_idx = 1, self%out_ch
                        self%db(c_out_idx) = self%db(c_out_idx) + dout(n, i_out, j_out, c_out_idx)
                        do kh = 1, self%kH
                            i_in = i_start + kh - 1
                            do kw = 1, self%kW
                                j_in = j_start + kw - 1
                                do c_in_idx = 1, C_in
                                    self%dW(kh, kw, c_in_idx, c_out_idx) = self%dW(kh, kw, c_in_idx, c_out_idx) + &
                                         xpad(n, i_in, j_in, c_in_idx) * dout(n, i_out, j_out, c_out_idx)
                                    dxpad(n, i_in, j_in, c_in_idx) = dxpad(n, i_in, j_in, c_in_idx) + &
                                         self%W(kh, kw, c_in_idx, c_out_idx) * dout(n, i_out, j_out, c_out_idx)
                                end do
                            end do
                        end do
                    end do
                end do
            end do
        end do

        allocate(dx(batch, H, W_in, C_in))
        if (self%pad > 0) then
            dx = dxpad(:, self%pad+1:self%pad+H, self%pad+1:self%pad+W_in, :)
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
