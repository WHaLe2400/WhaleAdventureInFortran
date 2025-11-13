module Conv_mod
    ! 卷积层 (CNN)，面向对象 Fortran 实现
    ! 标准且健壮的实现，使用局部变量。
    use iso_fortran_env, only: dp => real64! 规定实型变量的精度为双精度
    implicit none

    type :: ConvLayer
        ! 持久状态（权重和梯度）
        integer :: in_ch = 0, out_ch = 0, kH = 0, kW = 0
        integer :: stride = 1, pad = 0
        real(dp), allocatable :: W(:,:,:,:), b(:)      ! W(out_ch, in_ch, kH, kW)，b(out_ch)
        real(dp), allocatable :: dW(:,:,:,:), db(:)    ! 梯度

        ! 后向传播时缓存的输入
        real(dp), allocatable :: x_cache(:,:,:,:)

    contains
        procedure :: init => conv_init
        procedure :: forward => conv_forward
        procedure :: backward => conv_backward
        procedure :: update => conv_update
        procedure :: zero_grads => conv_zero_grads
    end type ConvLayer

contains

    subroutine conv_init(self, in_ch_, out_ch_, kH_, kW_, stride_, pad_, seed)
        ! 构建卷积层，初始化权重和偏置
        implicit none
        class(ConvLayer), intent(inout) :: self
        integer, intent(in) :: in_ch_, out_ch_, kH_, kW_, stride_, pad_
        integer, intent(in), optional :: seed(:)
        real(dp) :: scale

        self%in_ch = in_ch_; self%out_ch = out_ch_
        self%kH = kH_; self%kW = kW_
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

    function conv_forward(self, x_in) result(y)
        ! 前向传播
        implicit none
        class(ConvLayer), intent(inout) :: self
        real(dp), intent(in) :: x_in(:,:,:,:)   ! (N, 输入通道, H, W)
        real(dp), allocatable :: y(:,:,:,:)     ! (N, 输出通道, H_out, W_out)

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
            xpad(:,:, self%pad+1:self%pad+H, self%pad+1:self%pad+W_in) = x_in
        else
            xpad = x_in
        end if

        do n = 1, batch! 
            do c_out_idx = 1, self%out_ch
                do i_out = 1, H_out !在图像上竖直方向遍历
                    i_start = (i_out-1)*self%stride + 1
                    do j_out = 1, W_out !在图像上水平方向遍历
                        j_start = (j_out-1)*self%stride + 1
                        do c_in_idx = 1, C_in ! 遍历输入通道
                            do kh = 1, self%kH ! 遍历卷积核高度
                                i_in = i_start + kh - 1
                                do kw = 1, self%kW ! 遍历卷积核宽度
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
        real(dp), intent(in) :: dout(:,:,:,:)   ! (N, 输出通道, H_out, W_out)
        real(dp), allocatable :: dx(:,:,:,:)    ! (N, 输入通道, H, W)

        ! 本函数的局部变量
        integer :: batch, C_in, H, W_in
        integer :: H_out, W_out
        integer :: n, c_out_idx, c_in_idx, i_out, j_out, kh, kw
        integer :: i_start, j_start, i_in, j_in
        real(dp), allocatable :: xpad(:,:,:,:), dxpad(:,:,:,:)

        if (.not. allocated(self%x_cache)) stop "conv_backward: no cached input"

        batch = size(self%x_cache, 1)
        C_in = size(self%x_cache, 2)
        H = size(self%x_cache, 3)
        W_in = size(self%x_cache, 4)

        H_out = size(dout, 3)
        W_out = size(dout, 4)

        self%dW = 0.0_dp
        self%db = 0.0_dp

        allocate(xpad(batch, C_in, H + 2*self%pad, W_in + 2*self%pad))
        xpad = 0.0_dp
        if (self%pad > 0) then
            xpad(:,:, self%pad+1:self%pad+H, self%pad+1:self%pad+W_in) = self%x_cache
        else
            xpad = self%x_cache
        end if

        allocate(dxpad(batch, C_in, H + 2*self%pad, W_in + 2*self%pad))
        dxpad = 0.0_dp

        do n = 1, batch
            do c_out_idx = 1, self%out_ch
                do i_out = 1, H_out
                    i_start = (i_out-1)*self%stride + 1
                    do j_out = 1, W_out
                        j_start = (j_out-1)*self%stride + 1
                        self%db(c_out_idx) = self%db(c_out_idx) + dout(n, c_out_idx, i_out, j_out)
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
            dx = dxpad(:,:, self%pad+1:self%pad+H, self%pad+1:self%pad+W_in)
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

end module Conv_mod
