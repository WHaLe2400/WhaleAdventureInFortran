module Conv_mod
    ! Convolutional layer (CNN) in object-oriented Fortran
    ! Standard and robust implementation with local variables.
    use iso_fortran_env, only: dp => real64
    implicit none

    type :: ConvLayer
        ! Persistent state (weights and gradients)
        integer :: in_ch = 0, out_ch = 0, kH = 0, kW = 0
        integer :: stride = 1, pad = 0
        real(dp), allocatable :: W(:,:,:,:), b(:)      ! W(out_ch, in_ch, kH, kW), b(out_ch)
        real(dp), allocatable :: dW(:,:,:,:), db(:)    ! gradients

        ! Cached input for backward pass
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
        !构建卷积层，初始化权重和偏置
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
        implicit none
        class(ConvLayer), intent(inout) :: self
        real(dp), intent(in) :: x_in(:,:,:,:)   ! (N, in_ch, H, W)
        real(dp), allocatable :: y(:,:,:,:)     ! (N, out_ch, H_out, W_out)

        ! Local variables for this function
        integer :: N, C, H, W_in
        integer :: H_out, W_out
        integer :: n, cout, cin, i_out, j_out, kh, kw
        integer :: i_start, j_start, i_in, j_in
        real(dp), allocatable :: xpad(:,:,:,:)

        ! Cache input
        if (allocated(self%x_cache)) deallocate(self%x_cache)
        allocate(self%x_cache, source=x_in)

        N = size(x_in, 1)
        C = size(x_in, 2)
        H = size(x_in, 3)
        W_in = size(x_in, 4)

        H_out = (H + 2*self%pad - self%kH) / self%stride + 1
        W_out = (W_in + 2*self%pad - self%kW) / self%stride + 1

        allocate(y(N, self%out_ch, H_out, W_out))
        y = 0.0_dp

        allocate(xpad(N, C, H + 2*self%pad, W_in + 2*self%pad))
        xpad = 0.0_dp
        if (self%pad > 0) then
            xpad(:,:, self%pad+1:self%pad+H, self%pad+1:self%pad+W_in) = x_in
        else
            xpad = x_in
        end if

        do n = 1, N
            do cout = 1, self%out_ch
                do i_out = 1, H_out
                    i_start = (i_out-1)*self%stride + 1
                    do j_out = 1, W_out
                        j_start = (j_out-1)*self%stride + 1
                        do cin = 1, C
                            do kh = 1, self%kH
                                i_in = i_start + kh - 1
                                do kw = 1, self%kW
                                    j_in = j_start + kw - 1
                                    y(n, cout, i_out, j_out) = y(n, cout, i_out, j_out) + &
                                         self%W(cout, cin, kh, kw) * xpad(n, cin, i_in, j_in)
                                end do
                            end do
                        end do
                        y(n, cout, i_out, j_out) = y(n, cout, i_out, j_out) + self%b(cout)
                    end do
                end do
            end do
        end do

        deallocate(xpad)
    end function conv_forward

    function conv_backward(self, dout) result(dx)
        implicit none
        class(ConvLayer), intent(inout) :: self
        real(dp), intent(in) :: dout(:,:,:,:)   ! (N, out_ch, H_out, W_out)
        real(dp), allocatable :: dx(:,:,:,:)    ! (N, in_ch, H, W)

        ! Local variables for this function
        integer :: N, C, H, W_in
        integer :: H_out, W_out
        integer :: n, cout, cin, i_out, j_out, kh, kw
        integer :: i_start, j_start, i_in, j_in
        real(dp), allocatable :: xpad(:,:,:,:), dxpad(:,:,:,:)

        if (.not. allocated(self%x_cache)) stop "conv_backward: no cached input"

        N = size(self%x_cache, 1)
        C = size(self%x_cache, 2)
        H = size(self%x_cache, 3)
        W_in = size(self%x_cache, 4)

        H_out = size(dout, 3)
        W_out = size(dout, 4)

        self%dW = 0.0_dp
        self%db = 0.0_dp

        allocate(xpad(N, C, H + 2*self%pad, W_in + 2*self%pad))
        xpad = 0.0_dp
        if (self%pad > 0) then
            xpad(:,:, self%pad+1:self%pad+H, self%pad+1:self%pad+W_in) = self%x_cache
        else
            xpad = self%x_cache
        end if

        allocate(dxpad(N, C, H + 2*self%pad, W_in + 2*self%pad))
        dxpad = 0.0_dp

        do n = 1, N
            do cout = 1, self%out_ch
                do i_out = 1, H_out
                    i_start = (i_out-1)*self%stride + 1
                    do j_out = 1, W_out
                        j_start = (j_out-1)*self%stride + 1
                        self%db(cout) = self%db(cout) + dout(n, cout, i_out, j_out)
                        do cin = 1, C
                            do kh = 1, self%kH
                                i_in = i_start + kh - 1
                                do kw = 1, self%kW
                                    j_in = j_start + kw - 1
                                    self%dW(cout, cin, kh, kw) = self%dW(cout, cin, kh, kw) + &
                                         xpad(n, cin, i_in, j_in) * dout(n, cout, i_out, j_out)
                                    dxpad(n, cin, i_in, j_in) = dxpad(n, cin, i_in, j_in) + &
                                         self%W(cout, cin, kh, kw) * dout(n, cout, i_out, j_out)
                                end do
                            end do
                        end do
                    end do
                end do
            end do
        end do

        allocate(dx(N, C, H, W_in))
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


