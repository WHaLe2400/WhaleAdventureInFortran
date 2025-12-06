module Conv_mod
    ! 卷积层 (CNN) 实现，遵循 Fortran 2008 标准
    ! Fortran 友好数据格式: (H, W, C, N)
    use iso_fortran_env, only: dp => real64
    implicit none
    private

    public :: ConvLayer

    type :: ConvLayer
        private
        integer :: in_ch = 0, out_ch = 0, kH = 0, kW = 0
        integer :: stride = 1, pad = 0
        
        ! 参数: 权重 (kH, kW, in_ch, out_ch), 偏置 (out_ch)
        real(dp), allocatable :: W(:,:,:,:), b(:)
        
        ! 梯度: 与参数形状一致
        real(dp), allocatable :: dW(:,:,:,:), db(:)
        
        ! 缓存: 保存前向传播的输入，用于反向传播
        real(dp), allocatable :: x_cache(:,:,:,:)

    contains
        procedure, public :: init => conv_init
        procedure, public :: save => conv_save
        procedure, public :: load => conv_load
        procedure, public :: forward => conv_forward
        procedure, public :: backward => conv_backward
        procedure, public :: destroy => conv_destroy
        procedure, public :: zero_grads => conv_zero_grads
        procedure, public :: update => conv_update
    end type ConvLayer

contains

    ! -------------------------------------------------------------------------
    ! 初始化层
    ! -------------------------------------------------------------------------
    subroutine conv_init(self, in_ch, out_ch, kernel_size, stride, padding)
        class(ConvLayer), intent(inout) :: self
        integer, intent(in) :: in_ch, out_ch, kernel_size
        integer, intent(in), optional :: stride, padding
        real(dp) :: k

        self%in_ch = in_ch
        self%out_ch = out_ch
        self%kH = kernel_size
        self%kW = kernel_size
        
        if (present(stride)) self%stride = stride
        if (present(padding)) self%pad = padding

        ! 分配内存 (kH, kW, in_ch, out_ch)
        allocate(self%W(self%kH, self%kW, self%in_ch, self%out_ch))
        allocate(self%b(self%out_ch))
        allocate(self%dW(self%kH, self%kW, self%in_ch, self%out_ch))
        allocate(self%db(self%out_ch))

        ! He Uniform 初始化
        k = sqrt(6.0_dp / real(self%in_ch * self%kH * self%kW, dp))
        call random_number(self%W)
        self%W = k * (2.0_dp * self%W - 1.0_dp)

        self%b = 0.0_dp
        call self%zero_grads()
    end subroutine conv_init

    ! -------------------------------------------------------------------------
    ! 保存权重
    ! -------------------------------------------------------------------------
    subroutine conv_save(self, filename)
        class(ConvLayer), intent(in) :: self
        character(len=*), intent(in) :: filename
        integer :: unit
        open(newunit=unit, file=filename, status='replace', action='write', form='formatted')
        write(unit, *) self%out_ch, self%in_ch, self%kH, self%kW
        ! 直接写入整个数组
        write(unit, *) self%W
        write(unit, *) self%b
        close(unit)
    end subroutine conv_save

    ! -------------------------------------------------------------------------
    ! 加载权重
    ! -------------------------------------------------------------------------
    subroutine conv_load(self, filename)
        class(ConvLayer), intent(inout) :: self
        character(len=*), intent(in) :: filename
        integer :: unit
        integer :: out_ch_read, in_ch_read, kH_read, kW_read
        open(newunit=unit, file=filename, status='old', action='read', form='formatted')
        read(unit, *) out_ch_read, in_ch_read, kH_read, kW_read
        if (out_ch_read /= self%out_ch .or. in_ch_read /= self%in_ch .or. &
            kH_read /= self%kH .or. kW_read /= self%kW) then
            print *, "Error: Dimensions in file do not match layer dimensions."
            close(unit)
            return
        end if
        ! 直接读取整个数组
        read(unit, *) self%W
        read(unit, *) self%b
        close(unit)
        call self%zero_grads()
    end subroutine conv_load

    ! -------------------------------------------------------------------------
    ! 前向传播
    ! x: 输入 (H_in, W_in, C_in, N)
    ! 返回: 输出 (H_out, W_out, C_out, N)
    ! -------------------------------------------------------------------------
    function conv_forward(self, x) result(y)
        class(ConvLayer), intent(inout) :: self
        real(dp), intent(in) :: x(:,:,:,:)
        real(dp), allocatable :: y(:,:,:,:)
        
        integer :: H, W, C, N, H_out, W_out
        integer :: n_idx, cout, cin, h_idx, w_idx, kh_idx, kw_idx
        integer :: h_in_start, w_in_start
        real(dp), allocatable :: x_pad(:,:,:,:)

        H = size(x, 1)
        W = size(x, 2)
        C = size(x, 3)
        N = size(x, 4)

        H_out = (H + 2 * self%pad - self%kH) / self%stride + 1
        W_out = (W + 2 * self%pad - self%kW) / self%stride + 1

        allocate(y(H_out, W_out, self%out_ch, N))
        
        ! Padding
        allocate(x_pad(H + 2*self%pad, W + 2*self%pad, C, N))
        x_pad = 0.0_dp
        x_pad(self%pad+1:self%pad+H, self%pad+1:self%pad+W, :, :) = x

        if (allocated(self%x_cache)) deallocate(self%x_cache)
        self%x_cache = x

        ! 卷积: 循环顺序优化 (N -> C_out -> W_out -> H_out)
        do n_idx = 1, N
            do cout = 1, self%out_ch
                do w_idx = 1, W_out
                    do h_idx = 1, H_out
                        h_in_start = (h_idx - 1) * self%stride + 1
                        w_in_start = (w_idx - 1) * self%stride + 1
                        
                        ! 使用 sum 和 spread 进行向量化计算
                        y(h_idx, w_idx, cout, n_idx) = sum( &
                            x_pad(h_in_start : h_in_start + self%kH - 1, &
                                  w_in_start : w_in_start + self%kW - 1, &
                                  :, n_idx) * &
                            self%W(:,:,:,cout) &
                        )
                    end do
                end do
                ! 添加偏置 (向量化)
                y(:, :, cout, n_idx) = y(:, :, cout, n_idx) + self%b(cout)
            end do
        end do
        
        deallocate(x_pad)
    end function conv_forward

    ! -------------------------------------------------------------------------
    ! 反向传播
    ! grad_output: (H_out, W_out, C_out, N)
    ! 返回: grad_input (H_in, W_in, C_in, N)
    ! -------------------------------------------------------------------------
    function conv_backward(self, grad_output) result(grad_input)
        class(ConvLayer), intent(inout) :: self
        real(dp), intent(in) :: grad_output(:,:,:,:)
        real(dp), allocatable :: grad_input(:,:,:,:)
        
        integer :: H, W, C, N, H_out, W_out
        integer :: n_idx, cout, cin, h_idx, w_idx, kh_idx, kw_idx
        integer :: h_in_start, w_in_start
        real(dp), allocatable :: x_pad(:,:,:,:), dx_pad(:,:,:,:)

        H = size(self%x_cache, 1)
        W = size(self%x_cache, 2)
        C = size(self%x_cache, 3)
        N = size(self%x_cache, 4)
        
        H_out = size(grad_output, 1)
        W_out = size(grad_output, 2)

        allocate(x_pad(H + 2*self%pad, W + 2*self%pad, C, N))
        x_pad = 0.0_dp
        x_pad(self%pad+1:self%pad+H, self%pad+1:self%pad+W, :, :) = self%x_cache

        allocate(dx_pad(H + 2*self%pad, W + 2*self%pad, C, N))
        dx_pad = 0.0_dp

        ! 1. 偏置梯度 (对 H, W, N 求和)
        self%db = self%db + sum(sum(sum(grad_output, dim=4), dim=2), dim=1)

        ! 2. 权重梯度和输入梯度
        do n_idx = 1, N
            do cout = 1, self%out_ch
                do w_idx = 1, W_out
                    do h_idx = 1, H_out
                        h_in_start = (h_idx - 1) * self%stride + 1
                        w_in_start = (w_idx - 1) * self%stride + 1
                        
                        ! 权重梯度 dW
                        self%dW(:,:,:,cout) = self%dW(:,:,:,cout) + &
                            x_pad(h_in_start : h_in_start + self%kH - 1, &
                                  w_in_start : w_in_start + self%kW - 1, &
                                  :, n_idx) * &
                            grad_output(h_idx, w_idx, cout, n_idx)
                        
                        ! 输入梯度 dx_pad
                        dx_pad(h_in_start : h_in_start + self%kH - 1, &
                               w_in_start : w_in_start + self%kW - 1, &
                               :, n_idx) = &
                            dx_pad(h_in_start : h_in_start + self%kH - 1, &
                                   w_in_start : w_in_start + self%kW - 1, &
                                   :, n_idx) + &
                            self%W(:,:,:,cout) * grad_output(h_idx, w_idx, cout, n_idx)
                    end do
                end do
            end do
        end do

        ! 去除 Padding
        allocate(grad_input(H, W, C, N))
        grad_input = dx_pad(self%pad+1:self%pad+H, self%pad+1:self%pad+W, :, :)

        deallocate(x_pad, dx_pad)
    end function conv_backward

    ! -------------------------------------------------------------------------
    ! 清零梯度
    ! -------------------------------------------------------------------------
    subroutine conv_zero_grads(self)
        class(ConvLayer), intent(inout) :: self
        if (allocated(self%dW)) self%dW = 0.0_dp
        if (allocated(self%db)) self%db = 0.0_dp
    end subroutine conv_zero_grads
    ! -------------------------------------------------------------------------
    ! 更新参数 (SGD)
    ! -------------------------------------------------------------------------
    subroutine conv_update(self, lr)
        class(ConvLayer), intent(inout) :: self
        real(dp), intent(in) :: lr
        
        if (allocated(self%W)) self%W = self%W - lr * self%dW
        if (allocated(self%b)) self%b = self%b - lr * self%db
    end subroutine conv_update

    ! -------------------------------------------------------------------------
    ! 销毁层，释放内存
    ! -------------------------------------------------------------------------
    subroutine conv_destroy(self)
        class(ConvLayer), intent(inout) :: self
        if (allocated(self%W)) deallocate(self%W)
        if (allocated(self%b)) deallocate(self%b)
        if (allocated(self%dW)) deallocate(self%dW)
        if (allocated(self%db)) deallocate(self%db)
        if (allocated(self%x_cache)) deallocate(self%x_cache)
    end subroutine conv_destroy

end module Conv_mod
