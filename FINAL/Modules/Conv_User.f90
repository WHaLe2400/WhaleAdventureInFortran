module Conv_mod
    ! 卷积层 (CNN) 实现，遵循 Fortran 2008 标准
    ! 逻辑与 PyTorch Conv2d 保持一致 (N, C, H, W)
    use iso_fortran_env, only: dp => real64
    implicit none
    private

    public :: ConvLayer

    type :: ConvLayer
        private
        integer :: in_ch = 0, out_ch = 0, kH = 0, kW = 0
        integer :: stride = 1, pad = 0
        
        ! 参数: 权重 (out_ch, in_ch, kH, kW), 偏置 (out_ch)
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
    ! in_ch: 输入通道数
    ! out_ch: 输出通道数
    ! kernel_size: 卷积核大小 (假设为正方形)
    ! stride: 步长 (默认为 1)
    ! padding: 填充 (默认为 0)
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
        
        if (present(stride)) then
            self%stride = stride
        else
            self%stride = 1
        end if
        
        if (present(padding)) then
            self%pad = padding
        else
            self%pad = 0
        end if

        ! 分配内存
        allocate(self%W(self%out_ch, self%in_ch, self%kH, self%kW))
        allocate(self%b(self%out_ch))
        allocate(self%dW(self%out_ch, self%in_ch, self%kH, self%kW))
        allocate(self%db(self%out_ch))

        ! 初始化权重 (使用 Kaiming/He 初始化，适合 ReLU)
        ! PyTorch 默认是 Uniform(-sqrt(k), sqrt(k)) where k = 1/(in_ch*kernel*kernel)
        ! 这里我们使用 He 初始化: Normal(0, sqrt(2/n)) 或 Uniform 变体
        ! He Uniform: Uniform(-bound, bound), bound = sqrt(6 / fan_in)
        k = sqrt(6.0_dp / real(self%in_ch * self%kH * self%kW, dp))
        call random_number(self%W)
        self%W = k * (2.0_dp * self%W - 1.0_dp) ! Uniform(-k, k)

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

    ! -------------------------------------------------------------------------
    ! 前向传播
    ! x: 输入数据 (N, C_in, H_in, W_in)
    ! 返回: 输出数据 (N, C_out, H_out, W_out)
    ! -------------------------------------------------------------------------
    function conv_forward(self, x) result(y)
        class(ConvLayer), intent(inout) :: self
        real(dp), intent(in) :: x(:,:,:,:)
        real(dp), allocatable :: y(:,:,:,:)
        
        integer :: N, C, H, W, H_out, W_out
        integer :: n_idx, cout, cin, h_idx, w_idx, kh_idx, kw_idx
        integer :: h_in, w_in
        real(dp), allocatable :: x_pad(:,:,:,:)

        N = size(x, 1)
        C = size(x, 2)
        H = size(x, 3)
        W = size(x, 4)

        ! 计算输出尺寸
        H_out = (H + 2 * self%pad - self%kH) / self%stride + 1
        W_out = (W + 2 * self%pad - self%kW) / self%stride + 1

        allocate(y(N, self%out_ch, H_out, W_out))
        y = 0.0_dp

        ! 处理 Padding
        allocate(x_pad(N, C, H + 2*self%pad, W + 2*self%pad))
        x_pad = 0.0_dp
        x_pad(:, :, self%pad+1:self%pad+H, self%pad+1:self%pad+W) = x

        ! 缓存输入用于反向传播 (保存 padding 后的还是原始的? 通常保存原始的比较省内存，但为了方便这里保存原始的)
        if (allocated(self%x_cache)) deallocate(self%x_cache)
        allocate(self%x_cache, source=x)

        ! 卷积运算 (朴素实现，可优化)
        ! 循环顺序: Batch -> OutChannel -> OutH -> OutW -> InChannel -> KH -> KW
        do n_idx = 1, N
            do cout = 1, self%out_ch
                do h_idx = 1, H_out
                    do w_idx = 1, W_out
                        ! 计算输入特征图上的起始位置
                        h_in = (h_idx - 1) * self%stride + 1
                        w_in = (w_idx - 1) * self%stride + 1
                        
                        do cin = 1, self%in_ch
                            do kh_idx = 1, self%kH
                                do kw_idx = 1, self%kW
                                    y(n_idx, cout, h_idx, w_idx) = y(n_idx, cout, h_idx, w_idx) + &
                                        x_pad(n_idx, cin, h_in + kh_idx - 1, w_in + kw_idx - 1) * &
                                        self%W(cout, cin, kh_idx, kw_idx)
                                end do
                            end do
                        end do
                        ! 加上偏置
                        y(n_idx, cout, h_idx, w_idx) = y(n_idx, cout, h_idx, w_idx) + self%b(cout)
                    end do
                end do
            end do
        end do
        
        deallocate(x_pad)
    end function conv_forward

    ! -------------------------------------------------------------------------
    ! 反向传播
    ! grad_output: 输出梯度 (N, C_out, H_out, W_out)
    ! 返回: 输入梯度 (N, C_in, H_in, W_in)
    ! -------------------------------------------------------------------------
    function conv_backward(self, grad_output) result(grad_input)
        class(ConvLayer), intent(inout) :: self
        real(dp), intent(in) :: grad_output(:,:,:,:)
        real(dp), allocatable :: grad_input(:,:,:,:)
        
        integer :: N, C, H, W, H_out, W_out
        integer :: n_idx, cout, cin, h_idx, w_idx, kh_idx, kw_idx
        integer :: h_in, w_in
        real(dp), allocatable :: x_pad(:,:,:,:), dx_pad(:,:,:,:)

        N = size(self%x_cache, 1)
        C = size(self%x_cache, 2)
        H = size(self%x_cache, 3)
        W = size(self%x_cache, 4)
        
        H_out = size(grad_output, 3)
        W_out = size(grad_output, 4)

        ! 恢复 Padding 后的输入
        allocate(x_pad(N, C, H + 2*self%pad, W + 2*self%pad))
        x_pad = 0.0_dp
        x_pad(:, :, self%pad+1:self%pad+H, self%pad+1:self%pad+W) = self%x_cache

        ! 准备 Padding 后的输入梯度
        allocate(dx_pad(N, C, H + 2*self%pad, W + 2*self%pad))
        dx_pad = 0.0_dp

        ! 计算梯度
        ! dL/dW = sum(dL/dy * x)
        ! dL/dx = sum(dL/dy * W)
        ! dL/db = sum(dL/dy)
        
        ! 1. 偏置梯度 (对 Batch, H, W 求和)
        self%db = self%db + sum(sum(sum(grad_output, dim=4), dim=3), dim=1)

        ! 2. 权重梯度和输入梯度
        do n_idx = 1, N
            do cout = 1, self%out_ch
                do h_idx = 1, H_out
                    do w_idx = 1, W_out
                        h_in = (h_idx - 1) * self%stride + 1
                        w_in = (w_idx - 1) * self%stride + 1
                        
                        do cin = 1, self%in_ch
                            do kh_idx = 1, self%kH
                                do kw_idx = 1, self%kW
                                    ! 累加权重梯度
                                    self%dW(cout, cin, kh_idx, kw_idx) = self%dW(cout, cin, kh_idx, kw_idx) + &
                                        grad_output(n_idx, cout, h_idx, w_idx) * &
                                        x_pad(n_idx, cin, h_in + kh_idx - 1, w_in + kw_idx - 1)
                                    
                                    ! 累加输入梯度
                                    dx_pad(n_idx, cin, h_in + kh_idx - 1, w_in + kw_idx - 1) = &
                                        dx_pad(n_idx, cin, h_in + kh_idx - 1, w_in + kw_idx - 1) + &
                                        grad_output(n_idx, cout, h_idx, w_idx) * &
                                        self%W(cout, cin, kh_idx, kw_idx)
                                end do
                            end do
                        end do
                    end do
                end do
            end do
        end do

        ! 去除 Padding，得到原始输入的梯度
        allocate(grad_input(N, C, H, W))
        grad_input = dx_pad(:, :, self%pad+1:self%pad+H, self%pad+1:self%pad+W)

        deallocate(x_pad)
        deallocate(dx_pad)
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
