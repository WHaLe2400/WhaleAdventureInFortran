module LossFunc_mod
    use iso_fortran_env, only: dp => real64
    implicit none
    private

    public :: LossFunc

    type, public :: LossFunc
    private
        ! 用于反向传播的缓存
        real(dp), allocatable :: softmax_cache(:,:)
        real(dp), allocatable :: labels_cache(:,:)  ! 修正：改为2D以存储 one-hot 标签
        integer :: batch_size_cache = 0
    contains
        procedure, public :: forward => loss_forward
        procedure, public :: backward => loss_backward
        procedure, public :: destroy => loss_destroy
    end type LossFunc

contains

    !> @brief 计算 Softmax 和交叉熵损失
    !! @param self 损失函数对象
    !! @param logits 模型的原始输出 (批量大小, 类别数)
    !! @param labels 真实的标签 (批量大小, 类别数), one-hot 编码
    !! @return loss 标量损失值
    function loss_forward(self, logits, labels) result(loss)
        class(LossFunc), intent(inout) :: self
        real(dp), intent(in) :: logits(:,:)
        real(dp), intent(in) :: labels(:,:)  ! One-hot encoded labels (batch_size, num_classes)
        real(dp) :: loss

        integer :: batch_size, num_classes, i
        real(dp), allocatable :: max_logits(:), exp_logits(:,:)
        real(dp), allocatable :: sum_exp_logits(:)

        batch_size = size(logits, 1)
        num_classes = size(logits, 2)

        ! --- 缓存数据以备反向传播使用 ---
        self%batch_size_cache = batch_size
        if (allocated(self%labels_cache)) deallocate(self%labels_cache)
        allocate(self%labels_cache(batch_size, num_classes))  ! 修正：2D分配
        self%labels_cache = labels  ! 现在兼容

        ! --- 实现数值稳定的 Softmax ---
        ! 1. 找到每个样本 logits 的最大值
        allocate(max_logits(batch_size))
        do i = 1, batch_size
            max_logits(i) = maxval(logits(i, :))
        end do

        ! 2. 从 logits 中减去最大值以防止 exp() 溢出
        allocate(exp_logits(batch_size, num_classes))
        exp_logits = exp(logits - spread(max_logits, dim=2, ncopies=num_classes))

        ! 3. 计算每个样本的 exp 总和
        allocate(sum_exp_logits(batch_size))
        do i = 1, batch_size
            sum_exp_logits(i) = sum(exp_logits(i, :))
        end do

        ! 4. 计算 Softmax 概率并缓存
        if (allocated(self%softmax_cache)) deallocate(self%softmax_cache)
        allocate(self%softmax_cache(batch_size, num_classes))
        self%softmax_cache = exp_logits / spread(sum_exp_logits, dim=2, ncopies=num_classes)

        ! --- 计算交叉熵损失 ---
        loss = -sum(labels * log(self%softmax_cache + 1e-9_dp))

        ! 返回平均损失
        loss = loss / real(batch_size, dp)

        deallocate(max_logits, exp_logits, sum_exp_logits)
    end function loss_forward

    !> @brief 计算损失相对于模型输出(logits)的梯度
    !! @param self 损失函数对象
    !! @return grad_logits 损失对 logits 的梯度
    function loss_backward(self) result(grad_logits)
        class(LossFunc), intent(inout) :: self
        real(dp), allocatable :: grad_logits(:,:)

        if (.not. allocated(self%softmax_cache)) then
            print *, "LossFunc Error: backward() called before forward()."
            stop
        end if

        ! 梯度是 (softmax概率 - one-hot标签)，向量化计算
        grad_logits = (self%softmax_cache - self%labels_cache) / real(self%batch_size_cache, dp)

    end function loss_backward

    !> @brief 释放缓存的内存
    subroutine loss_destroy(self)
        class(LossFunc), intent(inout) :: self
        if (allocated(self%softmax_cache)) deallocate(self%softmax_cache)
        if (allocated(self%labels_cache)) deallocate(self%labels_cache)
        self%batch_size_cache = 0
    end subroutine loss_destroy

end module LossFunc_mod