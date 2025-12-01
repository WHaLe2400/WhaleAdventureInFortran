module FullConnect_mod
    use iso_fortran_env, only: dp => real64
    implicit none
    private

    ! 公共接口
    public :: FullConnectLayer

    ! 全连接层的类型定义
    type, public :: FullConnectLayer
        private
        integer :: input_size = 0
        integer :: output_size = 0
        real(dp), allocatable :: weights(:, :)
        real(dp), allocatable :: biases(:)
        ! 用于批量反向传播
        real(dp), allocatable :: input_cache(:, :)
        real(dp), allocatable :: grad_weights(:, :)
        real(dp), allocatable :: grad_biases(:)
    contains
        procedure, public :: init => fc_init
        procedure, public :: load => fc_load
        procedure, public :: save => fc_save
        procedure, public :: forward => fc_forward
        procedure, public :: backward => fc_backward
        procedure, public :: update => fc_update
        procedure, public :: destroy => fc_destroy
        procedure, public :: zero_grads => fc_zero_grads ! 新增
        ! Getter 函数
        procedure, public :: get_input_size => fc_get_input_size
        procedure, public :: get_output_size => fc_get_output_size
        procedure, public :: get_weights => fc_get_weights
        procedure, public :: get_biases => fc_get_biases
        procedure, public :: get_grad_weights => fc_get_grad_weights
        procedure, public :: get_grad_biases => fc_get_grad_biases
    end type FullConnectLayer

contains

    ! 初始化全连接层的子程序
    subroutine fc_init(self, input_dim, output_dim)
        class(FullConnectLayer), intent(inout) :: self
        integer, intent(in) :: input_dim
        integer, intent(in) :: output_dim

        ! 1. 如果已分配则释放 (必须先做这一步)
        call self%destroy()

        ! 2. 设置新的尺寸
        self%input_size = input_dim
        self%output_size = output_dim

        ! 3. 分配权重和偏置
        allocate(self%weights(self%output_size, self%input_size))
        allocate(self%biases(self%output_size))

        ! 为梯度分配存储空间
        allocate(self%grad_weights(self%output_size, self%input_size))
        allocate(self%grad_biases(self%output_size))
        ! input_cache 在前向传播中动态分配

        ! 使用随机数初始化，并将梯度置零
        call random_number(self%weights)
        self%weights = self%weights * sqrt(2.0_dp / real(input_dim, dp)) ! Kaiming He 初始化
        self%biases = 0.0_dp
        call self%zero_grads() ! 调用新的清零函数
    end subroutine fc_init


    ! 加载权重和偏置的子程序
    subroutine fc_load(self, filename)
        class(FullConnectLayer), intent(inout) :: self
        character(len=*), intent(in) :: filename
        integer :: unit, i
        open(newunit=unit, file=filename, status='old', action='read', form='formatted')
        do i = 1, self%output_size
            read(unit, *) self%weights(i, :)
        end do
        read(unit, *) self%biases
        close(unit)
        ! 不分配梯度，假设init已处理；仅清零
        call self%zero_grads()
    end subroutine fc_load

    subroutine fc_save(self, filename)
        class(FullConnectLayer), intent(in) :: self
        character(len=*), intent(in) :: filename
        integer :: unit, i
        open(newunit=unit, file=filename, status='replace', action='write', form='formatted')
        do i = 1, self%output_size
            write(unit, *) self%weights(i, :)
        end do
        write(unit, *) self%biases
        close(unit)
    end subroutine fc_save


    ! 执行批量前向传播的函数
    function fc_forward(self, input_batch) result(output_batch)
        class(FullConnectLayer), intent(inout) :: self
        real(dp), intent(in) :: input_batch(:, :) ! 形状: (批量大小, 输入大小)
        real(dp), allocatable :: output_batch(:, :)   ! 形状: (批量大小, 输出大小)

        integer :: batch_size

        ! 检查维度不匹配
        if (size(input_batch, 2) /= self%input_size) then
            print *, "Error: Input batch feature size (", size(input_batch, 2), &
                     ") does not match layer input size (", self%input_size, ")."
            allocate(output_batch(0, 0))
            return
        end if

        batch_size = size(input_batch, 1)

        ! 缓存输入以用于反向传播
        if (allocated(self%input_cache)) deallocate(self%input_cache)
        allocate(self%input_cache(batch_size, self%input_size))
        self%input_cache = input_batch

        ! 分配输出
        allocate(output_batch(batch_size, self%output_size))

        ! 执行矩阵-矩阵乘法并加上偏置
        ! output = input * weights' + biases
        output_batch = matmul(input_batch, transpose(self%weights))
        output_batch = output_batch + spread(self%biases, dim=1, ncopies=batch_size)
    end function fc_forward

    ! 执行批量反向传播的函数
    function fc_backward(self, grad_output_batch) result(grad_input_batch)
        class(FullConnectLayer), intent(inout) :: self
        real(dp), intent(in) :: grad_output_batch(:, :) ! 损失函数关于层输出的梯度。形状: (批量大小, 输出大小)
        real(dp), allocatable :: grad_input_batch(:, :)   ! 损失函数关于层输入的梯度。形状: (批量大小, 输入大小)

        integer :: batch_size

        ! 检查维度不匹配
        if (size(grad_output_batch, 2) /= self%output_size) then
            print *, "Error: Gradient output batch feature size (", size(grad_output_batch, 2), &
                     ") does not match layer output size (", self%output_size, ")."
            allocate(grad_input_batch(0, 0))
            return
        end if
        
        batch_size = size(grad_output_batch, 1)
        if (size(self%input_cache, 1) /= batch_size) then
            print *, "Error: Gradient output batch size (", batch_size, &
                     ") does not match cached input batch size (", size(self%input_cache, 1), ")."
            allocate(grad_input_batch(0, 0))
            return
        end if

        ! 1. 计算关于权重的梯度 (dLoss/dW) 并累加
        ! dLoss/dW = dLoss/dOut^T * In = In^T * dLoss/dOut (转置)
        ! grad_output_batch: (批量, 输出大小), input_cache: (批量, 输入大小)
        ! grad_weights: (输出大小, 输入大小)
        self%grad_weights = self%grad_weights + matmul(transpose(grad_output_batch), self%input_cache)

        ! 2. 计算关于偏置的梯度 (dLoss/dB) 并累加
        ! 沿批量维度求和梯度
        self%grad_biases = self%grad_biases + sum(grad_output_batch, dim=1)

        ! 3. 计算关于输入的梯度 (dLoss/dIn)
        ! dLoss/dIn = dLoss/dOut * W
        allocate(grad_input_batch(batch_size, self%input_size))
        grad_input_batch = matmul(grad_output_batch, self%weights)

    end function fc_backward

    ! 更新权重和偏置的子程序
    subroutine fc_update(self, learning_rate)
        class(FullConnectLayer), intent(inout) :: self
        real(dp), intent(in) :: learning_rate

        ! 使用梯度下降更新权重和偏置
        self%weights = self%weights - learning_rate * self%grad_weights
        self%biases = self%biases - learning_rate * self%grad_biases

    end subroutine fc_update

    ! 释放内存的子程序
    subroutine fc_destroy(self)
        class(FullConnectLayer), intent(inout) :: self
        if (allocated(self%weights)) deallocate(self%weights)
        if (allocated(self%biases)) deallocate(self%biases)
        if (allocated(self%input_cache)) deallocate(self%input_cache)
        if (allocated(self%grad_weights)) deallocate(self%grad_weights)
        if (allocated(self%grad_biases)) deallocate(self%grad_biases)
        self%input_size = 0
        self%output_size = 0
    end subroutine fc_destroy

    ! 新增：将梯度清零的子程序
    subroutine fc_zero_grads(self)
        class(FullConnectLayer), intent(inout) :: self
        if (allocated(self%grad_weights)) then
            self%grad_weights = 0.0_dp
        end if
        if (allocated(self%grad_biases)) then
            self%grad_biases = 0.0_dp
        end if
    end subroutine fc_zero_grads

    ! --- Getter 函数 ---

    function fc_get_input_size(self) result(val)
        class(FullConnectLayer), intent(in) :: self
        integer :: val
        val = self%input_size
    end function fc_get_input_size

    function fc_get_output_size(self) result(val)
        class(FullConnectLayer), intent(in) :: self
        integer :: val
        val = self%output_size
    end function fc_get_output_size

    function fc_get_weights(self) result(arr)
        class(FullConnectLayer), intent(in) :: self
        real(dp), allocatable :: arr(:,:)
        if (allocated(self%weights)) then
            arr = self%weights
        end if
    end function fc_get_weights

    function fc_get_biases(self) result(arr)
        class(FullConnectLayer), intent(in) :: self
        real(dp), allocatable :: arr(:)
        if (allocated(self%biases)) then
            arr = self%biases
        end if
    end function fc_get_biases

    function fc_get_grad_weights(self) result(arr)
        class(FullConnectLayer), intent(in) :: self
        real(dp), allocatable :: arr(:,:)
        if (allocated(self%grad_weights)) then
            arr = self%grad_weights
        end if
    end function fc_get_grad_weights

    function fc_get_grad_biases(self) result(arr)
        class(FullConnectLayer), intent(in) :: self
        real(dp), allocatable :: arr(:)
        if (allocated(self%grad_biases)) then
            arr = self%grad_biases
        end if
    end function fc_get_grad_biases

end module FullConnect_mod
