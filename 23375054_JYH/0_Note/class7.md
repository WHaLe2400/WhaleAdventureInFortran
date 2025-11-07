# 子程序

*在fortran中程序传递变量使用的是参数的地址，而不是变量的值，因此一定会对原有的值产生修改*

## 函数（function）
有回归值，用法：  
```fortran
result = function_name(arg1, arg2, ...)
```

定义方式：
```fortran
function function_name(arg1, arg2, ...) result(result_name)
    ! 声明参数和变量
    implicit none
    ! 参数声明
    integer :: arg1, arg2
    ! 结果声明
    integer :: result_name    
    ! 函数体
    result_name = arg1 + arg2
end function function_name
```

或者可以写作
```fortran
real function function_name(arg1, arg2, ...)
```
直接将function_name作为结果返回。

## 子例行程序（subroutine）

定义方式
```fortran
subroutine subroutine_name(arg1, arg2)
    ! 声明参数和变量
    implicit none
    ! 参数声明
    integer :: arg1, arg2
    ! 结果声明
    integer :: result_name
    ! 函数体
    result_name = arg1 + arg2
end subroutine subroutine_name
```

传递中可以没有形参，使用时通过`call subroutine_name()`调用。

    *`interface`方法？？？*


## 模块（module）
定义方式
```fortran
module module_name
    implicit none
    ! 声明变量、参数、子程序等
contains
    ! 定义子程序和函数
end module module_name
```

**注意**：在module中定义的变量会被初始化为0或空字符串。而在程序中定义的变量则是未初始化的，可能包含垃圾值。

通过模块实现面向对象编程：

下面给出一个小例子，演示如何用模块、派生类型（type extends）以及类型绑定过程（type-bound procedures）实现面向对象风格的代码。

```fortran
module ocean_mod
    implicit none
    private
    public :: animal, whale, make_whale

    type, public :: animal
        character(len=20) :: name = ''
    contains
        procedure :: speak => animal_speak
    end type animal

    type, extends(animal) :: whale
        real :: weight = 0.0
    contains
        procedure :: speak => whale_speak
    end type whale

contains

    subroutine animal_speak(this)
        class(animal), intent(in) :: this
        print '(A,1X,A)', 'Animal:', trim(this%name)
    end subroutine animal_speak

    subroutine whale_speak(this)
        class(whale), intent(in) :: this
        print '(A,1X,A,1X,A,F8.2)', 'Whale:', trim(this%name), ' Weight=', this%weight
    end subroutine whale_speak

    function make_whale(nm, w) result(wl)
        character(len=*), intent(in) :: nm
        real, intent(in) :: w
        type(whale) :: wl
        wl%name = nm
        wl%weight = w
    end function make_whale

end module ocean_mod

program test_oop
    use ocean_mod
    implicit none
    type(whale) :: moby

    moby = make_whale('Moby', 15000.0)
    call moby%speak()    ! 调用 whale 的类型绑定子程序（重写了 animal 的 speak）

    ! 也可以将 whale 看作更通用的 animal（多态调用）
    call (moby)%speak()
end program test_oop
```

要点说明：
- 使用 `type, extends(base)` 来实现继承。
- 在 `contains` 中用 `procedure :: name => proc_name` 将子程序绑定到类型上（type-bound procedure）。
- 使用 `class(type)` 可以实现多态参数，派生类型对象可传递给基类型接口。
- 在 `program` 中使用 `call obj%method()` 来调用类型绑定的过程。

可用命令编译运行该示例（在包含程序文件的目录下）:

```bash
gfortran -std=f2003 test_oop.f90 -o test_oop
./test_oop
```

输出示例应类似：

```
Whale: Moby  Weight=15000.00
```

这个例子是一个小型、易懂的面向对象示范；如果你希望我把示例另存为单独的 `.f90` 文件（并帮你编译运行），我可以继续执行。
