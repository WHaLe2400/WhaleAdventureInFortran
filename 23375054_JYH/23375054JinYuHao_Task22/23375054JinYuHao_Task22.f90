program Task22
    implicit none
    character(len=5) :: grad_str
    character(len=5) :: level_str
    integer :: grad_int
    print *, '请输入成绩(百分制或字母等级):'
    read *, grad_str
    if (verify(trim(grad_str), '0123456789') == 0) then
        ! grad_str 全是数字，假设是百分制分数
        print *, '输入的是百分制分数: ', trim(grad_str)
        read(grad_str, *) grad_int
        if (grad_int >= 90) then
            level_str = '优秀'
        elseif (grad_int >= 60) then
            level_str = '通过'
        else
            level_str = '不及格'
        end if
    else
        ! grad_str 包含非数字字符，假设是字母等级
        print *, '输入的是字母等级: ', trim(grad_str)
        select case (trim(grad_str))
        case ('A', 'a')
            level_str = '优秀'
        case ('B', 'b', 'C', 'c')
            level_str = '通过'
        case default
            level_str = '不及格'
        end select

    
    end if
    print *, '等级: ', level_str

end program Task22