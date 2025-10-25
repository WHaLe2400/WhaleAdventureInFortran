program test
    implicit none
    integer :: i, j
    integer, parameter :: rows = 28, cols = 28
    integer, parameter :: imgsize = rows*cols
    integer :: idx       ! 要读取的图片索引（1-based）
    integer(kind=1), dimension(imgsize) :: data
    character(len=200) :: filename
    integer :: iostat
    character(len = cols*2+1), dimension(rows) :: WhatAImage

    filename = '/root/0_FoRemote/WhaleAdventureInFortran/' // &
               '23375054_JYH/23375054JinYuHao_Final/1_DATA_Reread/t10k-images3-.bin'

    open(unit=101, file=filename, form='unformatted', access='stream', status='old', iostat=iostat)
    if (iostat /= 0) then
        print *, 'Error opening file. iostat=', iostat
        stop
    end if

    ! 指定要读取的图片索引（1-based），可以改为从用户读取
    idx = 128
    if (idx < 1) then
        print *, 'Invalid index'
        stop
    end if

    ! stream 访问中，pos 按字节偏移（这里每像素为 1 byte）
    read(101, pos=(idx-1)*imgsize + 1, iostat=iostat) data
    if (iostat /= 0) then
        if (iostat < 0) then
            print *, 'End of file reached.'
        else
            print *, 'Error reading file. iostat=', iostat
        end if
        close(101)
        stop
    end if

    ! 将每行像素转换为字符画并输出
    do i = 1, rows
        WhatAImage(i) = ''
        do j = 1, cols
            ! data 索引 (i-1)*cols + j
            if ( iand(int(data((i-1)*cols + j)), 255) > 64 ) then
                WhatAImage(i) = trim(WhatAImage(i)) // '##'
            else
                WhatAImage(i) = trim(WhatAImage(i)) // '--'
            end if
        end do
        print *, WhatAImage(i)
    end do

    close(101)
end program test