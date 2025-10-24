program read_uint8_bin
    implicit none
    integer, parameter :: bufsize = 1e5
    integer(kind=1), dimension(bufsize) :: data
    integer :: iunit, n, i
    character(len=200) :: filename

    filename = '/root/0_FoRemote/WhaleAdventureInFortran/' // &
    '23375054_JYH/23375054JinYuHao_Final/1_DATA_Reread/t10k-images3-.bin'
    open(unit=10, file=filename, form='unformatted', access='stream', status='old', iostat=iunit)
    if (iunit /= 0) then
        print *, 'Error opening file.'
        stop
    end if

    ! 读取指定图片（28x28），以1为起始索引
    print *, 'Enter image index (1-based):'
    read (*, *) i
    n = 28*28
    ! POS 从1开始，因此偏移为 (index-1)*784 + 1
    read(10, pos=(i-1)*n + 1, iostat=iunit) data(1:n)
    if (iunit < 0) then
        print *, 'End of file reached.'
    else if (iunit > 0) then
        print *, 'Error reading file.'
        stop
    else
        n = size(data)
        ! data 是有符号的 integer(kind=1)，转换为无符号 0..255 输出
        print *, 'Read', n, 'items. First 10:', [( iand(int(data(i), kind=4), 255), i=1, min((28*28),n) )]
    end if
    close(10)
end program read_uint8_bin