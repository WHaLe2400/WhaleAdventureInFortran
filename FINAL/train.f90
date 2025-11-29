Module Train_mod
    subroutine init()
        model%init()
        loss_func%destroy()
        train_data_loader%init()
        train_label_loader%init()
        test_data_loader%init()
        test_label_loader%init()
    end subroutine init
end Module Train_mod

program Train
    use iso_fortran_env, only: dp => real64
    use Train_mod
    use ModelCombine_mod
    use LoadData_mod
    use LossFunc_mod

    implicit none
    type(Model) :: model
    type(LossFunc) :: loss_func
    type(Data_Loader) :: train_data_loader, train_label_loader, test_data_loader, test_label_loader
    character(len=200) :: file_root = "/root/0_FoRemote/WhaleAdventureInFortran/FINAL/1_DATA_Reread"
    character(len=200) :: train_data_path = file_root // "/train-images3-.bin"
    character(len=200) :: train_label_path = file_root // "/train-labels1-.bin"
    character(len=200) :: test_data_path = file_root // "/t10k-images3-.binn"
    character(len=200) :: test_label_path = file_root // "/t10k-labels1-.bin"

    integer ::  epoch = 16,&
                batch_size = 64 & 
    real ::     learning_rate = 0.01
    

end program Train