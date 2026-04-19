from train_test import train_test

if __name__ == "__main__":
    data_folder = 'LGG2'
    view_list = [1, 2, 3]
    num_epoch_pretrain = 500
    num_epoch = 2500
    lr_e_pretrain = 1e-4
    lr_e = 5e-4
    lr_c = 1e-3


    if data_folder == 'LGG2':
        num_class = 2

    for i in range(25):
        train_test(data_folder, view_list, num_class,
                   lr_e_pretrain, lr_e, lr_c,
                   num_epoch_pretrain, num_epoch)
