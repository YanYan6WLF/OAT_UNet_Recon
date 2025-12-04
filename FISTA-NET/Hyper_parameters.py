import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Hyper-parameters management')
    parser.add_argument('--data_path', type=str, default=r'D:\MATLAB_Yan\Test\Mouse_6_v7', help='path of dataset for trainning and validation')
    # parser.add_argument('--validate_path', type=str, default=r'D:\MATLAB_Yan\Test\Mouse_6_v7_outcome', help='path of validation')
    parser.add_argument('--test_data_path', type=str, default=r'D:\MATLAB_Yan\Test\Mouse_6_v7_topredict', help='path of predict ')
    parser.add_argument('--saved_model_path', type=str, default='Reg_net_iter1_regularization.ckpt', help='file name of saved model')
    parser.add_argument('--save_test_path', type=str, default=r'D:\MATLAB_Yan\Test\Mouse_6_v7_save', help='path for saving results of testing')
    parser.add_argument('--save_validation_path', type=str, default=r'D:\MATLAB_Yan\Test\Mouse_6_v7_save', help='path for saving results of validation')

    parser.add_argument('--loadcp', type=bool, default=False, help='if load model')
    parser.add_argument('--resume', type=str, default='', help='path to resume checkpoint')

    parser.add_argument('--train_batch_size', type=int, default=32, help='batch_size of training')
    parser.add_argument('--validate_batch_size', type=int, default=12, help='batch_size of validation')
    parser.add_argument('--test_batch_size', type=int, default=12, help='batch_size of predicting')

    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--num_epochs', type=int, default=500, help='the number of epoches')
    parser.add_argument('--num_iters', type=int, default=10, help="number of total iterations")
    parser.add_argument('--size', type=int, default=125, help='image resolution in matlab')
    parser.add_argument('--tosize',type=int, default=125, help='resized image resolution')

    parser.add_argument('--learning_rate', type=float, default=0.004, help='learning rate')
    parser.add_argument('--weight_dacay',type=float, default=0.001, help='weight decay of optimizer learning rate')
    parser.add_argument('--L1_weight',type=float, default=0.1, help='L1_weight')
    parser.add_argument('--weight_loss_constraint',type=float, default=0.1, help='weight_loss_constraint')
    parser.add_argument('--weight_sparsity_constraint',type=float, default=0.1, help='weight_sparsity_constraint')
    parser.add_argument('--step_size',type=int, default=10, help='step size for lr update and printing processes')
    parser.add_argument('--gamma',type=float, default=0.9, help='gamma for lr update weigth')
    parser.add_argument('--data_range',type=int, default=1, help='data range of image for metric calculation')


    args = parser.parse_args()
    return args
