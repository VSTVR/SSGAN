import argparse
import os

from model import Model
import datasets.hdf5_loader as dataset


def argparser(is_train=True):

    def str2bool(v):
        return v.lower() == 'true'
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--prefix', type=str, default='default')
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None, 
                        choices=["mean_excep_nl_nn","mean_std_all","min_max_all","raw","raw_excep_nl_n"])
    parser.add_argument('--dump_result', type=str2bool, default=False)#保存保存点生成图像的h5py
    # Model
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_z', type=int, default=128) #噪声维度
    parser.add_argument('--norm_type', type=str, default='batch',
                        choices=['batch', 'instance', 'None'])
    parser.add_argument('--deconv_type', type=str, default='bilinear',
                        choices=['bilinear', 'nn', 'transpose'])

    # Training config {{{
    # ========
    # log
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--write_summary_step', type=int, default=200)
    #parser.add_argument('--ckpt_save_step', type=int, default=1)
    parser.add_argument('--test_sample_step', type=int, default=200)#这个训练阶段的测试集就是验证集
    parser.add_argument('--output_save_step', type=int, default=500)
    # learning
    #parser.add_argument('--max_sample', type=int, default=200, 
    #                    help='num of samples the model can see')
    parser.add_argument('--max_training_steps', type=int, default=30001)
    parser.add_argument('--learning_rate_g', type=float, default=2e-4)
    parser.add_argument('--learning_rate_d', type=float, default=1e-4)
    parser.add_argument('--update_rate', type=int, default=2) #这不是学习率的改变，而是决定生成器和辨别器训练时机的参数（比如训练一次生成器再训练一次辨别器）。
    # }}}

    # Testing config {{{
    # ========
    parser.add_argument('--data_id', nargs='*', default=None)
    # }}}

    config = parser.parse_args()

    dataset_path = os.path.join('drive/Colab_Notebooks/SSGAN-Tensorflow-master/datasets', config.dataset.lower())
    dataset_train, dataset_test = dataset.create_default_splits(dataset_path)
    dataset_train_unlabel, _ = dataset.create_default_splits_unlabel(dataset_path)

    img, label = dataset_train.get_data(dataset_train.ids[0])
    config.h = img.shape[0]
    config.w = img.shape[1]
    config.c = img.shape[2]
    config.num_class = label.shape[0] # 5

    # --- create model ---
    model = Model(config, debug_information=config.debug, is_train=is_train)

    return config, model, dataset_train, dataset_train_unlabel, dataset_test
