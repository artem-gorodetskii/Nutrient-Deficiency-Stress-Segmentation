# -*- coding: utf-8 -*-
# @Author: Artem Gorodetskii
# @Created Time: 3/22/2022 4:45 PM

class ModelConfig:
    """Configuration of Segmentation model. """

    # model
    input_channels = 9 # number of input chanels
    size = 512 # input image size
    norm_type = 'GN' # type of normalization. GN - Group Normalization, BN - Batch Noramlization

    # data
    data_path = 'Longitudinal_Nutrient_Deficiency' # path to dataset
    train_vaild_split_ratio = 0.15 # train-validation split ration

    # input normalization
    channels_avgs = [0.30222556, 0.56079715, 0.36130947, 0.26359808, 0.4109688, 0.37013307, 0.28755838, 0.5263754, 0.36376357] 
    channels_stds = [0.05933824, 0.08243724, 0.04012331, 0.08589337, 0.16466905, 0.061317742, 0.048356656, 0.058588393, 0.028817762]

    # training
    initial_lr = 0.001 # inital learning rate
    weight_decay = 1e-6
    n_epochs = 100
    grad_clip = 1.0
    gamma = 0.5
    milestones = [100, 300, 600, 900] # in steps
    log_every = 40 # in steps
    test_every = 40 # in steps
    BS = 2 # batch size
    savepath = '/pretrained/best_model_' + str(size) + '_' + str(BS) + '.pt' # path to save backups
    loadpath = 'pretrained/pretrained.pt' # path used to load pretrained model
    logs_dir = 'tb_logs' # path for tensorboard logs
    load_backup = False # if true the backup will be loaded before training
