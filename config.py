cfg = {
    
    'trainer':{
        'dataset':'/data/ysz_data/CLIC2020/',
        'train_split':'train',
        'test_split':'kodak',
        'num_workers':8,
        'epoches':100,
        'batch_size':16,
        'test_batch_size':1,
        'patch_size':(256,256),
        'lmbda':0.0250,
         #0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483, 0.0932, 0.18
        'cuda':True,
        'save_path':'/home/yinsz/dic_ac_allon_spaEG/ckpt/',
        'log':'./logdir/',
        'seed':100,
        'clip_max_norm':1.0,
        'pretrained':False,
    },

    'acceleration':{
        'if_acceleration':False,
        'acceleration_epoch':30,
        'weight_bit_depth':8,
        'activation_bit_depth':8,
    },
    'nm_sparse':{
        'if_nm_sparse':False,
        'sparse_epoch':10,
        'sparse_N':2,
        'sparse_M':4,
    },
    'compress_activation':{
        'if_compress_activation':True,
        'compression_bit_depth':8,
        'use_affine':True,
        'ep':'Sparse_EG', # 'Symmetrical_EG'
        'use_rans':False,
        'use_penalty':True,
        'penalty':1e-5,
        'adaptive_bit_depth_allocation':False,
        'adaptive_bit_depth_start': 3,
        'adaptive_bit_depth_end': 12,
    },

    'lr_schedule':{
        'lr':1e-4,
        'aux_lr':1e-3,
    },
    
}