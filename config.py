PRETRAIN_CONFIG = {
    'lr': 0.1, #initial learning rate
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'batch_size': 128,
    'epochs': 200,
    'iabn': True,
    'alpha': 4,
    'memory_type': "PBRS",
    'capacity': 64,
}

ONLINE_CONFIG = {
    "epochs": 1,
    "lr": 0.0001,
    "weight_decay": 0,
    "batch_size": 128,
    "use_learned_stats": True,
    'bn_momentum': 0.01,
    "update_interval": 64,
    "temp_factor": 1.0,
    "optimize": True,
}

# Config for CIFAR10, mostly for target domain
CIFAR10_CONFIG = {
    'file_path': './dataset/CIFAR-10-C',
    'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    'num_class': 10,
    'severity': 5,
    'distribution': 'dirichlet',
    'dir_beta': 0.1,
    'shuffle_criterion': 'class',
}