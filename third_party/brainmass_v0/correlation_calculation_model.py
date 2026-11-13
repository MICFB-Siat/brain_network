from BrainMass.model import MLP1, MLP2, BrainFly
import numpy as np
from BrainMass.utils import BNTF
import torch
import yaml
import os
from nilearn.connectome import ConnectivityMeasure
import argparse
print('Calculate Brain Network!')

def get_datasets(FLAGS):
    datasets_dir = ''
    if FLAGS.datasets == 'PPMI_scheafer-200':
        print('Dataset : PPMI')
        datasets_dir = 'D:\\datasets\\PPMI_schaefer-200'
    if FLAGS.datasets == 'PPMI_Our':
        print('Dataset : PPMI')
        datasets_dir = 'D:\\datasets\\PPMI_Our'
    return datasets_dir
config_path = 'D:\\Python Project\\Personalized_Functional_BrainNetwork_Construction\\Stage3-Adaptive_Correlation_Calculation_Train\\BrainMass\\config\\config.yaml'
save_path = 'D:\\datasets\\PPMI_FBN_Our'
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='')
FLAGS = parser.parse_args()
root = get_datasets(FLAGS)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
feature_dim = config['model']['feature_dim']
depth = config['model']['depth']
heads = config['model']['heads']
dim_feedforward = config['model']['dim_feedforward']
in_channels = config['model']['in_channels']
mlp_hidden_size = config['model']['hidden_size']
projection_size = config['model']['projection_size']
model_checkpoints_folder = str(config['savings']['checkpoints'])
MLP1 = MLP1(in_channels=in_channels, mlp_hidden_size=mlp_hidden_size, projection_size=projection_size)
BrainMass = BNTF(feature_dim=feature_dim, depth=depth, heads=heads, dim_feedforward=dim_feedforward)
MLP2 = MLP2(in_channels=projection_size, mlp_hidden_size=mlp_hidden_size, projection_size=in_channels)
params = torch.load(model_checkpoints_folder)
encoder = BrainFly(MLP1, BrainMass, MLP2).to(device)
encoder.load_state_dict(params)
encoder.eval()
correlation_measure = ConnectivityMeasure(kind='correlation')
os.makedirs(save_path, exist_ok=True)
for file in os.listdir(root):
    img = np.loadtxt(os.path.join(root, f'{file}'))
    file.replace('.txt', '')
    correlation_matrix = correlation_measure.fit_transform([img])[0]
    save_file = os.path.join(save_path, f'{file}_FBN.txt')
    correlation_matrix = torch.tensor(correlation_matrix, requires_grad=False, device=device)
    correlation_matrix = correlation_matrix.float()
    output = encoder.mlp1(correlation_matrix)
    for atten in encoder.brain_mass.attention_list:
        output = atten(output)
    output = encoder.mlp2(output)
    output = output.detach().cpu().numpy()
    result = correlation_measure.fit_transform([output])[0]
    np.savetxt(save_file, result)
    print(output.shape)
    print(f'file:{file} is saved in {save_file}')
