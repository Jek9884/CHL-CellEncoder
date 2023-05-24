import os
import os
import sys

import scanpy as sc
from torch.utils.data import DataLoader

from custom_dataset import TextSamplerDataset

sys.path.insert(1, 'scBERT')
from scBERT.performer_pytorch import PerformerLM, AutoregressiveWrapper
from utils import *
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler

#model hypper-parameters
CLASS = 7
SEED = 42
EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 10
LEARNING_RATE = 1e-4
GENE_NUM = 19022
SEQ_LEN = GENE_NUM+1
DIM_LATENT = 200


# select the GPU to use
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device(0)

data_path = "../dataset/all_tissue.h5ad"

data = sc.read_h5ad(data_path)
# remove all the columns having a count smaller than 3
sc.pp.filter_genes(data, min_counts = 3)
print(np.shape(data.X))
# pre-processing step
sc.pp.normalize_total(data, target_sum=1e4)
logged = sc.pp.log1p(data, base=2, copy=True)
condition = ~pd.isna(data.obs['cell_ontology_class'])
data = data[condition, :]
data = data.X

dataset = TextSamplerDataset(data, device)

data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    
# init model with checkpoint
model = PerformerLM(
    num_tokens = CLASS,
    dim = DIM_LATENT,
    depth = 6,
    max_seq_len = SEQ_LEN,
    heads = 10,
    g2v_position_emb = False
)

ckpt_path = "../data/panglao_pretrain.pth"
ckpt = torch.load(ckpt_path)

ckpt['model_state_dict'].pop('pos_emb.emb.weight')
model.load_state_dict(ckpt['model_state_dict'])
for param in model.parameters():
    param.requires_grad = False
for param in model.norm.parameters():
    param.requires_grad = True
for param in model.performer.net.layers[-2].parameters():
    param.requires_grad = True

model = AutoregressiveWrapper(model)

model = model.to(device)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler()
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=15,
    cycle_mult=2,
    max_lr=LEARNING_RATE,
    min_lr=1e-6,
    warmup_steps=5,
    gamma=0.9
)

# training phase
for i in range(1, EPOCHS+1):

    model.train()

    for index, data in enumerate(data_loader):
        index += 1

        if index % GRADIENT_ACCUMULATE_EVERY != 0:
            loss, _ = model(data, return_loss = True)
            scaler.scale(loss).backward()

        if index % GRADIENT_ACCUMULATE_EVERY == 0:
            loss, _ = model(data, return_loss = True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

# save model parameters
if not os.path.exists("ckpt_folder"):
    os.makedirs("ckpt_folder")
torch.save(
    {
        'model_performer_state_dict': model.net.state_dict(),
        'model_autowrapper_state_dict': model.state_dict(),
    },
    f'ckpt_folder/performer_best.pth'
)