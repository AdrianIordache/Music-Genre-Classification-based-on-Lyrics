import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from IPython.display import display

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

from sentence_transformers import SentenceTransformer, util

import warnings
warnings.filterwarnings("ignore")

SEED = 42

def seed_everything(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(seed = SEED)

train_data = pd.read_csv("../data/Lyrics-Genre-Train.csv") 
test_data  = pd.read_csv("../data/Lyrics-Genre-Test-GroundTruth.csv")

train_data.drop(["Song", "Song year", "Artist"], axis = 1, inplace = True)
test_data.drop(["Song", "Song year", "Artist"], axis = 1, inplace = True)

encoder = OrdinalEncoder()
encoder.fit(train_data['Genre'].values.reshape(-1, 1))
train_data['Genre'] = encoder.transform(train_data['Genre'].values.reshape(-1, 1)).astype(np.uint8)
test_data['Genre']  = encoder.transform(test_data['Genre'].values.reshape(-1, 1)).astype(np.uint8)

# display(train_data.head(n = 3))
# display(train_data.shape)

# display(test_data.head(n = 3))
# display(test_data.shape)

skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = SEED)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train_data, train_data['Genre'].values)):
    train_data.loc[valid_idx, 'Fold'] = fold

train_data['Fold'] = train_data['Fold'].astype(int)    

# ! pip install transformers
# ! pip install sentence-transformers

model = SentenceTransformer('stsb-roberta-large')

embeddings = pd.DataFrame(columns = ["Track_id", "Fold", "Genre"] + [f"Feature_{i}" for i in range(1024)])
for row, sample in test_data.iterrows():
	track_id = sample['Track_id']
	lyrics   = sample['Lyrics']
	label    = sample['Genre']

	embedding = model.encode(lyrics, convert_to_tensor = True).detach().cpu().numpy().tolist()
	row = [track_id, label, fold] + embedding
	embeddings.loc[len(embeddings)] = row
	
	# break

embeddings.to_csv("./data/train_embeddings.csv", index = False)
display(embeddings)