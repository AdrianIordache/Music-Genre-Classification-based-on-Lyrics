import os
import torch
import random
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder

from simpletransformers.classification import ClassificationModel

import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

SEED = 42
seed_everything(seed = SEED)

train_data = pd.read_csv("../data/Lyrics-Genre-Train.csv") 
test_data  = pd.read_csv("../data/Lyrics-Genre-Test-GroundTruth.csv")

train_data.drop(["Song", "Song year", "Artist", "Track_id"], axis = 1, inplace = True)
test_data.drop(["Song", "Song year", "Artist", "Track_id"],  axis = 1, inplace = True)

encoder = OrdinalEncoder()
encoder.fit(train_data['Genre'].values.reshape(-1, 1))
train_data['Genre'] = encoder.transform(train_data['Genre'].values.reshape(-1, 1)).astype(np.uint8)
test_data['Genre']  = encoder.transform(test_data['Genre'].values.reshape(-1, 1)).astype(np.uint8)

train_data = train_data.rename(columns = {"Lyrics": "text", "Genre": "label"})
test_data  = test_data.rename(columns = {"Lyrics": "text", "Genre": "label"})

train_data          = train_data[['text', 'label']]
train_data['text']  = train_data['text'].astype(str)
train_data['label'] = train_data['label'].astype(int)

test_data          = test_data[['text', 'label']]
test_data['text']  = test_data['text'].astype(str)
test_data['label'] = test_data['label'].astype(int)

args = {
    "output_dir": "outputs/",
    "cache_dir": "cache_dir/",

    "fp16": True,
    "fp16_opt_level": "O1",
    "max_seq_length": 128,
    "train_batch_size": 96,
    "gradient_accumulation_steps": 1,
    "eval_batch_size": 96,
    "num_train_epochs": 10,
    "weight_decay": 0,
    "learning_rate": 1e-5,

    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.06,
    "warmup_steps": 0,
    "max_grad_norm": 1.0,

    "logging_steps": 50,
    "save_steps": 2000,

    "overwrite_output_dir": True,
    "reprocess_input_data": True,
    "evaluate_during_training": True,
    "evaluate_during_training_silent": False,

    "n_gpu": 1,
}

model = ClassificationModel('roberta', 'roberta-base', num_labels = 10, args = args)

model.train_model(train_data, eval_df = test_data, accuracy = lambda truth, predictions: accuracy_score(
        truth, [round(p) for p in predictions]
    ))

result, model_outputs, wrong_predictions = model.eval_model(test_data, acc = accuracy_score)

# print(torch.argmax(torch.nn.functional.softmax(torch.tensor(model_outputs), dim = 1)), dim = 1)
print(result)
