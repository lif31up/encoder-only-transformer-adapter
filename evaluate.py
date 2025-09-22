import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import Config
from EmbeddedDataset import EmbeddedDataset
from model.BERT import BERT


def evaluate(model, testset, device):
  bert_config = Config()
  trainset = EmbeddedDataset(dataset=bert_config.textset, dim=bert_config.dim, embedder=bert_config.embedder, model=bert_config.embedder)
  model.to(device)
  model.eval()
  counts, n_problems = 0, len(testset)
  for feature, label in tqdm(DataLoader(trainset, batch_size=1, shuffle=True, pin_memory=True, num_workers=4)):
    feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
    pred = model.forward(feature)
    if torch.argmax(pred) == torch.argmax(label): counts += 1
  return counts, n_problems
# eval

if __name__ == "__main__":
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  my_data = torch.load('your path', map_location='cpu', weights_only=False)
  my_config = my_data['config']
  my_model = BERT(my_config)
  my_model.load_state_dict(my_data['state'])
  testset = EmbeddedDataset(dataset=my_config.testset_for_test, dim=my_config.dim, embedder=my_config.embedder, model=my_config.embedder)
  counts, n_problems = evaluate(my_model, testset, device)
  print(f"Accuracy: {counts / n_problems:.4f}")
# __name__