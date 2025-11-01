import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import Config
from EmbeddedDataset import EmbeddedDataset
from model.Transformer import Transformer


def evaluate(model, dataset, device):
  model.to(device)
  correct = 0
  for feature, label in tqdm(DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=4)):
    feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
    output = model.forward(feature)
    output = torch.softmax(output, dim=-1)
    if torch.argmax(output, dim=-1) == torch.argmax(label, dim=-1):
      correct += 1
  # for
  print(f"Accuracy: {correct / len(dataset):.4f}")
# eval

if __name__ == "__main__":
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  bert_config = Config()
  my_data = torch.load(
    '/content/drive/MyDrive/Colab Notebooks/BERT.bin', map_location=torch.device('cpu'), weights_only=False)
  my_model = Transformer(my_data['config'])
  my_model.load_state_dict(my_data["state"])
  testset = EmbeddedDataset(
    dataset=bert_config.testset_for_test, dim=bert_config.dim, tokenizer=bert_config.tokenizer, embedder=bert_config.embedder)
  testset.consolidate()
  evaluate(model=my_model, dataset=testset, device=device)
# __name__