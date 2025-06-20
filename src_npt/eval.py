import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from src_npt.BPEDataset import BPEDataset
from src_npt.model.BERT import BERT

def evaluate(MODEL, dataset):
  data = torch.load(MODEL, map_location="cpu")

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  tokenizer_config, model_config = data["config"]["tokenizer_config"], data["config"]["model"]
  bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  bert_tokenizer.truncation_side = "right"
  bert_tokenizer.padding_side = "right"
  bert_tokenizer.pad_token = bert_tokenizer.eos_token = "[PAD]"
  trainset = BPEDataset(dataset=dataset, dim=model_config["dim"], tokenizer=bert_tokenizer)

  model = BERT(model_config, init_weights=None).to(device)
  model.load_state_dict(data["state"])
  model.eval()

  # Training loop
  correct = 0
  for feature, label in tqdm(DataLoader(trainset, batch_size=1, shuffle=True, pin_memory=True, num_workers=4)):
    feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
    output = model.forward(input=feature)
    output = torch.softmax(output, dim=-1)
    if torch.argmax(output, dim=-1) == torch.argmax(label, dim=-1):
      correct += 1
  # for
  print(f"Accuracy: {correct / len(trainset):.4f}")
# eval

if __name__ == "__main__":
  from datasets import load_dataset

  dataset = load_dataset('imdb')['train'].shuffle(seed=42).select(range(100))
  evaluate("BERT.pth", dataset)  # Replace with the actual model path
# __name__