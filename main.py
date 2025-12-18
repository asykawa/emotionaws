from fastapi import FastAPI
from pydantic import BaseModel
from translate import Translator
import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer

classes = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride',
    'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]


class EmotionModels(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, output_dim=28):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])


vocab = torch.load("vocab_emoton.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmotionModels(len(vocab))
model.load_state_dict(torch.load("model_emotion.pth", map_location=device))
model.to(device)
model.eval()

tokenizer = get_tokenizer("basic_english")

translator = Translator(from_lang="ru", to_lang="en")


def text_to_ids(text: str):
    return [vocab.get(token, vocab.get("<unk>", 0)) for token in tokenizer(text)]


class TextSchema(BaseModel):
    word: str


emotion_app = FastAPI()


@emotion_app.post("/predict")
async def predict(text: TextSchema):
    if not text.word.strip():
        return {"class": "neutral", "error": "Empty input text"}

    try:
        translated_text = str(translator.translate(text.word))
    except Exception:
        translated_text = text.word

    ids = torch.tensor(text_to_ids(translated_text), dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(ids)
        label = torch.argmax(pred, dim=1).item()

    return {"class": classes[label]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(emotion_app, host="127.0.0.1", port=8080)
