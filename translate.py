import pandas as pd
import torch
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-ru"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Загрузка данных
df = pd.read_csv("dataset.1csv")

columns = ["resume_text", "job_description_text"]
new_df = pd.DataFrame(columns=columns)


# Перевод текста батчами
def translate_batch(texts, batch_size=8):
    translated = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
        batch = texts[i : i + batch_size]
        # Заменяем NaN на пустую строку
        batch = [" " if pd.isna(t) else t for t in batch]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            translated_tokens = model.generate(**inputs, max_length=512)
        translated_batch = tokenizer.batch_decode(
            translated_tokens, skip_special_tokens=True
        )
        translated.extend(translated_batch)
    return translated


# Перевод колонок
new_df["resume_text"] = translate_batch(df["resume_text"].tolist())
new_df["job_description_text"] = translate_batch(df["job_description_text"].tolist())

# Сохраняем результат
new_df.to_csv("translated_dataset.csv", index=False)
