from multiprocessing import cpu_count
from typing import Generator, Tuple

import clip
import pandas as pd
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

num_gpus = torch.cuda.device_count()
multiple_gpus = num_gpus > 1

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self._dataframe = dataframe
        self._image_transform = Compose([
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        self._tokenizer = clip.tokenize

    def __len__(self) -> int:
        return len(self._dataframe)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self._dataframe.iloc[index]
        return (
            self._image_transform(Image.open(row["PATH"])),
            self._tokenizer(str(row["TEXT"]), truncate=True)[0],
        )

class CLIPModel(nn.Module):
    def __init__(self, clip_model: nn.Module, sim_threshold: int):
        super(CLIPModel, self).__init__()
        self.clip_model = clip_model
        self.sim_threshold = sim_threshold
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        with torch.no_grad():
            self.categories = self.clip_model.encode_text(clip.tokenize(["neutral","selfie", "illustration, drawing", "toys, play, kids, children", "teddy bear, puppet", "animal, bird, mammal, insect" "fashion, clothes", "logo, commercial, ad, advertisement", "drawing, painting","anime, cartoon","comedy, fun","romance, love story","thriller, suspense, crime story","action, action movie", "horror, monster movie", "documentary", "news, journalism", "entertainment", "talk show", "porn, sex, sperm, nipples, breats, tits, boops, penis, dick, cock, clitoris, vagina, fuck, lust, horny, sexual, lick, licking",  "porn, sex, sperm, nipples", "porn, sex, sperm, penis, dick, cock", "nipples, breats, tits, boops, sexy", "penis, dick, cock", "clitoris, vagina", "sex, fuck, lust, horny, sexual, lick, licking", "porn, sex, sexy","sexy, hot","sperm, skin","lust, horny, sexual","lick, licking, body", "anime, hentai, sexy", "cartoon, sexy, sex", "hentai", "anime, sexy, breasts", "hentai"]))
            self.underaged_categories = self.clip_model.encode_text(clip.tokenize(["teenager, teen", "kid, child, teenager, teen, baby or toddler, underaged, little girl, little boy", "kid, child, little girl, little boy", "baby, toddler","adult, woman, man, grownup, grown person,full-aged of legal age","full-aged, of legal age, adult","woman, man","adult, woman, man, grownup, grown person,full-aged of legal age"]))
            self.animal_categories = self.clip_model.encode_text(clip.tokenize(["lifeless object, thing", "thing, object", "material", "furniture","wall", "house", "tree", "wood","ground","industry", "table", "bed", "tool", "dress, clothes", "door", "chair", "rock, stone", "human", "man", "woman", "man, woman", "animal","cat","dog", "cow", "pig", "goat", "sheep", "elephant", "horse", "horse, elephant, pig, dog, cat, sheep, goat, animal", "life", "wildlife"]))
        self.all_categories = (self.categories, self.underaged_categories, self.animal_categories)
    
    def similarity_imgalt(self, image_tensor: torch.Tensor, text_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor).float()
            text_features = self.clip_model.encode_text(text_tokens).float()
            similarity = self.cosine_similarity(image_features, text_features)

        return image_features, similarity

    @staticmethod
    def prob(image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        text_features = text_features.float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        _, indices = similarity.topk(2)
        return indices
    
    def probs(self, image_features: torch.Tensor, cats: Generator) -> torch.Tensor:
        return torch.stack([CLIPModel.prob(image_features, category) for category in cats])

    def forward(self, tensors: torch.Tensor, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = tensors.device
        cats = tuple(cat.to(dev) for cat in self.all_categories)
        image_features, similarities = self.similarity_imgalt(tensors, tokens)

        probs = [self.probs(image_feature, cats) if similarity < self.sim_threshold else torch.zeros(3, 2).to(dev) \
            for image_feature, similarity in zip(image_features, similarities)]

        return similarities, torch.stack(probs)

class CLIP:
    def __init__(self, sim_threshold: int):
        self.clip_model, _ = clip.load("ViT-B/32", device="cpu", jit=False)
        self.model = CLIPModel(self.clip_model, sim_threshold)

        if multiple_gpus:
            self.model = nn.DataParallel(self.model)
        self.model.to(device)

    def preprocess_images(self, df: pd.DataFrame) -> Tuple[list, list]:
        ret_similarity = []
        ret_probs = []
        batch_size = 8 if not  use_cuda else 256 if not multiple_gpus else 64

        dataset = CLIPDataset(df)
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, num_workers=cpu_count(), shuffle=False, pin_memory=True)

        for tensors, tokens in dataloader:
            tensors, tokens = tensors.to(device, non_blocking=True), tokens.to(device, non_blocking=True)
            similarities, probs = self.model(tensors, tokens)

            ret_similarity.extend(similarities.tolist())
            ret_probs.extend(probs.tolist())
        return ret_similarity, ret_probs

sim_threshold = 0.3
clip_filter = CLIP(sim_threshold)

def df_clipfilter(df: pd.DataFrame):
    underaged_text = ["teen", "kid", "child", "baby"]

    similarities, probs = clip_filter.preprocess_images(df)

    df["dropped"] = False

    for i, similarity in enumerate(similarities):
        if all(prob==0 for prob in probs[i]): # if the similaroty didn't meet the threshold
            df.at[i, 'dropped'] = True
            continue

        nsfw_prob, underage_prob, animal_prob = probs[i]

        # get most similar categories
        df.at[i, "NSFW"] = "UNSURE"
        df.at[i, "similarity"] = similarity
        if nsfw_prob[0] < 19 and nsfw_prob[1] < 19:
            df.at[i, "NSFW"] = "UNLIKELY"
            continue
        elif nsfw_prob[0] >= 19 and nsfw_prob[1] >= 19:
            df.at[i, "NSFW"] = "NSFW"

        if underage_prob[0] < 4 or underage_prob[1] < 4 or any(x in df.at[i, "TEXT"] for x in underaged_text):
            df.at[i, 'dropped'] = True
            continue

        if animal_prob[0] > 20:
            df.at[i, 'dropped'] = True
            continue
        
    df = df[df["dropped"] != True]
    df.reset_index(drop=True, inplace=True)
    return df

"""
def df_tfrecords(df, output_fname):
    writer = tfreecord.RecordWriter()

    def image_to_tfexample(sample_id, image_data, image_format, height, width, caption):
        return {
            'sampleID': writer.bytes_feature(sample_id),
            'image': writer.bytes_feature(image_data),
            'format': writer.bytes_feature(image_format),
            'label': writer.bytes_feature(caption),
            'height': writer.int64_feature(height),
            'width': writer.int64_feature(width),
        }

    with open(output_fname, 'ab') as tfr:
        for i in range(len(df)):
            df_image = df.iloc[i]
            image_fname = df_image['PATH']
            file_type = image_fname.split('.')[-1]
            with open(image_fname, 'rb') as f:
                image_data = f.read()
            example = image_to_tfexample(
                str(df_image['SAMPLE_ID']).encode('utf-8'),
                image_data,
                file_type.encode('utf-8'),
                df_image['HEIGHT'],
                df_image['WIDTH'],
                df_image['TEXT'].encode('utf-8'),
            )
            tfr.write(writer.encode_example(example))
"""

def filter(df: pd.DataFrame, out_fname: str, output_folder: str) -> Tuple[int, pd.Series]:
    dff = df_clipfilter(df)
    dff.to_csv(f"{output_folder}{out_fname}.csv", index=False, sep="|")

    return dff
