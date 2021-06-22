import argparse
import gc
from io import BytesIO
import os
import pickle
import shutil
import time
from glob import glob
from urllib.parse import urljoin, urlparse
from uuid import uuid1

import tractor
import trio
import ujson
from PIL import Image, ImageFile, UnidentifiedImageError
from copy import copy
import tqdm

import asks
asks.init("trio")

ImageFile.LOAD_TRUNCATED_IMAGES = True  # https://stackoverflow.com/a/47958486

import clip
import datasets
import torch
from anyascii import anyascii

device = "cuda" if torch.cuda.is_available() else "cpu"
datasets.set_caching_enabled(False)

class CLIP:
    def __init__(self):
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.categories = self.model.encode_text(clip.tokenize(["neutral","selfie", "illustration, drawing", "toys, play, kids, children", "teddy bear, puppet", "animal, bird, mammal, insect" "fashion, clothes", "logo, commercial, ad, advertisement", "drawing, painting","anime, cartoon","comedy, fun","romance, love story","thriller, suspense, crime story","action, action movie", "horror, monster movie", "documentary", "news, journalism", "entertainment", "talk show", "porn, sex, sperm, nipples, breats, tits, boops, penis, dick, cock, clitoris, vagina, fuck, lust, horny, sexual, lick, licking",  "porn, sex, sperm, nipples", "porn, sex, sperm, penis, dick, cock", "nipples, breats, tits, boops, sexy", "penis, dick, cock", "clitoris, vagina", "sex, fuck, lust, horny, sexual, lick, licking", "porn, sex, sexy","sexy, hot","sperm, skin","lust, horny, sexual","lick, licking, body", "anime, hentai, sexy", "cartoon, sexy, sex", "hentai", "anime, sexy, breasts", "hentai"]).to(device))
        self.underaged_categories = self.model.encode_text(clip.tokenize(["teenager, teen", "kid, child, teenager, teen, baby or toddler, underaged, little girl, little boy", "kid, child, little girl, little boy", "baby, toddler","adult, woman, man, grownup, grown person,full-aged of legal age","full-aged, of legal age, adult","woman, man","adult, woman, man, grownup, grown person,full-aged of legal age"]).to(device))
        self.animal_categories = self.model.encode_text(clip.tokenize(["lifeless object, thing", "thing, object", "material", "furniture","wall", "house", "tree", "wood","ground","industry", "table", "bed", "tool", "dress, clothes", "door", "chair", "rock, stone", "human", "man", "woman", "man, woman", "animal","cat","dog", "cow", "pig", "goat", "sheep", "elephant", "horse", "horse, elephant, pig, dog, cat, sheep, goat, animal", "life", "wildlife"]).to(device))


    def similarity_imgalt(self, batch):
        similarity = []
        images = [
            self.preprocess(Image.open(path)).unsqueeze(0).to(device)
            for path in batch["PATH"]
        ]
        max_texts = [anyascii(text)[:77] for text in batch["TEXT"]]
        texts = clip.tokenize(max_texts).to(device)

        with torch.no_grad():
            image_features = self.model.encode_image(
                torch.cat(images)
            ).float()
            text_features = self.model.encode_text(texts).float()

        for image_feat, text_feat in zip(image_features, text_features):
            similarity.append(
                float(
                    self.cosine_similarity(
                        torch.reshape(text_feat, (1, 512)),
                        torch.reshape(image_feat, (1, 512)),
                    )
                )
            )

        batch["similarity"] = similarity
        batch["image_features"] = image_features.detach().cpu().numpy()
        return batch

    def preprocess_images(self, df):
        im_dataset = datasets.Dataset.from_pandas(df)
        im_dataset = im_dataset.map(self.similarity_imgalt, batched=True, batch_size=512)
        return im_dataset["image_features"], im_dataset["similarity"]

    def prob(self, image_features, text_features):
        image_features = torch.as_tensor(image_features).to(device)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        similarity = (100.0 * image_features.float() @ text_features.T.float()).softmax(dim=-1)
        _, indices = similarity.topk(2)
        return indices

class FileData:
    def __init__(self, filename):
        self._filename = filename
        self._line_to_position = [0]
        self._length = 0

        with open(self._filename, "r") as f:
            while f.readline():
                self._line_to_position.append(f.tell())
                self._length += 1

    def __getitem__(self, line):
        return self._line_to_position[line]

    def __len__(self):
        return self._length

class TrioProgress(trio.abc.Instrument):

    def __init__(self, total, notebook_mode=False, **kwargs):
        if notebook_mode:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        self.tqdm = tqdm(total=total, desc="Downloaded: [ 0 ] / Links ", **kwargs)

    def task_exited(self, task):
        if task.custom_sleep_data == 0:
            self.tqdm.update(7)
        if task.custom_sleep_data == 1:
            self.tqdm.update(7)
            self.tqdm.desc = self.tqdm.desc.split(":")[0] + ": [ " + str( int(self.tqdm.desc.split(":")[1].split(" ")[2]) + 1 ) + " ] / Links "
            self.tqdm.refresh()

def chunk_using_generators(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def remove_bad_chars(text):
    return "".join(c for c in text if c.isprintable())


def parse_wat(content, start, line_count):
    import ftfy
    import pycld2 as cld2

    blocklist = open("blocklist-domain.txt").read().splitlines()
    valid_data = []
    content.seek(start)
    for _ in range(line_count):
        line = content.readline()
        if "IMG@" not in line:
            continue
        line_str = line.strip()
        data = ujson.loads(line_str)
        linklist = data["Envelope"]["Payload-Metadata"]["HTTP-Response-Metadata"][
            "HTML-Metadata"
        ]["Links"]
        base_url = os.path.dirname(
            data["Envelope"]["WARC-Header-Metadata"]["WARC-Target-URI"]
        )  # get base url
        license = "?"
        for e in linklist:
            if "url" in e and "creativecommons.org/licenses/" in e["url"]:
                license = e["url"]
            if "alt" not in e:
                continue
            url = e["url"]
            alt_text = ftfy.fix_text(e["alt"].replace("\n", " ")).strip()
            if url in [".svg", ".gif", "data:image", "javascript:"] or any(
                bl in url for bl in blocklist
            ):
                continue
            try:
                _, _, details = cld2.detect(alt_text)
            except Exception as e:
                alt_text = remove_bad_chars(alt_text)
                _, _, details = cld2.detect(alt_text)

            if details[0][1] == "en":
                if not url.startswith("http"):
                    url = urljoin(base_url, url)
                valid_data.append((url, alt_text, license))
    return [
        t for t in {tuple(i) for i in valid_data}
    ]  # Remove duplicate tuple from list


def process_img_content(response, alt_text, license, sample_id):
    img_output_folder = "save/images/"

    try:
        if len(response.content) < 5000:
            return
        img_data = BytesIO(response.content)
        with Image.open(img_data) as im:
            width, height = im.size
            im_format = im.format
            out_fname = f"{img_output_folder}{str(sample_id)}.{im_format.lower()}"
            if im_format not in ["JPEG", "JPG", "PNG", "WEBP"]:
                return
            if im.mode != "RGB":
                im = im.convert("RGB")
            im.save(out_fname)
    except (KeyError, UnidentifiedImageError):
        return

    return [str(sample_id), out_fname, response.url, alt_text, width, height, license]


async def request_image(datas, start_sampleid, instrument=None):
    tmp_data = []

    import asks
    asks.init("trio")

    session = asks.Session(connections=64)
    session.headers = {
        "User-Agent": "Googlebot-Image",
        "Accept-Language": "en-US",
        "Accept-Encoding": "gzip, deflate",
        "Referer": "https://www.google.com/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    async def _request(data, sample_id):
        url, alt_text, license = data
        task = trio.lowlevel.current_task()
        task.custom_sleep_data = None
        try:
            proces = process_img_content(
                await session.get(url, timeout=5, connection_timeout=40), alt_text, license, sample_id
            )
            if proces is not None:
                tmp_data.append(proces)
                task.custom_sleep_data = 1
        except Exception:
            return

    async with trio.open_nursery() as n:
        for data in datas:
            n.start_soon(_request, data, start_sampleid)
            start_sampleid += 1

    with open(f".tmp/{uuid1()}.json", "w") as f:
        ujson.dump(tmp_data, f)
    gc.collect()
    return

    #trio.run(_requestmain)
    #instruments=[instrument(len(datas), True)]


async def dl_wat(valid_data, first_sample_id):
    import pandas as pd
    
    # Download every image available
    processed_samples = []
    async with tractor.open_nursery() as n:
        for i, data in enumerate(chunk_using_generators(valid_data, 65536)):
            await n.run_in_actor(
                request_image, datas=data, start_sampleid = i * 65536 + first_sample_id
            )
        #trio.run(request_image, valid_data, first_sample_id, instruments=[TrioProgress(len(valid_data), True)] )

    for tmpf in glob(".tmp/*.json"):
        processed_samples.extend(ujson.load(open(tmpf)))
    return pd.DataFrame(
        processed_samples,
        columns=["SAMPLE_ID", "PATH", "URL", "TEXT", "HEIGHT", "WIDTH", "LICENSE"],
    )


def df_clipfilter(df):
    sim_threshold = 0.3
    underaged_text = ["teen", "kid", "child", "baby"]

    clip = CLIP()
    img_embedding, similarities = clip.preprocess_images(df)
    tmp_embed = copy(img_embedding)
    for i, img_embed in enumerate(tmp_embed):
        if similarities[i] < sim_threshold:
            df.drop(i, inplace=True)
            img_embedding.remove(img_embed)
            continue

        # get most similar categories
        nsfw_prob = clip.prob(img_embed, clip.categories)
        df.at[i, "NSFW"] = "UNSURE"
        df.at[i, "similarity"] = similarities[i]
        if nsfw_prob[0] < 19 and nsfw_prob[1] < 19:
            df.at[i, "NSFW"] = "UNLIKELY"
            continue
        elif nsfw_prob[0] >= 19 and nsfw_prob[1] >= 19:
            df.at[i, "NSFW"] = "NSFW"

        underage_prob = clip.prob(img_embed, clip.underaged_categories)
        if (
            underage_prob[0] < 4
            or underage_prob[1] < 4
            or any(x in df.at[i, "TEXT"] for x in underaged_text)
        ):
            df.drop(i, inplace=True)
            img_embedding.remove(img_embed)
            continue

        animal_prob = clip.prob(img_embed, clip.animal_categories)
        if animal_prob[0] > 20:
            df.drop(i, inplace=True)
            img_embedding.remove(img_embed)

    df.reset_index(drop=True, inplace=True)
    return df, img_embedding


def df_tfrecords(df, output_fname):
    import tensorflow as tf
    from tfr_image.utils import bytes_feature, int64_feature

    def image_to_tfexample(sample_id, image_data, image_format, height, width, caption):
        return tf.train.Example(
            features=tf.train.Features(
                feature={
                    "sampleID": bytes_feature(sample_id),
                    "image": bytes_feature(image_data),
                    "format": bytes_feature(image_format),
                    "label": bytes_feature(caption),
                    "height": int64_feature(height),
                    "width": int64_feature(width),
                }
            )
        )

    with tf.io.TFRecordWriter(output_fname) as tfrecord_writer:
        for i in range(len(df)):
            df_image = df.iloc[i]
            image_fname = df_image["PATH"]
            file_type = image_fname.split(".")[-1]
            with tf.io.gfile.GFile(image_fname, "rb") as f:
                image_data = f.read()
            example = image_to_tfexample(
                str(df_image["SAMPLE_ID"]).encode("utf_8"),
                image_data,
                file_type.encode("utf_8"),
                df_image["HEIGHT"],
                df_image["WIDTH"],
                df_image["TEXT"].encode("utf_8"),
            )
            tfrecord_writer.write(example.SerializeToString())


def upload_gdrive(output_filename):
    import requests

    client_id = (
        "648172777761-onv1nc5f93nhlhf63flsq6onrmjphpfo.apps.googleusercontent.com"
    )
    client_secret = "HZ4Zw-_jVJ-3mwicz1NM5W5x"
    refresh_token = "1//04N2Kysz1LObLCgYIARAAGAQSNwF-L9IrntHNWi2_nEVu2QX5fmlW0Ea0qA-ToBJLSdatDATYxiKcNFI8eZQ_fYN53gjF7b8MGmA"

    def refresh_gdrive_token():
        params = {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
        }

        authorization_url = "https://www.googleapis.com/oauth2/v4/token"

        r = requests.post(authorization_url, data=params)

        if r.ok:
            return r.json()["access_token"]
        else:
            return None

    access_t = refresh_gdrive_token()
    headers = {"Authorization": "Bearer " + access_t}
    para = {
        "name": output_filename.split("/")[-1],
        "parents": ["1CIgcIR7nX2xNBPB577jwEqbbwxAJR_nt"],
    }

    files = {
        "data": ("metadata", ujson.dumps(para), "application/json; charset=UTF-8"),
        "file": ("application/zip", open(output_filename, "rb")),
    }
    requests.post(
        "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
        headers=headers,
        files=files,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Crawling@Home Worker Script'
    )

    parser.add_argument('--name', '-n', type=str, default="ARKseal", help='Your name')
    parser.add_argument('--url', '-u', type=str, default="http://crawlingathome.duckdns.org/", help='The Crawling Server')

    args = parser.parse_args()

    import crawlingathome_client as cah

    client = cah.init(
        url=args.url, nickname=args.name
    )

    output_folder = "./save/"
    csv_output_folder = output_folder
    img_output_folder = output_folder + "images/"

    while client.jobCount() > 0:
        start = time.time()

        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        if os.path.exists(".tmp"):
            shutil.rmtree(".tmp")

        os.mkdir(output_folder)
        os.mkdir(img_output_folder)
        os.mkdir(".tmp")

        client.newJob()
        client.downloadShard()
        first_sample_id = int(client.start_id)
        last_sample_id = int(client.end_id)
        shard_of_chunk = client.shard_piece

        out_fname = f"FIRST_SAMPLE_ID_IN_SHARD_{str(first_sample_id)}_LAST_SAMPLE_ID_IN_SHARD_{str(last_sample_id)}_{shard_of_chunk}"
        print(
            f"[crawling@home] shard identification {out_fname}"
        )  # in case test fails, we need to remove bad data
        client.log("Processing shard")

        fd = FileData("shard.wat")

        if shard_of_chunk == 0:
            start_index = fd[0]
        if shard_of_chunk == 1:
            start_index = fd[int(len(fd) * 0.5)]

        lines = int(len(fd) * 0.5)

        with open("shard.wat", "r") as infile:
            parsed_data = parse_wat(infile, start_index, lines)

        client.log("Downloading images")
        time.sleep(10) #prevent damaging the progress bars
        dlparse_df = trio.run(dl_wat, parsed_data, first_sample_id, instruments=[TrioProgress(len(parsed_data), True)])
        dlparse_df.to_csv(output_folder + out_fname + ".csv", index=False, sep="|")
        print (f"[crawling@home] Downloaded {len(dlparse_df)} in {round(time.time() - start)} seconds")
        print (f"[crawling@home] Download efficiency {len(dlparse_df)/(time.time() - start)} img/sec")

        client.log("Dropping NSFW keywords")
        start2 = time.time()
        filtered_df, img_embeddings = df_clipfilter(dlparse_df)
        filtered_df.to_csv(output_folder + out_fname + ".csv", index=False, sep="|")
        print (f"[crawling@home] CLIP filtered {len(filtered_df)} in {round(time.time() - start2)} seconds")
        print (f"[crawling@home] CLIP efficiency {len(dlparse_df)/(time.time() - start2)} img/sec")

        img_embeds_sampleid = {}
        for i, img_embed_it in enumerate(img_embeddings):
            dfid_index = filtered_df.at[i, "SAMPLE_ID"]
            img_embeds_sampleid[str(dfid_index)] = img_embed_it
        with open(f"{output_folder}img_embeddings-{out_fname}.pkl", "wb") as f:
            pickle.dump(img_embeds_sampleid, f)

        client.log("Saving TFRs")
        df_tfrecords(
            filtered_df,
            f"{output_folder}crawling_at_home_{out_fname}__00000-of-00001.tfrecord",
        )
        upload_gdrive(f"{output_folder}img_embeddings-{out_fname}.pkl")
        upload_gdrive(
            f"{output_folder}crawling_at_home_{out_fname}__00000-of-00001.tfrecord"
        )
        upload_gdrive(output_folder + out_fname + ".csv")

        client._markjobasdone(len(filtered_df))
        print(f"[crawling@home] job completed in {round(time.time() - start)} seconds")
        print(f"[crawling@home] job efficiency {len(filtered_df)/(time.time() - start)} pairs/sec")
        break
    client.bye()