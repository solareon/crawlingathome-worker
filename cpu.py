import gc
import hashlib
import multiprocessing as mp
import os
import random
import shutil
import ssl
import sys
import tarfile
import time
import traceback
import warnings
from ctypes import c_bool, c_int
from functools import partial
from glob import glob
from io import BytesIO
from typing import Dict, Generator, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
from uuid import uuid1, uuid4

import asks
import ftfy
import numpy as np
import pandas as pd
import pycld2 as cld2
import requests
import trio
import ujson
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError
from random_user_agent.params import OperatingSystem, SoftwareName
from random_user_agent.user_agent import UserAgent

import crawlingathome_client as cah

ImageFile.LOAD_TRUNCATED_IMAGES = True  # https://stackoverflow.com/a/47958486

warnings.filterwarnings('ignore')

ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE


class DownloadProgressInstrument(trio.abc.Instrument):
    def __init__(self, processing_count: mp.Value, finished_count: mp.Value, http_error_count: mp.Value, image_error_count: mp.Value, lock: mp.Lock):
        self._processing_count = processing_count
        self._finished_count = finished_count
        self._http_error_count = http_error_count
        self._image_error_count = image_error_count
        self._lock = lock

    def task_exited(self, task: trio.lowlevel.Task):
        if task.custom_sleep_data in [0, 1, 2]:
            with self._lock:
                self._processing_count.value -= 1
                self._finished_count.value += 1
            if task.custom_sleep_data == 1:
                with self._lock:
                    self._http_error_count.value += 1
            if task.custom_sleep_data == 2:
                with self._lock:
                    self._image_error_count.value += 1


class FileData:
    def __init__(self, filename: str):
        self._filename = filename
        self._line_to_position = [0]
        self._length = 0

        with open(self._filename, 'r') as f:
            while f.readline():
                self._line_to_position.append(f.tell())
                self._length += 1
        gc.collect()

    def __getitem__(self, line: int) -> int:
        return self._line_to_position[line]

    def __len__(self) -> int:
        return self._length


class InvalidImageError(Exception):
    pass


def chunk_using_generators(lst: list, n: int) -> Generator[list, None, None]:
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def log(ex: Exception, msg: str = ''):
    if msg:
        msg = f'- {msg} '
    with open('err.log', 'a') as f:
        f.write(f'{ex.__class__.__name__} {msg} -> {ex}\n')


def remove_bad_chars(text: str) -> str:
    return ''.join(c for c in text if c.isprintable())


def bloom_server_filter(hashes: bytes, bloom_ip: str, key: Optional[str] = 'clipped', loc: Optional[str] = 'deduplicate') -> List[str]:
    files = {
        'file': ('hash.txt', BytesIO(hashes)),
        'key': (None, key),
    }

    for _ in range(5):
        response = requests.post(
            f'http://{bloom_ip}:8000/{loc}/', files=files)
        if response.status_code != 200:
            cah.print('Bloom Server Error, retrying...')
            time.sleep(1)
            continue
        return response.content.decode('utf-8').split('\n')

    cah.print('Cannot contact bloom server, shutting down worker')
    sys.exit(1)


def parse_wat_worker(file_name: str, start: int, line_count: int, bloom_ip1: Optional[str] = '116.202.162.146', bloom_ip2: Optional[str] = '94.130.167.172') -> Tuple[List[Tuple], int, int, int]:
    blocked_formats = ('.svg', '.gif', '.webp',
                       'data:image', 'javascript:', 'mailto:')

    valid_data = []
    added_urls = set()

    with open(file_name, 'r') as content:
        content.seek(start)
        for _ in range(line_count):
            line = content.readline()

            if 'IMG@' not in line:
                continue

            line_str = line.strip()
            data = ujson.loads(line_str)

            linklist = data['Envelope']['Payload-Metadata']['HTTP-Response-Metadata'][
                'HTML-Metadata'
            ]['Links']

            base_url = os.path.dirname(
                data['Envelope']['WARC-Header-Metadata']['WARC-Target-URI']
            )

            license = '?'
            for e in linklist:
                if 'url' in e and 'creativecommons.org/licenses/' in e['url']:
                    license = e['url']

                if 'alt' not in e:
                    continue

                url = e['url']

                if any(bf in url for bf in blocked_formats):
                    continue

                url_domain = 'unknown'
                try:
                    url_domain = urlparse(url).netloc
                except:
                    continue

                alt_text = ftfy.fix_text(e['alt'].replace('\n', ' ')).strip()
                try:
                    _, _, details = cld2.detect(alt_text)
                except:
                    alt_text = remove_bad_chars(alt_text)
                    _, _, details = cld2.detect(alt_text)

                if details[0][1] != 'en':
                    continue

                if not url.startswith('http'):
                    url = urljoin(base_url, url)

                if url in added_urls:
                    continue

                hash = hashlib.md5(
                    (url + alt_text).encode('utf-8')).hexdigest()

                valid_data.append((url, alt_text, license, url_domain, hash))

        orig_len = len(valid_data)
        valid_data = [
            t for t in {tuple(i) for i in valid_data}
        ]
        shard_dups = orig_len - len(valid_data)

        valid_hashes = bloom_server_filter(
            '\n'.join(data[-1].strip() for data in valid_data).encode('utf-8'), bloom_ip1, key='clipped', loc='deduplicate')

        before_clipped = len(valid_data)
        valid_data = [data
                      for data in valid_data if data[-1].strip() in valid_hashes]
        clipped = before_clipped - len(valid_data)

        valid_hashes = bloom_server_filter(
            '\n'.join(data[-1].strip() for data in valid_data).encode('utf-8'), bloom_ip2, key='parsed', loc='deduplicate')
        
        before_parsed_check = len(valid_data)
        valid_data = [data
                      for data in valid_data if data[-1].strip() in valid_hashes]
        parsed_check = before_parsed_check - len(valid_data)

        return valid_data, clipped, shard_dups, parsed_check


def parse_wat(file_name: str, shard: int, fd: Optional[FileData] = None, bloom_ip1: Optional[str] = '116.202.162.146', bloom_ip2: Optional[str] = '94.130.167.172') -> Tuple[List[Tuple], int, int, int]:
    if fd is None:
        fd = FileData(file_name)

    if shard == 0:
        start_line = 0
    elif shard == 1:
        start_line = len(fd)//2

    line_count = len(fd)//2

    return parse_wat_worker(file_name, fd[start_line], line_count, bloom_ip1=bloom_ip1, bloom_ip2=bloom_ip2)


def torchvision_resize(img: Image.Image) -> Image.Image:
    size = 224
    w, h = img.size

    short, long = (w, h) if w <= h else (h, w)
    if short == size:
        return img

    new_short, new_long = size, int(size * long / short)

    new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    return img.resize((new_w, new_h), Image.BICUBIC)


def torchvision_parse_fill(img: Image.Image) -> Dict[str, Tuple[float, ...]]:
    fill = 0
    name = 'fill'

    num_bands = len(img.getbands())
    if num_bands > 1:
        fill = tuple([fill] * num_bands)

    return {name: fill}


def torchvision_pad(img: Image.Image, padding: Tuple[int]) -> Image.Image:
    opts = torchvision_parse_fill(img)
    if img.mode == "P":
        palette = img.getpalette()
        image = ImageOps.expand(img, border=padding, **opts)
        image.putpalette(palette)
        return image

    return ImageOps.expand(img, border=padding, **opts)


def torchvision_crop(img: Image.Image,
                     top: int,
                     left: int,
                     height: int,
                     width: int,
                     ) -> Image.Image:

    return img.crop((left, top, left + width, top + height))


def torchvision_centercrop(img: Image.Image) -> Image.Image:
    size = (224, 224)
    output_size = size

    image_width, image_height = img.size
    crop_height, crop_width = output_size

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = (
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        )
        img = torchvision_pad(img, padding_ltrb)
        image_width, image_height = img.size
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top = round((image_height - crop_height) // 2)
    crop_left = round((image_width - crop_width) // 2)
    return torchvision_crop(img, crop_top, crop_left, crop_height, crop_width)


def clip_preprocessing(img: Image.Image) -> Image.Image:
    img = torchvision_resize(img)
    img = torchvision_centercrop(img)
    img = img.convert('RGB')
    return img


def process_img_content(response, alt_text: str, license: str, sample_id: int) -> Tuple[str, str, str, str, int, int, str]:
    img_output_folder = 'save/images/'

    try:
        if len(response.content) < 5000:
            raise InvalidImageError('image is too small')

        img_data = BytesIO(response.content)
        with Image.open(img_data) as im:
            width, height = im.size

            if width * height > 89478484:  # Might be a DOS decompression bomb
                raise InvalidImageError(
                    'image is too large (dos decompression)')

            im_format = im.format.lower()
            if im_format not in set(['jpeg', 'jpg', 'png', 'webp']):
                raise InvalidImageError('invalid image format')

            out_fname = f'{img_output_folder}{sample_id}.{im_format}'

            im = clip_preprocessing(im)
            im.save(out_fname)

        return (str(sample_id), out_fname, response.url, alt_text, width, height, license)
    except (AttributeError, KeyError, UnidentifiedImageError) as ex:
        raise InvalidImageError(
            f'Unidentified Image Error: {ex.__class__.__name__}')


async def request_image(datas: list, start_sampleid: int, user_agent: str, processing_count: mp.Value, lock: mp.Lock, connections: int = 165):
    tmp_data = []

    limit = trio.CapacityLimiter(2000)
    session = asks.Session(connections=connections, ssl_context=ssl_ctx)

    session.headers = {
        'User-Agent': user_agent,
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Referer': 'https://www.google.com',
        'DNT': '1',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }

    async def _request(data_index, sample_id):
        with lock:
            processing_count.value += 1

        task = trio.lowlevel.current_task()

        url, alt_text, license, _, _ = datas[data_index]

        try:
            async with limit:
                response = await session.get(url, timeout=10, connection_timeout=20)
            tmp_data.append(process_img_content(
                response, alt_text, license, sample_id
            ))
            task.custom_sleep_data = 0
        except InvalidImageError as ex:
            log(ex, msg='Image Error')
            task.custom_sleep_data = 1
        except Exception as ex:
            log(ex, msg='HTTP Error')
            task.custom_sleep_data = 2
        return

    async with trio.open_nursery() as n:
        for id, index in enumerate(range(len(datas)), start=start_sampleid):
            async with limit:
                n.start_soon(_request, index, id)

    with open(f'.tmp/dl-{uuid1()}.json', 'w') as f:
        ujson.dump(tmp_data, f)
    gc.collect()


def dl_wat_worker(data: list, start_sample_id: int, user_agent: str, processing_count: mp.Value, finished_count: mp.Value, http_error_count: mp.Value, image_error_count: mp.Value, lock: mp.Lock) -> None:
    trio.run(request_image, data, start_sample_id, user_agent, processing_count, lock, instruments=[
             DownloadProgressInstrument(processing_count, finished_count, http_error_count, image_error_count, lock)])


def dl_progress(len_data: int, processing_count: mp.Value, finished_count: mp.Value, http_error_count: mp.Value, image_error_count: mp.Value, update_tqdm: mp.Value, lock: mp.Lock, isnotebook: Optional[bool] = False) -> None:
    if isnotebook:
        from tqdm import tqdm
    else:
        from tqdm import tqdm

    progress_bar = tqdm(total=len_data, unit='links')
    while True:
        with lock:
            if not update_tqdm.value:
                break
            progress_bar.desc = f'Processing {processing_count.value} links, {http_error_count.value} http errors, {image_error_count.value} image errors'
            progress_bar.update(finished_count.value - progress_bar.n)
        time.sleep(1)
    progress_bar.close()


def dl_wat(valid_data: list, first_sample_id: str, user_agent_rotator: UserAgent, isnotebook: Optional[bool] = False) -> pd.DataFrame:
    # Download every image available
    processed_samples = []
    n_processes = mp.cpu_count()

    manager = mp.Manager()
    update_tqdm = manager.Value(c_bool, True)

    processing_count = manager.Value(c_int, 0)
    finished_count = manager.Value(c_int, 0)
    http_error_count = manager.Value(c_int, 0)
    image_error_count = manager.Value(c_int, 0)

    lock = manager.Lock()

    t = mp.Process(target=dl_progress, args=(
        len(valid_data), processing_count, finished_count, http_error_count, image_error_count, update_tqdm, lock, isnotebook))
    t.start()

    if n_processes == 1:
        dl_wat_worker(valid_data, first_sample_id, user_agent_rotator.get_random_user_agent(), processing_count,
                      finished_count, http_error_count, image_error_count, update_tqdm, lock)
    else:
        chunk_size = len(valid_data) // n_processes + 1
        worker = partial(dl_wat_worker, processing_count=processing_count,
                         finished_count=finished_count, http_error_count=http_error_count, image_error_count=image_error_count, lock=lock)

        with mp.Pool(n_processes) as pool:
            pool.starmap(worker, [(data, first_sample_id + i * chunk_size, user_agent_rotator.get_random_user_agent())
                                  for (i, data) in enumerate(chunk_using_generators(valid_data, chunk_size))])

    time.sleep(1)
    with lock:
        update_tqdm.value = False

    t.join()
    t.close()

    for tmpf in glob('.tmp/dl-*.json'):
        with open(tmpf, 'r') as f:
            processed_samples.extend(ujson.load(f))
    return pd.DataFrame(
        processed_samples,
        columns=['SAMPLE_ID', 'PATH', 'URL',
                 'TEXT', 'HEIGHT', 'WIDTH', 'LICENSE'],
    )


def upload(source: str, target: str):
    with tarfile.open(f'{source}.tar.gz', 'w:gz') as tar:
        tar.add(source, arcname=os.path.basename(source))

    result = 1
    while result:
        result = os.system(
            f'rsync -av {source}.tar.gz {target} > /dev/null 2>&1')
    if os.path.exists(f'{source}.tar.gz'):
        os.remove(f'{source}.tar.gz')


def safe_client_function(client_function, *args, **kwargs):
    while True:
        try:
            return client_function(*args, **kwargs)
        except cah.WorkerTimedOutError:
            cah.print('worker timed out, retrying')


def check_current_worker_version() -> None:
    while True:
        try:
            r = requests.get(
                'https://raw.githubusercontent.com/ARKseal/crawlingathome-worker/master/version.txt')
            r.raise_for_status()
            break
        except requests.HTTPError:
            pass

    version = r.content.decode('utf-8')
    current_version = os.getenv('CAHVERSION', default=None)

    if current_version is None:
        cah.print("Worker doesn't use the correct docker env")
        exit(-1)

    if version == current_version:
        return

    cah.print("Worker is outdated, please repull the docker image")
    sys.exit(1)


def main(name: str, url: str, debug: bool, isnotebook: bool, isdocker: bool) -> None:
    if isdocker:
        check_current_worker_version()

    client = cah.FullWATClient(
        url=url, nickname=name
    )

    if not hasattr(client, 'bye'):
        client.bye = client._c.bye

    output_folder = './save/'
    img_output_folder = output_folder + 'images/'

    uid = ''

    user_agent_rotator = UserAgent(software_names=[SoftwareName.CHROME.value], operating_systems=[
                                   OperatingSystem.LINUX.value], limit=2000)

    while safe_client_function(client.jobCount) > 0:
        if isdocker:
            check_current_worker_version()

        try:
            completing_arg = {}
            start = time.time()

            if not safe_client_function(client.isAlive):
                safe_client_function(client.recreate)

            safe_client_function(client.newJob)
            safe_client_function(client.downloadWat)

            fd = FileData('shard.wat')

            for shard_of_chunk in range(2):
                shutil.rmtree(output_folder, ignore_errors=True)
                shutil.rmtree(uid, ignore_errors=True)
                shutil.rmtree('.tmp', ignore_errors=True)

                os.mkdir(output_folder)
                os.mkdir(img_output_folder)
                os.mkdir('.tmp')

                first_sample_id = np.int64(
                    client.shards[shard_of_chunk][1]['start_id'])
                last_sample_id = np.int64(
                    client.shards[shard_of_chunk][1]['end_id'])

                out_fname = \
                    f'FIRST_SAMPLE_ID_IN_SHARD_{first_sample_id}_LAST_SAMPLE_ID_IN_SHARD_{last_sample_id}_{shard_of_chunk}'
                cah.print(
                    f'shard identification: {out_fname}'
                )  # in case test fails, we need to remove bad data

                safe_client_function(client.log, 'Processing shard')
                start_processing = time.time()

                parsed_data, cliped, shard_dups, parsed_check = parse_wat(
                    'shard.wat', shard_of_chunk, fd=fd)

                num_links = len(parsed_data)

                parsed_df = pd.DataFrame(parsed_data, columns=[
                                         'URL', 'TEXT', 'LICENSE', 'DOMAIN', 'HASH'])
                parsed_df = parsed_df.drop_duplicates(subset=['URL'])
                parsed_df.to_csv(
                    f'{output_folder}{out_fname}_parsed.csv', index=False, sep='|')

                random.shuffle(parsed_data)

                end_processing = time.time()
                cah.print(
                    f'Processed shard in {(end_processing-start_processing):.2f} seconds'
                    '\n\t'
                    f'cliped found: {cliped}, shard dups found: {shard_dups}, parsed check found: {parsed_check}')

                safe_client_function(client.log, 'Downloading images')
                start_dl = time.time()

                dlparse_df = dl_wat(
                    parsed_data, first_sample_id, user_agent_rotator, isnotebook)
                dlparse_df.to_csv(
                    f'{output_folder}{out_fname}.csv', index=False, sep='|')

                end_dl = time.time()
                cah.print(
                    f'Downloaded {len(dlparse_df)} images out of {num_links} links in {(end_dl - start_dl):.2f} seconds')
                cah.print(
                    f'Download efficiency: {(len(dlparse_df) / (end_dl - start_dl)):.2f} img/sec OR {(num_links / (end_dl - start_dl)):.2f} links/sec')

                safe_client_function(client.log, 'Uploading Temporary Job')

                uid = uuid4().hex
                shutil.move('save', uid)

                completing_arg[str(
                    client.shards[shard_of_chunk][0])] = f'rsync {uid}'
                upload(uid, client.upload_address)

            safe_client_function(client.completeJob, completing_arg)
            end = time.time()
            cah.print(
                f'job completed in {(end - start):.2f} seconds')
            cah.print(
                f'job efficiency {(len(dlparse_df) / (end - start)):.2f} pairs/sec')
        except KeyboardInterrupt:
            cah.print('stopping crawler')
            break
        except Exception as ex:
            cah.print(f'ERROR: {ex}')
            if debug:
                traceback.print_exc()
            try:
                if safe_client_function(client.isAlive):
                    safe_client_function(client.log, 'Error, restarting job')
            except:
                cah.print("Couldn't log to client")
        finally:
            if debug:
                break
    shutil.rmtree(uid, ignore_errors=True)
    shutil.rmtree(output_folder, ignore_errors=True)
    shutil.rmtree(img_output_folder, ignore_errors=True)
    shutil.rmtree('.tmp', ignore_errors=True)
    try:
        if safe_client_function(client.isAlive):
            safe_client_function(client.bye)
    except:
        pass
