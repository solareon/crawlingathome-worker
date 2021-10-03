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
from math import sqrt
from threading import Thread
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
from bloom_filter2 import BloomFilter
from PIL import Image, ImageFile, UnidentifiedImageError
from random_user_agent.params import OperatingSystem, SoftwareName
from random_user_agent.user_agent import UserAgent
from requests.adapters import HTTPAdapter

import crawlingathome_client as cah
from crawlingathome_client.temp import TempCPUWorker

ImageFile.LOAD_TRUNCATED_IMAGES = True  # https://stackoverflow.com/a/47958486

warnings.filterwarnings('ignore')

ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE


class DownloadProgressInstrument(trio.abc.Instrument):
    def __init__(self, processing_count, finished_count, http_error_count, image_error_count, lock):
        self._processing_count = processing_count
        self._finished_count = finished_count
        self._http_error_count = http_error_count
        self._image_error_count = image_error_count
        self._lock = lock

    def task_exited(self, task):
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
    def __init__(self, filename):
        self._filename = filename
        self._line_to_position = [0]
        self._length = 0

        with open(self._filename, 'r') as f:
            while f.readline():
                self._line_to_position.append(f.tell())
                self._length += 1
        gc.collect()

    def __getitem__(self, line):
        return self._line_to_position[line]

    def __len__(self):
        return self._length


class InvalidImageError(Exception):
    pass


def chunk_using_generators(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def log(ex: Exception, msg: str = ''):
    if msg:
        msg = f'- {msg} '
    with open('err.log', 'a') as f:
        f.write(f'{ex.__class__.__name__} {msg} -> {ex}\n')


def remove_bad_chars(text):
    return ''.join(c for c in text if c.isprintable())


def bloom_server_filter(hashes, bloom_ip):
    files = {
        'file': ('hash.txt', BytesIO(hashes)),
        'key': (None, 'clipped'),
    }

    failure = True
    for _ in range(5):
        response = requests.post(
            f'http://{bloom_ip}:8000/deduplicate/', files=files)
        if response.status_code != 200:
            cah.print('Bloom Server Error, retrying...')
            time.sleep(1)
            continue
        return response.content.decode('utf-8').split('\n')

    cah.print('Cannot contact bloom server, shutting down worker')
    sys.exit(1)


def parse_wat_worker(file_name, start, line_count, oneprocess=False, bloom_ip='116.202.162.146'):
    blocked_links = BloomFilter(max_elements=10_000_000, error_rate=0.01, filename=(
        'blocklists/failed-domains.bin', -1))

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
                    if url_domain in blocked_links:
                        continue
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

        valid_hashes = bloom_server_filter(
            '\n'.join([data[-1].strip() for data in valid_data]).encode('utf-8'), bloom_ip)

        orig_len = len(valid_data)
        valid_data = [
            t for t in {tuple(i) for i in valid_data}
        ]
        shard_dups = orig_len - len(valid_data)

        before_clipped = len(valid_data)
        valid_data = [data
                      for data in valid_data if data[-1].strip() in valid_hashes]
        clipped = before_clipped - len(valid_data)

        return valid_data, clipped, shard_dups


def parse_wat(file_name, shard, fd=None, bloom_ip='116.202.162.146'):
    if fd is None:
        fd = FileData(file_name)

    if shard == 0:
        start_line = 0
    elif shard == 1:
        start_line = len(fd)//2

    line_count = len(fd)//2

    return parse_wat_worker(file_name, fd[start_line], line_count)


def process_img_content(response, alt_text, license, sample_id):
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

            if width * height > 8294400:  # Resize Images larger than 4K
                ratio = sqrt(width * height / 8294400)
                width = int(width/ratio)
                height = int(height/ratio)
                im = im.resize((width, height), resample=Image.LANCZOS)

            im_format = im.format.lower()
            out_fname = f'{img_output_folder}{str(sample_id)}.{im_format}'

            if im_format not in set(['jpeg', 'jpg', 'png', 'webp']):
                raise InvalidImageError('invalid image format')

            if im.mode != 'RGB':
                im = im.convert('RGB')
            im.save(out_fname)

        return [str(sample_id), out_fname, response.url, alt_text, width, height, license]
    except (AttributeError, KeyError, UnidentifiedImageError) as ex:
        raise InvalidImageError(
            f'Unidentified Image Error: {ex.__class__.__name__}')


async def request_image(datas, start_sampleid, user_agent, processing_count, lock, connections=165):
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
        async with limit:
            with lock:
                processing_count.value += 1

            task = trio.lowlevel.current_task()

            url, alt_text, license, domain, hash = datas[data_index]
            try:
                tmp_data.append(process_img_content(
                    await session.get(url, timeout=10, connection_timeout=20), alt_text, license, sample_id
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


def dl_wat_worker(data, start_sample_id, user_agent, processing_count, finished_count, http_error_count, image_error_count, lock):
    trio.run(request_image, data, start_sample_id, user_agent, processing_count, lock, instruments=[
             DownloadProgressInstrument(processing_count, finished_count, http_error_count, image_error_count, lock)])


def dl_progress(len_data, processing_count, finished_count, http_error_count, image_error_count, update_tqdm, lock, isnotebook=False):
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


def dl_wat(valid_data, first_sample_id, user_agent_rotator, isnotebook=False):
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


def update_filter(session):
    start = time.time()
    shutil.rmtree('blocklists', ignore_errors=True)
    os.mkdir('blocklists')

    try:
        with session.get('http://the-eye.eu/public/AI/cahblacklists/failed-domains.bin', stream=True) as r:
            r.raise_for_status()
            with open(f'blocklists/failed-domains.bin', 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except requests.HTTPError as ex:
        cah.print(f'error in updating filters: {ex}')
        return

    end = time.time()
    cah.print(f'updated filters in {(end-start):.2f}')


def safe_client_function(client_function, *args, **kwargs):
    while True:
        try:
            return client_function(*args, **kwargs)
        except cah.WorkerTimedOutError:
            cah.print('worker timed out, retrying')


def check_current_worker_version():
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


def main(name, url, debug, isnotebook, isdocker):
    if isdocker:
        check_current_worker_version()

    client = TempCPUWorker(
        url=url, nickname=name
    )

    if not hasattr(client, 'bye'):
        client.bye = client._c.bye

    if not os.path.exists('blocklists'):
        os.mkdir('blocklists')

    output_folder = './save/'
    img_output_folder = output_folder + 'images/'

    uid = ''
    updater = None
    workers = mp.cpu_count()

    retry_15 = HTTPAdapter(max_retries=15)
    filter_session = requests.Session()
    filter_session.mount('http://', retry_15)

    update_filter(filter_session)

    user_agent_rotator = UserAgent(software_names=[SoftwareName.CHROME.value], operating_systems=[
                                   OperatingSystem.LINUX.value], limit=2000)

    while safe_client_function(client.jobCount) > 0:
        if isdocker:
            check_current_worker_version()

        try:
            completing_arg = {}
            start = time.time()

            updater = Thread(target=update_filter, args=(filter_session,))
            updater.start()

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

                updater.join()

                safe_client_function(client.log, 'Processing shard')
                start_processing = time.time()

                parsed_data, cliped, shard_dups = parse_wat(
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
                    f'cliped found: {cliped}, shard dups found: {shard_dups}')

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
    try:
        if updater is not None:
            updater.join()
    except:
        pass
    try:
        if safe_client_function(client.isAlive):
            safe_client_function(client.bye)
    except:
        pass
