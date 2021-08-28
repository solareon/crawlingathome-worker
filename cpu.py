import gc
import hashlib
import multiprocessing as mp
import os
import random
import shutil
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


class DownloadProgressInstrument(trio.abc.Instrument):
    def __init__(self, processing_count, finished_count, error_count, lock):
        self._processing_count = processing_count
        self._finished_count = finished_count
        self._error_count = error_count
        self._lock = lock

    def task_exited(self, task):
        if task.custom_sleep_data in [0, 1]:
            with self._lock:
                self._processing_count.value -= 1
                self._finished_count.value += 1
            if task.custom_sleep_data == 1:
                with self._lock:
                    self._error_count.value += 1


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


def chunk_using_generators(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def remove_bad_chars(text):
    return ''.join(c for c in text if c.isprintable())


def bloom_server_filter(hashes, bloom_ip):
    files = {
        'file': ('hash.txt', BytesIO(hashes)),
        'key': (None, 'clipped'),
    }

    failure = True
    for _ in range(10):
        response = requests.post(
            f'http://{bloom_ip}:8000/deduplicate/', files=files)
        if response.status_code != 200:
            cah.print('bloom server error, retrying...')
            time.sleep(1)
        else:
            failure = False
            break
    if failure:
        cah.print('crash, cannot contact the bloom server, please fix')
        sys.exit(1)

    return response.content.decode('utf-8').split('\n')


def parse_wat_worker(file_name, start, line_count, oneprocess=False, bloom_ip='116.202.162.146'):
    blocked_links = BloomFilter(max_elements=10_000_000, error_rate=0.01, filename=(
        'blocklists/failed-domains.bin', -1))

    blocked_formats = set(
        ['.svg', '.gif', '.webp', 'data:image', 'javascript:', 'mailto:'])

    valid_data = []

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
                if 'alt' not in e:
                    continue

                if 'url' in e and 'creativecommons.org/licenses/' in e['url']:
                    license = e['url']

                url = e['url']

                if any(bf in url for bf in blocked_formats):
                    continue

                try:
                    if urlparse(url).netloc in blocked_links:
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

                hash = hashlib.md5(
                    (url + alt_text).encode('utf-8')).hexdigest()

                valid_data.append((url, alt_text, license, hash))

        if oneprocess:
            valid_hashes = bloom_server_filter(
                '\n'.join([data[-1] for data in valid_data]).encode('utf-8'), bloom_ip)

            orig_len = len(valid_data)
            valid_data = [
                t for t in {tuple(i) for i in valid_data}
            ]
            shard_dups = orig_len - len(valid_data)

            before_clipped = len(valid_data)
            valid_data = [data[:-1]
                          for data in valid_data if data[-1].strip() in valid_hashes]
            clipped = before_clipped - len(valid_data)

            return valid_data, clipped, shard_dups

        with open(f'.tmp/pw-{uuid1()}.json', 'w') as f:
            ujson.dump(valid_data, f)
        gc.collect()


def parse_wat(file_name, shard, workers, bloom_ip='116.202.162.146'):
    fd = FileData(file_name)

    if shard == 0:
        start_line = 0
    elif shard == 1:
        start_line = len(fd)//2

    line_count = len(fd)//2

    if workers == 1:
        return parse_wat_worker(file_name, fd[start_line], line_count, oneprocess=True)

    lc = line_count//workers - 1
    with mp.Pool(workers) as pool:
        pool.starmap(parse_wat_worker, [
                     (file_name, fd[start_line + i*lc], lc) for i in range(workers)])

    valid_data = []
    for tmpf in glob('.tmp/pw-*.json'):
        with open(tmpf, 'r') as f:
            valid_data.extend(ujson.load(f))

    valid_hashes = bloom_server_filter(
        '\n'.join([data[-1] for data in valid_data]).encode('utf-8'), bloom_ip)

    orig_len = len(valid_data)
    valid_data = [
        t for t in {tuple(i) for i in valid_data}
    ]
    shard_dups = orig_len - len(valid_data)

    before_clipped = len(valid_data)
    valid_data = [data[:-1]
                  for data in valid_data if data[-1].strip() in valid_hashes]
    clipped = before_clipped - len(valid_data)

    del fd
    return valid_data, clipped, shard_dups


def process_img_content(response, alt_text, license, sample_id):
    img_output_folder = 'save/images/'

    try:
        if len(response.content) < 5000:
            return

        img_data = BytesIO(response.content)
        with Image.open(img_data) as im:
            width, height = im.size

            if width * height > 89478484:
                return

            if width * height > 8294400:  # if image is larger than 4K then attempt scale down
                ratio = sqrt(width * height / 8294400)
                width = int(width/ratio)
                height = int(height/ratio)
                im = im.resize((width, height))

            im_format = im.format
            out_fname = f'{img_output_folder}{str(sample_id)}.{im_format.lower()}'

            if im_format not in set(['JPEG', 'JPG', 'PNG', 'WEBP']):
                return

            if im.mode != 'RGB':
                im = im.convert('RGB')
            im.save(out_fname)
    except (KeyError, UnidentifiedImageError):
        return

    return [str(sample_id), out_fname, response.url, alt_text, width, height, license]


async def request_image(datas, start_sampleid, processing_count, lock):
    tmp_data = []
    session = asks.Session(connections=165)

    limit = trio.CapacityLimiter(1000)

    user_agent_rotator = UserAgent(software_names=[SoftwareName.CHROME.value], operating_systems=[
                                   OperatingSystem.LINUX.value], limit=2000)
    user_agent = user_agent_rotator.get_random_user_agent()

    session.headers = {
        'User-Agent': user_agent,
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Referer': 'https://www.google.com',
        "DNT": "1",
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }

    async def _request(data_index, sample_id):
        async with limit:
            with lock:
                processing_count.value += 1

            task = trio.lowlevel.current_task()

            url, alt_text, license = datas[data_index]
            try:
                proces = process_img_content(
                    await session.get(url, timeout=5), alt_text, license, sample_id
                )
                task.custom_sleep_data = 0
                if proces is not None:
                    tmp_data.append(proces)
            except Exception:
                task.custom_sleep_data = 1
        return

    async with trio.open_nursery() as n:
        for index in range(len(datas)):
            async with limit:
                n.start_soon(_request, index, start_sampleid)
            start_sampleid += 1

    with open(f'.tmp/dl-{uuid1()}.json', 'w') as f:
        ujson.dump(tmp_data, f)
    gc.collect()


def dl_wat_worker(data, start_sample_id, processing_count, finished_count, error_count, lock):
    trio.run(request_image, data, start_sample_id, processing_count, lock, instruments=[
             DownloadProgressInstrument(processing_count, finished_count, error_count, lock)])


def dl_progress(len_data, processing_count, finished_count, error_count, update_tqdm, lock, isnotebook=False):
    if isnotebook:
        from tqdm import tqdm
    else:
        from tqdm import tqdm

    progress_bar = tqdm(total=len_data, unit='links')
    while True:
        with lock:
            if not update_tqdm.value:
                break
            progress_bar.desc = f'Processing {processing_count.value} links, {error_count.value} errors'
            progress_bar.update(finished_count.value - progress_bar.n)
        time.sleep(1)
    progress_bar.close()


def dl_wat(valid_data, first_sample_id, isnotebook=False):
    # Download every image available
    processed_samples = []
    n_processes = mp.cpu_count()

    manager = mp.Manager()
    update_tqdm = manager.Value(c_bool, True)

    processing_count = manager.Value(c_int, 0)
    finished_count = manager.Value(c_int, 0)
    error_count = manager.Value(c_int, 0)

    lock = manager.Lock()

    t = mp.Process(target=dl_progress, args=(
        len(valid_data), processing_count, finished_count, error_count, update_tqdm, lock, isnotebook))
    t.start()

    if n_processes == 1:
        dl_wat_worker(valid_data, processing_count,
                      finished_count, error_count, update_tqdm, lock)
    else:
        chunk_size = len(valid_data) // n_processes + 1
        worker = partial(dl_wat_worker, processing_count=processing_count,
                         finished_count=finished_count, error_count=error_count, lock=lock)

        with mp.Pool(n_processes) as pool:
            pool.starmap(worker, [(data, first_sample_id + i * chunk_size)
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

    def getJobs():
        while True:
            try:
                return safe_client_function(client.jobCount)
            except:
                pass

    while getJobs() > 0:
        if isdocker:
            check_current_worker_version()

        try:
            completing_arg = {}
            start = time.time()

            updater = Thread(target=update_filter, args=(filter_session,))
            updater.start()

            if not safe_client_function(client.isAlive):
                safe_client_function(client.recreate)

            shutil.rmtree(output_folder, ignore_errors=True)
            shutil.rmtree(uid, ignore_errors=True)
            shutil.rmtree('.tmp', ignore_errors=True)

            os.mkdir(output_folder)
            os.mkdir(img_output_folder)
            os.mkdir('.tmp')

            safe_client_function(client.newJob)
            safe_client_function(client.downloadWat)

            for shard_of_chunk in range(2):
                shutil.rmtree(output_folder, ignore_errors=True)
                shutil.rmtree(uid, ignore_errors=True)
                shutil.rmtree('.tmp', ignore_errors=True)

                os.mkdir(output_folder)
                os.mkdir(img_output_folder)
                os.mkdir('.tmp')

                first_sample_id = np.int64(
                    client.shards[shard_of_chunk][1]["start_id"])
                last_sample_id = np.int64(
                    client.shards[shard_of_chunk][1]["end_id"])

                out_fname = \
                    f'FIRST_SAMPLE_ID_IN_SHARD_{first_sample_id}_LAST_SAMPLE_ID_IN_SHARD_{last_sample_id}_{shard_of_chunk}'
                cah.print(
                    f'shard identification: {out_fname}'
                )  # in case test fails, we need to remove bad data

                updater.join()

                safe_client_function(client.log, 'Processing shard')
                start_processing = time.time()

                parsed_data, cliped, shard_dups = parse_wat(
                    'shard.wat', shard_of_chunk, workers)

                num_links = len(parsed_data)

                random.shuffle(parsed_data)

                end_processing = time.time()
                cah.print(
                    f'Processed shard in {(end_processing-start_processing):.2f} seconds'
                    '\n\t'
                    f'cliped found: {cliped}, shard dups found: {shard_dups}')

                safe_client_function(client.log, 'Downloading images')
                start_dl = time.time()
                dlparse_df = dl_wat(parsed_data, first_sample_id, isnotebook)
                dlparse_df.to_csv(
                    f'{output_folder}{out_fname}.csv', index=False, sep='|')
                end_dl = time.time()

                cah.print(
                    f'Downloaded {len(dlparse_df)} images out of {num_links} links in {(end_dl - start_dl):.2f} seconds')
                cah.print(
                    f'Download efficiency: {(len(dlparse_df) / (end_dl - start_dl)):.2f} img/sec OR {(num_links / (end_dl - start_dl)):.2f} links/sec')

                safe_client_function(client.log, 'Uploading Temporary Job')

                uid = uuid4().hex
                shutil.copytree('save', uid)

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
                if client.isAlive():
                    client.log('Error, restarting job')
            except:
                cah.print("Couldn't log to client")
        finally:
            if debug:
                break
    try:
        if updater is not None:
            updater.join()
    except:
        pass
    try:
        if client.isAlive():
            client.bye()
    except:
        pass
