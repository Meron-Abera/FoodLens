"""Tests for src/data/download.py, using a tiny local archive instead of the real 5 GB download."""
import tarfile

import pytest

from src.data.download import download, is_downloaded


def make_fake_dataset(root):
    (root / 'food-101' / 'images' / 'pizza').mkdir(parents=True)
    (root / 'food-101' / 'meta').mkdir()
    (root / 'food-101' / 'meta' / 'train.txt').write_text('pizza/1\n')
    (root / 'food-101' / 'images' / 'pizza' / '1.jpg').write_bytes(b'not really a jpeg')


def test_is_downloaded(tmp_path):
    assert not is_downloaded(tmp_path)
    make_fake_dataset(tmp_path)
    assert is_downloaded(tmp_path)


def test_download_unpacks_archive_and_removes_it(tmp_path):
    source = tmp_path / 'source'
    make_fake_dataset(source)
    archive = tmp_path / 'food-101.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(source / 'food-101', arcname='food-101')

    data_dir = tmp_path / 'data'
    download(data_dir, url=archive.as_uri())
    assert is_downloaded(data_dir)
    assert not (data_dir / 'food-101.tar.gz').exists()


def test_download_skips_when_already_present(tmp_path):
    make_fake_dataset(tmp_path)
    download(tmp_path, url='https://invalid.example/should-not-be-requested')  # would fail if requested
    assert is_downloaded(tmp_path)


def test_download_rejects_archive_without_food101(tmp_path):
    source = tmp_path / 'source'
    (source / 'other').mkdir(parents=True)
    archive = tmp_path / 'bad.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(source / 'other', arcname='other')
    with pytest.raises(RuntimeError):
        download(tmp_path / 'data', url=archive.as_uri())
