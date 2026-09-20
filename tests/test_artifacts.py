"""Fail-closed checks for corrupted artifacts and ignored benchmark controls."""
import argparse
import os
import pathlib
import struct
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', default='build')
    args = parser.parse_args()
    suffix = '.exe' if os.name == 'nt' else ''
    fixture = str((pathlib.Path(args.build) / ('recserve_fixture' + suffix)).resolve())
    bench = str((pathlib.Path(args.build) / ('recserve_bench' + suffix)).resolve())
    def run(command, success):
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert (result.returncode == 0) == success, (command, result.stdout, result.stderr)
        assert result.returncode in (0, 1, 2) and 'Sanitizer' not in result.stderr, result.stderr
    for options in (['--kernel', 'typo'], ['--mode', 'typo'], ['--workers', '4'], ['--dim', '-1']):
        run([bench, *options], False)
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        catalog, index = root/'catalog.bin', root/'index.bin'
        run([fixture, '--items', '128', '--dim', '17', '--build-threads', '1',
             '--out-catalog', str(catalog), '--out-index', str(index)], True)
        command = [bench, '--load-catalog', str(catalog), '--load-index', str(index), '--n', '4',
                   '--trials', '1', '--warmup', '0', '--recall-probe', '0', '--json']
        run(command, True)
        run([*command, '--pin'], True)
        run([*command, '--no-arena'], True)
        cat_data, idx_data = catalog.read_bytes(), index.read_bytes()
        for offset, value in ((0, 0), (4, 0xFFFFFFFF), (8, 0xFFFFFFFF), (12, 0x7FC00000)):
            corrupt = bytearray(cat_data)
            struct.pack_into('<I', corrupt, offset, value)
            catalog.write_bytes(corrupt)
            run(command, False)
        catalog.write_bytes(cat_data[:-1])
        run(command, False)
        catalog.write_bytes(cat_data)
        for offset, value in ((0, 0), (4, 0xFFFFFFFF), (8, 18), (12, 0xFFFFFFFF),
                              (28, 0xFFFFFFFF), (32, 0xFFFFFFFF), (32 + 128*4, 0xFFFFFFFF)):
            corrupt = bytearray(idx_data)
            struct.pack_into('<I', corrupt, offset, value)
            index.write_bytes(corrupt)
            run(command, False)
        index.write_bytes(idx_data[:-1])
        run(command, False)
        index.unlink()
        run(command, False)
    print('artifact bounds, finite vectors, graph IDs, compatibility, missing index and CLI controls: PASS')


if __name__ == '__main__':
    main()
