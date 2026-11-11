"""The comparison scan must retain completed points if interrupted."""
import csv
import pytest

from bench_fasteq_compare import IncrementalRows


def test_incremental_rows_survive_interruption(tmp_path):
    path = tmp_path / 'partial.csv'
    row = {name: '' for name in IncrementalRows.FIELDS}
    row.update(mode='after', config='SMALL_P1', batch=512, precision='fp32')
    with pytest.raises(RuntimeError, match='interrupt'):
        with IncrementalRows(path) as sink:
            sink.append(row)
            with path.open(newline='') as stream:
                assert list(csv.DictReader(stream))[0]['config'] == 'SMALL_P1'
            raise RuntimeError('interrupt')
    with path.open(newline='') as stream:
        assert len(list(csv.DictReader(stream))) == 1
