import sys
import math
import numpy as np

sys.path.append('..')
import modules as mm


dataset_shapes = [(100, 20), (1000, 100), (10000, 500)]
batch_sizes = [1, 16, 100, 500, 512, 1000]
num_tests = 3
random_seed = 9


def _test_dataloader(dataset_shape, batch_size=1, shuffle=False, unsqueezed_y=True,
                     outer_iters=10, inner_iters=2, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    for _ in range(outer_iters):
        X = np.random.randn(*dataset_shape)
        if unsqueezed_y:
            y = np.random.randn(dataset_shape[0], 1)
            y[:, 0] = np.sort(y[:, 0])
        else:
            y = np.random.randn(dataset_shape[0])
            y = np.sort(y)

        dataloader = mm.DataLoader(X, y, batch_size=batch_size, shuffle=shuffle)
        debug_msg = 'Error in DataLoader: '
        assert len(dataloader) == math.ceil(dataset_shape[0] / batch_size), debug_msg + 'wrong len()'
        assert dataloader.num_samples() == dataset_shape[0], debug_msg + 'wrong num_samples()'

        old_X_collected, old_y_collected = None, None
        for _ in range(inner_iters):
            X_collected, y_collected = [], []

            for i, (X_batch, y_batch) in enumerate(dataloader):
                if i < len(dataloader) - 1:
                    msg = debug_msg + 'wrong mini-batch shape'
                    assert X_batch.shape == (batch_size, dataset_shape[1]), msg
                    if unsqueezed_y:
                        assert y_batch.shape == (batch_size, 1), msg
                    else:
                        assert y_batch.shape == (batch_size, ), msg
                else:
                    msg = debug_msg + 'wrong last mini-batch shape'
                    assert len(X_batch.shape) == 2 and X_batch.shape[0] <= batch_size and \
                           X_batch.shape[1] == dataset_shape[1], msg
                    assert y_batch.shape[0] <= batch_size, msg
                    assert X_batch.shape[0] == y_batch.shape[0], msg
                    if unsqueezed_y:
                        assert len(y_batch.shape) == 2 and y_batch.shape[1] == 1, msg
                    else:
                        assert len(y_batch.shape) == 1, msg

                X_collected += [X_batch]
                y_collected += [y_batch]

            X_collected = np.concatenate(X_collected, axis=0)
            y_collected = np.concatenate(y_collected, axis=0)

            msg = debug_msg + 'collected dataset has wrong shape'
            assert X_collected.shape == X.shape, msg
            assert y_collected.shape == y.shape, msg

            msg = debug_msg + 'collected dataset has wrong type'
            assert X_collected.dtype == X.dtype, msg
            assert y_collected.dtype == y.dtype, msg

            if shuffle:
                msg = debug_msg + 'dataset not shuffled'
                assert not (X == X_collected).all(), msg
                assert not (y == y_collected).all(), msg

                if old_X_collected is not None:
                    msg = debug_msg + 'dataset not shuffled'
                    assert not (X == old_X_collected).all(), msg
                    assert not (y == old_y_collected).all(), msg

                old_X_collected = X_collected
                old_y_collected = y_collected

                if unsqueezed_y:
                    ordering = np.argsort(y_collected[:, 0])
                else:
                    ordering = np.argsort(y_collected)

                y_collected = y_collected[ordering]
                X_collected = X_collected[ordering]

            msg = debug_msg + 'collected and initial datasets do not match'
            assert (X == X_collected).all(), msg
            assert (y == y_collected).all(), msg


def test_dataloader():
    print('test_dataloader ... ', end='')
    for dataset_shape in dataset_shapes:
        for batch_size in batch_sizes:
            for shuffle in (False, True):
                for unsqueezed_y in (False, True):
                    _test_dataloader(
                        dataset_shape, batch_size, shuffle, unsqueezed_y=unsqueezed_y,
                        outer_iters=num_tests, random_seed=dataset_shape[0] + random_seed
                    )

    print('OK')
