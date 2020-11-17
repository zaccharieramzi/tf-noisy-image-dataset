from pathlib import Path

import pytest

from tf_noisy_image_dataset.dataset import NoisyDatasetBuilder


data_path = Path(__file__).parent / 'data'
PATCH_SIZE = 96
@pytest.mark.parametrize('patch_size', [None, PATCH_SIZE])
@pytest.mark.parametrize('noise_config', [
    {},
    {'fixed_noise': True},
    {'noise_input': True},
])
def test_init_and_iter(patch_size, noise_config):
    ds_builder = NoisyDatasetBuilder(
        paths=data_path,
        extension='png',
        batch_size=1,
        patch_size=patch_size,
        noise_config=noise_config,
        to_grey=True,
    )
    iterator = ds_builder.noisy_image_ds.as_numpy_iterator()
    model_inputs, model_outputs = next(iterator)
    if noise_config.get('noise_input', False):
        assert len(model_inputs) == 2
        model_inputs = model_inputs[0]
    assert model_inputs.shape == model_outputs.shape
    assert model_inputs.shape[0] == 1
    assert model_inputs.shape[-1] == 1
    if patch_size is not None:
        assert model_inputs.shape[1] == PATCH_SIZE
        assert model_inputs.shape[2] == PATCH_SIZE
