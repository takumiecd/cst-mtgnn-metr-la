# CST MTGNN for METR-LA

English | [日本語](README.ja.md)

Predict the next hour of traffic speed at 207 METR-LA sensors. This project
compares the [MTGNN](https://arxiv.org/abs/2005.11650) learned sensor graph
with a graph whose scores come from one `CSTLinear`. The time-series
convolutions, graph propagation and forecast head use the original architecture.

## Forecast from the published weights

Use Python 3.11 or newer. The CLI downloads the model configuration and
Safetensors weights from [Hugging Face](https://huggingface.co/takumiecd/cst-mtgnn-metr-la)
and uses `torchcst.hub.from_pretrained()` to place the model on the selected device.

```bash
git clone https://github.com/takumiecd/cst-mtgnn-metr-la.git
cd cst-mtgnn-metr-la
python -m pip install -r requirements.txt
mkdir -p data
curl -fL 'https://zenodo.org/records/5146275/files/METR-LA.csv?download=1' -o data/METR-LA.csv
python infer.py data/METR-LA.csv --test-index 0 --sensor 773869 --variant cst
```

This returns 12 five-minute forecasts with observed values for the first held-out
forecast window. Set `--variant baseline` to run the original model on the same
input. Set `--device cuda` with a CUDA-enabled PyTorch installation. To use your
own CSV, provide a timestamp column followed by all 207 METR-LA sensor-ID
columns at five-minute intervals and omit `--test-index`. The CLI forecasts
from the last 12 rows and accepts any sensor-column order.

## Training and evaluation

The baseline follows the MIT-licensed [MTGNN source](https://github.com/nnzhan/MTGNN/tree/f811746fa7022ebf336f9ecd2434af5f365ecbf6)
in `upstream/`. The CST variant uses a fixed Chart built from the
[DCRNN sensor coordinates](https://github.com/liyaguang/DCRNN/blob/602afd9d767d3aa1c9b3eac51710d6aeee12c227/data/sensor_graph/graph_sensor_locations.csv).
The rank-one factors of a 512-Atom `CSTLinear` produce a 207×207 score matrix.
Both variants use the same directional score transform, top-20 masking,
self-loop handling and normalization. Ranking jitter is used during training
and evaluation uses deterministic top-k.

Both variants train all 12 horizons from random initialization, using seed 17,
batch size 64 and 100 epochs on one RTX 4090. The shared parameters use AdamW
with initial LR 0.001 and cosine decay. The CST Atom group starts at 0.0048.
Training and evaluation omit zero-speed missing targets. We split chronologically
at 70% and 80% of timestamps and retain target windows entirely inside one split.
Normalization is fitted on training inputs. Each model is tested using the
checkpoint with the lowest validation MAE averaged over all 12 horizons.

```bash
curl -fL 'https://raw.githubusercontent.com/liyaguang/DCRNN/602afd9d767d3aa1c9b3eac51710d6aeee12c227/data/sensor_graph/graph_sensor_locations.csv' -o data/locations.csv
for variant in baseline cst; do
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python train.py --variant "$variant" \
    --data data/METR-LA.csv --locations data/locations.csv --epochs 100 \
    --batch-size 64 --atoms 512 --seed 17 --output "output/paired/$variant"
done
python export.py --run output/paired --output output/weights
```

New code and released weights use [Apache-2.0](LICENSE). The vendored
MTGNN source uses its [MIT license](upstream/LICENSE).
