# CST MTGNN for METR-LA

English | [日本語](README.ja.md)

Predict the next hour of traffic speed at 207 METR-LA sensors. This project
compares the [MTGNN](https://arxiv.org/abs/2005.11650) learned sensor graph
with a graph whose scores come from one `CSTLinear`. The time-series
convolutions, graph propagation and forecast head use the original architecture.

[See the PEMS-BAY outage and delay evaluation](#feed-outages-and-latency).

## Three-seed results with shared sensor coordinates

**CST had lower test MAE in all three seeds when both models received the same
sensor coordinates.** Averaged across seeds, MAE fell by **5.5% across all 12
forecast steps** and by **7.6% at 60 minutes**.

| Forecast horizon | MTGNN + coordinates | CST + coordinates | MAE reduction | Seeds favoring CST |
| --- | ---: | ---: | ---: | ---: |
| 15 min | 2.919 | 2.803 | 4.0% | 3/3 |
| 30 min | 3.347 | 3.180 | 5.0% | 3/3 |
| 60 min | 3.861 | 3.568 | 7.6% | 3/3 |
| All 12 steps | 3.306 | 3.125 | 5.5% | 3/3 |

The test MAE is in miles per hour (lower is better). Here is the all-step MAE for
each pair.

| Seed | MTGNN + coordinates | CST + coordinates | MAE reduction |
| ---: | ---: | ---: | ---: |
| 17 | 3.332 | 3.130 | 6.1% |
| 29 | 3.281 | 3.121 | 4.9% |
| 43 | 3.305 | 3.123 | 5.5% |

MTGNN concatenates the scaled coordinates to its learned sensor ID embeddings.
CST uses the same locations in its fixed Chart. Each pair shares the initial
weights outside the graph generator, batch order, and data split. Both models
trained from scratch for 100 epochs on an RTX 4090, with batch size 64 and
checkpoint selection by validation MAE. The CST graph uses 512 Atoms, with an
initial Atom learning rate of 0.0048. The graph parameter count fell from
20,000 to 3,072, and mean training time rose from 10.2 to 11.2 minutes per run.

![Validation MAE by epoch for MTGNN and CST with shared sensor coordinates, with the mean and range across three seeds](results/coordinate-controlled/validation-curves.png)

The lines show the mean validation MAE across the three seeds. The shaded areas
span the minimum and maximum at each epoch.

[Per-seed results (CSV)](results/coordinate-controlled/test-by-seed.csv) · [Summary (JSON)](results/coordinate-controlled/summary.json)

## Forecast from the published weights

Use Python 3.11 or newer. The CLI downloads the model configuration and
Safetensors weights from [Hugging Face](https://huggingface.co/takumiecd/cst-mtgnn-metr-la)
and uses `torchcst.hub.from_pretrained()` to place the model on the selected device.
The two downloadable checkpoints come from the initial seed-17 comparison below.

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

### Reuse the learned graph for repeated forecasts

The learned graph is fixed once a model is in evaluation mode. For a worker
that forecasts repeatedly with the same weights and all 207 sensors, the
benchmark computes the adjacency once and supplies it through MTGNN's
predefined adjacency path. With the published checkpoints, the maximum
prediction difference was 0 mph for eight held-out input windows.

| Model, batch 1 | Build graph every call | Reuse graph |
| --- | ---: | ---: |
| MTGNN | 4.46 ms | 4.24 ms |
| CST MTGNN | 5.59 ms | 4.19 ms |

These are median inference times over 40 calls on an Apple M4 Pro CPU with
two PyTorch threads. The precomputed FP32 adjacency is 167.4 KiB. The
[benchmark](benchmark_inference_graph.py) also measures eight-window batches.

## Initial comparison for the published checkpoints

Masked test MAE in miles per hour. Lower is better.

| Model | 15 min | 30 min | 60 min | All 12 horizons | Graph parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| MTGNN | 2.938 | 3.401 | 3.960 | 3.355 | 19,840 |
| CST MTGNN | 2.809 | 3.190 | 3.573 | 3.132 | 3,072 |

The CST graph reduced graph-generator parameters by 84.5%. Its average
test MAE was 6.7% lower. The graph contains 884 positive directed edges
before self-loops, compared with 4,140 from the baseline. In a 60-update
RTX 4090 pilot after ten warmup updates, median update time was
16.08 ms for the baseline and 17.15 ms for CST. The full runs use one
training seed and a geographic prior in the CST graph.

![Training and validation MAE for both graph models](results/learning-curves.png)

[Machine-readable results](results/summary.json)

## Training and evaluation

The baseline follows the MIT-licensed [MTGNN source](https://github.com/nnzhan/MTGNN/tree/f811746fa7022ebf336f9ecd2434af5f365ecbf6)
in `upstream/`. The CST variant uses a fixed Chart built from the
[DCRNN sensor coordinates](https://github.com/liyaguang/DCRNN/blob/602afd9d767d3aa1c9b3eac51710d6aeee12c227/data/sensor_graph/graph_sensor_locations.csv).
The rank-one factors of a 512-Atom `CSTLinear` produce a 207×207 score matrix.
Both variants use the same directional score transform, top-20 masking,
self-loop handling and normalization. Ranking jitter is used during training
and evaluation uses deterministic top-k.

Each run trains all 12 horizons from random initialization, with batch size 64
and 100 epochs on one RTX 4090. The shared parameters use AdamW
with initial LR 0.001 and cosine decay. The CST Atom group starts at 0.0048.
Training and evaluation omit zero-speed missing targets. We split chronologically
at 70% and 80% of timestamps and retain target windows entirely inside one split.
Normalization is fitted on training inputs. Each model is tested using the
checkpoint with the lowest validation MAE averaged over all 12 horizons.

```bash
curl -fL 'https://raw.githubusercontent.com/liyaguang/DCRNN/602afd9d767d3aa1c9b3eac51710d6aeee12c227/data/sensor_graph/graph_sensor_locations.csv' -o data/locations.csv
for seed in 17 29 43; do
  for variant in baseline_coords cst; do
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python train.py --variant "$variant" \
      --data data/METR-LA.csv --locations data/locations.csv --epochs 100 \
      --batch-size 64 --atoms 512 --seed "$seed" \
      --output "output/coordinate-controlled/seed-$seed-$variant"
  done
done
```

To reproduce the initial comparison and export the two published variants, run
the following commands.

```bash
for variant in baseline cst; do
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python train.py --variant "$variant" \
    --data data/METR-LA.csv --locations data/locations.csv --epochs 100 \
    --batch-size 64 --atoms 512 --seed 17 --output "output/paired/$variant"
done
python export.py --run output/paired --output output/weights
```

## PEMS-BAY transfer experiment

Both models train from scratch on 325 Bay Area sensors using the same
[DCRNN coordinates](https://github.com/liyaguang/DCRNN/blob/602afd9d767d3aa1c9b3eac51710d6aeee12c227/data/sensor_graph/graph_sensor_locations_bay.csv)
and [PEMS-BAY speed series](https://zenodo.org/records/5146275).
The protocol uses a chronological 70/10/20 split, 10 epochs, batch size 64,
and two paired seeds on one RTX 4090. Each pair shares the initial non-graph
weights and data order. The model with the lowest validation MAE is evaluated
on 10,412 test windows. PEMS-BAY timestamps use Pacific local time, with
cadence checked in UTC through the March daylight-saving transition.

| Test MAE (mph), two-seed mean | MTGNN + coordinates | CST + coordinates | Reduction |
| --- | ---: | ---: | ---: |
| All 12 forecast steps | 1.741 | 1.616 | 7.1% |
| 60 minutes ahead | 2.207 | 1.986 | 10.0% |
| Measurements below 45 mph | 7.553 | 6.751 | 10.6% |

CST had lower overall MAE in both paired seeds. The persistence baseline
scored 2.170 mph, and the weekly median profile with a validation-tuned
correction from the latest measurement scored 1.894 mph. CST reduced the
graph parameters from 29,440 to 3,072. The pilot step median was 37.6 ms for
MTGNN and 38.8 ms for CST, with about 2.30 GiB allocated at peak in both runs.

[Validation curves](results/pems-bay/validation-curves.png) ·
[Per-seed results](results/pems-bay/test-by-seed.csv) ·
[Metrics and baselines](results/pems-bay/summary.json)

The first seed's Safetensors were reloaded on CPU and tested through the
forecast CLI. The export command takes paired runs in `baseline_coords/` and
`cst/`. The coordinate file is the DCRNN CSV with the header
`sensor_id,latitude,longitude`.

```bash
for seed in 17 29; do
  for variant in baseline_coords cst; do
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python train.py --dataset pems-bay \
      --variant "$variant" --data data/PEMS-BAY.csv --locations data/locations-bay.csv \
      --epochs 10 --batch-size 64 --atoms 512 --seed "$seed" \
      --output "output/pems-bay/s$seed/$variant"
  done
done
python export.py --run output/pems-bay/s17 --baseline-variant baseline_coords \
  --output output/pems-bay/weights-s17
```

### Full 100-epoch comparison

On one RTX 4090, we repeated the coordinate-controlled comparison from
scratch for 100 epochs per variant and seed (17, 29, 43). Every run uses the
same split, batch size 64, and the same seed-specific initial non-graph
weights and data order within each pair. Test MAE uses each run's
validation-selected checkpoint.

| Test MAE (mph), three-seed mean | MTGNN + coordinates | CST + coordinates | Reduction |
| --- | ---: | ---: | ---: |
| All 12 forecast steps | 1.745 | 1.586 | 9.1% |
| 15 minutes ahead | 1.413 | 1.333 | 5.6% |
| 30 minutes ahead | 1.794 | 1.654 | 7.8% |
| 60 minutes ahead | 2.206 | 1.928 | 12.6% |

CST had lower overall test MAE in all three seeds. The baseline selected
epochs 20, 11, and 10, while CST selected epochs 74, 89, and 81.
Average time per run was 36.43 min for the baseline and 38.03 min for CST.
Peak allocated memory was 2.296 GiB and 2.303 GiB, respectively.
All six checkpoints were verified by SHA-256.

[Validation curves](results/pems-bay-100/validation-curves.png) ·
[Per-seed results](results/pems-bay-100/test-by-seed.csv) ·
[Verified metrics and checkpoint hashes](results/pems-bay-100/summary.json)

### Forecast from the PEMS-BAY weights

The [PEMS-BAY checkpoint collection](https://huggingface.co/takumiecd/cst-mtgnn-pems-bay) contains the six
validation-selected weights from the 100-epoch runs. After installing
`requirements.txt`, run either model with the bundled synthetic input.

```bash
curl -fL https://huggingface.co/takumiecd/cst-mtgnn-pems-bay/resolve/main/example_input.csv -o example_input.csv
python infer.py example_input.csv --dataset pems-bay --seed 17 --variant cst
python infer.py example_input.csv --dataset pems-bay --seed 17 --variant baseline_coords
```

Use `--seed 17`, `--seed 29`, or `--seed 43`. The CLI downloads the selected
Safetensors and configuration, and returns 12 forecasts at five-minute
intervals for the first sensor unless you specify `--sensor`. Use
`--device cuda` with a CUDA-enabled PyTorch installation. To forecast with
your own CSV, supply all 325 sensor-ID columns and five-minute timestamps
in Pacific local time. The last 12 rows form the input window.

The coordinate-aware MTGNN can select different neighbors on CPU and CUDA
when graph scores tie at the top-20 cutoff. We report the benchmark MAE
from the RTX 4090 evaluation.

You can also download a checkpoint and forecast from local files.

```bash
hf download takumiecd/cst-mtgnn-pems-bay config.json example_input.csv \
  seed-43/cst.safetensors --local-dir ./pems-bay-weights
python infer.py pems-bay-weights/example_input.csv --variant cst \
  --config pems-bay-weights/config.json \
  --weights pems-bay-weights/seed-43/cst.safetensors
```

### Feed outages and latency

We replayed the 10,412 held-out forecast windows for all six validation-selected
checkpoints from the 100-epoch runs on one RTX 4090 and reproduced the original
clean-input MAE for each checkpoint. The inputs use the HTTP service's carry
policy. It carries the latest valid speed within the one-hour input window and
uses the training mean for gaps before the first valid reading.

Synthetic faults follow the same timeline for all models and seeds. For each
six-hour period, we select a random sensor and its 32 nearest neighbors by
coordinates. The 30- and 120-minute outages share an onset within the period.
Regional delivery latency applies to the selected group's measurements for the
full period. At each forecast origin, the replay withholds readings that have
yet to arrive and retains the original targets. A sensor counts as affected
when at least one otherwise valid reading is withheld from its input window.

The table reports three-seed mean MAE (mph) across all 12 horizons and, in
parentheses, the number of paired seeds with lower CST MAE.

| Feed condition | MTGNN + coordinates | CST | Overall MAE reduction | Affected-sensor MAE reduction |
| --- | ---: | ---: | ---: | ---: |
| No injected fault | 1.744 | 1.585 | 9.1% (3/3) | — |
| Regional outage, 30 min | 1.749 | 1.589 | 9.2% (3/3) | 11.4% (3/3) |
| Regional outage, 120 min | 1.823 | 1.658 | 9.0% (3/3) | 7.2% (3/3) |
| Regional delay, 15 min | 1.780 | 1.610 | 9.6% (3/3) | 11.7% (3/3) |
| Network-wide delay, 5 min | 1.904 | 1.681 | 11.7% (3/3) | 11.7% (3/3) |
| Network-wide delay, 15 min | 2.151 | 1.865 | 13.3% (3/3) | 13.3% (3/3) |
| Network-wide delay, 30 min | 2.537 | 2.201 | 13.2% (3/3) | 13.2% (3/3) |
| 120 min regional outage + 15 min network-wide delay | 2.233 | 1.946 | 12.9% (3/3) | 12.9% (3/3) |

With five minutes of network-wide delay, CST achieved 1.681 mph
MAE. The coordinate-aware MTGNN with no injected fault scored
1.744 mph. At 15 minutes of delay, CST MAE rose to
1.865 mph. Among the affected sensors during the two-hour
regional outage, MAE fell from 3.338 to
3.099 mph with CST.

To rerun the replay, pass the directories containing each group's
`results/<variant>-s<seed>/best.pt` and `metrics.json`.

```bash
python benchmark_pems_feed.py data/PEMS-BAY.csv data/locations-bay.csv \
  output/pems-bay-100-seeds-17-29 output/pems-bay-100-seed-43 \
  output/pems-bay-feed.json
```

[Reduction by scenario](results/pems-bay-feed-robustness/feed-robustness.png) ·
[Per-seed results](results/pems-bay-feed-robustness/test-by-seed.csv) ·
[All groups, horizon metrics and provenance](results/pems-bay-feed-robustness/summary.json) ·
[Replay code](benchmark_pems_feed.py) ·
[Hugging Face evaluation](https://huggingface.co/takumiecd/cst-mtgnn-metr-la#pems-bay-feed-outage-and-delay-evaluation)

### Serve all 325 PEMS-BAY sensors

Start the local HTTP service with the exported PEMS-BAY checkpoint.

```bash
python forecast_service.py --config output/pems-bay/weights-s17/config.json \
  --weights output/pems-bay/weights-s17/cst.safetensors
curl -fsS http://127.0.0.1:8080/health
curl -fsS -H 'Content-Type: application/json' --data-binary @request.json \
  http://127.0.0.1:8080/forecast
```

The request contains `timestamps` (12 consecutive five-minute ISO 8601
timestamps with UTC offsets), `sensor_ids` (known sensor IDs in column
order), and `speeds_mph` (12 rows with one reading per supplied sensor).
`null`, zero, and omitted sensor columns are treated as missing. The default
fills gaps with each sensor's latest observed reading within the input window.
Leading gaps and entirely missing sensors use the training mean. The response
returns all 325 sensors in checkpoint order, 12 future timestamps, 12 rows of
predicted speeds, and counts of missing readings and fully missing sensors.
The service loads the model and computes the adjacency once at startup.

On the Apple M4 Pro CPU with two PyTorch threads, the median of 40 complete
local HTTP requests was 9.24 ms and the 95th percentile was 9.85 ms.
The response matched the earlier single-sensor CLI for the same held-out input.
[Service verification](results/pems-bay/service.json)

We simulated an outage of 33 randomly selected sensors per input window,
with the same outage masks across models and policies. Validation selected
the default policy. The following are test MAEs for the affected sensors
on 10,412 windows across both CST seeds.

| Outage | Feed zeros | Fill from available readings |
| --- | ---: | ---: |
| Last 15 minutes | 36.79 mph | 1.90 mph |
| Entire 60-minute input window | 38.90 mph | 4.80 mph |

With no simulated outage, MAE for the same selected sensors was 1.62 mph.
[Outage results](results/pems-bay/outages.json) · [Benchmark](benchmark_sensor_outages.py)

New code and released weights use [Apache-2.0](LICENSE). The vendored
MTGNN source uses its [MIT license](upstream/LICENSE).
