# dog-face-landmarks-training

Training code, evaluation harness, and experiment journal for the dog facial
landmark models that ship in the
[dog_detection](https://github.com/hugocornellier/dog_detection) Flutter package.

Two models are produced here, both exported to TFLite:

- **Face localizer**, which finds and crops the dog's face
- **Landmark detector**, which predicts the 46-point DogFLW scheme on that crop

Dogs are substantially harder than cats. The same architecture that reaches 3.27
NME_IOD on CatFLW reaches 8.77 here, and the gap is structural rather than a
tuning problem. See the journal for the per-landmark analysis.

## Results

Accuracy is NME_IOD (normalized mean error, inter-ocular distance) on the DogFLW
validation split. Lower is better. The paper's ELD ensemble target is 6.52; this
project started near 40.

| Model | Backbone | NME_IOD | + flip TTA | + ms&flip TTA | TFLite |
|---|---|---|---|---|---|
| 3-model ensemble (256+320+384) | EfficientNetV2S | 8.91 | 8.67 | **8.04** | 3x 55 MB |
| `tight_margin_384` | EfficientNetV2S | **8.77** | 8.49 | 8.33 | 55 MB |
| `tight_margin_320` | EfficientNetV2S | 8.82 | 8.52 | 8.32 | 55 MB |
| `small_v3large_384_long` | MobileNetV3Large | **8.57** | | | **11 MB** |

`small_v3large_384_long` is what ships in the Flutter package: it is the best
accuracy that still runs in real time on a phone. It was trained after the
journal's main comparison table was written, so the journal documents it only in
the Round 8 export benchmark. Architecture across all of them is backbone +
4x Conv2DTranspose heatmap head + SoftArgmax2D.

The ensemble at 8.04 is the best result here, but it costs 18 forward passes and
is not something you would put on a phone.

**Where the error lives:** ears. Ear landmarks run NME 12 to 14 against roughly 5
for eyes, and ear tips reach 15 to 18. The train-val gap sits near 3.4 across
every model tried, which regularization did not move. Those two facts are the
whole story of why this number is not 6.52, and they are worked through properly
in [`LANDMARK_DETECTION_REPORT.md`](LANDMARK_DETECTION_REPORT.md).

That journal covers eight rounds of experiments, a ranked list of what failed,
and the per-landmark breakdown. [`NME_PUSH_PLAN.md`](NME_PUSH_PLAN.md) is the
diagnosis written at 9.11 of why the paradigm, not the backbone, was the
bottleneck. If you are picking this up, read those before this README.

## Read this before exporting any model

The static-vs-dynamic TFLite export choice depends on which runtime and which
delegate you are targeting, and it has inverted more than once:

- On **XNNPACK CPU**, static was a 1.58x win on flutter_litert 3.6.0 and a 2.13x
  loss on 3.7.0, with byte-identical accuracy.
- On **GPU via CompiledModel**, static is mandatory: a dynamic batch dimension
  makes every Conv2DTranspose build its output shape at run time, and GPU
  backends refuse the graph outright. Re-exporting from a batch-1 concrete
  function is what let the shipped models reach the GPU, worth 27.10 ms to
  3.82 ms on an M4 Max.

The models that ship today are the static exports, for the GPU reason. Benchmark
both exports against the exact runtime and delegate you ship against, never
against Python `tf.lite`.

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` pins `tensorflow-macos` and `tensorflow-metal`, so it installs
as-is only on Apple Silicon. On Linux or CUDA, substitute `tensorflow==2.15.0`
and drop the `-metal` package.

The DogFLW dataset is pulled automatically through `kagglehub` on first run, to
your own machine, under your own acceptance of Kaggle's terms. No dataset content
is redistributed in this repository.

## Training

```bash
# Best single model (384px)
python scripts/train_dog_face_landmarks.py --experiment tight_margin_384 \
  --out artifacts/tight_margin_384

# Resolution sweep
python scripts/train_dog_face_landmarks.py --experiment tight_margin_320 \
  --out artifacts/tight_margin_320
python scripts/train_dog_face_landmarks.py --experiment tight_margin_256 \
  --out artifacts/tight_margin_256

# Quick smoke test
python scripts/train_dog_face_landmarks.py --experiment tight_margin_256 \
  --epochs 5 --patience 50 --out artifacts/smoke
```

Evaluation:

```bash
python scripts/eval_all_models.py        # every model plus ensembles and TTA
python scripts/gen_landmark_examples.py  # prediction visualizations
```

There are 72 scripts in `scripts/`, most of them one-off experiments from a
specific round. The journal names the ones that matter.

## Weights

The trained weights are released on Hugging Face:

**[hugocornellier/dog-face-landmarks](https://huggingface.co/hugocornellier/dog-face-landmarks)**

That repository holds the face localizer, `dog_face_landmarks_full.tflite` (11 MB, ships in the Flutter package) and the EfficientNetV2-S 384 model (55 MB), the `.keras` sources for
fine-tuning, and the per-model training config and epoch logs. The model card
documents the input and output contract, which is the part you need to actually
use them.

Weights are **CC BY-NC 4.0**, non-commercial. See the License section below for
why, and note that the code here is Apache 2.0: the two are different.

## Dataset

Models here are trained on the
[DogFLW dataset](https://github.com/martvelge/DogFLW) by Martvel et al.,
Tech4Animals Lab, University of Haifa.

DogFLW is licensed
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Obtain it from
[Kaggle](https://www.kaggle.com/datasets/georgemartvel/dogflw) under its own
terms. No images, annotations, or derived crops from the dataset are included
here.

## License

This repository carries two licenses, because the code and the weights come from
different places.

- **Code** (everything in this repository): Apache License 2.0, see
  [`LICENSE`](LICENSE).
- **Trained weights** (released on Hugging Face, see above): CC BY-NC 4.0, see
  [`LICENSE-WEIGHTS`](LICENSE-WEIGHTS).

The weights are non-commercial at the request of the dataset authors, who asked
that weights derived from DogFLW annotations remain consistent with the
non-commercial terms of the source data. They granted permission to publish them
on that basis.

## Citation

If you use this work, please cite the DogFLW paper:

```bibtex
@article{martvel2025dog,
  title={Dog facial landmarks detection and its applications for facial analysis},
  author={Martvel, George and Zamansky, Anna and Pedretti, Giulia and Canori,
          Chiara and Shimshoni, Ilan and Bremhorst, Annika},
  journal={Scientific Reports},
  volume={15},
  number={1},
  pages={21886},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

## Acknowledgements

Thanks to George Martvel, Anna Zamansky, Giulia Pedretti, Chiara Canori, Ilan
Shimshoni, and Annika Bremhorst at the Tech4Animals Lab, University of Haifa, for
publishing DogFLW and for permission to release these weights.
