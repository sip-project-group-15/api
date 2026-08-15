# Data sources

Every labelled dataset considered for this project, why it is or is not useful,
and what it costs to fetch.

**The gap this list exists to document:** no public dataset contains rhinos and
the people threatening them, seen from the air, in RGB. That combination is
what our own drone footage has to supply. Everything below is a way of getting
a working, alert-capable model *before* that footage exists — it does not
remove the need for it.

Classes we train (`app/config.py`): `0 rhino`, `1 person`, `2 vehicle`,
`3 weapon` (declared, not trained — see [Weapons](#weapons)).

Merge them with `training/build_dataset.py`, which remaps classes by name and
caps per-class instances so a dense source cannot drown a sparse one.

---

## Tier 1 — use these now

Both download on demand through Ultralytics, so they need no manual step.

### African Wildlife
Ground-level photography of `buffalo, elephant, rhino, zebra`.

| | |
|---|---|
| Rhinos | **yes** — the only easy rhino source |
| People / vehicles | no |
| Size | ~100 MB, ~1.5k images |
| Fetch | `african-wildlife.yaml` (built into Ultralytics) |
| Measured yield | 399 rhino boxes train, 85 val |

Ground-level, so a model trained only on this will not generalise to top-down
drone frames — different silhouette, scale and background. It is a starting
point for the `rhino` class, not a finished training set.

### COCO val2017
Ground-level people and vehicles — the half the aerial sources cannot supply.

| | |
|---|---|
| People / vehicles | **yes** — ground level |
| Rhinos | no |
| Size | 816 MB images + 241 MB annotations |
| Fetch | `python training/fetch_coco.py` |
| Measured yield | **2,951 images — 10,777 person, 2,982 vehicle** |

`val2017` rather than `train2017`: 816 MB instead of 19 GB, still far more
instances than any cap takes, and genuinely unseen — the YOLO base weights were
pretrained on train2017, so none of this was in that.

Mapped `person→person` and `car`/`truck`/`bus`/`motorcycle→vehicle`. Only leaf
categories, so nothing is boxed twice. Crowd regions (`iscrowd`) are dropped —
one box over many overlapping people would train a single enormous `person`.

> **Why this is not optional.** The first trained model learned `rhino` from
> ground photos and `person`/`vehicle` from aerial drone footage. Measured on
> the real weights: rhino 95% detected at ground level, person/vehicle 92% of
> aerial images — but **nothing at all** for person/vehicle at ground level
> (max score 0.016). No single viewpoint saw all three classes, so it produced
> zero alerts in either domain.

### VisDrone
Real aerial imagery, 10 classes: `pedestrian, people, bicycle, car, van, truck,
tricycle, awning-tricycle, bus, motor`.

| | |
|---|---|
| People / vehicles | **yes** — correct aerial viewpoint |
| Rhinos | no |
| Size | 2.3 GB — 6,471 train / 548 val / 1,610 test |
| Fetch | `VisDrone.yaml` (built into Ultralytics) |
| Docs | https://docs.ultralytics.com/datasets/detect/visdrone |

Our remap:

```
pedestrian=1,people=1,car=2,van=2,truck=2,bus=2,motor=2
```

`bicycle`, `tricycle` and `awning-tricycle` are dropped.

> **Cap this source.** VisDrone frames carry hundreds of tiny vehicles and
> pedestrians each. Merged unrestricted against a few hundred rhino boxes it is
> roughly **1000:1**, and a detector trained on that stops predicting rhino
> entirely while its headline mAP still looks respectable. `--cap 1500` keeps
> it near 4:1.

---

## Tier 2 — aerial, with animals *and* people in the same frames

The most valuable data for this project, because it contains the actual
co-occurrence the scorer measures. All are large; take a subset.

| dataset | classes | annotations | size | link |
|---|---|---|---|---|
| **DAZZLE** | zebra, **person**, **vehicle** | 162,931 boxes, 306k frames | 96 GB | [DaRUS](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-5162) |
| **Koger Drones** | zebra, gazelle, waterbuck, buffalo, gelada, **human** | 40,532 boxes, 1,982 images | 65 GB | [Edmond](https://edmond.mpdl.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.EMRZGH) |
| **BIRDSAI** | **human** + elephant, lion, giraffe, rhino, hippo, zebra | 166,221 boxes, 61,994 frames | 3.7 GB real | [LILA BC](https://lila.science/datasets/conservationdrones/) |

**DAZZLE** matches our exact three-class structure and is the closest public
analogue to what we are building.

**BIRDSAI** deserves separate mention: long-wave thermal infrared, nighttime,
Southern Africa, built explicitly for anti-poaching, permissively licensed
(CDLA-permissive), and small enough for Colab. Do **not** mix it into RGB
training — thermal is a different domain and would poison the RGB model. It is
the obvious basis for a separate *night* model later, which is when poaching
actually happens.

---

## Tier 3 — aerial animals only

Useful for teaching the top-down animal silhouette even where the species
differ. None contain people.

| dataset | animals | annotations | size | link |
|---|---|---|---|---|
| Eikelboom Savanna | zebra, giraffe, elephant | 4,305 boxes / 561 images | 5.7 GB | [4TU](https://data.4tu.nl/articles/dataset/Improving_the_precision_and_accuracy_of_animal_population_estimates_with_aerial_image_object_detection/12713903/1) |
| Delplanque Mammals | buffalo, kob, warthog, waterbuck, elephant | 10,239 boxes / 1,297 images | 12 GB | [ULiège](https://dataverse.uliege.be/file.xhtml?fileId=11098&version=1.0) |
| Aerial Elephant (AED) | elephant | 15,581 **points**, not boxes | 16.3 GB | [Zenodo](https://zenodo.org/record/3234780) |
| SAVMAP / Kuzikus | mixed Namibian savanna | 1,183 boxes / 654 images | — | [EPFL paper](https://infoscience.epfl.ch/server/api/core/bitstreams/56e868af-c042-44d7-acc0-b7f67a04d2ff/content) |
| BuckTales | blackbuck antelope | 18,400+ boxes | 80 GB | [Edmond](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.JCZ9WK) |
| MMLA-Mpala | zebra, giraffe | ~617,000 boxes | 490 GB | [HuggingFace](https://huggingface.co/datasets/imageomics/mmla_mpala) |
| MMLA-OPC | zebra | ~163,000 boxes | 64 GB | [HuggingFace](https://huggingface.co/datasets/imageomics/mmla_opc) |
| MMLA-Wilds | zebra, giraffe, onager, dog | — | 21 GB | [HuggingFace](https://huggingface.co/datasets/imageomics/mmla_wilds) |
| WAID | sheep, cattle, seal, camel, kiang, zebra | boxes / 14,366 images | 1.5 GB | [GitHub](https://github.com/xiaohuicui/WAID/tree/main/WAID) |
| HerdNet | African mammals, aerial | models + data | — | [GitHub](https://github.com/Alexandre-Delplanque/HerdNet) |

**AED is point annotations, not boxes** — it needs conversion before YOLO can
use it, and a synthesised box size is a guess.

SAVMAP is the closest match to our deployment conditions: eBee UAV over Kuzikus
Wildlife Reserve, Namibia, 120–160m altitude, 4–8cm ground resolution. Note the
average annotation is **25 × 23 pixels** — a useful reality check on how small
these targets are, and an argument for keeping `imgsz=640` rather than lowering
it for speed.

---

## More rhinos

399 boxes is thin, and rhino recall is a floor on the whole system: lose the
rhino and the scorer loses the ruler it measures every distance with.

- **Open Images V7** — has a `Rhinoceros` class among its 600 boxable classes.
  Pull only that via FiftyOne:
  [Ultralytics docs](https://docs.ultralytics.com/datasets/detect/open-images-v7) ·
  [FiftyOne docs](https://docs.voxel51.com/dataset_zoo/datasets/open_images_v7.html)

  ```python
  fiftyone.zoo.load_zoo_dataset(
      "open-images-v7", split="train",
      label_types=["detections"], classes=["Rhinoceros"],
  )
  ```

- **Roboflow Universe** — [247 rhino-matching datasets](https://universe.roboflow.com/search?q=class:rhino),
  exporting directly in YOLOv8 format. Individually small and quality varies,
  but they stack:
  - [741 images](https://universe.roboflow.com/tappay-nhzuq/project-e9qjp-kmkth)
  - [200 images](https://universe.roboflow.com/ng-wei-xiang/rhino-dataset)
  - [54 images, Javan rhino](https://universe.roboflow.com/project-b9qen/rhino-sj7yu)
  - [10 images, thermal](https://universe.roboflow.com/simantika-choudhury/thermal-rhino-data/dataset/1)

---

## Weapons

Labelled weapon data exists, but **not of the kind this project needs**, and
the recommendation is not to train the class.

| dataset | contents | link |
|---|---|---|
| Roboflow weapon detection | 4,556 images, rifle / handgun / knife | [Universe](https://universe.roboflow.com/weapon-detect-qbsiw/yolo-weapon-detection) |
| Kaggle gun detection | firearms, YOLO v2/v3 labels | [Kaggle](https://www.kaggle.com/datasets/atulyakumar98/gundetection) |
| Orientation-aware weapons benchmark | oriented boxes | [arXiv](https://arxiv.org/pdf/2112.02221) |

All of it is frontal, close-range, indoor CCTV framing of handguns and knives.
Our case is a rifle carried through bush, seen from 80m up, perhaps 15 pixels
long and usually occluded by the person carrying it. The domain gap is wider
than the one between ground-level and aerial rhinos.

More decisive: **the anti-poaching drone literature does not attempt weapon
detection at all.** SPOT, BIRDSAI and the Tanzania miombo trials all detect
*humans*, because a person on foot inside a protected area — especially at
night — is already the signal.

- [AI for Social Impact / SPOT](https://arxiv.org/pdf/2001.00088)
- [Factors affecting poacher detection with drones](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8232034/)
- [TIR vs RGB detection, miombo woodlands, Tanzania](https://www.sciencedirect.com/science/article/abs/pii/S0006320718315726)
- [Handheld thermal imaging for poacher detection](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4481516/)

So `weapon` stays declared in `app/config.py` and treated by `app/threat.py` as
an escalator that can raise an alert's severity but can never raise one alone.
Effort is better spent on person recall.

---

## Curated lists

Worth revisiting as new datasets appear.

- [agentmorris/drone-wildlife-datasets](https://github.com/agentmorris/drone-wildlife-datasets) — the most complete list of aerial wildlife survey datasets
- [lila.science/aerialdata](http://lila.science/aerialdata)
- [LILA BC datasets](https://lila.science/datasets/) — camera-trap and aerial conservation data
- [LILA BC, other datasets](https://lila.science/otherdatasets/)

---

## Provenance

Directly verified: African Wildlife (downloaded and merged — the 399/85 figures
are measured, not quoted), VisDrone, BIRDSAI, and the curated lists.

Tier 2 and Tier 3 figures come from the `drone-wildlife-datasets` list and have
**not** been independently checked. Confirm sizes and licences before relying
on any of them.
