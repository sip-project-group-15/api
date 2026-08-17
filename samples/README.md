# Sample media

Test footage for the API and the frontend. Small enough to live in the repo,
excluded from the Docker image (see `.dockerignore`) so they never ship.

```bash
# through the running API
curl -X POST localhost:7000/videos/analyze -F "video=@samples/approach.mp4"

# or straight through the pipeline, with boxes drawn
python training/check_model.py samples/approach.mp4 --annotate out/
python training/check_model.py samples/rhino.jpg
```

## What each one is for

| file | contains | expected |
|---|---|---|
| `rhino.jpg` | one white rhino, side on | `rhino` ~0.96 — the clean case for the distance ruler |
| `rhino-pair.jpg` | a cow and her calf | two `rhino` boxes ~0.98 / ~0.92 — **no alert**, wildlife alone is not poaching |
| `tourists-vehicle.jpg` | a game-drive truck and a guide | `person` ~0.98, `vehicle` ~0.94 |
| `approach.mp4` | a person walking towards a rhino | alerts escalating **medium → high → critical**, closing rate positive |
| `keepers-and-rhino.mp4` | **real footage** — keepers walking beside a white rhino | `critical` 0.786 on 1 of 5 analysed frames |
| `patrol.mp4` | a slow pan over the game-drive scene | scores 0.21–0.26, **all under the 0.45 threshold** |

`patrol.mp4` is the one worth understanding. It is a **hard negative**: people
and a vehicle are detected clearly, but with no rhino in frame there is nothing
to measure a distance against, so proximity falls back to its unknown baseline
and nothing crosses the threshold. Tourists near a vehicle must not raise an
alert, and this is the clip that proves it still doesn't.

`approach.mp4` exercises the whole chain — detection, tracking across frames,
distance in body-lengths, the closing rate, and the severity escalation.

`keepers-and-rhino.mp4` is the only sample here that is **real footage of a
person beside a rhino**, and it is worth understanding precisely. The proximity
call is right — a person really is about 2.8m from the animal, and `critical` is
the correct band. But the men are the rhino's own Samburu keepers, not poachers.
The system is behaving exactly as designed, flagging a human at touching
distance and leaving intent to whoever reads the alert; do not present it as
"detected a poacher".

It also shows how thin real co-occurrence footage is. Across the source video,
a rhino and a person appear together in 20 frames out of 160 in that shot, and
1 of 255 sampled across the whole clip — so only one analysed frame alerts. The
flicker is the model's 0.42 person recall, not a bug in the sampling.

> **`approach.mp4` is composited, not real footage.** A person cut from
> `tourists-vehicle.jpg` is pasted into the `rhino.jpg` scene at decreasing
> distance. No public dataset contains a person near a rhino in RGB — that gap
> is documented in [`../training/DATASETS.md`](../training/DATASETS.md) and only
> real drone footage closes it. This clip proves the pipeline reasons
> correctly; it is **not** evidence about field accuracy.
>
> The person is also missed in roughly one analysed frame in eight, which is
> the model's 0.42 person recall showing up honestly. The track survives it —
> that is what `TRACK_MAX_MISSES` is for.

## Sources and licences

All stills are from Wikimedia Commons. Attribution is required; keep this
section with the files if they are reused.

**`rhino.jpg`** and **`rhino-pair.jpg`**
White rhinoceros in the Kalahari Desert, Namibia — by **Giles Laurent**,
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
[single](https://commons.wikimedia.org/wiki/File:081_White_rhinoceros_(male)_in_the_Kalahari_Desert_of_Namibia_Photo_by_Giles_Laurent.jpg)
·
[mother and calf](https://commons.wikimedia.org/wiki/File:075_White_rhinoceros_mother_and_her_newborn_baby_in_the_Kalahari_Desert_of_Namibia_Photo_by_Giles_Laurent.jpg)

**`tourists-vehicle.jpg`**
Bedford game drive truck, Pilanesberg National Park, South Africa — by
**Brian Snelson**,
[CC BY 2.0](https://creativecommons.org/licenses/by/2.0/).
[source](https://commons.wikimedia.org/wiki/File:Bedford_game_drive_truck,_Pilanesberg_National_Park,_South_Africa_-_001.jpg)

**`keepers-and-rhino.mp4`**
A 2.3-second cut from *Rhino Joins Tinder*, **public domain** (Voice of America
/ Africa54), via Wikimedia Commons. Filmed at Ol Pejeta Conservancy, Kenya.
[source](https://commons.wikimedia.org/wiki/File:Rhino_Joins_Tinder.webm)

**`approach.mp4`** and **`patrol.mp4`** are derived from the stills above and
inherit their licences — CC BY-SA 4.0 and CC BY 2.0 respectively.

Images were downscaled to 1280px wide and re-encoded; otherwise unmodified.

## What is missing

There is no aerial sample, and it is not for want of looking. Wikimedia Commons
carries the supplementary video from an actual [rhino anti-poaching RPAS
study](https://commons.wikimedia.org/wiki/File:Remotely-Piloted-Aircraft-Systems-as-a-Rhinoceros-Anti-Poaching-Tool-in-Africa-pone.0083873.s001.ogv)
— precisely the deployment case. Run through this model it produces **zero
detections across 83 sampled frames**: no rhinos, no people. There is no point
shipping a sample whose only lesson is that the model is blind to it.

When drone footage exists, a clip of a person approaching a rhino **from the
air** is the sample that would actually test what this system is for.

Watermarked stock footage must not be added here — it cannot be redistributed
with a public repo. `samples/gettyimages-*` is gitignored for that reason.
