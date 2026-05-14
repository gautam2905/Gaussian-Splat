# References (gitignored)

Local clones for study. Not bundled at build time; not imported as deps. Refresh via re-clone.

## ML upstreams (Task 7)

| Repo | Source | License | Use |
|---|---|---|---|
| TRELLIS.2 | https://github.com/microsoft/TRELLIS.2 | MIT | Object generator: image → mesh+PBR. Spike A + production object_gen (Task 15). |
| HunyuanWorld | https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0 | Tencent HunyuanWorld-1.0 Community License (territory-restricted; not EU/UK/KR) | Scene generator: text → 3D scene. Spike B + production scene_gen (Task 14) if pass. |
| supersplat | https://github.com/playcanvas/supersplat | MIT | Gaussian splat viewer; study only (we dropped splats per spec v4). |
| splatviz | https://github.com/Florian-Barthel/splatviz | MIT | Python splat viewer; study only. |

## Pre-existing study material (relocated from earlier 3DGS work)

`4DGaussians/`, `EasyVolcap/`, `gaussian-splatting/`, `gsplat/`, `fast-gaussian-rasterization/`, `UnityGaussianSplatting/`. Kept as reading material; not used at runtime.

## Refresh

```bash
cd /mnt/3D_Mapping/references
rm -rf TRELLIS.2 HunyuanWorld supersplat splatviz
git clone --depth=1 https://github.com/microsoft/TRELLIS.2.git TRELLIS.2
git clone --depth=1 https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0.git HunyuanWorld
git clone --depth=1 https://github.com/playcanvas/supersplat.git supersplat
git clone --depth=1 https://github.com/Florian-Barthel/splatviz.git splatviz
```
