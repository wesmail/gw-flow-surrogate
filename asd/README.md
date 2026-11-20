# Einstein Telescope sensitivity curves

Official amplitude spectral densities (ASD) from the ET design-comparison study
(Branchesi et al. 2023, [arXiv:2303.15923](https://arxiv.org/abs/2303.15923)),
taken from the public [gwfast](https://github.com/CosmoStatGW/gwfast) repository:

```text
https://github.com/CosmoStatGW/gwfast/tree/master/psds/ET_designs_comparison_paper/HFLF_cryo
```

| File | Design | Source file in gwfast |
|------|--------|------------------------|
| `ET_HFLF_cryo_15km.txt` | 2L (15 km arms), full HF+LF cryogenic | `ETLength15km.txt` |
| `ET_HFLF_cryo_10km.txt` | triangle (10 km), full HF+LF cryogenic | `ETLength10km.txt` |

**Columns** (whitespace-separated, no header):

1. frequency [Hz] — 1 Hz to 10 kHz (3000 rows)
2. ASD [1/√Hz]

SHA-256:

```text
9e50a06fefbb5fa1312a296cedabfc1377441f59806fc7fcb6b4c2b0b807f378  ET_HFLF_cryo_15km.txt
a8b84c228e37fdfbf49fcabc2115f4111b9bcaa0ac75b6db858c64a3e1233169  ET_HFLF_cryo_10km.txt
```
