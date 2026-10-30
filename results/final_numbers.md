# Final numbers recovered from the session log (VM ldrb was reclaimed on 2026-10-28, checkpoints lost)

## SID Sony test (598 full-resolution images)
| model | PSNR | SSIM | MS-SSIM | L1 | eval s |
|---|---|---|---|---|---|
| SID U-Net | 28.6197 | 0.7961 | 0.8764 | 0.0288 | 635 |
| Bridge U-Net | 28.3649 | 0.7886 | 0.8679 | 0.0297 | 517 |
| Bridge w/o distill | 28.9370 | 0.7981 | 0.8759 | 0.0276 | 339 |
| Bridge w/o FiLM | 28.1088 | 0.7872 | 0.8677 | 0.0306 | 286 |
| DDBM-Feature, single pass (N=0) | 25.5140 | 0.7329 | 0.8412 | 0.0438 | 338 |
| DDBM-Feature, iterative (N=3) | 18.0229 | 0.4489 | 0.7424 | 0.1101 | 408 |
| DDBM VE-BB, N=5 | 28.0262 | 0.7816 | 0.8621 | 0.0320 | 1044 |
| DDBM VE-BB, N=10 | 27.8971 | 0.7787 | 0.8558 | 0.0322 | 1620 |
| DDBM VE-BB, N=10 eta=0 (DBIM) | 27.8829 | 0.7755 | 0.8535 | 0.0325 | 2033 |
| DDBM VE-BB, N=20 | 27.7665 | 0.7759 | 0.8505 | 0.0324 | 3505 |
| DDBM official (sigma_max 80) | test eval lost with the VM | | | | |

## SID val (center 512 crops), final step
sid 33.07 @40k; bridge 32.35 @40k; bridge_nodistill 33.17 @40k; bridge_nofilm 32.02 @40k;
ddbm_feature 12.27 @40k (val uses iterative inference); ddbm (VE-BB) 31.43 @60k;
ddbm_official 24.09 @60k.

ddbm_official val PSNR by step: 2k 11.42, 4k 12.46, 6k 12.60, 8k 12.55, 10k 12.47, 12k 12.41,
14k 12.36, 16k 12.32, 18k 12.60, 20k 13.12, 22k 13.71, 24k 14.27, 26k 15.07, 28k 16.03, 30k 16.95,
32k 17.94, 34k 18.77, 36k 19.54, 38k 20.57, 40k 21.23, 42k 21.99, 44k 22.58, 46k 22.85, 48k 23.15,
50k 23.06, 52k 23.41, 54k 23.70, 56k 23.59, 58k 23.99, 60k 24.09.
Final ddbm_official val: SSIM 0.4046, MS-SSIM 0.7393.

## ELD, fine-tuned on scenes 1-6, tested on scenes 7-10 (24 pairs per camera, x100/x200, 1024 crops)
| init / mode | Canon 70D | Canon 700D | Nikon D850 |
|---|---|---|---|
| full fine-tune | 26.1409 / 0.7452 / 0.8388 | 26.0148 / 0.7133 / 0.8220 | 29.8130 / 0.8268 / 0.9053 |
| decoder+bridge (frozen encoder) | 26.1596 / 0.7403 / 0.8354 | 25.9866 / 0.7088 / 0.8185 | 29.3512 / 0.8109 / 0.8988 |
(PSNR / SSIM / MS-SSIM). ft val at 8k: full 27.32, decoder 27.17.

## Few-shot on ELD (mean PSNR over 3 cameras; K = adaptation scenes per camera, 12 pairs per scene)
| init, adaptation | K=0 | K=1 | K=3 | K=5 |
|---|---|---|---|---|
| SID, Adam 200 steps | 24.17 | 26.74 | 27.36 | 27.54 |
| Bridge, Adam 200 steps | 24.37 | 26.31 | 26.66 | 27.05 |
| Meta (MAML), Adam 200 steps | 24.77 | 26.48 | 26.86 | 27.15 |
| Bridge, inner SGD 30 steps | - | 24.40 | 24.53 | 24.25 |
| Meta (MAML), inner SGD 30 steps | - | 25.29 | 25.26 | 25.46 |
Canon 70D details: SID K0 23.73 / K1 25.83; Bridge K0 23.99 / K1 25.70; Meta K0 23.68 / K1 25.98;
Bridge inner K1 24.33, K3 24.25; Meta inner K1 24.79, K3 24.76.

## Meta-training (2000 iterations) final: query L1 after adaptation 0.0459, support L1 before 0.0510.

## Third run (2026-10-29, retrained after the VM loss; LPIPS = AlexNet, full resolution)
| model | PSNR | SSIM | MS-SSIM | LPIPS | eval s |
|---|---|---|---|---|---|
| SID U-Net, seed 0 (40k) | 28.6057 | 0.7962 | 0.8761 | 0.4592 | 335 |
| DDBM VE-BB, N=5 (60k) | 27.9673 | 0.7799 | 0.8615 | 0.3942 | 898 |
| DDBM VE-BB, N=10 | 27.8839 | 0.7784 | 0.8564 | 0.3942 | 902 |
| DDBM official sigma_max 1 (60k), N=5 | 23.5857 | 0.4060 | 0.5754 | 0.6407 | 1032 |
| DDBM official sigma_max 1, N=10 | 25.5749 | 0.5854 | 0.7208 | 0.4645 | 2518 |
| DDBM official sigma_max 1, N=20 | 26.2394 | 0.6407 | 0.7621 | 0.3973 | 3477 |
Final val: sid 33.15 @40k; ddbm 31.24 @60k; ddbm_official_s1 27.01 @60k.
Second run's SID seed 1 (json overwritten): 28.6145 / 0.7964 / 0.8762.
