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

## Fourth run (2026-10-30, full DDBM: released VP bridge and sampler, 303k iterations; best val 28.19 @294k)
| model | PSNR | SSIM | MS-SSIM | LPIPS | eval s |
|---|---|---|---|---|---|
| DDBM full VP, N=20 (59 NFE) | 26.9735 | 0.6753 | 0.7874 | 0.3164 | 2413 |
| DDBM full VP, N=40 (119 NFE) | 27.3477 | 0.6963 | 0.8036 | 0.2909 | 4586 |
Timing, network only, A100-SXM4-40GB: N=20 3.560 s, N=40 7.188 s per full-resolution image.
Full VP N=40 vs SID U-Net by ratio (PSNR, LPIPS): x100 29.03/29.73, 0.206/0.326; x250 25.98/27.22, 0.287/0.595; x300 27.10/28.79, 0.359/0.458.
Scene-mean differences, full VP N=40 minus SID U-Net (mean, s.e., scenes where VP scores higher): PSNR -1.27, 0.21, 7/50; SSIM -0.097, 0.007, 0/50; LPIPS -0.150, 0.023, 5/50.
Mean luminance gradient relative to the target on the six crops: SID 0.54, simplified N=10 0.59, full VP N=40 1.20, sigma_max 1 N=10 2.3.

## Fifth run (2026-11-10): blend x = (1-w) SID U-Net + w full DDBM (N=40, one sample per image, seeded by name)
Weight chosen on validation (2 pairs per val scene, 40 pairs, full resolution): max PSNR at w=0.25 (28.38 vs 28.22 dB for the U-Net).
Test, all 598 pairs:
| w | PSNR | SSIM | MS-SSIM | LPIPS |
|---|---|---|---|---|
| 0 (U-Net) | 28.6057 | 0.7962 | 0.8761 | 0.4592 |
| 0.125 | 28.7847 | 0.7971 | 0.8771 | 0.4126 |
| 0.25 | 28.8582 | 0.7927 | 0.8745 | 0.3443 |
| 0.375 | 28.8217 | 0.7837 | 0.8687 | 0.2952 |
| 0.5 | 28.6795 | 0.7710 | 0.8599 | 0.2688 |
| 0.625 | 28.4440 | 0.7553 | 0.8485 | 0.2605 |
| 0.75 | 28.1319 | 0.7372 | 0.8351 | 0.2642 |
| 0.875 | 27.7616 | 0.7174 | 0.8200 | 0.2753 |
| 1 (DDBM) | 27.3498 | 0.6963 | 0.8036 | 0.2909 |
w=0 reproduces the U-Net evaluation exactly; w=1 matches the fourth run's DDBM N=40 to 0.002 dB.
Scene-mean differences to the U-Net (mean, s.e., scenes where the blend scores higher): w=0.125 PSNR +0.176, 0.022, 47/50; SSIM +0.0010, 0.0004, 36/50; MS-SSIM +0.0010, 0.0001, 45/50; LPIPS -0.043, 0.005, 0/50. w=0.25 PSNR +0.246, 0.046, 42/50; SSIM -0.0032, 0.0010, 13/50; MS-SSIM -0.0015, 0.0004, 17/50; LPIPS -0.106, 0.012, 0/50.
Per ratio, w=0.25 vs U-Net: x100 30.07/29.73 dB, LPIPS 0.255/0.326; x250 27.51/27.22, 0.406/0.595; x300 28.95/28.79, 0.366/0.458.
Cost: 10.4 compute units (A100, about 1.6 h including fetching val+test scenes and prep).

## Sixth run (2026-11-10): full DDBM started from the SID U-Net output (--unet-start), 302k iterations, final checkpoint
Same recipe as the fourth run (VP bridge, EMA 0.9999, released 40-step sampler); only x_T changes from the naive rendering to the U-Net output.
The best 10-step val PSNR came at 10k (25.81), then fell to about 24 by 50k and recovered to 28.07 at 302k; with 40 steps
the 54k model was already better than the 10k one on 20 val crops (25.50 vs 25.01 dB, LPIPS 0.376 vs 0.571), so the final checkpoint is tested.
| model (test, 598 pairs, N=40) | PSNR | SSIM | MS-SSIM | LPIPS |
|---|---|---|---|---|
| full DDBM, naive start (fourth run) | 27.3477 | 0.6963 | 0.8036 | 0.2909 |
| full DDBM, U-Net start | 27.2805 | 0.6997 | 0.8105 | 0.2839 |
Scene-mean differences, U-Net start minus naive start (mean, s.e., scenes higher): PSNR -0.049, 0.176, 27/50; SSIM +0.004, 0.003, 32/50; MS-SSIM +0.007, 0.002, 36/50; LPIPS -0.008, 0.005, 18/50.
Per ratio (PSNR / LPIPS), U-Net start vs naive start: x100 28.53/0.217 vs 29.03/0.206; x250 25.81/0.283 vs 25.98/0.287; x300 27.44/0.337 vs 27.10/0.359.
Timing: 7.210 s per full image including the U-Net (119 + 1 NFE). The 10-step test evaluation was stopped part-way and is not reported.
Cost: about 55 compute units (189.6 -> 134.7), including the one-time copy of the dataset to Drive.
Without the three misaligned test scenes 10034, 10045, 10172 (562 pairs; the SID README asks to leave them out of quantitative results), PSNR / SSIM / MS-SSIM / LPIPS:
SID U-Net 28.9562 / 0.7957 / 0.8796 / 0.4696; simplified DDBM N=5 28.3285 / 0.7789 / 0.8644 / 0.4011; full DDBM N=40 27.6505 / 0.6927 / 0.8047 / 0.2914;
full DDBM from U-Net N=40 27.5631 / 0.6969 / 0.8119 / 0.2842; blend w=0.25 29.2228 / 0.7917 / 0.8780 / 0.3498; blend w=0.625 28.7926 / 0.7530 / 0.8511 / 0.2619.
