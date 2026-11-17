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

## Seventh run (2026-11-11): one-step estimates and sample averages of the two full DDBMs (colab/multisample.py, eval only)
One-step estimate D(x_T, T) on all 598 test pairs (naive start = fourth-run DDBM, U-Net start = sixth-run DDBM; one evaluation each):
| estimate | PSNR | SSIM | MS-SSIM | LPIPS |
|---|---|---|---|---|
| SID U-Net | 28.6057 | 0.7962 | 0.8761 | 0.4592 |
| one-step, naive start | 28.7896 | 0.7911 | 0.8760 | 0.3871 |
| one-step, U-Net start | 28.8125 | 0.8000 | 0.8816 | 0.4373 |
| U-Net + one-step naive, 1:1 | 29.1367 | 0.8006 | 0.8812 | 0.4286 |
| average of the two one-step estimates | 29.1937 | 0.8006 | 0.8827 | 0.4202 |
Scene-mean differences to the U-Net (mean, s.e., scenes higher): average of the two one-step estimates PSNR +0.573, 0.099, 41/50; SSIM +0.0042, 0.0015, 37/50; MS-SSIM +0.0064, 0.0005, 49/50; LPIPS -0.0386, 0.0030, 1/50.
100 test pairs (two per scene, 50 scenes; ratios x100/x250/x300 = 29/29/42), four 40-step samples from each DDBM (ms_mean.jsonl), PSNR / SSIM / MS-SSIM / LPIPS:
U-Net 28.79 / 0.804 / 0.883 / 0.450; one sample (mean of draws) 27.61 / 0.707 / 0.813 / 0.297; mean of 2 samples 28.39 / 0.751 / 0.848 / 0.262; mean of 4 28.85 / 0.777 / 0.868 / 0.263;
mean of 4+4 samples of the two DDBMs 29.22 / 0.794 / 0.881 / 0.289; one-step naive 29.15 / 0.800 / 0.884 / 0.376; average of the two one-step estimates 29.50 / 0.810 / 0.890 / 0.410; U-Net + mean of the 8 samples 1:1 29.25 / 0.807 / 0.887 / 0.372.
Scene-mean PSNR difference to the U-Net: mean of 4 +0.056 (s.e. 0.207, 20/50 scenes); mean of 4+4 +0.432 (0.101, 36/50); average of one-step estimates +0.710 (0.109, 47/50); U-Net + 8-sample mean +0.460 (0.053, 48/50). 4+4 mean minus the one-step average: -0.278 (0.033).
Variance model MSE(K) = MSE_inf + V/K per image (fit on single draws and the mean of 4, held-out K=2 predicted within 0.002 dB): K=inf PSNR 29.37 (naive start), 29.00 (U-Net start); variance share of a single sample 0.32 / 0.29 (an exact posterior sampler: 0.5); single sample is 1.8 / 1.5 dB below the limit.
Cost: 120.1 compute units left after the run (134.1 -> 120.1, A100 about 2 h with the one-time restore from Drive).

## Eighth run (2026-11-15): ensemble control with a second SID U-Net
Run on Colab account 3 under VPS supervision (colab/vps/driver.sh; A100, resumed once from step 33400 after the final pull failed and the first VM was lost). Second SID U-Net, seed 1, same recipe (40k iterations, MS-SSIM from 20k): best validation PSNR 32.974 dB (first U-Net 32.98 at 38k; seed 1 checkpoint is runs/sid_s1/best.pt, log runs/sid_s1/log.jsonl). Scored on all 598 test pairs with colab/unet_ensemble.py (results/results/ens_unet.jsonl) and compared by colab/compare_ens.py, PSNR / SSIM / MS-SSIM / LPIPS:
| SID U-Net, seed 0 | 28.606 | 0.796 | 0.876 | 0.459 |
| SID U-Net, seed 1 | 28.690 | 0.797 | 0.876 | 0.453 |
| average of the two U-Nets | 28.764 | 0.799 | 0.879 | 0.459 |
| average of the two DDBM one-step estimates | 29.194 | 0.801 | 0.883 | 0.420 |
Scene-mean PSNR differences (mean, s.e., scenes higher of 50): U-Net pair minus seed 0 +0.16 (0.02, 46); seed 1 minus seed 0 +0.08 (0.04, 30); DDBM pair minus seed 0 +0.57 (0.10, 41); DDBM pair minus U-Net pair +0.42 (0.09, 38), MS-SSIM +0.004 (46/50), LPIPS -0.039 (49/50). Conclusion: a plain two-network ensemble explains about a quarter of the DDBM pair's PSNR gain and none of its LPIPS gain. Balance after the run: 2311.4 units on account 3.

## Ninth run (2026-11-15): zero-shot transfer to the ELD cameras (evaluation only)
colab/run_crosscam.sh on Colab account 3 under VPS supervision (A100, 12 minutes, 0.9 units; ELD from the public Drive file with gdown, no mount). Models trained on SID Sony only, full ELD images, scenes 7-10, ratios 100 and 200, 24 pairs per camera (72 pairs, 12 scenes), colab/crosscam.py -> results/results/crosscam.jsonl. PSNR over all 72 pairs / LPIPS: U-Net seed 0 24.86 / 0.379; seed 1 24.98 / 0.375; U-Net pair 24.98 / 0.380; DDBM one-step naive start 20.98 / 0.494 (Canon 700D 16.76); DDBM one-step U-Net start 25.07 / 0.358; average of the two DDBM one-step estimates 23.48 / 0.418. Scene-mean PSNR differences to the seed-0 U-Net: U-Net pair +0.12 (s.e. 0.02, 12/12 scenes); U-Net-start DDBM one-step +0.21 (0.03, 12/12), LPIPS -0.021; DDBM pair -1.38 (0.71, 4/12); naive start -3.88 (1.11, 2/12). So the SID-best DDBM pair does not transfer; the U-Net-start one-step estimate does. Cause of the naive-start failure not isolated (hypothesis: naive rendering depends on white balance, black level, gamma of the sensor).
Diagnostic (colab/diag_naive.py, results/results/diag_naive.jsonl, run on the same 72 ELD pairs): mean of image-mean ratio to the target, naive rendering x_T / SID U-Net output: Nikon D850 2.8 / 1.10, Canon 70D 3.8 / 1.23, Canon 700D 4.3 / 1.35; on the 700D the naive rendering's blue/red channel ratio (relative to the target) is 1.5. Order matches the DDBM naive-start failure (D850 24.70 dB, 70D 21.47, 700D 16.76). Not measured on SID (no SID data on that VM): consistency with the explanation, not proof. 0.9 units.

## Tenth run (2026-11-15): few-shot adaptation of the U-Net and the DDBM on ELD (colab/fewshot_ddbm.py)
Run via the VPS driver on an A100 (colab/run_fewshot.sh, about 25 minutes, ~3 units). 200 Adam steps (lr 2e-5, bs 4, 512 crops) on K support scenes (1, 3, 6) per camera, test = scenes 7-10 full images, 72 pairs per K, results/results/fewshot_ddbm.jsonl. U = SID U-Net fine-tuned with L1; D = DDBM from the U-Net (runs/ddbm_vp_unet) fine-tuned with its bridge loss, one-step estimate; DU = unadapted DDBM fed with the adapted U-Net's output. PSNR / LPIPS pooled: K=0 U 24.86 / 0.379, D = DU 25.07 / 0.358; K=1 U 27.87 / 0.333, D 25.92 / 0.346, DU 27.78 / 0.312; K=3 U 28.13 / 0.330, D 25.82 / 0.349, DU 28.14 / 0.309; K=6 U 28.70 / 0.320, D 25.74 / 0.354, DU 28.66 / 0.301. DU vs U: LPIPS lower by 0.021 / 0.021 / 0.019, on 12/12 scenes at every K; PSNR change -0.09 / +0.01 / -0.04. Single seed, one untuned recipe: the DDBM barely adapts under it; adapting only the U-Net and feeding the frozen DDBM keeps the perceptual gain.

## Eleventh run (2026-11-16): sweep of the DDBM adaptation recipe at K=3 (colab/run_fewshot2.sh)
lr 1e-4 and 3e-4 (5x and 15x the 2e-5 of the tenth run) with 200 and 600 steps, results/results/fewshot_lr*.jsonl, pooled over 3 cameras (72 pairs), PSNR / LPIPS for U, D, DU: lr 2e-5 st200 28.13/0.330, 25.82/0.349, 28.14/0.309; 1e-4 st200 28.17/0.327, 25.50/0.355, 28.15/0.305; 1e-4 st600 27.87/0.328, 25.57/0.347, 27.91/0.307; 3e-4 st200 27.80/0.351, 24.09/0.371, 27.83/0.318; 3e-4 st600 27.82/0.344, 24.94/0.356, 27.87/0.316. The adapted DDBM's one-step estimate stays at 24.1-25.8 dB whatever the setting; DU stays best on LPIPS. About 12 units (a hung final pack call kept the idle A100 for about 20 minutes; the driver's pack and download timeouts are now 700 s and 900 s).

## Twelfth run (2026-11-16): adapting the DDBM with an L1 loss on its one-step estimate (colab/run_fewshot3.sh, --objective readout)
200 steps, lr 2e-5 and 1e-4, K = 1, 3, 6, results/results/fewshot_readout_lr*.jsonl, pooled over 3 cameras (72 pairs), PSNR / LPIPS: D (adapted DDBM, one step) lr 2e-5: K=1 27.04 / 0.314, K=3 27.53 / 0.310, K=6 27.39 / 0.312; lr 1e-4: 27.21 / 0.312, 27.34 / 0.307, 26.70 / 0.322 (bridge loss: 25.92 / 0.346, 25.82 / 0.349, 25.74 / 0.354). Adapted U-Net U: 27.87 / 0.333, 28.13 / 0.330, 28.70 / 0.320. DU stays 27.78 / 0.312, 28.14 / 0.309, 28.66 / 0.301. So the DDBM does adapt if trained on its own one-step output: PSNR 0.6-1.3 dB below the adapted U-Net, LPIPS 0.02 lower; the bridge loss alone does not adapt it. Cost about 5 units.

## Fourteenth run (2026-11-16): the SID side of the brightness diagnostic (colab/diag_naive_sid.py, results/results/diag_naive_sid.jsonl)
Run on Colab account 3 via the VPS driver with a Drive mount (colab/run_diag_sid.sh, about 3 units). SID Sony test pairs, two per scene (100 pairs): naive rendering x_T / target mean ratio 1.23 (PSNR 18.4 dB against the target; rgb ratios 1.06, 1.29, 1.59, blue/red 1.5); SID U-Net output 1.01 (28.8 dB). On ELD: x_T 2.8 / 3.8 / 4.3 (D850 / 70D / 700D), U-Net 1.10-1.35. The colour cast is the same on both (blue/red 1.5), so the difference is the brightness: the ELD start images are 2 to 4 times brighter than anything in training. Not yet tested: whether rescaling x_T fixes the DDBM (colab/diag_gain.py, blocked by the ELD download quota until about 2026-11-17 01:30 GMT).

## Fifteenth run (2026-11-16): smoke test of colab/custom_camera.py on SID Sony ARW files laid out as a camera of one's own
colab/run_custom_test.sh (VPS driver, Drive mount, A100, about 2 units): 6 SID test scenes as scene-01..06 with ref.ARW (the long exposure) and 3 short ARW frames each; prep, adapt (K=3 scenes, 60 steps, with the unadapted DDBM from runs/ddbm_vp_unet) and enhance (4256x2848 PNGs, tiled) all ran. Held-out scenes 04-06, centre 1024 crop, PSNR / LPIPS: zero-shot U-Net 30.77 / 0.307, with DDBM 31.06 / 0.280; after 60 steps on 3 scenes of the same (in-distribution) camera: U-Net 29.63 / 0.326, with DDBM 29.86 / 0.298. Fine-tuning a camera the model already knows on 3 scenes hurts by 1.1 dB, the DDBM on top still helps (+0.2 dB, -0.03 LPIPS). Outputs in results/custom_smoke/. Not tested: a real 61 MP A7CR file (rawpy/LibRaw support for ILCE-7CR, memory at 9504x6336).

## Sixteenth run (2026-11-17): the SID-trained U-Net on a new Sony sensor, SIED Sony (A7R III, 3840x2160 crops), zero-shot and K-shot
Data: SIED (Jiang et al., ICCV 2025), Sony 0.001 illuminance level, got from the authors' OneDrive (Sony_gt.zip 3.3 GB, Sony_0.001.zip 10.7 GB, downloaded by the user to the Mac and copied via the VPS to Colab and to the Colab account's Drive (MyDrive/SIED)). Low-light frames are bare uint16 mosaics (black 512, white 16383), references are sRGB JPGs; the exposure ratio is NOT in the files and was estimated per image from the reference (mean linearised reference / mean normalised mosaic, clipped to 1..300; test median 153), so this is an adaptation study, not blind use. colab/sied_prep.py, colab/custom_camera.py adapt --train-pairs --test-pairs --ks, colab/run_sied_exp.sh; 180 test images (the authors' test list), 1024 centre crops, support = every 15th of the 1500 training images, 200 Adam steps lr 2e-5, one seed; results/custom_smoke/sied_result.json. PSNR / LPIPS, U-Net and frozen DDBM on its output (one-step):
K=0: 15.82 / 0.646 and 15.87 / 0.614; K=10: 22.00 / 0.567 and 22.22 / 0.535; K=30: 22.14 / 0.589 and 22.27 / 0.541; K=100: 22.90 / 0.559 and 23.19 / 0.528. DDBM minus U-Net, paired over 180 images: PSNR +0.05, +0.22, +0.13, +0.29 dB; LPIPS -0.032, -0.032, -0.048, -0.030 (lower on 165, 154, 168, 148 of 180). Confounds: the targets are JPG renderings, not rawpy renderings (a rendering shift is part of the K=0 drop), the noise is synthetic, the ratio is estimated, and train and test images may come from the same scenes (not checked). Cost about 1 unit.

## Seventeenth run (2026-11-17): grid of few-shot adaptation recipes with three seeds (colab/adapt_grid.py, colab/run_adapt_grid.sh)
Ran on one Colab G4 (NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, 10.05 units/h; A100 was unavailable at the time), SIED and ELD in parallel, about 1.4 h, about 19 units; results/results/adapt_grid_{sied,eld}.jsonl (per-image PSNR, LPIPS for U and DU). Recipes: base (whole network, L1, constant lr 2e-5, 200 steps), cos (L1 + 0.1 MS-SSIM, cosine), dec (decoder only, lr 1e-4, 200 steps), dec_long (400 steps), enc (encoder only, lr 2e-5), wd (whole network, wd 1e-2, lr 5e-5, cosine). U-Net PSNR, mean (sd over 3 seeds):
ELD (3 cameras, 72 test pairs, zero-shot 24.86): K=1 base 27.99 (0.15), cos 27.88, dec 28.15, dec_long 28.40 (0.07), enc 27.35, wd 28.13; K=6 base 28.66 (0.11), cos 28.53, dec 28.61, dec_long 28.75, enc 28.15, wd 28.78 (0.06).
SIED (A7R III, 180 test images, zero-shot 15.82): K=10 base 21.90 (0.09), cos 21.82, dec 21.96, dec_long 21.94, enc 20.16, wd 22.27 (0.10); K=30 base 22.22, cos 21.83, dec 22.15, dec_long 22.41, enc 20.03, wd 22.35; K=100 base 22.95 (0.05), cos 22.73, dec 23.11, dec_long 23.35 (0.04), enc 20.65, wd 23.17.
Conclusions: the best recipes (decoder only for longer, or whole network with weight decay) add 0.1-0.4 dB over the report's recipe, above the seed noise (sd up to 0.18 dB) at K=1 on ELD and at K=10 and 100 on SIED; the encoder alone is clearly worse (-0.5 dB on ELD, -2 dB on SIED); MS-SSIM + cosine alone does not help. The frozen DDBM on top: SIED PSNR +0.15 to +0.25 and LPIPS about -0.04 for every recipe; ELD PSNR about -0.05 to -0.1 and LPIPS about -0.02. ELD part is in the report (Table 8); SIED part is not (confounds, see the sixteenth run).

## Eighteenth run (2026-11-17): LRID_simplified (Redmi K30 phone, Sony IMX686 sensor; Hugging Face hansen97/LRID_simplified, CC BY-NC 4.0): real-pair adaptation and calibrated-noise synthesis
Data on the held Colab G4 (RTX PRO 6000 Blackwell, 97,887 MiB): indoor_x3 (22 scenes; noisy ISO-6400 DNG frames at digital gains 1, 2, 4; multi-frame fused clean raw npy), bias dark frames and resources. DNG 3472x4624, RGGB, 10 bit (black 64, white 1023). colab/lrid_prep.py renders the targets itself (bilinear demosaic, camera white balance, exposure to mean linear 0.18, gamma 1/2.2; no colour matrix), ratio = exposure x clean mean / noisy mean from the clean raw (oracle, median 4.1, range 1 to 20); 12 train scenes (36 pairs), 10 test scenes (30 pairs), 1024 crops. Real-pair adaptation (colab/custom_camera.py adapt, 200 steps lr 2e-5, one seed), PSNR / LPIPS, U-Net and frozen DDBM on top: 0 pairs 20.64 / 0.422 and 21.30 / 0.374; 3 pairs (1 scene) 28.34 / 0.341 and 28.14 / 0.302; 6 pairs 28.94 / 0.340 and 28.65 / 0.303; 12 pairs 29.00 / 0.334 and 28.88 / 0.292; 36 pairs 29.46 / 0.331 and 29.11 / 0.292 (results/results/lrid_real_ks.json).
Calibrated-noise synthesis (colab/lrid_synth.py, results/results/lrid_synth.json): var = K x + sigma^2 with sigma^2 per channel 1.8e-4 from 12 dark frames (temporal variance) and K = 0.00875 from ONE real noisy/clean pair (scene 000, gain 1); the U-Net is fine-tuned on the clean raw of K scenes with fresh noise at every step (real-frame brightness and ratio), scored on the same real noisy test frames. Mean over 3 seeds, PSNR (sd) / LPIPS, U-Net and with the frozen DDBM: 1 scene 27.09 (0.22) / 0.334 and 27.24 / 0.288; 2 scenes 27.48 / 0.336 and 27.59 / 0.286; 4 scenes 27.61 (0.70) / 0.339 and 27.77 / 0.284; 12 scenes 27.73 / 0.339 and 27.84 / 0.289. So calibrated synthetic noise recovers about 85% of what one real scene gives (20.6 to 27.1 against 28.3 dB) but stays 1.2 to 1.7 dB below real noisy pairs and gains little from more scenes, i.e. the Gaussian model misses part of the real noise. Confounds: own rendering of the targets, oracle ratio, 30 test pairs from 10 scenes, one seed for the real-pair row, phone sensor (not a full-frame Sony body). Not in the report.
Shot-noise scale sweep (colab/lrid_synth.py --k-scale, scenes 2 and 4, 3 seeds, U-Net PSNR mean): K x 0.5: 27.65 and 27.77; x 1.0: 27.48 and 27.61; x 1.5: 27.12 and 27.37; x 2.0: 26.84 and 26.98. A lower noise level is slightly better and an overestimate hurts by up to 0.8 dB, but no scale closes the 1.2 to 1.7 dB gap to real noisy pairs (28.9 to 29.0 at 6 and 12 pairs), so the missing part is the structure of the real noise (spatial correlation, row and fixed-pattern noise, non-Gaussian tails) or the fusion artefacts of the clean targets, not its magnitude.
Richer noise models (colab/lrid_synth.py --noise, scenes 2 and 4, 3 seeds, U-Net PSNR / with frozen DDBM): Gaussian 27.48 / 27.59 and 27.61 / 27.77; real dark-frame residuals as read noise (keeps spatial correlation and row noise) 27.58 / 27.59 and 27.59 / 27.82; plus the fixed pattern of the dark frames 27.61 / 27.52 and 27.72 / 27.84; plus Poisson shot noise 27.69 / 27.55 and 27.79 / 27.87. Real noisy pairs at the same scene counts: 28.94 (6 pairs, 2 scenes) and 29.00 (12 pairs, 4 scenes). The finer noise models add 0.1 to 0.2 dB only, so the 1.2 to 1.3 dB gap to real pairs is not read-noise structure; the remaining candidates are what real pairs also contain (residual misalignment and brightness changes between the noisy frame and the fused clean target, any non-linear processing of the phone's RAW) and the one-seed real-pair reference. Results in results/results/lrid_synth*.json.
Cross test (colab/lrid_synth.py --cross, 2 scenes, 3 seeds, U-Net PSNR on the 30 test pairs, real noisy frames and synthetic noisy frames made from the clean test raw with the dark-frame + fixed-pattern + Poisson model; results/results/lrid_cross.json): zero-shot 20.64 on real and 21.05 on synthetic frames (the synthetic frames are about as hard as the real ones for an unadapted network); trained on real pairs: 29.01 on real, 28.69 on synthetic; trained on synthetic noise: 27.69 on real, 28.83 on synthetic. The real-pair model nearly keeps its score on synthetic noise (-0.3 dB), the synthetic model loses 1.1 dB on real noise: real noise contains structure that the calibrated model does not produce, and training on real noise covers the synthetic noise as well, not the other way round.
Mix test (colab/lrid_synth.py --mix, 3 seeds, U-Net PSNR / LPIPS on the 30 real test pairs, with the frozen DDBM in brackets; results/results/lrid_mix_1real_*synth.json): one real scene (3 pairs) alone 28.31 / 0.339 (28.07 / 0.301); one real scene plus synthetic noise (dark-frame residuals, fixed pattern, Poisson shot) on the clean raw of 4 scenes, half and half in every batch: 28.82 / 0.335 (28.59 / 0.293); plus synthetic on 12 scenes: 29.05 / 0.333 (29.04 / 0.294). For reference, real pairs of 4 scenes give 29.00 and of 12 scenes 29.46 (one seed). So one real pair set plus calibrated synthetic noise on more clean images is worth about four real scenes. Cost of the whole LRID series about 15 units of G4 time.
