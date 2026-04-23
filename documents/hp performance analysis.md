# Hyperparameter Performance Analysis

This document analyzes the impact of hyperparameter (HP) tuning on model performance across different datasets and methods.

## CNN Performance: Standard vs Tuned

| Dataset | Method | Standard Acc | Tuned Acc | Improvement |
| :--- | :--- | :---: | :---: | :---: |
| 06_friedman1 | Ours | 0.8341 | 0.8683 | +0.0341 |
| 06_friedman1 | NCTD | 0.6390 | 0.6585 | +0.0195 |
| 07_friedman2 | Ours | 0.8829 | 0.8780 | -0.0049 |
| 07_friedman2 | NCTD | 0.8732 | 0.8390 | -0.0341 |
| 08_friedman3 | Ours | 0.7024 | 0.7463 | +0.0439 |
| 08_friedman3 | NCTD | 0.5659 | 0.5171 | -0.0488 |
| 13_rotated_rastrigin_50d | Ours | 0.6000 | 0.5000 | -0.1000 |
| 13_rotated_rastrigin_50d | NCTD | 0.5000 | 0.5000 | +0.0000 |
| digen10_8322 | Ours | 0.6150 | 0.8100 | +0.1950 |
| digen10_8322 | NCTD | 0.5250 | 0.5700 | +0.0450 |
| digen11_7270 | Ours | 0.6800 | 0.7100 | +0.0300 |
| digen11_7270 | NCTD | 0.5550 | 0.5150 | -0.0400 |
| digen12_8322 | Ours | 0.4900 | 0.6800 | +0.1900 |
| digen12_8322 | NCTD | 0.5050 | 0.5450 | +0.0400 |
| digen13_769 | Ours | 0.7250 | 0.8150 | +0.0900 |
| digen13_769 | NCTD | 0.6500 | 0.5950 | -0.0550 |
| digen14_769 | Ours | 0.4800 | 0.6450 | +0.1650 |
| digen14_769 | NCTD | 0.5050 | 0.5500 | +0.0450 |
| digen15_5311 | Ours | 0.7400 | 0.7650 | +0.0250 |
| digen15_5311 | NCTD | 0.5650 | 0.5800 | +0.0150 |
| digen16_5390 | Ours | 0.7350 | 0.8200 | +0.0850 |
| digen16_5390 | NCTD | 0.5650 | 0.5900 | +0.0250 |
| digen17_6949 | Ours | 0.4850 | 0.6100 | +0.1250 |
| digen17_6949 | NCTD | 0.5200 | 0.4750 | -0.0450 |
| digen18_5578 | Ours | 0.5100 | 0.6000 | +0.0900 |
| digen18_5578 | NCTD | 0.5850 | 0.6050 | +0.0200 |
| digen19_7270 | Ours | 0.6700 | 0.8500 | +0.1800 |
| digen19_7270 | NCTD | 0.4950 | 0.5300 | +0.0350 |
| digen1_6265 | Ours | 0.9000 | 0.9150 | +0.0150 |
| digen1_6265 | NCTD | 0.6600 | 0.8750 | +0.2150 |
| digen20_5191 | Ours | 0.6400 | 0.7600 | +0.1200 |
| digen20_5191 | NCTD | 0.5350 | 0.5500 | +0.0150 |
| digen21_6265 | Ours | 0.5000 | 0.5850 | +0.0850 |
| digen21_6265 | NCTD | 0.5150 | 0.4500 | -0.0650 |
| digen22_2433 | Ours | 0.7400 | 0.8600 | +0.1200 |
| digen22_2433 | NCTD | 0.5300 | 0.5050 | -0.0250 |
| digen23_5191 | Ours | 0.5050 | 0.4900 | -0.0150 |
| digen23_5191 | NCTD | 0.4600 | 0.5200 | +0.0600 |
| digen24_2433 | Ours | 0.5800 | 0.5850 | +0.0050 |
| digen24_2433 | NCTD | 0.5750 | 0.5500 | -0.0250 |
| digen25_2433 | Ours | 0.5400 | 0.6400 | +0.1000 |
| digen25_2433 | NCTD | 0.5700 | 0.5850 | +0.0150 |
| digen26_7270 | Ours | 0.5500 | 0.6950 | +0.1450 |
| digen26_7270 | NCTD | 0.5250 | 0.5650 | +0.0400 |
| digen27_860 | Ours | 0.4850 | 0.5300 | +0.0450 |
| digen27_860 | NCTD | 0.5100 | 0.5100 | +0.0000 |
| digen28_769 | Ours | 0.5800 | 0.5850 | +0.0050 |
| digen28_769 | NCTD | 0.5450 | 0.5600 | +0.0150 |
| digen29_8322 | Ours | 0.5750 | 0.5950 | +0.0200 |
| digen29_8322 | NCTD | 0.5100 | 0.5250 | +0.0150 |
| digen2_6949 | Ours | 0.5700 | 0.5400 | -0.0300 |
| digen2_6949 | NCTD | 0.6000 | 0.5450 | -0.0550 |
| digen30_4426 | Ours | 0.4950 | 0.6350 | +0.1400 |
| digen30_4426 | NCTD | 0.4950 | 0.5250 | +0.0300 |
| digen31_2433 | Ours | 0.6450 | 0.8350 | +0.1900 |
| digen31_2433 | NCTD | 0.5000 | 0.5000 | +0.0000 |
| digen32_5191 | Ours | 0.5500 | 0.6750 | +0.1250 |
| digen32_5191 | NCTD | 0.5600 | 0.5200 | -0.0400 |
| digen33_769 | Ours | 0.7300 | 0.9100 | +0.1800 |
| digen33_769 | NCTD | 0.6150 | 0.6950 | +0.0800 |
| digen34_769 | Ours | 0.5350 | 0.6250 | +0.0900 |
| digen34_769 | NCTD | 0.5000 | 0.5500 | +0.0500 |
| digen35_4426 | Ours | 0.5400 | 0.5200 | -0.0200 |
| digen35_4426 | NCTD | 0.5300 | 0.4800 | -0.0500 |
| digen36_466 | Ours | 0.5850 | 0.7400 | +0.1550 |
| digen36_466 | NCTD | 0.5600 | 0.5500 | -0.0100 |
| digen37_769 | Ours | 0.7450 | 0.7850 | +0.0400 |
| digen37_769 | NCTD | 0.5850 | 0.6000 | +0.0150 |
| digen38_4426 | Ours | 0.5900 | 0.6350 | +0.0450 |
| digen38_4426 | NCTD | 0.4800 | 0.5650 | +0.0850 |
| digen39_5578 | Ours | 0.7800 | 0.8800 | +0.1000 |
| digen39_5578 | NCTD | 0.6450 | 0.6200 | -0.0250 |
| digen3_769 | Ours | 0.9500 | 0.9600 | +0.0100 |
| digen3_769 | NCTD | 0.6950 | 0.9600 | +0.2650 |
| digen40_5390 | Ours | 0.5200 | 0.5250 | +0.0050 |
| digen40_5390 | NCTD | 0.4950 | 0.5150 | +0.0200 |
| digen4_860 | Ours | 0.5150 | 0.5600 | +0.0450 |
| digen4_860 | NCTD | 0.5150 | 0.5750 | +0.0600 |
| digen5_6949 | Ours | 0.5550 | 0.8050 | +0.2500 |
| digen5_6949 | NCTD | 0.5450 | 0.5750 | +0.0300 |
| digen6_466 | Ours | 0.5700 | 0.4950 | -0.0750 |
| digen6_466 | NCTD | 0.5200 | 0.5200 | +0.0000 |
| digen7_6949 | Ours | 0.5100 | 0.6100 | +0.1000 |
| digen7_6949 | NCTD | 0.5600 | 0.5550 | -0.0050 |
| digen8_4426 | Ours | 0.5550 | 0.6700 | +0.1150 |
| digen8_4426 | NCTD | 0.5250 | 0.5000 | -0.0250 |
| digen9_7270 | Ours | 0.6000 | 0.6400 | +0.0400 |
| digen9_7270 | NCTD | 0.6350 | 0.6100 | -0.0250 |

## EfficientNet Performance: Tuned
(Note: No standard baseline available for EfficientNet)

| Dataset | Method | Tuned Acc |
| :--- | :--- | :---: |
| 06_friedman1 | Ours | 0.4732 |
| 06_friedman1 | NCTD | 0.5171 |
| 07_friedman2 | Ours | 0.5024 |
| 07_friedman2 | NCTD | 0.5756 |
| 08_friedman3 | Ours | 0.4976 |
| 08_friedman3 | NCTD | 0.4976 |
| 13_rotated_rastrigin_50d | Ours | 0.8000 |
| 13_rotated_rastrigin_50d | NCTD | 0.6000 |
| digen10_8322 | Ours | 0.5000 |
| digen10_8322 | NCTD | 0.4750 |
| digen11_7270 | Ours | 0.5000 |
| digen11_7270 | NCTD | 0.5200 |
| digen12_8322 | Ours | 0.5550 |
| digen12_8322 | NCTD | 0.5100 |
| digen13_769 | Ours | 0.4850 |
| digen13_769 | NCTD | 0.5500 |
| digen14_769 | Ours | 0.5250 |
| digen14_769 | NCTD | 0.5100 |
| digen15_5311 | Ours | 0.4900 |
| digen15_5311 | NCTD | 0.5000 |
| digen16_5390 | Ours | 0.5400 |
| digen16_5390 | NCTD | 0.5050 |
| digen17_6949 | Ours | 0.5450 |
| digen17_6949 | NCTD | 0.5000 |
| digen18_5578 | Ours | 0.5300 |
| digen18_5578 | NCTD | 0.5100 |
| digen19_7270 | Ours | 0.5500 |
| digen19_7270 | NCTD | 0.4800 |
| digen1_6265 | Ours | 0.5100 |
| digen1_6265 | NCTD | 0.5000 |
| digen20_5191 | Ours | 0.5150 |
| digen20_5191 | NCTD | 0.4250 |
| digen21_6265 | Ours | 0.5050 |
| digen21_6265 | NCTD | 0.5200 |
| digen22_2433 | Ours | 0.5250 |
| digen22_2433 | NCTD | 0.5000 |
| digen23_5191 | Ours | 0.5000 |
| digen23_5191 | NCTD | 0.5000 |
| digen24_2433 | Ours | 0.5000 |
| digen24_2433 | NCTD | 0.4550 |
| digen25_2433 | Ours | 0.5000 |
| digen25_2433 | NCTD | 0.4850 |
| digen26_7270 | Ours | 0.4900 |
| digen26_7270 | NCTD | 0.5250 |
| digen27_860 | Ours | 0.4600 |
| digen27_860 | NCTD | 0.5050 |
| digen28_769 | Ours | 0.5000 |
| digen28_769 | NCTD | 0.5000 |
| digen29_8322 | Ours | 0.4550 |
| digen29_8322 | NCTD | 0.5400 |
| digen2_6949 | Ours | 0.5000 |
| digen2_6949 | NCTD | 0.4950 |
| digen30_4426 | Ours | 0.5300 |
| digen30_4426 | NCTD | 0.4600 |
| digen31_2433 | Ours | 0.5600 |
| digen31_2433 | NCTD | 0.5000 |
| digen32_5191 | Ours | 0.5000 |
| digen32_5191 | NCTD | 0.4850 |
| digen33_769 | Ours | 0.5100 |
| digen33_769 | NCTD | 0.5000 |
| digen34_769 | Ours | 0.5000 |
| digen34_769 | NCTD | 0.5200 |
| digen35_4426 | Ours | 0.5300 |
| digen35_4426 | NCTD | 0.4750 |
| digen36_466 | Ours | 0.4950 |
| digen36_466 | NCTD | 0.5000 |
| digen37_769 | Ours | 0.4750 |
| digen37_769 | NCTD | 0.5650 |
| digen38_4426 | Ours | 0.5100 |
| digen38_4426 | NCTD | 0.4750 |
| digen39_5578 | Ours | 0.5050 |
| digen39_5578 | NCTD | 0.5000 |
| digen3_769 | Ours | 0.5000 |
| digen3_769 | NCTD | 0.5800 |
| digen40_5390 | Ours | 0.5100 |
| digen40_5390 | NCTD | 0.5000 |
| digen4_860 | Ours | 0.5600 |
| digen4_860 | NCTD | 0.4900 |
| digen5_6949 | Ours | 0.5300 |
| digen5_6949 | NCTD | 0.4700 |
| digen6_466 | Ours | 0.5050 |
| digen6_466 | NCTD | 0.5550 |
| digen7_6949 | Ours | 0.5000 |
| digen7_6949 | NCTD | 0.5000 |
| digen8_4426 | Ours | 0.5000 |
| digen8_4426 | NCTD | 0.5000 |
| digen9_7270 | Ours | 0.4800 |
| digen9_7270 | NCTD | 0.5450 |
