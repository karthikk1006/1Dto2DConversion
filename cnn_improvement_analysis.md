# CNN Accuracy Comparison Analysis (Ours vs NCTD)

## 📊 Executive Summary & Dataset Tally

This report compares the **Ours** approach against the **NCTD** baseline across **54 total datasets**.

### 📈 Global Performance Breakdown
- **Ours-List (39 datasets)**: Ours > NCTD
- **NCTD-List (7 datasets)**: NCTD > Ours
- **Equal-List (4 datasets)**: Ours == NCTD (or < 0.1% difference)
- **Error-List (4 datasets)**: Excluded due to technical failures in the NCTD pipeline.

### 🔢 Total Dataset Tally
| Category | Datasets Count | Names |
| :--- | :--- | :--- |
| **Digen** | 40 | `digen1` to `digen40` |
| **DS (Benchmark)** | 10 | `DS01` to `DS10` |
| **Friedman** | 3 | `06_friedman1` to `08_friedman3` |
| **Rastrigin** | 1 | `13_rotated_rastrigin_50d` |
| **TOTAL** | **54** | |

---

## 🏆 Grouped Dataset Lists

### 🚀 Ours-List (Superior Performance)
`06_friedman1`, `07_friedman2`, `08_friedman3`, `digen1_6265`, `digen5_6949`, `digen7_6949`, `digen8_4426`, `digen9_7270`, `digen10_8322`, `digen11_7270`, `digen12_8322`, `digen13_769`, `digen14_769`, `digen15_5311`, `digen16_5390`, `digen17_6949`, `digen19_7270`, `digen20_5191`, `digen21_6265`, `digen22_2433`, `digen24_2433`, `digen25_2433`, `digen26_7270`, `digen27_860`, `digen28_769`, `digen29_8322`, `digen30_4426`, `digen31_2433`, `digen32_5191`, `digen33_769`, `digen34_769`, `digen35_4426`, `digen36_466`, `digen37_769`, `digen38_4426`, `digen39_5578`, `digen40_5390`, `DS02_student_dropout`, `DS07_ringnorm`

### 📉 NCTD-List (Baseline Superiority)
`digen2_6949`, `digen4_860`, `digen6_466`, `digen18_5578`, `digen23_5191`, `DS03_adult`, `DS06_dengue_chikungunya`

### ⚖️ Equal-List (Tied/Marginal)
`13_rotated_rastrigin_50d`, `digen3_769`, `DS01_breast_cancer`, `DS04_heart`

### ⚠️ Error-List (Excluded due to Baseline Failures)
- **DS05_thyroid**: Stratified split failed (too few samples in class [5]).
- **DS08_isolet**: Crashed with CUDA unknown error after trial 0.
- **DS09_madelon**: Failed entirely with CUDA unknown error.
- **DS10_relathee**: Failed entirely with CUDA Out of Memory (OOM).

---

## 🔍 Detailed Comparison Report

This section provides the side-by-side accuracy and hyperparameter logs for each dataset.

## 🚀 Significant Improvements (Ours > NCTD)

### Dataset: `06_friedman1`
- **Improvement**: `+0.2098`
- **Ours Accuracy**: `0.8683`
- **NCTD Accuracy**: `0.6585`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.0027943714526832615,
    "batch_size": 32,
    "weight_decay": 0.0007429819400817155,
    "optimizer": "adam",
    "scheduler": "step",
    "epochs": 36,
    "scheduler_step_size": 6,
    "scheduler_gamma": 0.6658958461948439,
    "dropout_rate": 0.49265963972389465
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 5.152754415794431e-05,
    "batch_size": 64,
    "weight_decay": 0.00670250148148957,
    "optimizer": "adam",
    "scheduler": "none",
    "epochs": 25,
    "dropout_rate": 0.42031273089088894
}
```

---

### Dataset: `07_friedman2`
- **Improvement**: `+0.0390`
- **Ours Accuracy**: `0.8780`
- **NCTD Accuracy**: `0.8390`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.0024651159546273406,
    "batch_size": 64,
    "weight_decay": 0.0002458475723258273,
    "optimizer": "adam",
    "scheduler": "none",
    "epochs": 22,
    "dropout_rate": 0.37006535294716314
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.008105086696681708,
    "batch_size": 32,
    "weight_decay": 0.00024398834939928397,
    "optimizer": "adam",
    "scheduler": "none",
    "epochs": 21,
    "dropout_rate": 0.3656864078416148
}
```

---

### Dataset: `08_friedman3`
- **Improvement**: `+0.2293`
- **Ours Accuracy**: `0.7463`
- **NCTD Accuracy**: `0.5171`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.0023411552886301763,
    "batch_size": 128,
    "weight_decay": 0.0017961311976996248,
    "optimizer": "adam",
    "scheduler": "cosine",
    "epochs": 33,
    "scheduler_t_max": 39,
    "dropout_rate": 0.3808599820609112
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.0002825549110188169,
    "batch_size": 128,
    "weight_decay": 0.0015782360747235514,
    "optimizer": "sgd",
    "scheduler": "cosine",
    "epochs": 29,
    "momentum": 0.9635259071557937,
    "scheduler_t_max": 16,
    "dropout_rate": 0.17032225693858719
}
```

---

### Dataset: `digen1_6265`
- **Improvement**: `+0.0400`
- **Ours Accuracy**: `0.9150`
- **NCTD Accuracy**: `0.8750`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.0005314552750278099,
    "batch_size": 128,
    "weight_decay": 1.003646088086405e-05,
    "optimizer": "adam",
    "scheduler": "none",
    "epochs": 40,
    "dropout_rate": 0.40282400003704766
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.005390077982757105,
    "batch_size": 32,
    "weight_decay": 0.007052857755582413,
    "optimizer": "sgd",
    "scheduler": "none",
    "epochs": 49,
    "momentum": 0.9347147075802519,
    "dropout_rate": 0.24397028761663367
}
```

---

### Dataset: `digen5_6949`
- **Improvement**: `+0.2300`
- **Ours Accuracy**: `0.8050`
- **NCTD Accuracy**: `0.5750`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.0016840215354452766,
    "batch_size": 32,
    "weight_decay": 0.00855915981775034,
    "optimizer": "adam",
    "scheduler": "cosine",
    "epochs": 42,
    "scheduler_t_max": 22,
    "dropout_rate": 0.0931220991178211
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.0011061283352884952,
    "batch_size": 32,
    "weight_decay": 6.517533058903267e-05,
    "optimizer": "sgd",
    "scheduler": "cosine",
    "epochs": 31,
    "momentum": 0.8665533009472689,
    "scheduler_t_max": 17,
    "dropout_rate": 0.44467044245473786
}
```

---

### Dataset: `digen7_6949`
- **Improvement**: `+0.0550`
- **Ours Accuracy**: `0.6100`
- **NCTD Accuracy**: `0.5550`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.00043020062026513325,
    "batch_size": 64,
    "weight_decay": 6.055608986503227e-05,
    "optimizer": "adam",
    "scheduler": "cosine",
    "epochs": 43,
    "scheduler_t_max": 33,
    "dropout_rate": 0.2792573888437295
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.00539164015897414,
    "batch_size": 128,
    "weight_decay": 6.027825655731741e-05,
    "optimizer": "sgd",
    "scheduler": "step",
    "epochs": 43,
    "momentum": 0.8962726932922822,
    "scheduler_step_size": 10,
    "scheduler_gamma": 0.795878553106193,
    "dropout_rate": 0.12587263104656193
}
```

---

### Dataset: `digen8_4426`
- **Improvement**: `+0.1700`
- **Ours Accuracy**: `0.6700`
- **NCTD Accuracy**: `0.5000`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.0009738162668324438,
    "batch_size": 64,
    "weight_decay": 0.0002954093309795997,
    "optimizer": "adam",
    "scheduler": "step",
    "epochs": 50,
    "scheduler_step_size": 11,
    "scheduler_gamma": 0.5956499454727469,
    "dropout_rate": 0.1722424987417876
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.0009078013878043269,
    "batch_size": 32,
    "weight_decay": 0.00021401883646071662,
    "optimizer": "adam",
    "scheduler": "cosine",
    "epochs": 30,
    "scheduler_t_max": 20,
    "dropout_rate": 0.07593874882838689
}
```

---

### Dataset: `digen9_7270`
- **Improvement**: `+0.0300`
- **Ours Accuracy**: `0.6400`
- **NCTD Accuracy**: `0.6100`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.00690410936535934,
    "batch_size": 32,
    "weight_decay": 0.0018459098571671447,
    "optimizer": "adam",
    "scheduler": "cosine",
    "epochs": 48,
    "scheduler_t_max": 24,
    "dropout_rate": 0.19903813356549058
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 9.747591879533747e-05,
    "batch_size": 32,
    "weight_decay": 0.003802378893015482,
    "optimizer": "adam",
    "scheduler": "step",
    "epochs": 40,
    "scheduler_step_size": 13,
    "scheduler_gamma": 0.7000959259067979,
    "dropout_rate": 0.21044664851058678
}
```

---

### Dataset: `digen10_8322`
- **Improvement**: `+0.2400`
- **Ours Accuracy**: `0.8100`
- **NCTD Accuracy**: `0.5700`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.003404753654120649,
    "batch_size": 32,
    "weight_decay": 0.00048809153014087766,
    "optimizer": "adam",
    "scheduler": "none",
    "epochs": 48,
    "dropout_rate": 0.14793817797393327
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.006563883247636987,
    "batch_size": 64,
    "weight_decay": 0.00037798392721707013,
    "optimizer": "adam",
    "scheduler": "none",
    "epochs": 39,
    "dropout_rate": 0.09342747645316096
}
```

---

### Dataset: `digen11_7270`
- **Improvement**: `+0.1950`
- **Ours Accuracy**: `0.7100`
- **NCTD Accuracy**: `0.5150`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.004627556450702633,
    "batch_size": 32,
    "weight_decay": 0.0008434673039299558,
    "optimizer": "adam",
    "scheduler": "exponential",
    "epochs": 45,
    "scheduler_gamma": 0.9816511249506391,
    "dropout_rate": 0.13141835618693892
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.0033671444635090812,
    "batch_size": 32,
    "weight_decay": 2.995939425075911e-05,
    "optimizer": "adam",
    "scheduler": "exponential",
    "epochs": 26,
    "scheduler_gamma": 0.9608097145206937,
    "dropout_rate": 0.04062853394231662
}
```

---

### Dataset: `digen12_8322`
- **Improvement**: `+0.1350`
- **Ours Accuracy**: `0.6800`
- **NCTD Accuracy**: `0.5450`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.0017915290327442093,
    "batch_size": 32,
    "weight_decay": 7.08158302558749e-05,
    "optimizer": "adam",
    "scheduler": "cosine",
    "epochs": 47,
    "scheduler_t_max": 39,
    "dropout_rate": 0.42717793958619577
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.0010935413315542974,
    "batch_size": 128,
    "weight_decay": 1.4165404814303273e-05,
    "optimizer": "adam",
    "scheduler": "step",
    "epochs": 42,
    "scheduler_step_size": 13,
    "scheduler_gamma": 0.8709336228943543,
    "dropout_rate": 0.23826692344620762
}
```

---

### Dataset: `digen13_769`
- **Improvement**: `+0.2200`
- **Ours Accuracy**: `0.8150`
- **NCTD Accuracy**: `0.5950`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.005735451028919333,
    "batch_size": 32,
    "weight_decay": 0.00041433483361010627,
    "optimizer": "adam",
    "scheduler": "step",
    "epochs": 45,
    "scheduler_step_size": 12,
    "scheduler_gamma": 0.24845536641348434,
    "dropout_rate": 0.3778658745802847
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 4.633860603600197e-05,
    "batch_size": 64,
    "weight_decay": 8.571848107726374e-05,
    "optimizer": "adam",
    "scheduler": "cosine",
    "epochs": 33,
    "scheduler_t_max": 10,
    "dropout_rate": 0.17887931869677623
}
```

---

### Dataset: `digen14_769`
- **Improvement**: `+0.0950`
- **Ours Accuracy**: `0.6450`
- **NCTD Accuracy**: `0.5500`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.00046552791478096097,
    "batch_size": 32,
    "weight_decay": 0.005298464276239448,
    "optimizer": "adam",
    "scheduler": "exponential",
    "epochs": 36,
    "scheduler_gamma": 0.9753066674790523,
    "dropout_rate": 0.04998798565558438
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.000804209907527267,
    "batch_size": 32,
    "weight_decay": 0.00020423717959060082,
    "optimizer": "adam",
    "scheduler": "exponential",
    "epochs": 32,
    "scheduler_gamma": 0.9853125012833117,
    "dropout_rate": 0.3664951840733684
}
```

---

### Dataset: `digen15_5311`
- **Improvement**: `+0.1850`
- **Ours Accuracy**: `0.7650`
- **NCTD Accuracy**: `0.5800`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.00030682398041453256,
    "batch_size": 64,
    "weight_decay": 0.0009250654563740028,
    "optimizer": "adam",
    "scheduler": "none",
    "epochs": 26,
    "dropout_rate": 0.47308808363737964
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 9.820094287627021e-05,
    "batch_size": 128,
    "weight_decay": 0.0007821938957540177,
    "optimizer": "adam",
    "scheduler": "step",
    "epochs": 48,
    "scheduler_step_size": 15,
    "scheduler_gamma": 0.734165144379114,
    "dropout_rate": 0.1409900058553286
}
```

---

### Dataset: `digen16_5390`
- **Improvement**: `+0.2300`
- **Ours Accuracy**: `0.8200`
- **NCTD Accuracy**: `0.5900`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.000689785073655669,
    "batch_size": 32,
    "weight_decay": 0.00012057039952199534,
    "optimizer": "adam",
    "scheduler": "cosine",
    "epochs": 40,
    "scheduler_t_max": 26,
    "dropout_rate": 0.07503362360699556
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.002000042461652854,
    "batch_size": 32,
    "weight_decay": 0.00038913289546199485,
    "optimizer": "adam",
    "scheduler": "step",
    "epochs": 24,
    "scheduler_step_size": 11,
    "scheduler_gamma": 0.773778439454945,
    "dropout_rate": 0.2012361880545035
}
```

---

### Dataset: `digen17_6949`
- **Improvement**: `+0.1350`
- **Ours Accuracy**: `0.6100`
- **NCTD Accuracy**: `0.4750`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.0004645429726112362,
    "batch_size": 32,
    "weight_decay": 0.004346018914949364,
    "optimizer": "adam",
    "scheduler": "exponential",
    "epochs": 35,
    "scheduler_gamma": 0.9859844355055027,
    "dropout_rate": 0.4508801655491483
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.002107110340521451,
    "batch_size": 128,
    "weight_decay": 0.00026254468931330855,
    "optimizer": "adam",
    "scheduler": "none",
    "epochs": 25,
    "dropout_rate": 0.47855848139959845
}
```

---

### Dataset: `digen19_7270`
- **Improvement**: `+0.3200`
- **Ours Accuracy**: `0.8500`
- **NCTD Accuracy**: `0.5300`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.005765929924936018,
    "batch_size": 32,
    "weight_decay": 1.1028290191959253e-05,
    "optimizer": "adam",
    "scheduler": "step",
    "epochs": 29,
    "scheduler_step_size": 5,
    "scheduler_gamma": 0.5875073023365212,
    "dropout_rate": 0.49271279215347796
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.001933102412177674,
    "batch_size": 32,
    "weight_decay": 0.0001838898735573377,
    "optimizer": "adam",
    "scheduler": "exponential",
    "epochs": 24,
    "scheduler_gamma": 0.9671681616523421,
    "dropout_rate": 0.032376520444130474
}
```

---

### Dataset: `digen20_5191`
- **Improvement**: `+0.2100`
- **Ours Accuracy**: `0.7600`
- **NCTD Accuracy**: `0.5500`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.005519376748387782,
    "batch_size": 32,
    "weight_decay": 0.0004025715235345736,
    "optimizer": "adam",
    "scheduler": "none",
    "epochs": 33,
    "dropout_rate": 0.3400758553966482
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.00027422468985969476,
    "batch_size": 64,
    "weight_decay": 4.3993245768659524e-05,
    "optimizer": "adam",
    "scheduler": "cosine",
    "epochs": 39,
    "scheduler_t_max": 25,
    "dropout_rate": 0.16284780900965834
}
```

---

### Dataset: `digen21_6265`
- **Improvement**: `+0.1350`
- **Ours Accuracy**: `0.5850`
- **NCTD Accuracy**: `0.4500`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.0009558612753755265,
    "batch_size": 64,
    "weight_decay": 1.4858980948993898e-05,
    "optimizer": "adam",
    "scheduler": "exponential",
    "epochs": 26,
    "scheduler_gamma": 0.9749317459926184,
    "dropout_rate": 0.2820218965123068
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.00028999415081096605,
    "batch_size": 64,
    "weight_decay": 0.00118186071429948,
    "optimizer": "adam",
    "scheduler": "step",
    "epochs": 42,
    "scheduler_step_size": 11,
    "scheduler_gamma": 0.2807565526489586,
    "dropout_rate": 0.19459066165244804
}
```

---

### Dataset: `digen22_2433`
- **Improvement**: `+0.3550`
- **Ours Accuracy**: `0.8600`
- **NCTD Accuracy**: `0.5050`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.006841414224131644,
    "batch_size": 64,
    "weight_decay": 0.003172386787956889,
    "optimizer": "adam",
    "scheduler": "step",
    "epochs": 33,
    "scheduler_step_size": 11,
    "scheduler_gamma": 0.4553854051387491,
    "dropout_rate": 0.30668504023229654
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 5.3502512374426595e-05,
    "batch_size": 32,
    "weight_decay": 0.0001315205651606738,
    "optimizer": "adam",
    "scheduler": "step",
    "epochs": 25,
    "scheduler_step_size": 12,
    "scheduler_gamma": 0.8113412993435752,
    "dropout_rate": 0.021308574628801813
}
```

---

### Dataset: `digen24_2433`
- **Improvement**: `+0.0350`
- **Ours Accuracy**: `0.5850`
- **NCTD Accuracy**: `0.5500`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.001758332412593033,
    "batch_size": 64,
    "weight_decay": 0.004145420829191754,
    "optimizer": "adam",
    "scheduler": "exponential",
    "epochs": 46,
    "scheduler_gamma": 0.9600438352251438,
    "dropout_rate": 0.3262866696371001
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.00012253559469509905,
    "batch_size": 32,
    "weight_decay": 0.00316019559321008,
    "optimizer": "adam",
    "scheduler": "step",
    "epochs": 36,
    "scheduler_step_size": 15,
    "scheduler_gamma": 0.24323185657656418,
    "dropout_rate": 0.20181504197628813
}
```

---

### Dataset: `digen25_2433`
- **Improvement**: `+0.0550`
- **Ours Accuracy**: `0.6400`
- **NCTD Accuracy**: `0.5850`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.000559557199106714,
    "batch_size": 32,
    "weight_decay": 0.0005493165657633907,
    "optimizer": "adam",
    "scheduler": "step",
    "epochs": 44,
    "scheduler_step_size": 6,
    "scheduler_gamma": 0.6184198848457068,
    "dropout_rate": 0.24170243437713024
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.0007580727686661631,
    "batch_size": 32,
    "weight_decay": 7.708678608765492e-05,
    "optimizer": "adam",
    "scheduler": "cosine",
    "epochs": 29,
    "scheduler_t_max": 39,
    "dropout_rate": 0.16919934559473437
}
```

---

### Dataset: `digen26_7270`
- **Improvement**: `+0.1300`
- **Ours Accuracy**: `0.6950`
- **NCTD Accuracy**: `0.5650`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.0006679220634441973,
    "batch_size": 64,
    "weight_decay": 1.1548695008112546e-05,
    "optimizer": "adam",
    "scheduler": "exponential",
    "epochs": 45,
    "scheduler_gamma": 0.974719809865034,
    "dropout_rate": 0.4787568236425813
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.0017307934245042165,
    "batch_size": 64,
    "weight_decay": 0.0008189557282041664,
    "optimizer": "sgd",
    "scheduler": "step",
    "epochs": 45,
    "momentum": 0.9541975885381628,
    "scheduler_step_size": 14,
    "scheduler_gamma": 0.7204792714245539,
    "dropout_rate": 0.46513369486350803
}
```

---

### Dataset: `digen27_860`
- **Improvement**: `+0.0200`
- **Ours Accuracy**: `0.5300`
- **NCTD Accuracy**: `0.5100`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.0007317037525570994,
    "batch_size": 64,
    "weight_decay": 0.0009060300156690793,
    "optimizer": "adam",
    "scheduler": "step",
    "epochs": 38,
    "scheduler_step_size": 9,
    "scheduler_gamma": 0.8705596390726511,
    "dropout_rate": 0.21522077518670896
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.006456328772172276,
    "batch_size": 32,
    "weight_decay": 0.0009532180368722788,
    "optimizer": "adam",
    "scheduler": "step",
    "epochs": 21,
    "scheduler_step_size": 9,
    "scheduler_gamma": 0.6286733754544857,
    "dropout_rate": 0.1798265771021657
}
```

---

### Dataset: `digen28_769`
- **Improvement**: `+0.0250`
- **Ours Accuracy**: `0.5850`
- **NCTD Accuracy**: `0.5600`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.002133236295786527,
    "batch_size": 64,
    "weight_decay": 4.851573847784861e-05,
    "optimizer": "adam",
    "scheduler": "none",
    "epochs": 30,
    "dropout_rate": 0.3124809448132393
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.007086835442634417,
    "batch_size": 128,
    "weight_decay": 5.293522224328394e-05,
    "optimizer": "adam",
    "scheduler": "none",
    "epochs": 41,
    "dropout_rate": 0.34925279709154483
}
```

---

### Dataset: `digen29_8322`
- **Improvement**: `+0.0700`
- **Ours Accuracy**: `0.5950`
- **NCTD Accuracy**: `0.5250`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.0011080470700185778,
    "batch_size": 32,
    "weight_decay": 0.0003515270158589132,
    "optimizer": "adam",
    "scheduler": "step",
    "epochs": 48,
    "scheduler_step_size": 15,
    "scheduler_gamma": 0.19549631530156553,
    "dropout_rate": 0.17949430669226124
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.005230838644308987,
    "batch_size": 32,
    "weight_decay": 0.00017792950885832414,
    "optimizer": "adam",
    "scheduler": "step",
    "epochs": 43,
    "scheduler_step_size": 10,
    "scheduler_gamma": 0.8455283900938653,
    "dropout_rate": 0.20367459707619132
}
```

---

### Dataset: `digen30_4426`
- **Improvement**: `+0.1100`
- **Ours Accuracy**: `0.6350`
- **NCTD Accuracy**: `0.5250`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.00032642996947381003,
    "batch_size": 64,
    "weight_decay": 0.0018951129155452442,
    "optimizer": "adam",
    "scheduler": "step",
    "epochs": 50,
    "scheduler_step_size": 15,
    "scheduler_gamma": 0.7234116253108025,
    "dropout_rate": 0.3308254683709223
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.0002795608832129128,
    "batch_size": 64,
    "weight_decay": 1.2797259851496663e-05,
    "optimizer": "adam",
    "scheduler": "exponential",
    "epochs": 29,
    "scheduler_gamma": 0.9693046933291456,
    "dropout_rate": 0.41908336170465527
}
```

---

### Dataset: `digen31_2433`
- **Improvement**: `+0.3350`
- **Ours Accuracy**: `0.8350`
- **NCTD Accuracy**: `0.5000`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.005085746213379573,
    "batch_size": 32,
    "weight_decay": 0.006000789698167152,
    "optimizer": "sgd",
    "scheduler": "step",
    "epochs": 31,
    "momentum": 0.9770376352240974,
    "scheduler_step_size": 12,
    "scheduler_gamma": 0.4575775555066955,
    "dropout_rate": 0.3349731572302141
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.0037465575368525674,
    "batch_size": 64,
    "weight_decay": 0.00010665603759029371,
    "optimizer": "sgd",
    "scheduler": "exponential",
    "epochs": 50,
    "momentum": 0.913152988836159,
    "scheduler_gamma": 0.9884690432219086,
    "dropout_rate": 0.3190095554459596
}
```

---

### Dataset: `digen32_5191`
- **Improvement**: `+0.1550`
- **Ours Accuracy**: `0.6750`
- **NCTD Accuracy**: `0.5200`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.00322432297860479,
    "batch_size": 64,
    "weight_decay": 0.0007545994726824543,
    "optimizer": "adam",
    "scheduler": "cosine",
    "epochs": 50,
    "scheduler_t_max": 34,
    "dropout_rate": 0.1320298828304515
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.001946502048425891,
    "batch_size": 128,
    "weight_decay": 0.00018390542188137472,
    "optimizer": "adam",
    "scheduler": "step",
    "epochs": 20,
    "scheduler_step_size": 9,
    "scheduler_gamma": 0.7142312357160384,
    "dropout_rate": 0.332375576572579
}
```

---

### Dataset: `digen33_769`
- **Improvement**: `+0.2150`
- **Ours Accuracy**: `0.9100`
- **NCTD Accuracy**: `0.6950`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.005575895168719141,
    "batch_size": 32,
    "weight_decay": 0.003550429204377407,
    "optimizer": "adam",
    "scheduler": "step",
    "epochs": 47,
    "scheduler_step_size": 10,
    "scheduler_gamma": 0.2835412051899393,
    "dropout_rate": 0.03456187233532905
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.00020735105983986417,
    "batch_size": 32,
    "weight_decay": 8.519763961898053e-05,
    "optimizer": "adam",
    "scheduler": "none",
    "epochs": 44,
    "dropout_rate": 0.3260777919886364
}
```

---

### Dataset: `digen34_769`
- **Improvement**: `+0.0750`
- **Ours Accuracy**: `0.6250`
- **NCTD Accuracy**: `0.5500`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.0007480270669664834,
    "batch_size": 128,
    "weight_decay": 0.005111525611960847,
    "optimizer": "adam",
    "scheduler": "none",
    "epochs": 49,
    "dropout_rate": 0.2662642151300669
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.00017160732671708226,
    "batch_size": 128,
    "weight_decay": 3.7995892387413407e-05,
    "optimizer": "adam",
    "scheduler": "exponential",
    "epochs": 39,
    "scheduler_gamma": 0.9847204911889607,
    "dropout_rate": 0.33660862909057365
}
```

---

### Dataset: `digen35_4426`
- **Improvement**: `+0.0400`
- **Ours Accuracy**: `0.5200`
- **NCTD Accuracy**: `0.4800`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.0012593636648863878,
    "batch_size": 128,
    "weight_decay": 0.0005059241714724792,
    "optimizer": "adam",
    "scheduler": "cosine",
    "epochs": 46,
    "scheduler_t_max": 35,
    "dropout_rate": 0.22899642382042576
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 9.609125229410837e-05,
    "batch_size": 32,
    "weight_decay": 1.644349777459903e-05,
    "optimizer": "adam",
    "scheduler": "exponential",
    "epochs": 24,
    "scheduler_gamma": 0.9849761800209733,
    "dropout_rate": 0.4388166251476084
}
```

---

### Dataset: `digen36_466`
- **Improvement**: `+0.1900`
- **Ours Accuracy**: `0.7400`
- **NCTD Accuracy**: `0.5500`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.009905806517786518,
    "batch_size": 32,
    "weight_decay": 0.0002655941803336335,
    "optimizer": "adam",
    "scheduler": "step",
    "epochs": 37,
    "scheduler_step_size": 12,
    "scheduler_gamma": 0.7606911891805211,
    "dropout_rate": 0.0971876821329675
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.009321037210607192,
    "batch_size": 128,
    "weight_decay": 0.003387162059605974,
    "optimizer": "sgd",
    "scheduler": "none",
    "epochs": 29,
    "momentum": 0.9836142174235196,
    "dropout_rate": 0.0317254410068032
}
```

---

### Dataset: `digen37_769`
- **Improvement**: `+0.1850`
- **Ours Accuracy**: `0.7850`
- **NCTD Accuracy**: `0.6000`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.009609730463776034,
    "batch_size": 32,
    "weight_decay": 0.0005797070135588095,
    "optimizer": "adam",
    "scheduler": "cosine",
    "epochs": 39,
    "scheduler_t_max": 38,
    "dropout_rate": 0.3255354464461402
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.009804813884143065,
    "batch_size": 32,
    "weight_decay": 1.7860749809712622e-05,
    "optimizer": "adam",
    "scheduler": "exponential",
    "epochs": 48,
    "scheduler_gamma": 0.9563601209171761,
    "dropout_rate": 0.4549150232269776
}
```

---

### Dataset: `digen38_4426`
- **Improvement**: `+0.0700`
- **Ours Accuracy**: `0.6350`
- **NCTD Accuracy**: `0.5650`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.001452496413026206,
    "batch_size": 32,
    "weight_decay": 1.8363725063037114e-05,
    "optimizer": "adam",
    "scheduler": "step",
    "epochs": 47,
    "scheduler_step_size": 9,
    "scheduler_gamma": 0.7862209116075976,
    "dropout_rate": 0.20881778340901086
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.0006569341100011731,
    "batch_size": 32,
    "weight_decay": 0.0001073780702049272,
    "optimizer": "adam",
    "scheduler": "step",
    "epochs": 30,
    "scheduler_step_size": 10,
    "scheduler_gamma": 0.3680695305879956,
    "dropout_rate": 0.10037054479605617
}
```

---

### Dataset: `digen39_5578`
- **Improvement**: `+0.2600`
- **Ours Accuracy**: `0.8800`
- **NCTD Accuracy**: `0.6200`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.00033213696725135894,
    "batch_size": 32,
    "weight_decay": 9.133448199273545e-05,
    "optimizer": "adam",
    "scheduler": "exponential",
    "epochs": 49,
    "scheduler_gamma": 0.971810992222033,
    "dropout_rate": 0.4214449849811201
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.0008621607371646705,
    "batch_size": 64,
    "weight_decay": 0.0005295514460169657,
    "optimizer": "adam",
    "scheduler": "cosine",
    "epochs": 27,
    "scheduler_t_max": 37,
    "dropout_rate": 0.22356588723440304
}
```

---

### Dataset: `digen40_5390`
- **Improvement**: `+0.0100`
- **Ours Accuracy**: `0.5250`
- **NCTD Accuracy**: `0.5150`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.009817261850303318,
    "batch_size": 32,
    "weight_decay": 0.00016467040004771076,
    "optimizer": "sgd",
    "scheduler": "cosine",
    "epochs": 29,
    "momentum": 0.8566198862231779,
    "scheduler_t_max": 34,
    "dropout_rate": 0.13888838918846116
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.0002557172086379378,
    "batch_size": 64,
    "weight_decay": 0.006136782127543582,
    "optimizer": "sgd",
    "scheduler": "step",
    "epochs": 46,
    "momentum": 0.8630646015300286,
    "scheduler_step_size": 11,
    "scheduler_gamma": 0.11163222998678068,
    "dropout_rate": 0.03522996957973715
}
```

---

### Dataset: `DS02_student_dropout`
- **Improvement**: `+0.0090`
- **Ours Accuracy**: `0.7991`
- **NCTD Accuracy**: `0.7901`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.0006498627839821731,
    "batch_size": 32,
    "weight_decay": 7.128636209648619e-05,
    "optimizer": "adam",
    "scheduler": "exponential",
    "epochs": 47,
    "scheduler_gamma": 0.9793349501802932,
    "dropout_rate": 0.47090779265324656
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 7.601614646752211e-05,
    "batch_size": 64,
    "weight_decay": 2.2023410387145425e-05,
    "optimizer": "adam",
    "scheduler": "exponential",
    "epochs": 47,
    "scheduler_gamma": 0.9596725964892531,
    "dropout_rate": 0.11457779312798183
}
```

---

### Dataset: `DS07_ringnorm`
- **Improvement**: `+0.0095`
- **Ours Accuracy**: `0.9919`
- **NCTD Accuracy**: `0.9824`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.00041090616794067075,
    "batch_size": 64,
    "weight_decay": 0.0003760395899838903,
    "optimizer": "adam",
    "scheduler": "cosine",
    "epochs": 50,
    "scheduler_t_max": 38,
    "dropout_rate": 0.12430512872059894
}
```

#### Hyperparameters (NCTD baseline):
```json
{
    "learning_rate": 0.0006283671561676565,
    "batch_size": 64,
    "weight_decay": 2.4342938822146067e-05,
    "optimizer": "adam",
    "scheduler": "step",
    "epochs": 29,
    "scheduler_step_size": 15,
    "scheduler_gamma": 0.5609629711454908,
    "dropout_rate": 0.17835198485403547
}
```

---


## ⚖️ NCTD Better/Equal or Error Cases

### Dataset: `13_rotated_rastrigin_50d`
> [!NOTE]
> Ours is NOT the best performer in this case.

- **Difference**: `0.0000`
- **Ours Accuracy**: `0.5000`
- **NCTD Accuracy**: `0.5000`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.00407088744291861,
    "batch_size": 32,
    "weight_decay": 0.009437486920685035,
    "optimizer": "sgd",
    "scheduler": "exponential",
    "epochs": 39,
    "momentum": 0.9427074969594389,
    "scheduler_gamma": 0.9679357461124886,
    "dropout_rate": 0.34095435497342486
}
```

#### Hyperparameters (NCTD winner/equal):
```json
{
    "learning_rate": 1.2836154834385626e-05,
    "batch_size": 64,
    "weight_decay": 0.00596933125288995,
    "optimizer": "adam",
    "scheduler": "none",
    "epochs": 26,
    "dropout_rate": 0.1319315513239795
}
```

---

### Dataset: `digen2_6949`
> [!NOTE]
> Ours is NOT the best performer in this case.

- **Difference**: `-0.0050`
- **Ours Accuracy**: `0.5400`
- **NCTD Accuracy**: `0.5450`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.002361927776461118,
    "batch_size": 32,
    "weight_decay": 0.001189685211927671,
    "optimizer": "sgd",
    "scheduler": "step",
    "epochs": 49,
    "momentum": 0.8590998853064948,
    "scheduler_step_size": 14,
    "scheduler_gamma": 0.6109777239983218,
    "dropout_rate": 0.068636095521588
}
```

#### Hyperparameters (NCTD winner/equal):
```json
{
    "learning_rate": 0.0010888948826637452,
    "batch_size": 128,
    "weight_decay": 0.0003468307009720268,
    "optimizer": "sgd",
    "scheduler": "step",
    "epochs": 26,
    "momentum": 0.9091302980913425,
    "scheduler_step_size": 11,
    "scheduler_gamma": 0.725985472971611,
    "dropout_rate": 0.2396865915007021
}
```

---

### Dataset: `digen3_769`
> [!NOTE]
> Ours is NOT the best performer in this case.

- **Difference**: `0.0000`
- **Ours Accuracy**: `0.9600`
- **NCTD Accuracy**: `0.9600`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.004999461561174165,
    "batch_size": 64,
    "weight_decay": 4.881892907706292e-05,
    "optimizer": "adam",
    "scheduler": "exponential",
    "epochs": 30,
    "scheduler_gamma": 0.9835910121811069,
    "dropout_rate": 0.37578583864288717
}
```

#### Hyperparameters (NCTD winner/equal):
```json
{
    "learning_rate": 0.005507380641804208,
    "batch_size": 32,
    "weight_decay": 0.0027218743685769047,
    "optimizer": "adam",
    "scheduler": "step",
    "epochs": 48,
    "scheduler_step_size": 9,
    "scheduler_gamma": 0.8814146174734436,
    "dropout_rate": 0.2026834184264521
}
```

---

### Dataset: `digen4_860`
> [!NOTE]
> Ours is NOT the best performer in this case.

- **Difference**: `-0.0150`
- **Ours Accuracy**: `0.5600`
- **NCTD Accuracy**: `0.5750`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.0001169081787364292,
    "batch_size": 32,
    "weight_decay": 0.0025027632784148162,
    "optimizer": "adam",
    "scheduler": "none",
    "epochs": 39,
    "dropout_rate": 0.47380572630684337
}
```

#### Hyperparameters (NCTD winner/equal):
```json
{
    "learning_rate": 0.0018055329247029196,
    "batch_size": 32,
    "weight_decay": 0.0004521008544423144,
    "optimizer": "sgd",
    "scheduler": "cosine",
    "epochs": 32,
    "momentum": 0.983342378747331,
    "scheduler_t_max": 12,
    "dropout_rate": 0.1617969590522632
}
```

---

### Dataset: `digen6_466`
> [!NOTE]
> Ours is NOT the best performer in this case.

- **Difference**: `-0.0250`
- **Ours Accuracy**: `0.4950`
- **NCTD Accuracy**: `0.5200`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.00028443399547461995,
    "batch_size": 32,
    "weight_decay": 0.005806828984285978,
    "optimizer": "adam",
    "scheduler": "step",
    "epochs": 43,
    "scheduler_step_size": 5,
    "scheduler_gamma": 0.8412734505837545,
    "dropout_rate": 0.4775552121909811
}
```

#### Hyperparameters (NCTD winner/equal):
```json
{
    "learning_rate": 2.1970846024672676e-05,
    "batch_size": 32,
    "weight_decay": 0.0006976718834376773,
    "optimizer": "adam",
    "scheduler": "step",
    "epochs": 48,
    "scheduler_step_size": 7,
    "scheduler_gamma": 0.5266272400733181,
    "dropout_rate": 0.07240606420812773
}
```

---

### Dataset: `digen18_5578`
> [!NOTE]
> Ours is NOT the best performer in this case.

- **Difference**: `-0.0050`
- **Ours Accuracy**: `0.6000`
- **NCTD Accuracy**: `0.6050`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.0004827825126063408,
    "batch_size": 32,
    "weight_decay": 0.004052485070936014,
    "optimizer": "adam",
    "scheduler": "none",
    "epochs": 41,
    "dropout_rate": 0.07183599426766114
}
```

#### Hyperparameters (NCTD winner/equal):
```json
{
    "learning_rate": 0.003510584148839932,
    "batch_size": 64,
    "weight_decay": 0.00017248990547270892,
    "optimizer": "adam",
    "scheduler": "exponential",
    "epochs": 35,
    "scheduler_gamma": 0.9851659610616159,
    "dropout_rate": 0.4800364767901657
}
```

---

### Dataset: `digen23_5191`
> [!NOTE]
> Ours is NOT the best performer in this case.

- **Difference**: `-0.0300`
- **Ours Accuracy**: `0.4900`
- **NCTD Accuracy**: `0.5200`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.0016732660603612896,
    "batch_size": 32,
    "weight_decay": 0.0004634063597138018,
    "optimizer": "adam",
    "scheduler": "cosine",
    "epochs": 27,
    "scheduler_t_max": 10,
    "dropout_rate": 0.4090676021543359
}
```

#### Hyperparameters (NCTD winner/equal):
```json
{
    "learning_rate": 0.0005826420581803579,
    "batch_size": 128,
    "weight_decay": 5.446532721829738e-05,
    "optimizer": "adam",
    "scheduler": "exponential",
    "epochs": 42,
    "scheduler_gamma": 0.984388762042119,
    "dropout_rate": 0.46614166948161073
}
```

---

### Dataset: `DS01_breast_cancer`
> [!NOTE]
> Ours is NOT the best performer in this case.

- **Difference**: `0.0000`
- **Ours Accuracy**: `0.6571`
- **NCTD Accuracy**: `0.6571`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.0001329291894316216,
    "batch_size": 32,
    "weight_decay": 2.9380279387035334e-05,
    "optimizer": "adam",
    "scheduler": "none",
    "epochs": 50,
    "dropout_rate": 0.41622132040021087
}
```

#### Hyperparameters (NCTD winner/equal):
```json
{
    "learning_rate": 0.0001329291894316216,
    "batch_size": 32,
    "weight_decay": 2.9380279387035334e-05,
    "optimizer": "adam",
    "scheduler": "none",
    "epochs": 50,
    "dropout_rate": 0.41622132040021087
}
```

---

### Dataset: `DS03_adult`
> [!NOTE]
> Ours is NOT the best performer in this case.

- **Difference**: `-0.0047`
- **Ours Accuracy**: `0.5934`
- **NCTD Accuracy**: `0.5982`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.007215842818543611,
    "batch_size": 64,
    "weight_decay": 0.0034625003363405833,
    "optimizer": "sgd",
    "scheduler": "step",
    "epochs": 32,
    "momentum": 0.9700779132575867,
    "scheduler_step_size": 8,
    "scheduler_gamma": 0.1897450393389425,
    "dropout_rate": 0.08708975709932396
}
```

#### Hyperparameters (NCTD winner/equal):
```json
{
    "learning_rate": 0.0002451616602530067,
    "batch_size": 32,
    "weight_decay": 0.0008848851230686015,
    "optimizer": "adam",
    "scheduler": "exponential",
    "epochs": 49,
    "scheduler_gamma": 0.9708310533718633,
    "dropout_rate": 0.35679281205181296
}
```

---

### Dataset: `DS04_heart`
> [!NOTE]
> Ours is NOT the best performer in this case.

- **Difference**: `0.0000`
- **Ours Accuracy**: `0.9130`
- **NCTD Accuracy**: `0.9130`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.006533897137117582,
    "batch_size": 32,
    "weight_decay": 0.008881480471200678,
    "optimizer": "sgd",
    "scheduler": "exponential",
    "epochs": 41,
    "momentum": 0.9025278202925315,
    "scheduler_gamma": 0.9897883650454791,
    "dropout_rate": 0.2812276841599592
}
```

#### Hyperparameters (NCTD winner/equal):
```json
{
    "learning_rate": 0.0008850464618769227,
    "batch_size": 32,
    "weight_decay": 0.00013714544118141698,
    "optimizer": "adam",
    "scheduler": "none",
    "epochs": 21,
    "dropout_rate": 0.2083078865814134
}
```

---

### Dataset: `DS05_thyroid`
> [!WARNING]
> Stratified split failed (too few samples in class [5]).

- **Difference**: `0.0000`
- **Ours Accuracy**: `0.0000`
- **NCTD Accuracy**: `0.0000`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.0001329291894316216,
    "batch_size": 32,
    "weight_decay": 2.9380279387035334e-05,
    "optimizer": "adam",
    "scheduler": "none",
    "epochs": 50,
    "dropout_rate": 0.41622132040021087
}
```

#### Hyperparameters (NCTD winner/equal):
```json
{
    "learning_rate": 0.0001329291894316216,
    "batch_size": 32,
    "weight_decay": 2.9380279387035334e-05,
    "optimizer": "adam",
    "scheduler": "none",
    "epochs": 50,
    "dropout_rate": 0.41622132040021087
}
```

---

### Dataset: `DS06_dengue_chikungunya`
> [!NOTE]
> Ours is NOT the best performer in this case.

- **Difference**: `-0.0012`
- **Ours Accuracy**: `0.6275`
- **NCTD Accuracy**: `0.6286`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.004935110550656103,
    "batch_size": 64,
    "weight_decay": 7.832793252825395e-05,
    "optimizer": "adam",
    "scheduler": "cosine",
    "epochs": 24,
    "scheduler_t_max": 29,
    "dropout_rate": 0.13888103644788335
}
```

#### Hyperparameters (NCTD winner/equal):
```json
{
    "learning_rate": 0.00011312192077970348,
    "batch_size": 64,
    "weight_decay": 0.00021635836190406327,
    "optimizer": "adam",
    "scheduler": "cosine",
    "epochs": 36,
    "scheduler_t_max": 34,
    "dropout_rate": 0.15178491954208467
}
```

---

### Dataset: `DS08_isolet`
> [!WARNING]
> Crashed with CUDA unknown error after trial 0.

- **Difference**: `0.5808`
- **Ours Accuracy**: `0.7564`
- **NCTD Accuracy**: `0.1756`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.007571661863922665,
    "batch_size": 64,
    "weight_decay": 4.496880686511051e-05,
    "optimizer": "adam",
    "scheduler": "cosine",
    "epochs": 40,
    "scheduler_t_max": 33,
    "dropout_rate": 0.2627231916502448
}
```

#### Hyperparameters (NCTD winner/equal):
```json
{
    "learning_rate": 0.0001329291894316216,
    "batch_size": 32,
    "weight_decay": 2.9380279387035334e-05,
    "optimizer": "adam",
    "scheduler": "none",
    "epochs": 50,
    "dropout_rate": 0.41622132040021087
}
```

---

### Dataset: `DS09_madelon`
> [!WARNING]
> Failed entirely with CUDA unknown error.

- **Difference**: `0.5692`
- **Ours Accuracy**: `0.5692`
- **NCTD Accuracy**: `0.0000`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 4.335281794951564e-05,
    "batch_size": 128,
    "weight_decay": 0.00037520558551242813,
    "optimizer": "adam",
    "scheduler": "none",
    "epochs": 34,
    "dropout_rate": 0.3925879806965068
}
```

#### Hyperparameters (NCTD winner/equal):
```json
{
    "learning_rate": 0.0001329291894316216,
    "batch_size": 32,
    "weight_decay": 2.9380279387035334e-05,
    "optimizer": "adam",
    "scheduler": "none",
    "epochs": 50,
    "dropout_rate": 0.41622132040021087
}
```

---

### Dataset: `DS10_relathee`
> [!WARNING]
> Failed entirely with CUDA Out of Memory (OOM).

- **Difference**: `0.6224`
- **Ours Accuracy**: `0.6224`
- **NCTD Accuracy**: `0.0000`

#### Hyperparameters (Ours):
```json
{
    "learning_rate": 0.00019099646675394696,
    "batch_size": 128,
    "weight_decay": 1.575717846857783e-05,
    "optimizer": "adam",
    "scheduler": "step",
    "epochs": 37,
    "scheduler_step_size": 15,
    "scheduler_gamma": 0.6115070393162243,
    "dropout_rate": 0.2495266067832364
}
```

#### Hyperparameters (NCTD winner/equal):
```json
{
    "learning_rate": 0.0001329291894316216,
    "batch_size": 32,
    "weight_decay": 2.9380279387035334e-05,
    "optimizer": "adam",
    "scheduler": "none",
    "epochs": 50,
    "dropout_rate": 0.41622132040021087
}
```

---
