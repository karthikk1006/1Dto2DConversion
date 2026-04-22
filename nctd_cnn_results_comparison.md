# NCTD CNN Comparative Results and Hyperparameters

## Dataset: `DS01_breast_cancer`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
    "accuracy": 0.6571428571428571
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
    "accuracy": 0.6571428571428571
}
```

---

## Dataset: `DS02_student_dropout`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
    "accuracy": 0.7900677200902935
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
    "accuracy": 0.7990970654627539
}
```

---

## Dataset: `DS03_adult`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
    "accuracy": 0.598157625383828
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
    "accuracy": 0.5934493346980553
}
```

---

## Dataset: `DS04_heart`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
    "accuracy": 0.9130434782608695
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
    "accuracy": 0.9130434782608695
}
```

---

## Dataset: `DS05_thyroid`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
    "accuracy": 0.0
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
    "accuracy": 0.0
}
```

---

## Dataset: `DS06_dengue_chikungunya`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
    "accuracy": 0.6286379511059371
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
    "accuracy": 0.6274738067520372
}
```

---

## Dataset: `DS07_ringnorm`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
    "accuracy": 0.9824324324324324
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
    "accuracy": 0.9918918918918919
}
```

---

## Dataset: `DS08_isolet`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
    "accuracy": 0.17564102564102563
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
    "accuracy": 0.7564102564102564
}
```

---

## Dataset: `DS09_madelon`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
    "accuracy": 0.0
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
    "accuracy": 0.5692307692307692
}
```

---

## Dataset: `DS10_relathee`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
    "accuracy": 0.0
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
    "accuracy": 0.6223776223776224
}
```

---

