# Complete Tuned Results and Hyperparameters Dump

## Dataset: `06_friedman1`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.6585365853658537,
  "f1": 0.6568627450980392,
  "f1_macro": 0.6585284599276604,
  "f1_weighted": 0.6585365853658537,
  "precision": 0.6568627450980392,
  "recall": 0.6568627450980392,
  "balanced_accuracy": 0.6585284599276604,
  "roc_auc": 0.7272035027603274,
  "mcc": 0.31705691985532075,
  "cohen_kappa": 0.3170569198553208,
  "num_classes": 2,
  "num_samples": 205,
  "per_class_report": {
    "0": {
      "precision": 0.6601941747572816,
      "recall": 0.6601941747572816,
      "f1-score": 0.6601941747572816,
      "support": 103.0
    },
    "1": {
      "precision": 0.6568627450980392,
      "recall": 0.6568627450980392,
      "f1-score": 0.6568627450980392,
      "support": 102.0
    },
    "accuracy": 0.6585365853658537,
    "macro avg": {
      "precision": 0.6585284599276604,
      "recall": 0.6585284599276604,
      "f1-score": 0.6585284599276604,
      "support": 205.0
    },
    "weighted avg": {
      "precision": 0.6585365853658537,
      "recall": 0.6585365853658537,
      "f1-score": 0.6585365853658537,
      "support": 205.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.8682926829268293,
  "f1": 0.8656716417910447,
  "f1_macro": 0.8682425194601157,
  "f1_weighted": 0.8682550603267941,
  "precision": 0.8787878787878788,
  "recall": 0.8529411764705882,
  "balanced_accuracy": 0.8682181610508282,
  "roc_auc": 0.9474585950885208,
  "mcc": 0.7368572631603794,
  "cohen_kappa": 0.736541482221905,
  "num_classes": 2,
  "num_samples": 205,
  "per_class_report": {
    "0": {
      "precision": 0.8584905660377359,
      "recall": 0.883495145631068,
      "f1-score": 0.8708133971291866,
      "support": 103.0
    },
    "1": {
      "precision": 0.8787878787878788,
      "recall": 0.8529411764705882,
      "f1-score": 0.8656716417910447,
      "support": 102.0
    },
    "accuracy": 0.8682926829268293,
    "macro avg": {
      "precision": 0.8686392224128073,
      "recall": 0.8682181610508282,
      "f1-score": 0.8682425194601157,
      "support": 205.0
    },
    "weighted avg": {
      "precision": 0.8685897167719534,
      "recall": 0.8682926829268293,
      "f1-score": 0.8682550603267941,
      "support": 205.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 2.4228089807522295e-05,
  "batch_size": 128,
  "weight_decay": 0.0010900544962857228,
  "optimizer": "sgd",
  "scheduler": "exponential",
  "epochs": 31,
  "momentum": 0.9697938087205705,
  "scheduler_gamma": 0.9753437848891986
}
```

**Metrics:**
```json
{
  "accuracy": 0.5170731707317073,
  "f1": 0.4277456647398844,
  "f1_macro": 0.5050120728762713,
  "f1_weighted": 0.5053889821842537,
  "precision": 0.5211267605633803,
  "recall": 0.3627450980392157,
  "balanced_accuracy": 0.5163240053302874,
  "roc_auc": 0.5021892252046449,
  "mcc": 0.03430787730681197,
  "cohen_kappa": 0.03269624898717882,
  "num_classes": 2,
  "num_samples": 205,
  "per_class_report": {
    "0": {
      "precision": 0.5149253731343284,
      "recall": 0.6699029126213593,
      "f1-score": 0.5822784810126582,
      "support": 103.0
    },
    "1": {
      "precision": 0.5211267605633803,
      "recall": 0.3627450980392157,
      "f1-score": 0.4277456647398844,
      "support": 102.0
    },
    "accuracy": 0.5170731707317073,
    "macro avg": {
      "precision": 0.5180260668488543,
      "recall": 0.5163240053302874,
      "f1-score": 0.5050120728762713,
      "support": 205.0
    },
    "weighted avg": {
      "precision": 0.5180109415136616,
      "recall": 0.5170731707317073,
      "f1-score": 0.5053889821842537,
      "support": 205.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.0017774880883941375,
  "batch_size": 128,
  "weight_decay": 0.000515598761266142,
  "optimizer": "sgd",
  "scheduler": "cosine",
  "epochs": 23,
  "momentum": 0.9876526518420065,
  "scheduler_t_max": 14
}
```

**Metrics:**
```json
{
  "accuracy": 0.47317073170731705,
  "f1": 0.47058823529411764,
  "f1_macro": 0.47315819531696174,
  "f1_weighted": 0.47317073170731705,
  "precision": 0.47058823529411764,
  "recall": 0.47058823529411764,
  "balanced_accuracy": 0.47315819531696174,
  "roc_auc": 0.4506948410432134,
  "mcc": -0.05368360936607653,
  "cohen_kappa": -0.05368360936607641,
  "num_classes": 2,
  "num_samples": 205,
  "per_class_report": {
    "0": {
      "precision": 0.47572815533980584,
      "recall": 0.47572815533980584,
      "f1-score": 0.47572815533980584,
      "support": 103.0
    },
    "1": {
      "precision": 0.47058823529411764,
      "recall": 0.47058823529411764,
      "f1-score": 0.47058823529411764,
      "support": 102.0
    },
    "accuracy": 0.47317073170731705,
    "macro avg": {
      "precision": 0.47315819531696174,
      "recall": 0.47315819531696174,
      "f1-score": 0.47315819531696174,
      "support": 205.0
    },
    "weighted avg": {
      "precision": 0.47317073170731705,
      "recall": 0.47317073170731705,
      "f1-score": 0.47317073170731705,
      "support": 205.0
    }
  }
}
```

---

## Dataset: `07_friedman2`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.8390243902439024,
  "f1": 0.8216216216216217,
  "f1_macro": 0.8374774774774776,
  "f1_weighted": 0.8375548231157989,
  "precision": 0.9156626506024096,
  "recall": 0.7450980392156863,
  "balanced_accuracy": 0.8385684370835713,
  "roc_auc": 0.933466590519703,
  "mcc": 0.6897253704130746,
  "cohen_kappa": 0.6777497260991759,
  "num_classes": 2,
  "num_samples": 205,
  "per_class_report": {
    "0": {
      "precision": 0.7868852459016393,
      "recall": 0.9320388349514563,
      "f1-score": 0.8533333333333334,
      "support": 103.0
    },
    "1": {
      "precision": 0.9156626506024096,
      "recall": 0.7450980392156863,
      "f1-score": 0.8216216216216217,
      "support": 102.0
    },
    "accuracy": 0.8390243902439024,
    "macro avg": {
      "precision": 0.8512739482520244,
      "recall": 0.8385684370835713,
      "f1-score": 0.8374774774774776,
      "support": 205.0
    },
    "weighted avg": {
      "precision": 0.850959857021047,
      "recall": 0.8390243902439024,
      "f1-score": 0.8375548231157989,
      "support": 205.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.8780487804878049,
  "f1": 0.8768472906403941,
  "f1_macro": 0.8780371718902453,
  "f1_weighted": 0.8780429761890252,
  "precision": 0.8811881188118812,
  "recall": 0.8725490196078431,
  "balanced_accuracy": 0.8780220826194556,
  "roc_auc": 0.946126023224824,
  "mcc": 0.7561161385995614,
  "cohen_kappa": 0.7560801484936462,
  "num_classes": 2,
  "num_samples": 205,
  "per_class_report": {
    "0": {
      "precision": 0.875,
      "recall": 0.883495145631068,
      "f1-score": 0.8792270531400966,
      "support": 103.0
    },
    "1": {
      "precision": 0.8811881188118812,
      "recall": 0.8725490196078431,
      "f1-score": 0.8768472906403941,
      "support": 102.0
    },
    "accuracy": 0.8780487804878049,
    "macro avg": {
      "precision": 0.8780940594059405,
      "recall": 0.8780220826194556,
      "f1-score": 0.8780371718902453,
      "support": 205.0
    },
    "weighted avg": {
      "precision": 0.8780789664332287,
      "recall": 0.8780487804878049,
      "f1-score": 0.8780429761890252,
      "support": 205.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.0008262064512644598,
  "batch_size": 128,
  "weight_decay": 0.0006842079485840883,
  "optimizer": "sgd",
  "scheduler": "cosine",
  "epochs": 45,
  "momentum": 0.9280437045809077,
  "scheduler_t_max": 26
}
```

**Metrics:**
```json
{
  "accuracy": 0.5756097560975609,
  "f1": 0.6640926640926641,
  "f1_macro": 0.5439668618476565,
  "f1_weighted": 0.5433808823245101,
  "precision": 0.5477707006369427,
  "recall": 0.8431372549019608,
  "balanced_accuracy": 0.5769084332762231,
  "roc_auc": 0.5884732533790216,
  "mcc": 0.1816152083675736,
  "cohen_kappa": 0.153415294061803,
  "num_classes": 2,
  "num_samples": 205,
  "per_class_report": {
    "0": {
      "precision": 0.6666666666666666,
      "recall": 0.3106796116504854,
      "f1-score": 0.423841059602649,
      "support": 103.0
    },
    "1": {
      "precision": 0.5477707006369427,
      "recall": 0.8431372549019608,
      "f1-score": 0.6640926640926641,
      "support": 102.0
    },
    "accuracy": 0.5756097560975609,
    "macro avg": {
      "precision": 0.6072186836518046,
      "recall": 0.5769084332762231,
      "f1-score": 0.5439668618476565,
      "support": 205.0
    },
    "weighted avg": {
      "precision": 0.6075086738128527,
      "recall": 0.5756097560975609,
      "f1-score": 0.5433808823245101,
      "support": 205.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.006672042784388996,
  "batch_size": 32,
  "weight_decay": 0.004542555569860527,
  "optimizer": "adam",
  "scheduler": "step",
  "epochs": 25,
  "scheduler_step_size": 15,
  "scheduler_gamma": 0.35378329419582044
}
```

**Metrics:**
```json
{
  "accuracy": 0.5024390243902439,
  "f1": 0.0,
  "f1_macro": 0.3344155844155844,
  "f1_weighted": 0.3360468799493189,
  "precision": 0.0,
  "recall": 0.0,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.5,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 205,
  "per_class_report": {
    "0": {
      "precision": 0.5024390243902439,
      "recall": 1.0,
      "f1-score": 0.6688311688311688,
      "support": 103.0
    },
    "1": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 102.0
    },
    "accuracy": 0.5024390243902439,
    "macro avg": {
      "precision": 0.25121951219512195,
      "recall": 0.5,
      "f1-score": 0.3344155844155844,
      "support": 205.0
    },
    "weighted avg": {
      "precision": 0.2524449732302201,
      "recall": 0.5024390243902439,
      "f1-score": 0.3360468799493189,
      "support": 205.0
    }
  }
}
```

---

## Dataset: `08_friedman3`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.5170731707317073,
  "f1": 0.5857740585774058,
  "f1_macro": 0.5034133450781766,
  "f1_weighted": 0.5030115855001316,
  "precision": 0.5109489051094891,
  "recall": 0.6862745098039216,
  "balanced_accuracy": 0.5178945364553589,
  "roc_auc": 0.5523510375023796,
  "mcc": 0.03800619626781572,
  "cohen_kappa": 0.03572955765667307,
  "num_classes": 2,
  "num_samples": 205,
  "per_class_report": {
    "0": {
      "precision": 0.5294117647058824,
      "recall": 0.34951456310679613,
      "f1-score": 0.42105263157894735,
      "support": 103.0
    },
    "1": {
      "precision": 0.5109489051094891,
      "recall": 0.6862745098039216,
      "f1-score": 0.5857740585774058,
      "support": 102.0
    },
    "accuracy": 0.5170731707317073,
    "macro avg": {
      "precision": 0.5201803349076857,
      "recall": 0.5178945364553589,
      "f1-score": 0.5034133450781766,
      "support": 205.0
    },
    "weighted avg": {
      "precision": 0.520225366272555,
      "recall": 0.5170731707317073,
      "f1-score": 0.5030115855001316,
      "support": 205.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.7463414634146341,
  "f1": 0.75,
  "f1_macro": 0.7462871287128713,
  "f1_weighted": 0.746269017145617,
  "precision": 0.7358490566037735,
  "recall": 0.7647058823529411,
  "balanced_accuracy": 0.7464306110793832,
  "roc_auc": 0.8194365124690653,
  "mcc": 0.49314293765599637,
  "cohen_kappa": 0.4927674153026266,
  "num_classes": 2,
  "num_samples": 205,
  "per_class_report": {
    "0": {
      "precision": 0.7575757575757576,
      "recall": 0.7281553398058253,
      "f1-score": 0.7425742574257426,
      "support": 103.0
    },
    "1": {
      "precision": 0.7358490566037735,
      "recall": 0.7647058823529411,
      "f1-score": 0.75,
      "support": 102.0
    },
    "accuracy": 0.7463414634146341,
    "macro avg": {
      "precision": 0.7467124070897655,
      "recall": 0.7464306110793832,
      "f1-score": 0.7462871287128713,
      "support": 205.0
    },
    "weighted avg": {
      "precision": 0.7467653990433557,
      "recall": 0.7463414634146341,
      "f1-score": 0.746269017145617,
      "support": 205.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.00966026758913522,
  "batch_size": 64,
  "weight_decay": 0.005642516904996625,
  "optimizer": "adam",
  "scheduler": "step",
  "epochs": 42,
  "scheduler_step_size": 15,
  "scheduler_gamma": 0.8765433201532511
}
```

**Metrics:**
```json
{
  "accuracy": 0.4975609756097561,
  "f1": 0.6644951140065146,
  "f1_macro": 0.3322475570032573,
  "f1_weighted": 0.3306268372129975,
  "precision": 0.4975609756097561,
  "recall": 1.0,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.5,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 205,
  "per_class_report": {
    "0": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 103.0
    },
    "1": {
      "precision": 0.4975609756097561,
      "recall": 1.0,
      "f1-score": 0.6644951140065146,
      "support": 102.0
    },
    "accuracy": 0.4975609756097561,
    "macro avg": {
      "precision": 0.24878048780487805,
      "recall": 0.5,
      "f1-score": 0.3322475570032573,
      "support": 205.0
    },
    "weighted avg": {
      "precision": 0.2475669244497323,
      "recall": 0.4975609756097561,
      "f1-score": 0.3306268372129975,
      "support": 205.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.008139464782035607,
  "batch_size": 32,
  "weight_decay": 0.0002112118624132588,
  "optimizer": "adam",
  "scheduler": "none",
  "epochs": 32
}
```

**Metrics:**
```json
{
  "accuracy": 0.4975609756097561,
  "f1": 0.6644951140065146,
  "f1_macro": 0.3322475570032573,
  "f1_weighted": 0.3306268372129975,
  "precision": 0.4975609756097561,
  "recall": 1.0,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.5044736341138397,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 205,
  "per_class_report": {
    "0": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 103.0
    },
    "1": {
      "precision": 0.4975609756097561,
      "recall": 1.0,
      "f1-score": 0.6644951140065146,
      "support": 102.0
    },
    "accuracy": 0.4975609756097561,
    "macro avg": {
      "precision": 0.24878048780487805,
      "recall": 0.5,
      "f1-score": 0.3322475570032573,
      "support": 205.0
    },
    "weighted avg": {
      "precision": 0.2475669244497323,
      "recall": 0.4975609756097561,
      "f1-score": 0.3306268372129975,
      "support": 205.0
    }
  }
}
```

---

## Dataset: `13_rotated_rastrigin_50d`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.5,
  "f1": 0.6666666666666666,
  "f1_macro": 0.3333333333333333,
  "f1_weighted": 0.3333333333333333,
  "precision": 0.5,
  "recall": 1.0,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.24,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 10,
  "per_class_report": {
    "0": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 5.0
    },
    "1": {
      "precision": 0.5,
      "recall": 1.0,
      "f1-score": 0.6666666666666666,
      "support": 5.0
    },
    "accuracy": 0.5,
    "macro avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.3333333333333333,
      "support": 10.0
    },
    "weighted avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.3333333333333333,
      "support": 10.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.5,
  "f1": 0.0,
  "f1_macro": 0.3333333333333333,
  "f1_weighted": 0.3333333333333333,
  "precision": 0.0,
  "recall": 0.0,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.76,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 10,
  "per_class_report": {
    "0": {
      "precision": 0.5,
      "recall": 1.0,
      "f1-score": 0.6666666666666666,
      "support": 5.0
    },
    "1": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 5.0
    },
    "accuracy": 0.5,
    "macro avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.3333333333333333,
      "support": 10.0
    },
    "weighted avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.3333333333333333,
      "support": 10.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.0004609061905413901,
  "batch_size": 64,
  "weight_decay": 2.9153571095446216e-05,
  "optimizer": "adam",
  "scheduler": "exponential",
  "epochs": 28,
  "scheduler_gamma": 0.9610851912285155
}
```

**Metrics:**
```json
{
  "accuracy": 0.6,
  "f1": 0.6,
  "f1_macro": 0.6,
  "f1_weighted": 0.6,
  "precision": 0.6,
  "recall": 0.6,
  "balanced_accuracy": 0.6,
  "roc_auc": 0.56,
  "mcc": 0.2,
  "cohen_kappa": 0.19999999999999996,
  "num_classes": 2,
  "num_samples": 10,
  "per_class_report": {
    "0": {
      "precision": 0.6,
      "recall": 0.6,
      "f1-score": 0.6,
      "support": 5.0
    },
    "1": {
      "precision": 0.6,
      "recall": 0.6,
      "f1-score": 0.6,
      "support": 5.0
    },
    "accuracy": 0.6,
    "macro avg": {
      "precision": 0.6,
      "recall": 0.6,
      "f1-score": 0.6,
      "support": 10.0
    },
    "weighted avg": {
      "precision": 0.6,
      "recall": 0.6,
      "f1-score": 0.6,
      "support": 10.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.00417223635401638,
  "batch_size": 64,
  "weight_decay": 0.001096083850763953,
  "optimizer": "sgd",
  "scheduler": "step",
  "epochs": 22,
  "momentum": 0.8968651039441777,
  "scheduler_step_size": 8,
  "scheduler_gamma": 0.49896548822763365
}
```

**Metrics:**
```json
{
  "accuracy": 0.8,
  "f1": 0.8,
  "f1_macro": 0.8,
  "f1_weighted": 0.8,
  "precision": 0.8,
  "recall": 0.8,
  "balanced_accuracy": 0.8,
  "roc_auc": 0.78,
  "mcc": 0.6,
  "cohen_kappa": 0.6,
  "num_classes": 2,
  "num_samples": 10,
  "per_class_report": {
    "0": {
      "precision": 0.8,
      "recall": 0.8,
      "f1-score": 0.8,
      "support": 5.0
    },
    "1": {
      "precision": 0.8,
      "recall": 0.8,
      "f1-score": 0.8,
      "support": 5.0
    },
    "accuracy": 0.8,
    "macro avg": {
      "precision": 0.8,
      "recall": 0.8,
      "f1-score": 0.8,
      "support": 10.0
    },
    "weighted avg": {
      "precision": 0.8,
      "recall": 0.8,
      "f1-score": 0.8,
      "support": 10.0
    }
  }
}
```

---

## Dataset: `digen10_8322`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.57,
  "f1": 0.36764705882352944,
  "f1_macro": 0.5209447415329769,
  "f1_weighted": 0.5209447415329769,
  "precision": 0.6944444444444444,
  "recall": 0.25,
  "balanced_accuracy": 0.5700000000000001,
  "roc_auc": 0.5687,
  "mcc": 0.18220272220337375,
  "cohen_kappa": 0.14,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5426829268292683,
      "recall": 0.89,
      "f1-score": 0.6742424242424242,
      "support": 100.0
    },
    "1": {
      "precision": 0.6944444444444444,
      "recall": 0.25,
      "f1-score": 0.36764705882352944,
      "support": 100.0
    },
    "accuracy": 0.57,
    "macro avg": {
      "precision": 0.6185636856368564,
      "recall": 0.5700000000000001,
      "f1-score": 0.5209447415329769,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.6185636856368564,
      "recall": 0.57,
      "f1-score": 0.5209447415329769,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.81,
  "f1": 0.8061224489795918,
  "f1_macro": 0.8099239695878351,
  "f1_weighted": 0.8099239695878352,
  "precision": 0.8229166666666666,
  "recall": 0.79,
  "balanced_accuracy": 0.81,
  "roc_auc": 0.9011,
  "mcc": 0.6204965959947126,
  "cohen_kappa": 0.62,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.7980769230769231,
      "recall": 0.83,
      "f1-score": 0.8137254901960784,
      "support": 100.0
    },
    "1": {
      "precision": 0.8229166666666666,
      "recall": 0.79,
      "f1-score": 0.8061224489795918,
      "support": 100.0
    },
    "accuracy": 0.81,
    "macro avg": {
      "precision": 0.8104967948717949,
      "recall": 0.81,
      "f1-score": 0.8099239695878351,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.8104967948717947,
      "recall": 0.81,
      "f1-score": 0.8099239695878352,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.007018307965385619,
  "batch_size": 128,
  "weight_decay": 2.60891460335668e-05,
  "optimizer": "adam",
  "scheduler": "exponential",
  "epochs": 23,
  "scheduler_gamma": 0.9851749548564822
}
```

**Metrics:**
```json
{
  "accuracy": 0.475,
  "f1": 0.5643153526970954,
  "f1_macro": 0.45196899710326466,
  "f1_weighted": 0.4519689971032647,
  "precision": 0.48226950354609927,
  "recall": 0.68,
  "balanced_accuracy": 0.47500000000000003,
  "roc_auc": 0.47180000000000005,
  "mcc": -0.05481942074202942,
  "cohen_kappa": -0.050000000000000044,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.4576271186440678,
      "recall": 0.27,
      "f1-score": 0.33962264150943394,
      "support": 100.0
    },
    "1": {
      "precision": 0.48226950354609927,
      "recall": 0.68,
      "f1-score": 0.5643153526970954,
      "support": 100.0
    },
    "accuracy": 0.475,
    "macro avg": {
      "precision": 0.4699483110950835,
      "recall": 0.47500000000000003,
      "f1-score": 0.45196899710326466,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.46994831109508356,
      "recall": 0.475,
      "f1-score": 0.4519689971032647,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.007320279943063365,
  "batch_size": 64,
  "weight_decay": 0.00018466568213334008,
  "optimizer": "adam",
  "scheduler": "exponential",
  "epochs": 33,
  "scheduler_gamma": 0.9885301439779657
}
```

**Metrics:**
```json
{
  "accuracy": 0.5,
  "f1": 0.038461538461538464,
  "f1_macro": 0.3503118503118503,
  "f1_weighted": 0.35031185031185025,
  "precision": 0.5,
  "recall": 0.02,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.5,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5,
      "recall": 0.98,
      "f1-score": 0.6621621621621622,
      "support": 100.0
    },
    "1": {
      "precision": 0.5,
      "recall": 0.02,
      "f1-score": 0.038461538461538464,
      "support": 100.0
    },
    "accuracy": 0.5,
    "macro avg": {
      "precision": 0.5,
      "recall": 0.5,
      "f1-score": 0.3503118503118503,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5,
      "recall": 0.5,
      "f1-score": 0.35031185031185025,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen11_7270`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.515,
  "f1": 0.5488372093023256,
  "f1_macro": 0.5122564424890006,
  "f1_weighted": 0.5122564424890007,
  "precision": 0.5130434782608696,
  "recall": 0.59,
  "balanced_accuracy": 0.515,
  "roc_auc": 0.5345,
  "mcc": 0.03034330424545042,
  "cohen_kappa": 0.030000000000000027,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5176470588235295,
      "recall": 0.44,
      "f1-score": 0.4756756756756757,
      "support": 100.0
    },
    "1": {
      "precision": 0.5130434782608696,
      "recall": 0.59,
      "f1-score": 0.5488372093023256,
      "support": 100.0
    },
    "accuracy": 0.515,
    "macro avg": {
      "precision": 0.5153452685421995,
      "recall": 0.515,
      "f1-score": 0.5122564424890006,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5153452685421995,
      "recall": 0.515,
      "f1-score": 0.5122564424890007,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.71,
  "f1": 0.7156862745098039,
  "f1_macro": 0.7098839535814325,
  "f1_weighted": 0.7098839535814326,
  "precision": 0.7019230769230769,
  "recall": 0.73,
  "balanced_accuracy": 0.71,
  "roc_auc": 0.7877,
  "mcc": 0.4203364037383537,
  "cohen_kappa": 0.42000000000000004,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.71875,
      "recall": 0.69,
      "f1-score": 0.7040816326530612,
      "support": 100.0
    },
    "1": {
      "precision": 0.7019230769230769,
      "recall": 0.73,
      "f1-score": 0.7156862745098039,
      "support": 100.0
    },
    "accuracy": 0.71,
    "macro avg": {
      "precision": 0.7103365384615384,
      "recall": 0.71,
      "f1-score": 0.7098839535814325,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.7103365384615384,
      "recall": 0.71,
      "f1-score": 0.7098839535814326,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.0034634992372185123,
  "batch_size": 128,
  "weight_decay": 2.704287701386257e-05,
  "optimizer": "sgd",
  "scheduler": "none",
  "epochs": 27,
  "momentum": 0.87933421406805
}
```

**Metrics:**
```json
{
  "accuracy": 0.52,
  "f1": 0.5,
  "f1_macro": 0.5192307692307692,
  "f1_weighted": 0.5192307692307692,
  "precision": 0.5217391304347826,
  "recall": 0.48,
  "balanced_accuracy": 0.52,
  "roc_auc": 0.51745,
  "mcc": 0.0401286176952564,
  "cohen_kappa": 0.040000000000000036,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5185185185185185,
      "recall": 0.56,
      "f1-score": 0.5384615384615384,
      "support": 100.0
    },
    "1": {
      "precision": 0.5217391304347826,
      "recall": 0.48,
      "f1-score": 0.5,
      "support": 100.0
    },
    "accuracy": 0.52,
    "macro avg": {
      "precision": 0.5201288244766505,
      "recall": 0.52,
      "f1-score": 0.5192307692307692,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5201288244766505,
      "recall": 0.52,
      "f1-score": 0.5192307692307692,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.004878609727520485,
  "batch_size": 32,
  "weight_decay": 0.0008538170847219266,
  "optimizer": "adam",
  "scheduler": "step",
  "epochs": 36,
  "scheduler_step_size": 14,
  "scheduler_gamma": 0.8618082957829611
}
```

**Metrics:**
```json
{
  "accuracy": 0.5,
  "f1": 0.0,
  "f1_macro": 0.3333333333333333,
  "f1_weighted": 0.33333333333333326,
  "precision": 0.0,
  "recall": 0.0,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.5,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5,
      "recall": 1.0,
      "f1-score": 0.6666666666666666,
      "support": 100.0
    },
    "1": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 100.0
    },
    "accuracy": 0.5,
    "macro avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.3333333333333333,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.33333333333333326,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen12_8322`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.545,
  "f1": 0.43478260869565216,
  "f1_macro": 0.5270147353101692,
  "f1_weighted": 0.5270147353101692,
  "precision": 0.5737704918032787,
  "recall": 0.35,
  "balanced_accuracy": 0.5449999999999999,
  "roc_auc": 0.5248,
  "mcc": 0.09773951773486139,
  "cohen_kappa": 0.08999999999999997,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5323741007194245,
      "recall": 0.74,
      "f1-score": 0.6192468619246861,
      "support": 100.0
    },
    "1": {
      "precision": 0.5737704918032787,
      "recall": 0.35,
      "f1-score": 0.43478260869565216,
      "support": 100.0
    },
    "accuracy": 0.545,
    "macro avg": {
      "precision": 0.5530722962613516,
      "recall": 0.5449999999999999,
      "f1-score": 0.5270147353101692,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5530722962613516,
      "recall": 0.545,
      "f1-score": 0.5270147353101692,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.68,
  "f1": 0.6981132075471698,
  "f1_macro": 0.6788438378161381,
  "f1_weighted": 0.6788438378161381,
  "precision": 0.6607142857142857,
  "recall": 0.74,
  "balanced_accuracy": 0.6799999999999999,
  "roc_auc": 0.7636000000000001,
  "mcc": 0.36262033381142117,
  "cohen_kappa": 0.36,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.7045454545454546,
      "recall": 0.62,
      "f1-score": 0.6595744680851063,
      "support": 100.0
    },
    "1": {
      "precision": 0.6607142857142857,
      "recall": 0.74,
      "f1-score": 0.6981132075471698,
      "support": 100.0
    },
    "accuracy": 0.68,
    "macro avg": {
      "precision": 0.6826298701298701,
      "recall": 0.6799999999999999,
      "f1-score": 0.6788438378161381,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.6826298701298701,
      "recall": 0.68,
      "f1-score": 0.6788438378161381,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.0053034867066309305,
  "batch_size": 128,
  "weight_decay": 0.00012938063567039966,
  "optimizer": "adam",
  "scheduler": "cosine",
  "epochs": 36,
  "scheduler_t_max": 26
}
```

**Metrics:**
```json
{
  "accuracy": 0.51,
  "f1": 0.3194444444444444,
  "f1_macro": 0.4683159722222222,
  "f1_weighted": 0.4683159722222222,
  "precision": 0.5227272727272727,
  "recall": 0.23,
  "balanced_accuracy": 0.51,
  "roc_auc": 0.51415,
  "mcc": 0.02414022747926338,
  "cohen_kappa": 0.020000000000000018,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5064102564102564,
      "recall": 0.79,
      "f1-score": 0.6171875,
      "support": 100.0
    },
    "1": {
      "precision": 0.5227272727272727,
      "recall": 0.23,
      "f1-score": 0.3194444444444444,
      "support": 100.0
    },
    "accuracy": 0.51,
    "macro avg": {
      "precision": 0.5145687645687645,
      "recall": 0.51,
      "f1-score": 0.4683159722222222,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5145687645687645,
      "recall": 0.51,
      "f1-score": 0.4683159722222222,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.0019433894476321523,
  "batch_size": 128,
  "weight_decay": 6.483264941272445e-05,
  "optimizer": "adam",
  "scheduler": "exponential",
  "epochs": 37,
  "scheduler_gamma": 0.9503131490815689
}
```

**Metrics:**
```json
{
  "accuracy": 0.555,
  "f1": 0.46706586826347307,
  "f1_macro": 0.5425458096682172,
  "f1_weighted": 0.5425458096682172,
  "precision": 0.582089552238806,
  "recall": 0.39,
  "balanced_accuracy": 0.5549999999999999,
  "roc_auc": 0.5572,
  "mcc": 0.11652777748983194,
  "cohen_kappa": 0.10999999999999999,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5413533834586466,
      "recall": 0.72,
      "f1-score": 0.6180257510729614,
      "support": 100.0
    },
    "1": {
      "precision": 0.582089552238806,
      "recall": 0.39,
      "f1-score": 0.46706586826347307,
      "support": 100.0
    },
    "accuracy": 0.555,
    "macro avg": {
      "precision": 0.5617214678487263,
      "recall": 0.5549999999999999,
      "f1-score": 0.5425458096682172,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5617214678487263,
      "recall": 0.555,
      "f1-score": 0.5425458096682172,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen13_769`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.595,
  "f1": 0.6523605150214592,
  "f1_macro": 0.5836652874508493,
  "f1_weighted": 0.5836652874508493,
  "precision": 0.5714285714285714,
  "recall": 0.76,
  "balanced_accuracy": 0.595,
  "roc_auc": 0.6638000000000001,
  "mcc": 0.20127525202789154,
  "cohen_kappa": 0.18999999999999995,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.6417910447761194,
      "recall": 0.43,
      "f1-score": 0.5149700598802395,
      "support": 100.0
    },
    "1": {
      "precision": 0.5714285714285714,
      "recall": 0.76,
      "f1-score": 0.6523605150214592,
      "support": 100.0
    },
    "accuracy": 0.595,
    "macro avg": {
      "precision": 0.6066098081023454,
      "recall": 0.595,
      "f1-score": 0.5836652874508493,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.6066098081023454,
      "recall": 0.595,
      "f1-score": 0.5836652874508493,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.815,
  "f1": 0.8310502283105022,
  "f1_macro": 0.8133151694038698,
  "f1_weighted": 0.8133151694038699,
  "precision": 0.7647058823529411,
  "recall": 0.91,
  "balanced_accuracy": 0.815,
  "roc_auc": 0.9367,
  "mcc": 0.641688947919748,
  "cohen_kappa": 0.63,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.8888888888888888,
      "recall": 0.72,
      "f1-score": 0.7955801104972375,
      "support": 100.0
    },
    "1": {
      "precision": 0.7647058823529411,
      "recall": 0.91,
      "f1-score": 0.8310502283105022,
      "support": 100.0
    },
    "accuracy": 0.815,
    "macro avg": {
      "precision": 0.826797385620915,
      "recall": 0.815,
      "f1-score": 0.8133151694038698,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.826797385620915,
      "recall": 0.815,
      "f1-score": 0.8133151694038699,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.0016868459188467764,
  "batch_size": 64,
  "weight_decay": 0.00160904518948303,
  "optimizer": "adam",
  "scheduler": "cosine",
  "epochs": 39,
  "scheduler_t_max": 28
}
```

**Metrics:**
```json
{
  "accuracy": 0.55,
  "f1": 0.5754716981132075,
  "f1_macro": 0.5483741469289443,
  "f1_weighted": 0.5483741469289443,
  "precision": 0.5446428571428571,
  "recall": 0.61,
  "balanced_accuracy": 0.55,
  "roc_auc": 0.5541,
  "mcc": 0.10072787050317254,
  "cohen_kappa": 0.09999999999999998,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5568181818181818,
      "recall": 0.49,
      "f1-score": 0.5212765957446809,
      "support": 100.0
    },
    "1": {
      "precision": 0.5446428571428571,
      "recall": 0.61,
      "f1-score": 0.5754716981132075,
      "support": 100.0
    },
    "accuracy": 0.55,
    "macro avg": {
      "precision": 0.5507305194805194,
      "recall": 0.55,
      "f1-score": 0.5483741469289443,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5507305194805194,
      "recall": 0.55,
      "f1-score": 0.5483741469289443,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.00010820831043437573,
  "batch_size": 64,
  "weight_decay": 0.0009205301775483412,
  "optimizer": "adam",
  "scheduler": "step",
  "epochs": 44,
  "scheduler_step_size": 15,
  "scheduler_gamma": 0.8073540618870338
}
```

**Metrics:**
```json
{
  "accuracy": 0.485,
  "f1": 0.46632124352331605,
  "f1_macro": 0.48436835123025707,
  "f1_weighted": 0.48436835123025707,
  "precision": 0.4838709677419355,
  "recall": 0.45,
  "balanced_accuracy": 0.485,
  "roc_auc": 0.4876,
  "mcc": -0.03007377122020926,
  "cohen_kappa": -0.030000000000000027,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.48598130841121495,
      "recall": 0.52,
      "f1-score": 0.5024154589371981,
      "support": 100.0
    },
    "1": {
      "precision": 0.4838709677419355,
      "recall": 0.45,
      "f1-score": 0.46632124352331605,
      "support": 100.0
    },
    "accuracy": 0.485,
    "macro avg": {
      "precision": 0.48492613807657525,
      "recall": 0.485,
      "f1-score": 0.48436835123025707,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.4849261380765752,
      "recall": 0.485,
      "f1-score": 0.48436835123025707,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen14_769`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.55,
  "f1": 0.5212765957446809,
  "f1_macro": 0.5483741469289443,
  "f1_weighted": 0.5483741469289443,
  "precision": 0.5568181818181818,
  "recall": 0.49,
  "balanced_accuracy": 0.55,
  "roc_auc": 0.5667,
  "mcc": 0.10072787050317254,
  "cohen_kappa": 0.09999999999999998,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5446428571428571,
      "recall": 0.61,
      "f1-score": 0.5754716981132075,
      "support": 100.0
    },
    "1": {
      "precision": 0.5568181818181818,
      "recall": 0.49,
      "f1-score": 0.5212765957446809,
      "support": 100.0
    },
    "accuracy": 0.55,
    "macro avg": {
      "precision": 0.5507305194805194,
      "recall": 0.55,
      "f1-score": 0.5483741469289443,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5507305194805194,
      "recall": 0.55,
      "f1-score": 0.5483741469289443,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.645,
  "f1": 0.6321243523316062,
  "f1_macro": 0.6445645916247403,
  "f1_weighted": 0.6445645916247403,
  "precision": 0.6559139784946236,
  "recall": 0.61,
  "balanced_accuracy": 0.645,
  "roc_auc": 0.6796000000000001,
  "mcc": 0.2907131217953562,
  "cohen_kappa": 0.29000000000000004,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.6355140186915887,
      "recall": 0.68,
      "f1-score": 0.6570048309178744,
      "support": 100.0
    },
    "1": {
      "precision": 0.6559139784946236,
      "recall": 0.61,
      "f1-score": 0.6321243523316062,
      "support": 100.0
    },
    "accuracy": 0.645,
    "macro avg": {
      "precision": 0.6457139985931062,
      "recall": 0.645,
      "f1-score": 0.6445645916247403,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.6457139985931062,
      "recall": 0.645,
      "f1-score": 0.6445645916247403,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.005823669601998489,
  "batch_size": 32,
  "weight_decay": 0.001353842165699621,
  "optimizer": "adam",
  "scheduler": "none",
  "epochs": 24
}
```

**Metrics:**
```json
{
  "accuracy": 0.51,
  "f1": 0.26865671641791045,
  "f1_macro": 0.45011783189316573,
  "f1_weighted": 0.45011783189316573,
  "precision": 0.5294117647058824,
  "recall": 0.18,
  "balanced_accuracy": 0.51,
  "roc_auc": 0.5575,
  "mcc": 0.026621743403250103,
  "cohen_kappa": 0.020000000000000018,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5060240963855421,
      "recall": 0.84,
      "f1-score": 0.631578947368421,
      "support": 100.0
    },
    "1": {
      "precision": 0.5294117647058824,
      "recall": 0.18,
      "f1-score": 0.26865671641791045,
      "support": 100.0
    },
    "accuracy": 0.51,
    "macro avg": {
      "precision": 0.5177179305457122,
      "recall": 0.51,
      "f1-score": 0.45011783189316573,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5177179305457122,
      "recall": 0.51,
      "f1-score": 0.45011783189316573,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.0017995302024314413,
  "batch_size": 128,
  "weight_decay": 4.606798672508228e-05,
  "optimizer": "sgd",
  "scheduler": "cosine",
  "epochs": 30,
  "momentum": 0.8225694234793116,
  "scheduler_t_max": 18
}
```

**Metrics:**
```json
{
  "accuracy": 0.525,
  "f1": 0.5539906103286385,
  "f1_macro": 0.5229846099771535,
  "f1_weighted": 0.5229846099771536,
  "precision": 0.5221238938053098,
  "recall": 0.59,
  "balanced_accuracy": 0.525,
  "roc_auc": 0.5035000000000001,
  "mcc": 0.0504279317388775,
  "cohen_kappa": 0.050000000000000044,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5287356321839081,
      "recall": 0.46,
      "f1-score": 0.4919786096256685,
      "support": 100.0
    },
    "1": {
      "precision": 0.5221238938053098,
      "recall": 0.59,
      "f1-score": 0.5539906103286385,
      "support": 100.0
    },
    "accuracy": 0.525,
    "macro avg": {
      "precision": 0.525429762994609,
      "recall": 0.525,
      "f1-score": 0.5229846099771535,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5254297629946089,
      "recall": 0.525,
      "f1-score": 0.5229846099771536,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen15_5311`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.58,
  "f1": 0.5434782608695652,
  "f1_macro": 0.5772946859903382,
  "f1_weighted": 0.5772946859903382,
  "precision": 0.5952380952380952,
  "recall": 0.5,
  "balanced_accuracy": 0.5800000000000001,
  "roc_auc": 0.6084999999999999,
  "mcc": 0.16208817969462155,
  "cohen_kappa": 0.16000000000000003,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5689655172413793,
      "recall": 0.66,
      "f1-score": 0.6111111111111112,
      "support": 100.0
    },
    "1": {
      "precision": 0.5952380952380952,
      "recall": 0.5,
      "f1-score": 0.5434782608695652,
      "support": 100.0
    },
    "accuracy": 0.58,
    "macro avg": {
      "precision": 0.5821018062397373,
      "recall": 0.5800000000000001,
      "f1-score": 0.5772946859903382,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5821018062397373,
      "recall": 0.58,
      "f1-score": 0.5772946859903382,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.765,
  "f1": 0.7638190954773869,
  "f1_macro": 0.7649941248531213,
  "f1_weighted": 0.7649941248531212,
  "precision": 0.7676767676767676,
  "recall": 0.76,
  "balanced_accuracy": 0.765,
  "roc_auc": 0.8547000000000001,
  "mcc": 0.5300265019876657,
  "cohen_kappa": 0.53,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.7623762376237624,
      "recall": 0.77,
      "f1-score": 0.7661691542288557,
      "support": 100.0
    },
    "1": {
      "precision": 0.7676767676767676,
      "recall": 0.76,
      "f1-score": 0.7638190954773869,
      "support": 100.0
    },
    "accuracy": 0.765,
    "macro avg": {
      "precision": 0.765026502650265,
      "recall": 0.765,
      "f1-score": 0.7649941248531213,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.7650265026502651,
      "recall": 0.765,
      "f1-score": 0.7649941248531212,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.007671720383370134,
  "batch_size": 32,
  "weight_decay": 0.0046279747797099155,
  "optimizer": "adam",
  "scheduler": "step",
  "epochs": 31,
  "scheduler_step_size": 11,
  "scheduler_gamma": 0.6403904549461596
}
```

**Metrics:**
```json
{
  "accuracy": 0.5,
  "f1": 0.0,
  "f1_macro": 0.3333333333333333,
  "f1_weighted": 0.33333333333333326,
  "precision": 0.0,
  "recall": 0.0,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.5,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5,
      "recall": 1.0,
      "f1-score": 0.6666666666666666,
      "support": 100.0
    },
    "1": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 100.0
    },
    "accuracy": 0.5,
    "macro avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.3333333333333333,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.33333333333333326,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 1.2678625217516229e-05,
  "batch_size": 64,
  "weight_decay": 0.009309181741623956,
  "optimizer": "adam",
  "scheduler": "cosine",
  "epochs": 47,
  "scheduler_t_max": 17
}
```

**Metrics:**
```json
{
  "accuracy": 0.49,
  "f1": 0.6506849315068494,
  "f1_macro": 0.35312024353120247,
  "f1_weighted": 0.3531202435312024,
  "precision": 0.4947916666666667,
  "recall": 0.95,
  "balanced_accuracy": 0.49,
  "roc_auc": 0.49,
  "mcc": -0.05103103630798288,
  "cohen_kappa": -0.020000000000000018,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.375,
      "recall": 0.03,
      "f1-score": 0.05555555555555555,
      "support": 100.0
    },
    "1": {
      "precision": 0.4947916666666667,
      "recall": 0.95,
      "f1-score": 0.6506849315068494,
      "support": 100.0
    },
    "accuracy": 0.49,
    "macro avg": {
      "precision": 0.43489583333333337,
      "recall": 0.49,
      "f1-score": 0.35312024353120247,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.43489583333333337,
      "recall": 0.49,
      "f1-score": 0.3531202435312024,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen16_5390`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.59,
  "f1": 0.5444444444444444,
  "f1_macro": 0.5858585858585859,
  "f1_weighted": 0.5858585858585859,
  "precision": 0.6125,
  "recall": 0.49,
  "balanced_accuracy": 0.59,
  "roc_auc": 0.5959,
  "mcc": 0.18371173070873836,
  "cohen_kappa": 0.18000000000000005,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.575,
      "recall": 0.69,
      "f1-score": 0.6272727272727273,
      "support": 100.0
    },
    "1": {
      "precision": 0.6125,
      "recall": 0.49,
      "f1-score": 0.5444444444444444,
      "support": 100.0
    },
    "accuracy": 0.59,
    "macro avg": {
      "precision": 0.59375,
      "recall": 0.59,
      "f1-score": 0.5858585858585859,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.59375,
      "recall": 0.59,
      "f1-score": 0.5858585858585859,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.82,
  "f1": 0.8301886792452831,
  "f1_macro": 0.8193496587715776,
  "f1_weighted": 0.8193496587715776,
  "precision": 0.7857142857142857,
  "recall": 0.88,
  "balanced_accuracy": 0.8200000000000001,
  "roc_auc": 0.9299,
  "mcc": 0.6446583712203042,
  "cohen_kappa": 0.64,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.8636363636363636,
      "recall": 0.76,
      "f1-score": 0.8085106382978723,
      "support": 100.0
    },
    "1": {
      "precision": 0.7857142857142857,
      "recall": 0.88,
      "f1-score": 0.8301886792452831,
      "support": 100.0
    },
    "accuracy": 0.82,
    "macro avg": {
      "precision": 0.8246753246753247,
      "recall": 0.8200000000000001,
      "f1-score": 0.8193496587715776,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.8246753246753247,
      "recall": 0.82,
      "f1-score": 0.8193496587715776,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 3.057806840051588e-05,
  "batch_size": 128,
  "weight_decay": 8.1022010640811e-05,
  "optimizer": "adam",
  "scheduler": "cosine",
  "epochs": 20,
  "scheduler_t_max": 19
}
```

**Metrics:**
```json
{
  "accuracy": 0.505,
  "f1": 0.5074626865671642,
  "f1_macro": 0.5049876246906173,
  "f1_weighted": 0.5049876246906172,
  "precision": 0.504950495049505,
  "recall": 0.51,
  "balanced_accuracy": 0.505,
  "roc_auc": 0.4955,
  "mcc": 0.010000500037503125,
  "cohen_kappa": 0.010000000000000009,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5050505050505051,
      "recall": 0.5,
      "f1-score": 0.5025125628140703,
      "support": 100.0
    },
    "1": {
      "precision": 0.504950495049505,
      "recall": 0.51,
      "f1-score": 0.5074626865671642,
      "support": 100.0
    },
    "accuracy": 0.505,
    "macro avg": {
      "precision": 0.5050005000500051,
      "recall": 0.505,
      "f1-score": 0.5049876246906173,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5050005000500051,
      "recall": 0.505,
      "f1-score": 0.5049876246906172,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.0028598662288333017,
  "batch_size": 64,
  "weight_decay": 0.0008417236189758302,
  "optimizer": "adam",
  "scheduler": "exponential",
  "epochs": 36,
  "scheduler_gamma": 0.9662694783657075
}
```

**Metrics:**
```json
{
  "accuracy": 0.54,
  "f1": 0.5106382978723404,
  "f1_macro": 0.5383380168606985,
  "f1_weighted": 0.5383380168606985,
  "precision": 0.5454545454545454,
  "recall": 0.48,
  "balanced_accuracy": 0.54,
  "roc_auc": 0.5374,
  "mcc": 0.08058229640253803,
  "cohen_kappa": 0.07999999999999996,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5357142857142857,
      "recall": 0.6,
      "f1-score": 0.5660377358490566,
      "support": 100.0
    },
    "1": {
      "precision": 0.5454545454545454,
      "recall": 0.48,
      "f1-score": 0.5106382978723404,
      "support": 100.0
    },
    "accuracy": 0.54,
    "macro avg": {
      "precision": 0.5405844155844155,
      "recall": 0.54,
      "f1-score": 0.5383380168606985,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5405844155844156,
      "recall": 0.54,
      "f1-score": 0.5383380168606985,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen17_6949`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.475,
  "f1": 0.304635761589404,
  "f1_macro": 0.44147450730072607,
  "f1_weighted": 0.44147450730072607,
  "precision": 0.45098039215686275,
  "recall": 0.23,
  "balanced_accuracy": 0.475,
  "roc_auc": 0.4842000000000001,
  "mcc": -0.057357707125141495,
  "cohen_kappa": -0.050000000000000044,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.48322147651006714,
      "recall": 0.72,
      "f1-score": 0.5783132530120482,
      "support": 100.0
    },
    "1": {
      "precision": 0.45098039215686275,
      "recall": 0.23,
      "f1-score": 0.304635761589404,
      "support": 100.0
    },
    "accuracy": 0.475,
    "macro avg": {
      "precision": 0.46710093433346495,
      "recall": 0.475,
      "f1-score": 0.44147450730072607,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.467100934333465,
      "recall": 0.475,
      "f1-score": 0.44147450730072607,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.61,
  "f1": 0.6776859504132231,
  "f1_macro": 0.5920075321686369,
  "f1_weighted": 0.5920075321686369,
  "precision": 0.5774647887323944,
  "recall": 0.82,
  "balanced_accuracy": 0.61,
  "roc_auc": 0.69215,
  "mcc": 0.24241780349669298,
  "cohen_kappa": 0.21999999999999997,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.6896551724137931,
      "recall": 0.4,
      "f1-score": 0.5063291139240507,
      "support": 100.0
    },
    "1": {
      "precision": 0.5774647887323944,
      "recall": 0.82,
      "f1-score": 0.6776859504132231,
      "support": 100.0
    },
    "accuracy": 0.61,
    "macro avg": {
      "precision": 0.6335599805730938,
      "recall": 0.61,
      "f1-score": 0.5920075321686369,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.6335599805730938,
      "recall": 0.61,
      "f1-score": 0.5920075321686369,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.009643728097287153,
  "batch_size": 128,
  "weight_decay": 0.0007489257769878014,
  "optimizer": "adam",
  "scheduler": "exponential",
  "epochs": 46,
  "scheduler_gamma": 0.9689684058455093
}
```

**Metrics:**
```json
{
  "accuracy": 0.5,
  "f1": 0.0,
  "f1_macro": 0.3333333333333333,
  "f1_weighted": 0.33333333333333326,
  "precision": 0.0,
  "recall": 0.0,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.5,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5,
      "recall": 1.0,
      "f1-score": 0.6666666666666666,
      "support": 100.0
    },
    "1": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 100.0
    },
    "accuracy": 0.5,
    "macro avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.3333333333333333,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.33333333333333326,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.0008038281830754283,
  "batch_size": 32,
  "weight_decay": 9.385608143757267e-05,
  "optimizer": "adam",
  "scheduler": "exponential",
  "epochs": 33,
  "scheduler_gamma": 0.9716280105518036
}
```

**Metrics:**
```json
{
  "accuracy": 0.545,
  "f1": 0.3546099290780142,
  "f1_macro": 0.5016292888633315,
  "f1_weighted": 0.5016292888633315,
  "precision": 0.6097560975609756,
  "recall": 0.25,
  "balanced_accuracy": 0.5449999999999999,
  "roc_auc": 0.545,
  "mcc": 0.1114684645619942,
  "cohen_kappa": 0.08999999999999997,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5283018867924528,
      "recall": 0.84,
      "f1-score": 0.6486486486486487,
      "support": 100.0
    },
    "1": {
      "precision": 0.6097560975609756,
      "recall": 0.25,
      "f1-score": 0.3546099290780142,
      "support": 100.0
    },
    "accuracy": 0.545,
    "macro avg": {
      "precision": 0.5690289921767142,
      "recall": 0.5449999999999999,
      "f1-score": 0.5016292888633315,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5690289921767142,
      "recall": 0.545,
      "f1-score": 0.5016292888633315,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen18_5578`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.605,
  "f1": 0.6183574879227053,
  "f1_macro": 0.6045155315261195,
  "f1_weighted": 0.6045155315261195,
  "precision": 0.5981308411214953,
  "recall": 0.64,
  "balanced_accuracy": 0.605,
  "roc_auc": 0.6247,
  "mcc": 0.2105163985414648,
  "cohen_kappa": 0.20999999999999996,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.6129032258064516,
      "recall": 0.57,
      "f1-score": 0.5906735751295337,
      "support": 100.0
    },
    "1": {
      "precision": 0.5981308411214953,
      "recall": 0.64,
      "f1-score": 0.6183574879227053,
      "support": 100.0
    },
    "accuracy": 0.605,
    "macro avg": {
      "precision": 0.6055170334639735,
      "recall": 0.605,
      "f1-score": 0.6045155315261195,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.6055170334639735,
      "recall": 0.605,
      "f1-score": 0.6045155315261195,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.6,
  "f1": 0.5348837209302325,
  "f1_macro": 0.5920032639738881,
  "f1_weighted": 0.5920032639738882,
  "precision": 0.6388888888888888,
  "recall": 0.46,
  "balanced_accuracy": 0.6,
  "roc_auc": 0.6227,
  "mcc": 0.20833333333333334,
  "cohen_kappa": 0.19999999999999996,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.578125,
      "recall": 0.74,
      "f1-score": 0.6491228070175439,
      "support": 100.0
    },
    "1": {
      "precision": 0.6388888888888888,
      "recall": 0.46,
      "f1-score": 0.5348837209302325,
      "support": 100.0
    },
    "accuracy": 0.6,
    "macro avg": {
      "precision": 0.6085069444444444,
      "recall": 0.6,
      "f1-score": 0.5920032639738881,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.6085069444444444,
      "recall": 0.6,
      "f1-score": 0.5920032639738882,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.0008213296658239122,
  "batch_size": 64,
  "weight_decay": 0.0002430714697065806,
  "optimizer": "adam",
  "scheduler": "exponential",
  "epochs": 27,
  "scheduler_gamma": 0.9701922707326482
}
```

**Metrics:**
```json
{
  "accuracy": 0.51,
  "f1": 0.19672131147540983,
  "f1_macro": 0.4221016629319495,
  "f1_weighted": 0.4221016629319495,
  "precision": 0.5454545454545454,
  "recall": 0.12,
  "balanced_accuracy": 0.51,
  "roc_auc": 0.51025,
  "mcc": 0.03196013860502966,
  "cohen_kappa": 0.020000000000000018,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5056179775280899,
      "recall": 0.9,
      "f1-score": 0.6474820143884892,
      "support": 100.0
    },
    "1": {
      "precision": 0.5454545454545454,
      "recall": 0.12,
      "f1-score": 0.19672131147540983,
      "support": 100.0
    },
    "accuracy": 0.51,
    "macro avg": {
      "precision": 0.5255362614913177,
      "recall": 0.51,
      "f1-score": 0.4221016629319495,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5255362614913177,
      "recall": 0.51,
      "f1-score": 0.4221016629319495,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.0027834022376348666,
  "batch_size": 128,
  "weight_decay": 0.00019086378700027513,
  "optimizer": "adam",
  "scheduler": "none",
  "epochs": 29
}
```

**Metrics:**
```json
{
  "accuracy": 0.53,
  "f1": 0.25396825396825395,
  "f1_macro": 0.4554512802687985,
  "f1_weighted": 0.4554512802687985,
  "precision": 0.6153846153846154,
  "recall": 0.16,
  "balanced_accuracy": 0.53,
  "roc_auc": 0.5345,
  "mcc": 0.08920515501750788,
  "cohen_kappa": 0.06000000000000005,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5172413793103449,
      "recall": 0.9,
      "f1-score": 0.656934306569343,
      "support": 100.0
    },
    "1": {
      "precision": 0.6153846153846154,
      "recall": 0.16,
      "f1-score": 0.25396825396825395,
      "support": 100.0
    },
    "accuracy": 0.53,
    "macro avg": {
      "precision": 0.5663129973474801,
      "recall": 0.53,
      "f1-score": 0.4554512802687985,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5663129973474801,
      "recall": 0.53,
      "f1-score": 0.4554512802687985,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen19_7270`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.53,
  "f1": 0.3561643835616438,
  "f1_macro": 0.49304282170208175,
  "f1_weighted": 0.49304282170208175,
  "precision": 0.5652173913043478,
  "recall": 0.26,
  "balanced_accuracy": 0.53,
  "roc_auc": 0.5503,
  "mcc": 0.07128726847826362,
  "cohen_kappa": 0.06000000000000005,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5194805194805194,
      "recall": 0.8,
      "f1-score": 0.6299212598425197,
      "support": 100.0
    },
    "1": {
      "precision": 0.5652173913043478,
      "recall": 0.26,
      "f1-score": 0.3561643835616438,
      "support": 100.0
    },
    "accuracy": 0.53,
    "macro avg": {
      "precision": 0.5423489553924337,
      "recall": 0.53,
      "f1-score": 0.49304282170208175,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5423489553924337,
      "recall": 0.53,
      "f1-score": 0.49304282170208175,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.85,
  "f1": 0.85,
  "f1_macro": 0.85,
  "f1_weighted": 0.85,
  "precision": 0.85,
  "recall": 0.85,
  "balanced_accuracy": 0.85,
  "roc_auc": 0.9271,
  "mcc": 0.7,
  "cohen_kappa": 0.7,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.85,
      "recall": 0.85,
      "f1-score": 0.85,
      "support": 100.0
    },
    "1": {
      "precision": 0.85,
      "recall": 0.85,
      "f1-score": 0.85,
      "support": 100.0
    },
    "accuracy": 0.85,
    "macro avg": {
      "precision": 0.85,
      "recall": 0.85,
      "f1-score": 0.85,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.85,
      "recall": 0.85,
      "f1-score": 0.85,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.004143137578990764,
  "batch_size": 128,
  "weight_decay": 7.274500246879845e-05,
  "optimizer": "adam",
  "scheduler": "none",
  "epochs": 42
}
```

**Metrics:**
```json
{
  "accuracy": 0.48,
  "f1": 0.44680851063829785,
  "f1_macro": 0.4781212364512244,
  "f1_weighted": 0.4781212364512244,
  "precision": 0.4772727272727273,
  "recall": 0.42,
  "balanced_accuracy": 0.48,
  "roc_auc": 0.47315,
  "mcc": -0.040291148201269014,
  "cohen_kappa": -0.040000000000000036,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.48214285714285715,
      "recall": 0.54,
      "f1-score": 0.5094339622641509,
      "support": 100.0
    },
    "1": {
      "precision": 0.4772727272727273,
      "recall": 0.42,
      "f1-score": 0.44680851063829785,
      "support": 100.0
    },
    "accuracy": 0.48,
    "macro avg": {
      "precision": 0.47970779220779225,
      "recall": 0.48,
      "f1-score": 0.4781212364512244,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.4797077922077922,
      "recall": 0.48,
      "f1-score": 0.4781212364512244,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 2.1872607772170024e-05,
  "batch_size": 64,
  "weight_decay": 0.0009289365035413335,
  "optimizer": "sgd",
  "scheduler": "exponential",
  "epochs": 22,
  "momentum": 0.8276874948367743,
  "scheduler_gamma": 0.9856468038656864
}
```

**Metrics:**
```json
{
  "accuracy": 0.55,
  "f1": 0.5945945945945946,
  "f1_macro": 0.5444883085332524,
  "f1_weighted": 0.5444883085332524,
  "precision": 0.5409836065573771,
  "recall": 0.66,
  "balanced_accuracy": 0.55,
  "roc_auc": 0.5483,
  "mcc": 0.1025115460130912,
  "cohen_kappa": 0.09999999999999998,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5641025641025641,
      "recall": 0.44,
      "f1-score": 0.4943820224719101,
      "support": 100.0
    },
    "1": {
      "precision": 0.5409836065573771,
      "recall": 0.66,
      "f1-score": 0.5945945945945946,
      "support": 100.0
    },
    "accuracy": 0.55,
    "macro avg": {
      "precision": 0.5525430853299707,
      "recall": 0.55,
      "f1-score": 0.5444883085332524,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5525430853299705,
      "recall": 0.55,
      "f1-score": 0.5444883085332524,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen1_6265`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.875,
  "f1": 0.8663101604278075,
  "f1_macro": 0.8744696342045141,
  "f1_weighted": 0.8744696342045141,
  "precision": 0.9310344827586207,
  "recall": 0.81,
  "balanced_accuracy": 0.875,
  "roc_auc": 0.9371,
  "mcc": 0.7564189760831626,
  "cohen_kappa": 0.75,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.831858407079646,
      "recall": 0.94,
      "f1-score": 0.8826291079812206,
      "support": 100.0
    },
    "1": {
      "precision": 0.9310344827586207,
      "recall": 0.81,
      "f1-score": 0.8663101604278075,
      "support": 100.0
    },
    "accuracy": 0.875,
    "macro avg": {
      "precision": 0.8814464449191333,
      "recall": 0.875,
      "f1-score": 0.8744696342045141,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.8814464449191334,
      "recall": 0.875,
      "f1-score": 0.8744696342045141,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.915,
  "f1": 0.9109947643979057,
  "f1_macro": 0.9148275257396228,
  "f1_weighted": 0.9148275257396228,
  "precision": 0.9560439560439561,
  "recall": 0.87,
  "balanced_accuracy": 0.915,
  "roc_auc": 0.9619999999999999,
  "mcc": 0.8333820599391434,
  "cohen_kappa": 0.83,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.8807339449541285,
      "recall": 0.96,
      "f1-score": 0.9186602870813397,
      "support": 100.0
    },
    "1": {
      "precision": 0.9560439560439561,
      "recall": 0.87,
      "f1-score": 0.9109947643979057,
      "support": 100.0
    },
    "accuracy": 0.915,
    "macro avg": {
      "precision": 0.9183889504990423,
      "recall": 0.915,
      "f1-score": 0.9148275257396228,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.9183889504990423,
      "recall": 0.915,
      "f1-score": 0.9148275257396228,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.008552172318237232,
  "batch_size": 32,
  "weight_decay": 0.0013305971673770354,
  "optimizer": "adam",
  "scheduler": "cosine",
  "epochs": 33,
  "scheduler_t_max": 12
}
```

**Metrics:**
```json
{
  "accuracy": 0.5,
  "f1": 0.6666666666666666,
  "f1_macro": 0.3333333333333333,
  "f1_weighted": 0.33333333333333326,
  "precision": 0.5,
  "recall": 1.0,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.5,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 100.0
    },
    "1": {
      "precision": 0.5,
      "recall": 1.0,
      "f1-score": 0.6666666666666666,
      "support": 100.0
    },
    "accuracy": 0.5,
    "macro avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.3333333333333333,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.33333333333333326,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 8.836128588715876e-05,
  "batch_size": 128,
  "weight_decay": 0.0004452180190726698,
  "optimizer": "sgd",
  "scheduler": "none",
  "epochs": 41,
  "momentum": 0.8240115233618003
}
```

**Metrics:**
```json
{
  "accuracy": 0.51,
  "f1": 0.24615384615384617,
  "f1_macro": 0.4415954415954416,
  "f1_weighted": 0.4415954415954416,
  "precision": 0.5333333333333333,
  "recall": 0.16,
  "balanced_accuracy": 0.51,
  "roc_auc": 0.5519000000000001,
  "mcc": 0.028005601680560196,
  "cohen_kappa": 0.020000000000000018,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5058823529411764,
      "recall": 0.86,
      "f1-score": 0.6370370370370371,
      "support": 100.0
    },
    "1": {
      "precision": 0.5333333333333333,
      "recall": 0.16,
      "f1-score": 0.24615384615384617,
      "support": 100.0
    },
    "accuracy": 0.51,
    "macro avg": {
      "precision": 0.5196078431372548,
      "recall": 0.51,
      "f1-score": 0.4415954415954416,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5196078431372549,
      "recall": 0.51,
      "f1-score": 0.4415954415954416,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen20_5191`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.55,
  "f1": 0.5408163265306123,
  "f1_macro": 0.5498199279711885,
  "f1_weighted": 0.5498199279711885,
  "precision": 0.5520833333333334,
  "recall": 0.53,
  "balanced_accuracy": 0.55,
  "roc_auc": 0.5998000000000001,
  "mcc": 0.10008009612817945,
  "cohen_kappa": 0.09999999999999998,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5480769230769231,
      "recall": 0.57,
      "f1-score": 0.5588235294117647,
      "support": 100.0
    },
    "1": {
      "precision": 0.5520833333333334,
      "recall": 0.53,
      "f1-score": 0.5408163265306123,
      "support": 100.0
    },
    "accuracy": 0.55,
    "macro avg": {
      "precision": 0.5500801282051282,
      "recall": 0.55,
      "f1-score": 0.5498199279711885,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5500801282051282,
      "recall": 0.55,
      "f1-score": 0.5498199279711885,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.76,
  "f1": 0.7241379310344828,
  "f1_macro": 0.7558742752517547,
  "f1_weighted": 0.7558742752517545,
  "precision": 0.8513513513513513,
  "recall": 0.63,
  "balanced_accuracy": 0.76,
  "roc_auc": 0.8778,
  "mcc": 0.5385204638677067,
  "cohen_kappa": 0.52,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.7063492063492064,
      "recall": 0.89,
      "f1-score": 0.7876106194690266,
      "support": 100.0
    },
    "1": {
      "precision": 0.8513513513513513,
      "recall": 0.63,
      "f1-score": 0.7241379310344828,
      "support": 100.0
    },
    "accuracy": 0.76,
    "macro avg": {
      "precision": 0.7788502788502789,
      "recall": 0.76,
      "f1-score": 0.7558742752517547,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.7788502788502788,
      "recall": 0.76,
      "f1-score": 0.7558742752517545,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 1.914141997980014e-05,
  "batch_size": 64,
  "weight_decay": 7.691097722196137e-05,
  "optimizer": "adam",
  "scheduler": "exponential",
  "epochs": 26,
  "scheduler_gamma": 0.9563924559864929
}
```

**Metrics:**
```json
{
  "accuracy": 0.425,
  "f1": 0.3915343915343915,
  "f1_macro": 0.4232553474259635,
  "f1_weighted": 0.42325534742596355,
  "precision": 0.4157303370786517,
  "recall": 0.37,
  "balanced_accuracy": 0.425,
  "roc_auc": 0.4187,
  "mcc": -0.15091581949331018,
  "cohen_kappa": -0.1499999999999999,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.43243243243243246,
      "recall": 0.48,
      "f1-score": 0.4549763033175355,
      "support": 100.0
    },
    "1": {
      "precision": 0.4157303370786517,
      "recall": 0.37,
      "f1-score": 0.3915343915343915,
      "support": 100.0
    },
    "accuracy": 0.425,
    "macro avg": {
      "precision": 0.4240813847555421,
      "recall": 0.425,
      "f1-score": 0.4232553474259635,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.4240813847555421,
      "recall": 0.425,
      "f1-score": 0.42325534742596355,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.008267172532294201,
  "batch_size": 32,
  "weight_decay": 0.004348682167519488,
  "optimizer": "adam",
  "scheduler": "none",
  "epochs": 28
}
```

**Metrics:**
```json
{
  "accuracy": 0.515,
  "f1": 0.6339622641509434,
  "f1_macro": 0.45772187281621246,
  "f1_weighted": 0.4577218728162124,
  "precision": 0.509090909090909,
  "recall": 0.84,
  "balanced_accuracy": 0.515,
  "roc_auc": 0.5021,
  "mcc": 0.039477101697586135,
  "cohen_kappa": 0.030000000000000027,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5428571428571428,
      "recall": 0.19,
      "f1-score": 0.2814814814814815,
      "support": 100.0
    },
    "1": {
      "precision": 0.509090909090909,
      "recall": 0.84,
      "f1-score": 0.6339622641509434,
      "support": 100.0
    },
    "accuracy": 0.515,
    "macro avg": {
      "precision": 0.525974025974026,
      "recall": 0.515,
      "f1-score": 0.45772187281621246,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5259740259740259,
      "recall": 0.515,
      "f1-score": 0.4577218728162124,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen21_6265`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.45,
  "f1": 0.46078431372549017,
  "f1_macro": 0.4497799119647859,
  "f1_weighted": 0.4497799119647859,
  "precision": 0.4519230769230769,
  "recall": 0.47,
  "balanced_accuracy": 0.44999999999999996,
  "roc_auc": 0.5075000000000001,
  "mcc": -0.10008009612817945,
  "cohen_kappa": -0.10000000000000009,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.4479166666666667,
      "recall": 0.43,
      "f1-score": 0.4387755102040816,
      "support": 100.0
    },
    "1": {
      "precision": 0.4519230769230769,
      "recall": 0.47,
      "f1-score": 0.46078431372549017,
      "support": 100.0
    },
    "accuracy": 0.45,
    "macro avg": {
      "precision": 0.4499198717948718,
      "recall": 0.44999999999999996,
      "f1-score": 0.4497799119647859,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.4499198717948718,
      "recall": 0.45,
      "f1-score": 0.4497799119647859,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.585,
  "f1": 0.6556016597510373,
  "f1_macro": 0.5667945405673426,
  "f1_weighted": 0.5667945405673426,
  "precision": 0.5602836879432624,
  "recall": 0.79,
  "balanced_accuracy": 0.585,
  "roc_auc": 0.6431,
  "mcc": 0.18638603052290004,
  "cohen_kappa": 0.17000000000000004,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.6440677966101694,
      "recall": 0.38,
      "f1-score": 0.4779874213836478,
      "support": 100.0
    },
    "1": {
      "precision": 0.5602836879432624,
      "recall": 0.79,
      "f1-score": 0.6556016597510373,
      "support": 100.0
    },
    "accuracy": 0.585,
    "macro avg": {
      "precision": 0.602175742276716,
      "recall": 0.585,
      "f1-score": 0.5667945405673426,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.6021757422767159,
      "recall": 0.585,
      "f1-score": 0.5667945405673426,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.0008158461602315837,
  "batch_size": 64,
  "weight_decay": 1.7908119199334897e-05,
  "optimizer": "adam",
  "scheduler": "cosine",
  "epochs": 45,
  "scheduler_t_max": 38
}
```

**Metrics:**
```json
{
  "accuracy": 0.52,
  "f1": 0.2727272727272727,
  "f1_macro": 0.45725915875169604,
  "f1_weighted": 0.45725915875169604,
  "precision": 0.5625,
  "recall": 0.18,
  "balanced_accuracy": 0.52,
  "roc_auc": 0.5169999999999999,
  "mcc": 0.0545544725589981,
  "cohen_kappa": 0.040000000000000036,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5119047619047619,
      "recall": 0.86,
      "f1-score": 0.6417910447761194,
      "support": 100.0
    },
    "1": {
      "precision": 0.5625,
      "recall": 0.18,
      "f1-score": 0.2727272727272727,
      "support": 100.0
    },
    "accuracy": 0.52,
    "macro avg": {
      "precision": 0.5372023809523809,
      "recall": 0.52,
      "f1-score": 0.45725915875169604,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5372023809523809,
      "recall": 0.52,
      "f1-score": 0.45725915875169604,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 4.129881756118029e-05,
  "batch_size": 64,
  "weight_decay": 0.000261315502917518,
  "optimizer": "adam",
  "scheduler": "exponential",
  "epochs": 50,
  "scheduler_gamma": 0.989989658016094
}
```

**Metrics:**
```json
{
  "accuracy": 0.505,
  "f1": 0.36129032258064514,
  "f1_macro": 0.478604344963792,
  "f1_weighted": 0.478604344963792,
  "precision": 0.509090909090909,
  "recall": 0.28,
  "balanced_accuracy": 0.505,
  "roc_auc": 0.505,
  "mcc": 0.011197850219117086,
  "cohen_kappa": 0.010000000000000009,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.503448275862069,
      "recall": 0.73,
      "f1-score": 0.5959183673469388,
      "support": 100.0
    },
    "1": {
      "precision": 0.509090909090909,
      "recall": 0.28,
      "f1-score": 0.36129032258064514,
      "support": 100.0
    },
    "accuracy": 0.505,
    "macro avg": {
      "precision": 0.5062695924764891,
      "recall": 0.505,
      "f1-score": 0.478604344963792,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.506269592476489,
      "recall": 0.505,
      "f1-score": 0.478604344963792,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen22_2433`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.505,
  "f1": 0.5925925925925926,
  "f1_macro": 0.4810096720924746,
  "f1_weighted": 0.4810096720924746,
  "precision": 0.5034965034965035,
  "recall": 0.72,
  "balanced_accuracy": 0.505,
  "roc_auc": 0.5689,
  "mcc": 0.011076296005915018,
  "cohen_kappa": 0.010000000000000009,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5087719298245614,
      "recall": 0.29,
      "f1-score": 0.36942675159235666,
      "support": 100.0
    },
    "1": {
      "precision": 0.5034965034965035,
      "recall": 0.72,
      "f1-score": 0.5925925925925926,
      "support": 100.0
    },
    "accuracy": 0.505,
    "macro avg": {
      "precision": 0.5061342166605325,
      "recall": 0.505,
      "f1-score": 0.4810096720924746,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5061342166605325,
      "recall": 0.505,
      "f1-score": 0.4810096720924746,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.86,
  "f1": 0.875,
  "f1_macro": 0.8579545454545454,
  "f1_weighted": 0.8579545454545454,
  "precision": 0.7903225806451613,
  "recall": 0.98,
  "balanced_accuracy": 0.86,
  "roc_auc": 0.9511000000000001,
  "mcc": 0.7416770790872962,
  "cohen_kappa": 0.72,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.9736842105263158,
      "recall": 0.74,
      "f1-score": 0.8409090909090909,
      "support": 100.0
    },
    "1": {
      "precision": 0.7903225806451613,
      "recall": 0.98,
      "f1-score": 0.875,
      "support": 100.0
    },
    "accuracy": 0.86,
    "macro avg": {
      "precision": 0.8820033955857385,
      "recall": 0.86,
      "f1-score": 0.8579545454545454,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.8820033955857386,
      "recall": 0.86,
      "f1-score": 0.8579545454545454,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.005952821449364302,
  "batch_size": 32,
  "weight_decay": 0.0006011469696870622,
  "optimizer": "adam",
  "scheduler": "none",
  "epochs": 22
}
```

**Metrics:**
```json
{
  "accuracy": 0.5,
  "f1": 0.0,
  "f1_macro": 0.3333333333333333,
  "f1_weighted": 0.33333333333333326,
  "precision": 0.0,
  "recall": 0.0,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.5,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5,
      "recall": 1.0,
      "f1-score": 0.6666666666666666,
      "support": 100.0
    },
    "1": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 100.0
    },
    "accuracy": 0.5,
    "macro avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.3333333333333333,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.33333333333333326,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.004993948849590196,
  "batch_size": 32,
  "weight_decay": 0.005584673441934934,
  "optimizer": "adam",
  "scheduler": "none",
  "epochs": 47
}
```

**Metrics:**
```json
{
  "accuracy": 0.525,
  "f1": 0.6735395189003437,
  "f1_macro": 0.4009899429364104,
  "f1_weighted": 0.4009899429364103,
  "precision": 0.5130890052356021,
  "recall": 0.98,
  "balanced_accuracy": 0.525,
  "roc_auc": 0.5072,
  "mcc": 0.12059576754873692,
  "cohen_kappa": 0.050000000000000044,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.7777777777777778,
      "recall": 0.07,
      "f1-score": 0.12844036697247707,
      "support": 100.0
    },
    "1": {
      "precision": 0.5130890052356021,
      "recall": 0.98,
      "f1-score": 0.6735395189003437,
      "support": 100.0
    },
    "accuracy": 0.525,
    "macro avg": {
      "precision": 0.6454333915066899,
      "recall": 0.525,
      "f1-score": 0.4009899429364104,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.64543339150669,
      "recall": 0.525,
      "f1-score": 0.4009899429364103,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen23_5191`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.52,
  "f1": 0.5102040816326531,
  "f1_macro": 0.5198079231692677,
  "f1_weighted": 0.5198079231692677,
  "precision": 0.5208333333333334,
  "recall": 0.5,
  "balanced_accuracy": 0.52,
  "roc_auc": 0.5159,
  "mcc": 0.04003203845127178,
  "cohen_kappa": 0.040000000000000036,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5192307692307693,
      "recall": 0.54,
      "f1-score": 0.5294117647058824,
      "support": 100.0
    },
    "1": {
      "precision": 0.5208333333333334,
      "recall": 0.5,
      "f1-score": 0.5102040816326531,
      "support": 100.0
    },
    "accuracy": 0.52,
    "macro avg": {
      "precision": 0.5200320512820513,
      "recall": 0.52,
      "f1-score": 0.5198079231692677,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5200320512820513,
      "recall": 0.52,
      "f1-score": 0.5198079231692677,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.49,
  "f1": 0.6530612244897959,
  "f1_macro": 0.3453985367731998,
  "f1_weighted": 0.3453985367731998,
  "precision": 0.4948453608247423,
  "recall": 0.96,
  "balanced_accuracy": 0.49,
  "roc_auc": 0.46630000000000005,
  "mcc": -0.05862103817605492,
  "cohen_kappa": -0.020000000000000018,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.3333333333333333,
      "recall": 0.02,
      "f1-score": 0.03773584905660377,
      "support": 100.0
    },
    "1": {
      "precision": 0.4948453608247423,
      "recall": 0.96,
      "f1-score": 0.6530612244897959,
      "support": 100.0
    },
    "accuracy": 0.49,
    "macro avg": {
      "precision": 0.4140893470790378,
      "recall": 0.49,
      "f1-score": 0.3453985367731998,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.41408934707903783,
      "recall": 0.49,
      "f1-score": 0.3453985367731998,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.007414816278267339,
  "batch_size": 32,
  "weight_decay": 0.004136795076029481,
  "optimizer": "adam",
  "scheduler": "step",
  "epochs": 21,
  "scheduler_step_size": 5,
  "scheduler_gamma": 0.43969241297171446
}
```

**Metrics:**
```json
{
  "accuracy": 0.5,
  "f1": 0.6666666666666666,
  "f1_macro": 0.3333333333333333,
  "f1_weighted": 0.33333333333333326,
  "precision": 0.5,
  "recall": 1.0,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.5,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 100.0
    },
    "1": {
      "precision": 0.5,
      "recall": 1.0,
      "f1-score": 0.6666666666666666,
      "support": 100.0
    },
    "accuracy": 0.5,
    "macro avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.3333333333333333,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.33333333333333326,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.004969105117535124,
  "batch_size": 128,
  "weight_decay": 0.0003471438139455212,
  "optimizer": "adam",
  "scheduler": "step",
  "epochs": 45,
  "scheduler_step_size": 5,
  "scheduler_gamma": 0.8661197684282873
}
```

**Metrics:**
```json
{
  "accuracy": 0.5,
  "f1": 0.6621621621621622,
  "f1_macro": 0.3503118503118503,
  "f1_weighted": 0.35031185031185025,
  "precision": 0.5,
  "recall": 0.98,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.5,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5,
      "recall": 0.02,
      "f1-score": 0.038461538461538464,
      "support": 100.0
    },
    "1": {
      "precision": 0.5,
      "recall": 0.98,
      "f1-score": 0.6621621621621622,
      "support": 100.0
    },
    "accuracy": 0.5,
    "macro avg": {
      "precision": 0.5,
      "recall": 0.5,
      "f1-score": 0.3503118503118503,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5,
      "recall": 0.5,
      "f1-score": 0.35031185031185025,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen24_2433`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.55,
  "f1": 0.48863636363636365,
  "f1_macro": 0.5434253246753247,
  "f1_weighted": 0.5434253246753247,
  "precision": 0.5657894736842105,
  "recall": 0.43,
  "balanced_accuracy": 0.55,
  "roc_auc": 0.6209000000000001,
  "mcc": 0.10301070542879114,
  "cohen_kappa": 0.09999999999999998,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5403225806451613,
      "recall": 0.67,
      "f1-score": 0.5982142857142857,
      "support": 100.0
    },
    "1": {
      "precision": 0.5657894736842105,
      "recall": 0.43,
      "f1-score": 0.48863636363636365,
      "support": 100.0
    },
    "accuracy": 0.55,
    "macro avg": {
      "precision": 0.5530560271646858,
      "recall": 0.55,
      "f1-score": 0.5434253246753247,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5530560271646858,
      "recall": 0.55,
      "f1-score": 0.5434253246753247,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.585,
  "f1": 0.5608465608465608,
  "f1_macro": 0.5837408159683042,
  "f1_weighted": 0.5837408159683041,
  "precision": 0.5955056179775281,
  "recall": 0.53,
  "balanced_accuracy": 0.585,
  "roc_auc": 0.5961000000000001,
  "mcc": 0.17103792875908488,
  "cohen_kappa": 0.17000000000000004,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5765765765765766,
      "recall": 0.64,
      "f1-score": 0.6066350710900474,
      "support": 100.0
    },
    "1": {
      "precision": 0.5955056179775281,
      "recall": 0.53,
      "f1-score": 0.5608465608465608,
      "support": 100.0
    },
    "accuracy": 0.585,
    "macro avg": {
      "precision": 0.5860410972770523,
      "recall": 0.585,
      "f1-score": 0.5837408159683042,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5860410972770523,
      "recall": 0.585,
      "f1-score": 0.5837408159683041,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.00037216884474090405,
  "batch_size": 32,
  "weight_decay": 0.004768993278108469,
  "optimizer": "adam",
  "scheduler": "exponential",
  "epochs": 49,
  "scheduler_gamma": 0.9704461283127532
}
```

**Metrics:**
```json
{
  "accuracy": 0.455,
  "f1": 0.32298136645962733,
  "f1_macro": 0.43345721042646634,
  "f1_weighted": 0.4334572104264664,
  "precision": 0.4262295081967213,
  "recall": 0.26,
  "balanced_accuracy": 0.455,
  "roc_auc": 0.455,
  "mcc": -0.09773951773486139,
  "cohen_kappa": -0.09000000000000008,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.4676258992805755,
      "recall": 0.65,
      "f1-score": 0.5439330543933054,
      "support": 100.0
    },
    "1": {
      "precision": 0.4262295081967213,
      "recall": 0.26,
      "f1-score": 0.32298136645962733,
      "support": 100.0
    },
    "accuracy": 0.455,
    "macro avg": {
      "precision": 0.44692770373864843,
      "recall": 0.455,
      "f1-score": 0.43345721042646634,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.4469277037386484,
      "recall": 0.455,
      "f1-score": 0.4334572104264664,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.005977556157363177,
  "batch_size": 32,
  "weight_decay": 0.006135673845182855,
  "optimizer": "adam",
  "scheduler": "none",
  "epochs": 38
}
```

**Metrics:**
```json
{
  "accuracy": 0.5,
  "f1": 0.6666666666666666,
  "f1_macro": 0.3333333333333333,
  "f1_weighted": 0.33333333333333326,
  "precision": 0.5,
  "recall": 1.0,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.5,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 100.0
    },
    "1": {
      "precision": 0.5,
      "recall": 1.0,
      "f1-score": 0.6666666666666666,
      "support": 100.0
    },
    "accuracy": 0.5,
    "macro avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.3333333333333333,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.33333333333333326,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen25_2433`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.585,
  "f1": 0.5911330049261084,
  "f1_macro": 0.5849066039858968,
  "f1_weighted": 0.5849066039858968,
  "precision": 0.5825242718446602,
  "recall": 0.6,
  "balanced_accuracy": 0.585,
  "roc_auc": 0.5569,
  "mcc": 0.17007655167625865,
  "cohen_kappa": 0.17000000000000004,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5876288659793815,
      "recall": 0.57,
      "f1-score": 0.5786802030456852,
      "support": 100.0
    },
    "1": {
      "precision": 0.5825242718446602,
      "recall": 0.6,
      "f1-score": 0.5911330049261084,
      "support": 100.0
    },
    "accuracy": 0.585,
    "macro avg": {
      "precision": 0.5850765689120208,
      "recall": 0.585,
      "f1-score": 0.5849066039858968,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5850765689120209,
      "recall": 0.585,
      "f1-score": 0.5849066039858968,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.64,
  "f1": 0.6435643564356436,
  "f1_macro": 0.63996399639964,
  "f1_weighted": 0.63996399639964,
  "precision": 0.6372549019607843,
  "recall": 0.65,
  "balanced_accuracy": 0.64,
  "roc_auc": 0.688,
  "mcc": 0.280056016805602,
  "cohen_kappa": 0.28,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.6428571428571429,
      "recall": 0.63,
      "f1-score": 0.6363636363636364,
      "support": 100.0
    },
    "1": {
      "precision": 0.6372549019607843,
      "recall": 0.65,
      "f1-score": 0.6435643564356436,
      "support": 100.0
    },
    "accuracy": 0.64,
    "macro avg": {
      "precision": 0.6400560224089635,
      "recall": 0.64,
      "f1-score": 0.63996399639964,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.6400560224089636,
      "recall": 0.64,
      "f1-score": 0.63996399639964,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.00529886697701392,
  "batch_size": 64,
  "weight_decay": 1.6878360709792015e-05,
  "optimizer": "adam",
  "scheduler": "exponential",
  "epochs": 42,
  "scheduler_gamma": 0.9759538240721806
}
```

**Metrics:**
```json
{
  "accuracy": 0.485,
  "f1": 0.5690376569037657,
  "f1_macro": 0.46464305205436734,
  "f1_weighted": 0.4646430520543673,
  "precision": 0.4892086330935252,
  "recall": 0.68,
  "balanced_accuracy": 0.485,
  "roc_auc": 0.4850000000000001,
  "mcc": -0.03257983924495379,
  "cohen_kappa": -0.030000000000000027,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.47540983606557374,
      "recall": 0.29,
      "f1-score": 0.36024844720496896,
      "support": 100.0
    },
    "1": {
      "precision": 0.4892086330935252,
      "recall": 0.68,
      "f1-score": 0.5690376569037657,
      "support": 100.0
    },
    "accuracy": 0.485,
    "macro avg": {
      "precision": 0.48230923457954944,
      "recall": 0.485,
      "f1-score": 0.46464305205436734,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.48230923457954944,
      "recall": 0.485,
      "f1-score": 0.4646430520543673,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.007603780640249192,
  "batch_size": 32,
  "weight_decay": 0.001547779065652449,
  "optimizer": "adam",
  "scheduler": "step",
  "epochs": 33,
  "scheduler_step_size": 11,
  "scheduler_gamma": 0.5451890543194704
}
```

**Metrics:**
```json
{
  "accuracy": 0.5,
  "f1": 0.6666666666666666,
  "f1_macro": 0.3333333333333333,
  "f1_weighted": 0.33333333333333326,
  "precision": 0.5,
  "recall": 1.0,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.5,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 100.0
    },
    "1": {
      "precision": 0.5,
      "recall": 1.0,
      "f1-score": 0.6666666666666666,
      "support": 100.0
    },
    "accuracy": 0.5,
    "macro avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.3333333333333333,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.33333333333333326,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen26_7270`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.565,
  "f1": 0.5756097560975609,
  "f1_macro": 0.5647279549718573,
  "f1_weighted": 0.5647279549718575,
  "precision": 0.5619047619047619,
  "recall": 0.59,
  "balanced_accuracy": 0.565,
  "roc_auc": 0.5378,
  "mcc": 0.1301628053236573,
  "cohen_kappa": 0.13,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5684210526315789,
      "recall": 0.54,
      "f1-score": 0.5538461538461539,
      "support": 100.0
    },
    "1": {
      "precision": 0.5619047619047619,
      "recall": 0.59,
      "f1-score": 0.5756097560975609,
      "support": 100.0
    },
    "accuracy": 0.565,
    "macro avg": {
      "precision": 0.5651629072681704,
      "recall": 0.565,
      "f1-score": 0.5647279549718573,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5651629072681704,
      "recall": 0.565,
      "f1-score": 0.5647279549718575,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.695,
  "f1": 0.6514285714285715,
  "f1_macro": 0.6901587301587302,
  "f1_weighted": 0.6901587301587301,
  "precision": 0.76,
  "recall": 0.57,
  "balanced_accuracy": 0.695,
  "roc_auc": 0.7556000000000002,
  "mcc": 0.4027902680055714,
  "cohen_kappa": 0.39,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.656,
      "recall": 0.82,
      "f1-score": 0.7288888888888889,
      "support": 100.0
    },
    "1": {
      "precision": 0.76,
      "recall": 0.57,
      "f1-score": 0.6514285714285715,
      "support": 100.0
    },
    "accuracy": 0.695,
    "macro avg": {
      "precision": 0.708,
      "recall": 0.695,
      "f1-score": 0.6901587301587302,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.7080000000000001,
      "recall": 0.695,
      "f1-score": 0.6901587301587301,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.0005093907768217243,
  "batch_size": 32,
  "weight_decay": 1.0048072871875507e-05,
  "optimizer": "adam",
  "scheduler": "none",
  "epochs": 30
}
```

**Metrics:**
```json
{
  "accuracy": 0.525,
  "f1": 0.6360153256704981,
  "f1_macro": 0.4762810441302131,
  "f1_weighted": 0.4762810441302131,
  "precision": 0.515527950310559,
  "recall": 0.83,
  "balanced_accuracy": 0.525,
  "roc_auc": 0.5249999999999999,
  "mcc": 0.06309933217282221,
  "cohen_kappa": 0.050000000000000044,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5641025641025641,
      "recall": 0.22,
      "f1-score": 0.31654676258992803,
      "support": 100.0
    },
    "1": {
      "precision": 0.515527950310559,
      "recall": 0.83,
      "f1-score": 0.6360153256704981,
      "support": 100.0
    },
    "accuracy": 0.525,
    "macro avg": {
      "precision": 0.5398152572065615,
      "recall": 0.525,
      "f1-score": 0.4762810441302131,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5398152572065615,
      "recall": 0.525,
      "f1-score": 0.4762810441302131,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.00010846815628899961,
  "batch_size": 32,
  "weight_decay": 0.0038581788363267334,
  "optimizer": "sgd",
  "scheduler": "cosine",
  "epochs": 43,
  "momentum": 0.8267954380730993,
  "scheduler_t_max": 35
}
```

**Metrics:**
```json
{
  "accuracy": 0.49,
  "f1": 0.0,
  "f1_macro": 0.3288590604026846,
  "f1_weighted": 0.3288590604026846,
  "precision": 0.0,
  "recall": 0.0,
  "balanced_accuracy": 0.49,
  "roc_auc": 0.49,
  "mcc": -0.10050378152592121,
  "cohen_kappa": -0.020000000000000018,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.494949494949495,
      "recall": 0.98,
      "f1-score": 0.6577181208053692,
      "support": 100.0
    },
    "1": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 100.0
    },
    "accuracy": 0.49,
    "macro avg": {
      "precision": 0.2474747474747475,
      "recall": 0.49,
      "f1-score": 0.3288590604026846,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.2474747474747475,
      "recall": 0.49,
      "f1-score": 0.3288590604026846,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen27_860`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.51,
  "f1": 0.10909090909090909,
  "f1_macro": 0.38557993730407525,
  "f1_weighted": 0.38557993730407525,
  "precision": 0.6,
  "recall": 0.06,
  "balanced_accuracy": 0.51,
  "roc_auc": 0.5048,
  "mcc": 0.04588314677411235,
  "cohen_kappa": 0.020000000000000018,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5052631578947369,
      "recall": 0.96,
      "f1-score": 0.6620689655172414,
      "support": 100.0
    },
    "1": {
      "precision": 0.6,
      "recall": 0.06,
      "f1-score": 0.10909090909090909,
      "support": 100.0
    },
    "accuracy": 0.51,
    "macro avg": {
      "precision": 0.5526315789473684,
      "recall": 0.51,
      "f1-score": 0.38557993730407525,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5526315789473685,
      "recall": 0.51,
      "f1-score": 0.38557993730407525,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.53,
  "f1": 0.5523809523809524,
  "f1_macro": 0.5288220551378446,
  "f1_weighted": 0.5288220551378446,
  "precision": 0.5272727272727272,
  "recall": 0.58,
  "balanced_accuracy": 0.53,
  "roc_auc": 0.5582,
  "mcc": 0.06030226891555272,
  "cohen_kappa": 0.06000000000000005,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5333333333333333,
      "recall": 0.48,
      "f1-score": 0.5052631578947369,
      "support": 100.0
    },
    "1": {
      "precision": 0.5272727272727272,
      "recall": 0.58,
      "f1-score": 0.5523809523809524,
      "support": 100.0
    },
    "accuracy": 0.53,
    "macro avg": {
      "precision": 0.5303030303030303,
      "recall": 0.53,
      "f1-score": 0.5288220551378446,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5303030303030303,
      "recall": 0.53,
      "f1-score": 0.5288220551378446,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 8.120046630460172e-05,
  "batch_size": 64,
  "weight_decay": 0.00012566548602598005,
  "optimizer": "sgd",
  "scheduler": "exponential",
  "epochs": 37,
  "momentum": 0.915897946968016,
  "scheduler_gamma": 0.9627284227282321
}
```

**Metrics:**
```json
{
  "accuracy": 0.505,
  "f1": 0.47058823529411764,
  "f1_macro": 0.5028997514498756,
  "f1_weighted": 0.5028997514498756,
  "precision": 0.5057471264367817,
  "recall": 0.44,
  "balanced_accuracy": 0.505,
  "roc_auc": 0.5057,
  "mcc": 0.010085586347775502,
  "cohen_kappa": 0.010000000000000009,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.504424778761062,
      "recall": 0.57,
      "f1-score": 0.5352112676056338,
      "support": 100.0
    },
    "1": {
      "precision": 0.5057471264367817,
      "recall": 0.44,
      "f1-score": 0.47058823529411764,
      "support": 100.0
    },
    "accuracy": 0.505,
    "macro avg": {
      "precision": 0.5050859525989218,
      "recall": 0.505,
      "f1-score": 0.5028997514498756,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5050859525989219,
      "recall": 0.505,
      "f1-score": 0.5028997514498756,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 1.763121597370391e-05,
  "batch_size": 128,
  "weight_decay": 0.003714568863488223,
  "optimizer": "adam",
  "scheduler": "exponential",
  "epochs": 45,
  "scheduler_gamma": 0.9542978930364682
}
```

**Metrics:**
```json
{
  "accuracy": 0.46,
  "f1": 0.5263157894736842,
  "f1_macro": 0.44920440636474906,
  "f1_weighted": 0.44920440636474906,
  "precision": 0.46875,
  "recall": 0.6,
  "balanced_accuracy": 0.45999999999999996,
  "roc_auc": 0.4592,
  "mcc": -0.08333333333333333,
  "cohen_kappa": -0.08000000000000007,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.4444444444444444,
      "recall": 0.32,
      "f1-score": 0.37209302325581395,
      "support": 100.0
    },
    "1": {
      "precision": 0.46875,
      "recall": 0.6,
      "f1-score": 0.5263157894736842,
      "support": 100.0
    },
    "accuracy": 0.46,
    "macro avg": {
      "precision": 0.4565972222222222,
      "recall": 0.45999999999999996,
      "f1-score": 0.44920440636474906,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.4565972222222222,
      "recall": 0.46,
      "f1-score": 0.44920440636474906,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen28_769`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.56,
  "f1": 0.6206896551724138,
  "f1_macro": 0.548440065681445,
  "f1_weighted": 0.548440065681445,
  "precision": 0.5454545454545454,
  "recall": 0.72,
  "balanced_accuracy": 0.56,
  "roc_auc": 0.5817,
  "mcc": 0.1266600992762247,
  "cohen_kappa": 0.12,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5882352941176471,
      "recall": 0.4,
      "f1-score": 0.47619047619047616,
      "support": 100.0
    },
    "1": {
      "precision": 0.5454545454545454,
      "recall": 0.72,
      "f1-score": 0.6206896551724138,
      "support": 100.0
    },
    "accuracy": 0.56,
    "macro avg": {
      "precision": 0.5668449197860963,
      "recall": 0.56,
      "f1-score": 0.548440065681445,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5668449197860963,
      "recall": 0.56,
      "f1-score": 0.548440065681445,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.585,
  "f1": 0.5608465608465608,
  "f1_macro": 0.5837408159683042,
  "f1_weighted": 0.5837408159683041,
  "precision": 0.5955056179775281,
  "recall": 0.53,
  "balanced_accuracy": 0.585,
  "roc_auc": 0.5968,
  "mcc": 0.17103792875908488,
  "cohen_kappa": 0.17000000000000004,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5765765765765766,
      "recall": 0.64,
      "f1-score": 0.6066350710900474,
      "support": 100.0
    },
    "1": {
      "precision": 0.5955056179775281,
      "recall": 0.53,
      "f1-score": 0.5608465608465608,
      "support": 100.0
    },
    "accuracy": 0.585,
    "macro avg": {
      "precision": 0.5860410972770523,
      "recall": 0.585,
      "f1-score": 0.5837408159683042,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5860410972770523,
      "recall": 0.585,
      "f1-score": 0.5837408159683041,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.006548190073370541,
  "batch_size": 64,
  "weight_decay": 0.000533946252032412,
  "optimizer": "adam",
  "scheduler": "none",
  "epochs": 25
}
```

**Metrics:**
```json
{
  "accuracy": 0.5,
  "f1": 0.6666666666666666,
  "f1_macro": 0.3333333333333333,
  "f1_weighted": 0.33333333333333326,
  "precision": 0.5,
  "recall": 1.0,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.5,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 100.0
    },
    "1": {
      "precision": 0.5,
      "recall": 1.0,
      "f1-score": 0.6666666666666666,
      "support": 100.0
    },
    "accuracy": 0.5,
    "macro avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.3333333333333333,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.33333333333333326,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.004905723378020242,
  "batch_size": 32,
  "weight_decay": 0.0046156195984003055,
  "optimizer": "adam",
  "scheduler": "none",
  "epochs": 39
}
```

**Metrics:**
```json
{
  "accuracy": 0.5,
  "f1": 0.6666666666666666,
  "f1_macro": 0.3333333333333333,
  "f1_weighted": 0.33333333333333326,
  "precision": 0.5,
  "recall": 1.0,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.5,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 100.0
    },
    "1": {
      "precision": 0.5,
      "recall": 1.0,
      "f1-score": 0.6666666666666666,
      "support": 100.0
    },
    "accuracy": 0.5,
    "macro avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.3333333333333333,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.33333333333333326,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen29_8322`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.525,
  "f1": 0.6468401486988847,
  "f1_macro": 0.46082465450211413,
  "f1_weighted": 0.4608246545021142,
  "precision": 0.514792899408284,
  "recall": 0.87,
  "balanced_accuracy": 0.525,
  "roc_auc": 0.548,
  "mcc": 0.06907896231799035,
  "cohen_kappa": 0.050000000000000044,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5806451612903226,
      "recall": 0.18,
      "f1-score": 0.2748091603053435,
      "support": 100.0
    },
    "1": {
      "precision": 0.514792899408284,
      "recall": 0.87,
      "f1-score": 0.6468401486988847,
      "support": 100.0
    },
    "accuracy": 0.525,
    "macro avg": {
      "precision": 0.5477190303493034,
      "recall": 0.525,
      "f1-score": 0.46082465450211413,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5477190303493034,
      "recall": 0.525,
      "f1-score": 0.4608246545021142,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.595,
  "f1": 0.6367713004484304,
  "f1_macro": 0.5895720909021813,
  "f1_weighted": 0.5895720909021813,
  "precision": 0.5772357723577236,
  "recall": 0.71,
  "balanced_accuracy": 0.595,
  "roc_auc": 0.631,
  "mcc": 0.1952341035514183,
  "cohen_kappa": 0.18999999999999995,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.6233766233766234,
      "recall": 0.48,
      "f1-score": 0.5423728813559322,
      "support": 100.0
    },
    "1": {
      "precision": 0.5772357723577236,
      "recall": 0.71,
      "f1-score": 0.6367713004484304,
      "support": 100.0
    },
    "accuracy": 0.595,
    "macro avg": {
      "precision": 0.6003061978671735,
      "recall": 0.595,
      "f1-score": 0.5895720909021813,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.6003061978671735,
      "recall": 0.595,
      "f1-score": 0.5895720909021813,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 1.8146003725733405e-05,
  "batch_size": 64,
  "weight_decay": 0.0021669050737351953,
  "optimizer": "sgd",
  "scheduler": "exponential",
  "epochs": 36,
  "momentum": 0.8136382818975089,
  "scheduler_gamma": 0.9665909088884455
}
```

**Metrics:**
```json
{
  "accuracy": 0.54,
  "f1": 0.4457831325301205,
  "f1_macro": 0.5263103696838636,
  "f1_weighted": 0.5263103696838637,
  "precision": 0.5606060606060606,
  "recall": 0.37,
  "balanced_accuracy": 0.54,
  "roc_auc": 0.55065,
  "mcc": 0.08506788201182268,
  "cohen_kappa": 0.07999999999999996,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5298507462686567,
      "recall": 0.71,
      "f1-score": 0.6068376068376068,
      "support": 100.0
    },
    "1": {
      "precision": 0.5606060606060606,
      "recall": 0.37,
      "f1-score": 0.4457831325301205,
      "support": 100.0
    },
    "accuracy": 0.54,
    "macro avg": {
      "precision": 0.5452284034373587,
      "recall": 0.54,
      "f1-score": 0.5263103696838636,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5452284034373587,
      "recall": 0.54,
      "f1-score": 0.5263103696838637,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.0010917495198059346,
  "batch_size": 64,
  "weight_decay": 2.8376934625600546e-05,
  "optimizer": "adam",
  "scheduler": "exponential",
  "epochs": 29,
  "scheduler_gamma": 0.9798628320684055
}
```

**Metrics:**
```json
{
  "accuracy": 0.455,
  "f1": 0.4293193717277487,
  "f1_macro": 0.45389413562463987,
  "f1_weighted": 0.4538941356246399,
  "precision": 0.45054945054945056,
  "recall": 0.41,
  "balanced_accuracy": 0.45499999999999996,
  "roc_auc": 0.45749999999999996,
  "mcc": -0.09036672939099145,
  "cohen_kappa": -0.09000000000000008,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.45871559633027525,
      "recall": 0.5,
      "f1-score": 0.4784688995215311,
      "support": 100.0
    },
    "1": {
      "precision": 0.45054945054945056,
      "recall": 0.41,
      "f1-score": 0.4293193717277487,
      "support": 100.0
    },
    "accuracy": 0.455,
    "macro avg": {
      "precision": 0.45463252343986293,
      "recall": 0.45499999999999996,
      "f1-score": 0.45389413562463987,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.45463252343986293,
      "recall": 0.455,
      "f1-score": 0.4538941356246399,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen2_6949`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.545,
  "f1": 0.5603864734299517,
  "f1_macro": 0.5444419413781882,
  "f1_weighted": 0.5444419413781882,
  "precision": 0.5420560747663551,
  "recall": 0.58,
  "balanced_accuracy": 0.5449999999999999,
  "roc_auc": 0.5578000000000001,
  "mcc": 0.09022131366062779,
  "cohen_kappa": 0.08999999999999997,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5483870967741935,
      "recall": 0.51,
      "f1-score": 0.5284974093264249,
      "support": 100.0
    },
    "1": {
      "precision": 0.5420560747663551,
      "recall": 0.58,
      "f1-score": 0.5603864734299517,
      "support": 100.0
    },
    "accuracy": 0.545,
    "macro avg": {
      "precision": 0.5452215857702742,
      "recall": 0.5449999999999999,
      "f1-score": 0.5444419413781882,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5452215857702742,
      "recall": 0.545,
      "f1-score": 0.5444419413781882,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.54,
  "f1": 0.5818181818181818,
  "f1_macro": 0.5353535353535354,
  "f1_weighted": 0.5353535353535354,
  "precision": 0.5333333333333333,
  "recall": 0.64,
  "balanced_accuracy": 0.54,
  "roc_auc": 0.5593,
  "mcc": 0.08164965809277261,
  "cohen_kappa": 0.07999999999999996,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.55,
      "recall": 0.44,
      "f1-score": 0.4888888888888889,
      "support": 100.0
    },
    "1": {
      "precision": 0.5333333333333333,
      "recall": 0.64,
      "f1-score": 0.5818181818181818,
      "support": 100.0
    },
    "accuracy": 0.54,
    "macro avg": {
      "precision": 0.5416666666666667,
      "recall": 0.54,
      "f1-score": 0.5353535353535354,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5416666666666667,
      "recall": 0.54,
      "f1-score": 0.5353535353535354,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.0026561191173597306,
  "batch_size": 64,
  "weight_decay": 7.150254799876703e-05,
  "optimizer": "adam",
  "scheduler": "exponential",
  "epochs": 33,
  "scheduler_gamma": 0.9898629222986798
}
```

**Metrics:**
```json
{
  "accuracy": 0.495,
  "f1": 0.6622073578595318,
  "f1_macro": 0.3311036789297659,
  "f1_weighted": 0.3311036789297659,
  "precision": 0.49748743718592964,
  "recall": 0.99,
  "balanced_accuracy": 0.495,
  "roc_auc": 0.495,
  "mcc": -0.0708881205008336,
  "cohen_kappa": -0.010000000000000009,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 100.0
    },
    "1": {
      "precision": 0.49748743718592964,
      "recall": 0.99,
      "f1-score": 0.6622073578595318,
      "support": 100.0
    },
    "accuracy": 0.495,
    "macro avg": {
      "precision": 0.24874371859296482,
      "recall": 0.495,
      "f1-score": 0.3311036789297659,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.24874371859296482,
      "recall": 0.495,
      "f1-score": 0.3311036789297659,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.00650716419443199,
  "batch_size": 32,
  "weight_decay": 3.947303964874848e-05,
  "optimizer": "adam",
  "scheduler": "step",
  "epochs": 26,
  "scheduler_step_size": 7,
  "scheduler_gamma": 0.7176176305052315
}
```

**Metrics:**
```json
{
  "accuracy": 0.5,
  "f1": 0.0,
  "f1_macro": 0.3333333333333333,
  "f1_weighted": 0.33333333333333326,
  "precision": 0.0,
  "recall": 0.0,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.5,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5,
      "recall": 1.0,
      "f1-score": 0.6666666666666666,
      "support": 100.0
    },
    "1": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 100.0
    },
    "accuracy": 0.5,
    "macro avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.3333333333333333,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.33333333333333326,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen30_4426`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.525,
  "f1": 0.5177664974619289,
  "f1_macro": 0.5248931009477132,
  "f1_weighted": 0.5248931009477132,
  "precision": 0.5257731958762887,
  "recall": 0.51,
  "balanced_accuracy": 0.525,
  "roc_auc": 0.5353000000000001,
  "mcc": 0.0500225151988996,
  "cohen_kappa": 0.050000000000000044,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5242718446601942,
      "recall": 0.54,
      "f1-score": 0.5320197044334976,
      "support": 100.0
    },
    "1": {
      "precision": 0.5257731958762887,
      "recall": 0.51,
      "f1-score": 0.5177664974619289,
      "support": 100.0
    },
    "accuracy": 0.525,
    "macro avg": {
      "precision": 0.5250225202682415,
      "recall": 0.525,
      "f1-score": 0.5248931009477132,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5250225202682415,
      "recall": 0.525,
      "f1-score": 0.5248931009477132,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.635,
  "f1": 0.6473429951690821,
  "f1_macro": 0.6345523266000851,
  "f1_weighted": 0.6345523266000851,
  "precision": 0.6261682242990654,
  "recall": 0.67,
  "balanced_accuracy": 0.635,
  "roc_auc": 0.6836000000000001,
  "mcc": 0.27066394098188334,
  "cohen_kappa": 0.27,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.6451612903225806,
      "recall": 0.6,
      "f1-score": 0.6217616580310881,
      "support": 100.0
    },
    "1": {
      "precision": 0.6261682242990654,
      "recall": 0.67,
      "f1-score": 0.6473429951690821,
      "support": 100.0
    },
    "accuracy": 0.635,
    "macro avg": {
      "precision": 0.635664757310823,
      "recall": 0.635,
      "f1-score": 0.6345523266000851,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.635664757310823,
      "recall": 0.635,
      "f1-score": 0.6345523266000851,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 8.918436773300318e-05,
  "batch_size": 64,
  "weight_decay": 0.0007689717366737,
  "optimizer": "adam",
  "scheduler": "exponential",
  "epochs": 21,
  "scheduler_gamma": 0.9843332164756952
}
```

**Metrics:**
```json
{
  "accuracy": 0.46,
  "f1": 0.425531914893617,
  "f1_macro": 0.45804897631473307,
  "f1_weighted": 0.458048976314733,
  "precision": 0.45454545454545453,
  "recall": 0.4,
  "balanced_accuracy": 0.46,
  "roc_auc": 0.45699999999999996,
  "mcc": -0.08058229640253803,
  "cohen_kappa": -0.08000000000000007,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.4642857142857143,
      "recall": 0.52,
      "f1-score": 0.49056603773584906,
      "support": 100.0
    },
    "1": {
      "precision": 0.45454545454545453,
      "recall": 0.4,
      "f1-score": 0.425531914893617,
      "support": 100.0
    },
    "accuracy": 0.46,
    "macro avg": {
      "precision": 0.4594155844155844,
      "recall": 0.46,
      "f1-score": 0.45804897631473307,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.45941558441558444,
      "recall": 0.46,
      "f1-score": 0.458048976314733,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 1.2453772933355997e-05,
  "batch_size": 128,
  "weight_decay": 0.0006096133279213816,
  "optimizer": "sgd",
  "scheduler": "cosine",
  "epochs": 47,
  "momentum": 0.8237462394503742,
  "scheduler_t_max": 26
}
```

**Metrics:**
```json
{
  "accuracy": 0.53,
  "f1": 0.5392156862745098,
  "f1_macro": 0.529811924769908,
  "f1_weighted": 0.529811924769908,
  "precision": 0.5288461538461539,
  "recall": 0.55,
  "balanced_accuracy": 0.53,
  "roc_auc": 0.5308999999999999,
  "mcc": 0.06004805767690767,
  "cohen_kappa": 0.06000000000000005,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.53125,
      "recall": 0.51,
      "f1-score": 0.5204081632653061,
      "support": 100.0
    },
    "1": {
      "precision": 0.5288461538461539,
      "recall": 0.55,
      "f1-score": 0.5392156862745098,
      "support": 100.0
    },
    "accuracy": 0.53,
    "macro avg": {
      "precision": 0.5300480769230769,
      "recall": 0.53,
      "f1-score": 0.529811924769908,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.530048076923077,
      "recall": 0.53,
      "f1-score": 0.529811924769908,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen31_2433`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.5,
  "f1": 0.6666666666666666,
  "f1_macro": 0.3333333333333333,
  "f1_weighted": 0.33333333333333326,
  "precision": 0.5,
  "recall": 1.0,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.5390999999999999,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 100.0
    },
    "1": {
      "precision": 0.5,
      "recall": 1.0,
      "f1-score": 0.6666666666666666,
      "support": 100.0
    },
    "accuracy": 0.5,
    "macro avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.3333333333333333,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.33333333333333326,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.835,
  "f1": 0.8307692307692308,
  "f1_macro": 0.8348968105065666,
  "f1_weighted": 0.8348968105065666,
  "precision": 0.8526315789473684,
  "recall": 0.81,
  "balanced_accuracy": 0.835,
  "roc_auc": 0.9119,
  "mcc": 0.6708390735911568,
  "cohen_kappa": 0.6699999999999999,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.819047619047619,
      "recall": 0.86,
      "f1-score": 0.8390243902439024,
      "support": 100.0
    },
    "1": {
      "precision": 0.8526315789473684,
      "recall": 0.81,
      "f1-score": 0.8307692307692308,
      "support": 100.0
    },
    "accuracy": 0.835,
    "macro avg": {
      "precision": 0.8358395989974937,
      "recall": 0.835,
      "f1-score": 0.8348968105065666,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.8358395989974937,
      "recall": 0.835,
      "f1-score": 0.8348968105065666,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.009667489984598567,
  "batch_size": 64,
  "weight_decay": 0.002214306952444457,
  "optimizer": "adam",
  "scheduler": "exponential",
  "epochs": 42,
  "scheduler_gamma": 0.9841239741872437
}
```

**Metrics:**
```json
{
  "accuracy": 0.5,
  "f1": 0.0,
  "f1_macro": 0.3333333333333333,
  "f1_weighted": 0.33333333333333326,
  "precision": 0.0,
  "recall": 0.0,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.5,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5,
      "recall": 1.0,
      "f1-score": 0.6666666666666666,
      "support": 100.0
    },
    "1": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 100.0
    },
    "accuracy": 0.5,
    "macro avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.3333333333333333,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.33333333333333326,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.0067876835721272484,
  "batch_size": 128,
  "weight_decay": 0.0016577250260804359,
  "optimizer": "adam",
  "scheduler": "step",
  "epochs": 32,
  "scheduler_step_size": 9,
  "scheduler_gamma": 0.7315986964480146
}
```

**Metrics:**
```json
{
  "accuracy": 0.56,
  "f1": 0.45,
  "f1_macro": 0.5416666666666666,
  "f1_weighted": 0.5416666666666666,
  "precision": 0.6,
  "recall": 0.36,
  "balanced_accuracy": 0.56,
  "roc_auc": 0.5599999999999999,
  "mcc": 0.13093073414159542,
  "cohen_kappa": 0.12,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5428571428571428,
      "recall": 0.76,
      "f1-score": 0.6333333333333333,
      "support": 100.0
    },
    "1": {
      "precision": 0.6,
      "recall": 0.36,
      "f1-score": 0.45,
      "support": 100.0
    },
    "accuracy": 0.56,
    "macro avg": {
      "precision": 0.5714285714285714,
      "recall": 0.56,
      "f1-score": 0.5416666666666666,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5714285714285714,
      "recall": 0.56,
      "f1-score": 0.5416666666666666,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen32_5191`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.52,
  "f1": 0.5932203389830508,
  "f1_macro": 0.5039272426622571,
  "f1_weighted": 0.5039272426622571,
  "precision": 0.5147058823529411,
  "recall": 0.7,
  "balanced_accuracy": 0.52,
  "roc_auc": 0.5529,
  "mcc": 0.04287464628562721,
  "cohen_kappa": 0.040000000000000036,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.53125,
      "recall": 0.34,
      "f1-score": 0.4146341463414634,
      "support": 100.0
    },
    "1": {
      "precision": 0.5147058823529411,
      "recall": 0.7,
      "f1-score": 0.5932203389830508,
      "support": 100.0
    },
    "accuracy": 0.52,
    "macro avg": {
      "precision": 0.5229779411764706,
      "recall": 0.52,
      "f1-score": 0.5039272426622571,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5229779411764706,
      "recall": 0.52,
      "f1-score": 0.5039272426622571,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.675,
  "f1": 0.7325102880658436,
  "f1_macro": 0.6592487746061702,
  "f1_weighted": 0.6592487746061702,
  "precision": 0.6223776223776224,
  "recall": 0.89,
  "balanced_accuracy": 0.675,
  "roc_auc": 0.737,
  "mcc": 0.38767036020702567,
  "cohen_kappa": 0.35,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.8070175438596491,
      "recall": 0.46,
      "f1-score": 0.5859872611464968,
      "support": 100.0
    },
    "1": {
      "precision": 0.6223776223776224,
      "recall": 0.89,
      "f1-score": 0.7325102880658436,
      "support": 100.0
    },
    "accuracy": 0.675,
    "macro avg": {
      "precision": 0.7146975831186357,
      "recall": 0.675,
      "f1-score": 0.6592487746061702,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.7146975831186357,
      "recall": 0.675,
      "f1-score": 0.6592487746061702,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.009597599212559295,
  "batch_size": 64,
  "weight_decay": 0.0038806892186347476,
  "optimizer": "adam",
  "scheduler": "cosine",
  "epochs": 22,
  "scheduler_t_max": 30
}
```

**Metrics:**
```json
{
  "accuracy": 0.485,
  "f1": 0.037383177570093455,
  "f1_macro": 0.3429236706963095,
  "f1_weighted": 0.34292367069630947,
  "precision": 0.2857142857142857,
  "recall": 0.02,
  "balanced_accuracy": 0.485,
  "roc_auc": 0.485,
  "mcc": -0.08161943426864147,
  "cohen_kappa": -0.030000000000000027,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.49222797927461137,
      "recall": 0.95,
      "f1-score": 0.6484641638225256,
      "support": 100.0
    },
    "1": {
      "precision": 0.2857142857142857,
      "recall": 0.02,
      "f1-score": 0.037383177570093455,
      "support": 100.0
    },
    "accuracy": 0.485,
    "macro avg": {
      "precision": 0.38897113249444853,
      "recall": 0.485,
      "f1-score": 0.3429236706963095,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.3889711324944486,
      "recall": 0.485,
      "f1-score": 0.34292367069630947,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.007435147830000951,
  "batch_size": 32,
  "weight_decay": 0.001535936374865693,
  "optimizer": "adam",
  "scheduler": "none",
  "epochs": 41
}
```

**Metrics:**
```json
{
  "accuracy": 0.5,
  "f1": 0.6666666666666666,
  "f1_macro": 0.3333333333333333,
  "f1_weighted": 0.33333333333333326,
  "precision": 0.5,
  "recall": 1.0,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.5,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 100.0
    },
    "1": {
      "precision": 0.5,
      "recall": 1.0,
      "f1-score": 0.6666666666666666,
      "support": 100.0
    },
    "accuracy": 0.5,
    "macro avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.3333333333333333,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.33333333333333326,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen33_769`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.695,
  "f1": 0.7162790697674418,
  "f1_macro": 0.6932746700188561,
  "f1_weighted": 0.6932746700188561,
  "precision": 0.6695652173913044,
  "recall": 0.77,
  "balanced_accuracy": 0.6950000000000001,
  "roc_auc": 0.7319,
  "mcc": 0.3944629551908554,
  "cohen_kappa": 0.39,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.7294117647058823,
      "recall": 0.62,
      "f1-score": 0.6702702702702703,
      "support": 100.0
    },
    "1": {
      "precision": 0.6695652173913044,
      "recall": 0.77,
      "f1-score": 0.7162790697674418,
      "support": 100.0
    },
    "accuracy": 0.695,
    "macro avg": {
      "precision": 0.6994884910485933,
      "recall": 0.6950000000000001,
      "f1-score": 0.6932746700188561,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.6994884910485933,
      "recall": 0.695,
      "f1-score": 0.6932746700188561,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.91,
  "f1": 0.9108910891089109,
  "f1_macro": 0.90999099909991,
  "f1_weighted": 0.90999099909991,
  "precision": 0.9019607843137255,
  "recall": 0.92,
  "balanced_accuracy": 0.91,
  "roc_auc": 0.9763,
  "mcc": 0.8201640492164058,
  "cohen_kappa": 0.8200000000000001,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.9183673469387755,
      "recall": 0.9,
      "f1-score": 0.9090909090909091,
      "support": 100.0
    },
    "1": {
      "precision": 0.9019607843137255,
      "recall": 0.92,
      "f1-score": 0.9108910891089109,
      "support": 100.0
    },
    "accuracy": 0.91,
    "macro avg": {
      "precision": 0.9101640656262505,
      "recall": 0.91,
      "f1-score": 0.90999099909991,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.9101640656262506,
      "recall": 0.91,
      "f1-score": 0.90999099909991,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.009371575807279703,
  "batch_size": 32,
  "weight_decay": 0.00521689732250151,
  "optimizer": "adam",
  "scheduler": "exponential",
  "epochs": 39,
  "scheduler_gamma": 0.958625828420422
}
```

**Metrics:**
```json
{
  "accuracy": 0.5,
  "f1": 0.6666666666666666,
  "f1_macro": 0.3333333333333333,
  "f1_weighted": 0.33333333333333326,
  "precision": 0.5,
  "recall": 1.0,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.5,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 100.0
    },
    "1": {
      "precision": 0.5,
      "recall": 1.0,
      "f1-score": 0.6666666666666666,
      "support": 100.0
    },
    "accuracy": 0.5,
    "macro avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.3333333333333333,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.33333333333333326,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.0006259985899263418,
  "batch_size": 64,
  "weight_decay": 0.007594841238889879,
  "optimizer": "sgd",
  "scheduler": "step",
  "epochs": 33,
  "momentum": 0.9682038979652862,
  "scheduler_step_size": 5,
  "scheduler_gamma": 0.8401115115144371
}
```

**Metrics:**
```json
{
  "accuracy": 0.51,
  "f1": 0.5242718446601942,
  "f1_macro": 0.5095586027424682,
  "f1_weighted": 0.5095586027424682,
  "precision": 0.5094339622641509,
  "recall": 0.54,
  "balanced_accuracy": 0.51,
  "roc_auc": 0.51,
  "mcc": 0.020036097492521526,
  "cohen_kappa": 0.020000000000000018,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5106382978723404,
      "recall": 0.48,
      "f1-score": 0.4948453608247423,
      "support": 100.0
    },
    "1": {
      "precision": 0.5094339622641509,
      "recall": 0.54,
      "f1-score": 0.5242718446601942,
      "support": 100.0
    },
    "accuracy": 0.51,
    "macro avg": {
      "precision": 0.5100361300682457,
      "recall": 0.51,
      "f1-score": 0.5095586027424682,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5100361300682457,
      "recall": 0.51,
      "f1-score": 0.5095586027424682,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen34_769`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.55,
  "f1": 0.6052631578947368,
  "f1_macro": 0.5410036719706243,
  "f1_weighted": 0.5410036719706243,
  "precision": 0.5390625,
  "recall": 0.69,
  "balanced_accuracy": 0.5499999999999999,
  "roc_auc": 0.613,
  "mcc": 0.10416666666666667,
  "cohen_kappa": 0.09999999999999998,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5694444444444444,
      "recall": 0.41,
      "f1-score": 0.47674418604651164,
      "support": 100.0
    },
    "1": {
      "precision": 0.5390625,
      "recall": 0.69,
      "f1-score": 0.6052631578947368,
      "support": 100.0
    },
    "accuracy": 0.55,
    "macro avg": {
      "precision": 0.5542534722222222,
      "recall": 0.5499999999999999,
      "f1-score": 0.5410036719706243,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5542534722222222,
      "recall": 0.55,
      "f1-score": 0.5410036719706243,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.625,
  "f1": 0.6153846153846154,
  "f1_macro": 0.6247654784240151,
  "f1_weighted": 0.624765478424015,
  "precision": 0.631578947368421,
  "recall": 0.6,
  "balanced_accuracy": 0.625,
  "roc_auc": 0.6732,
  "mcc": 0.2503130871608794,
  "cohen_kappa": 0.25,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.6190476190476191,
      "recall": 0.65,
      "f1-score": 0.6341463414634146,
      "support": 100.0
    },
    "1": {
      "precision": 0.631578947368421,
      "recall": 0.6,
      "f1-score": 0.6153846153846154,
      "support": 100.0
    },
    "accuracy": 0.625,
    "macro avg": {
      "precision": 0.62531328320802,
      "recall": 0.625,
      "f1-score": 0.6247654784240151,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.62531328320802,
      "recall": 0.625,
      "f1-score": 0.624765478424015,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 8.346236479831111e-05,
  "batch_size": 128,
  "weight_decay": 0.00411937262386707,
  "optimizer": "adam",
  "scheduler": "step",
  "epochs": 37,
  "scheduler_step_size": 6,
  "scheduler_gamma": 0.10621088402905983
}
```

**Metrics:**
```json
{
  "accuracy": 0.52,
  "f1": 0.5514018691588785,
  "f1_macro": 0.517636418450407,
  "f1_weighted": 0.517636418450407,
  "precision": 0.5175438596491229,
  "recall": 0.59,
  "balanced_accuracy": 0.52,
  "roc_auc": 0.5081,
  "mcc": 0.040397858162338846,
  "cohen_kappa": 0.040000000000000036,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5232558139534884,
      "recall": 0.45,
      "f1-score": 0.4838709677419355,
      "support": 100.0
    },
    "1": {
      "precision": 0.5175438596491229,
      "recall": 0.59,
      "f1-score": 0.5514018691588785,
      "support": 100.0
    },
    "accuracy": 0.52,
    "macro avg": {
      "precision": 0.5203998368013056,
      "recall": 0.52,
      "f1-score": 0.517636418450407,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5203998368013056,
      "recall": 0.52,
      "f1-score": 0.517636418450407,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.008211545037830734,
  "batch_size": 64,
  "weight_decay": 0.0003476316887714105,
  "optimizer": "adam",
  "scheduler": "cosine",
  "epochs": 49,
  "scheduler_t_max": 39
}
```

**Metrics:**
```json
{
  "accuracy": 0.5,
  "f1": 0.6666666666666666,
  "f1_macro": 0.3333333333333333,
  "f1_weighted": 0.33333333333333326,
  "precision": 0.5,
  "recall": 1.0,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.5,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 100.0
    },
    "1": {
      "precision": 0.5,
      "recall": 1.0,
      "f1-score": 0.6666666666666666,
      "support": 100.0
    },
    "accuracy": 0.5,
    "macro avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.3333333333333333,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.33333333333333326,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen35_4426`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.48,
  "f1": 0.5047619047619047,
  "f1_macro": 0.4786967418546366,
  "f1_weighted": 0.47869674185463656,
  "precision": 0.4818181818181818,
  "recall": 0.53,
  "balanced_accuracy": 0.48,
  "roc_auc": 0.48350000000000004,
  "mcc": -0.04020151261036848,
  "cohen_kappa": -0.040000000000000036,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.4777777777777778,
      "recall": 0.43,
      "f1-score": 0.45263157894736844,
      "support": 100.0
    },
    "1": {
      "precision": 0.4818181818181818,
      "recall": 0.53,
      "f1-score": 0.5047619047619047,
      "support": 100.0
    },
    "accuracy": 0.48,
    "macro avg": {
      "precision": 0.4797979797979798,
      "recall": 0.48,
      "f1-score": 0.4786967418546366,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.4797979797979798,
      "recall": 0.48,
      "f1-score": 0.47869674185463656,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.52,
  "f1": 0.616,
  "f1_macro": 0.488,
  "f1_weighted": 0.488,
  "precision": 0.5133333333333333,
  "recall": 0.77,
  "balanced_accuracy": 0.52,
  "roc_auc": 0.5024,
  "mcc": 0.046188021535170064,
  "cohen_kappa": 0.040000000000000036,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.54,
      "recall": 0.27,
      "f1-score": 0.36,
      "support": 100.0
    },
    "1": {
      "precision": 0.5133333333333333,
      "recall": 0.77,
      "f1-score": 0.616,
      "support": 100.0
    },
    "accuracy": 0.52,
    "macro avg": {
      "precision": 0.5266666666666666,
      "recall": 0.52,
      "f1-score": 0.488,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5266666666666666,
      "recall": 0.52,
      "f1-score": 0.488,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.001466857212243702,
  "batch_size": 128,
  "weight_decay": 7.386193745464118e-05,
  "optimizer": "adam",
  "scheduler": "exponential",
  "epochs": 46,
  "scheduler_gamma": 0.9574631207436846
}
```

**Metrics:**
```json
{
  "accuracy": 0.475,
  "f1": 0.3137254901960784,
  "f1_macro": 0.4443121378105898,
  "f1_weighted": 0.4443121378105898,
  "precision": 0.4528301886792453,
  "recall": 0.24,
  "balanced_accuracy": 0.475,
  "roc_auc": 0.48865000000000003,
  "mcc": -0.05664654183701029,
  "cohen_kappa": -0.050000000000000044,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.48299319727891155,
      "recall": 0.71,
      "f1-score": 0.5748987854251012,
      "support": 100.0
    },
    "1": {
      "precision": 0.4528301886792453,
      "recall": 0.24,
      "f1-score": 0.3137254901960784,
      "support": 100.0
    },
    "accuracy": 0.475,
    "macro avg": {
      "precision": 0.4679116929790784,
      "recall": 0.475,
      "f1-score": 0.4443121378105898,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.4679116929790784,
      "recall": 0.475,
      "f1-score": 0.4443121378105898,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.0006449428556048363,
  "batch_size": 128,
  "weight_decay": 0.0004104696300194134,
  "optimizer": "sgd",
  "scheduler": "none",
  "epochs": 35,
  "momentum": 0.8190146110762182
}
```

**Metrics:**
```json
{
  "accuracy": 0.53,
  "f1": 0.6666666666666666,
  "f1_macro": 0.4350282485875706,
  "f1_weighted": 0.4350282485875706,
  "precision": 0.5164835164835165,
  "recall": 0.94,
  "balanced_accuracy": 0.53,
  "roc_auc": 0.5341,
  "mcc": 0.10482848367219183,
  "cohen_kappa": 0.06000000000000005,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.6666666666666666,
      "recall": 0.12,
      "f1-score": 0.2033898305084746,
      "support": 100.0
    },
    "1": {
      "precision": 0.5164835164835165,
      "recall": 0.94,
      "f1-score": 0.6666666666666666,
      "support": 100.0
    },
    "accuracy": 0.53,
    "macro avg": {
      "precision": 0.5915750915750916,
      "recall": 0.53,
      "f1-score": 0.4350282485875706,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5915750915750916,
      "recall": 0.53,
      "f1-score": 0.4350282485875706,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen36_466`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.55,
  "f1": 0.3283582089552239,
  "f1_macro": 0.4950061721467849,
  "f1_weighted": 0.49500617214678483,
  "precision": 0.6470588235294118,
  "recall": 0.22,
  "balanced_accuracy": 0.55,
  "roc_auc": 0.5862,
  "mcc": 0.13310871701625052,
  "cohen_kappa": 0.09999999999999998,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5301204819277109,
      "recall": 0.88,
      "f1-score": 0.6616541353383458,
      "support": 100.0
    },
    "1": {
      "precision": 0.6470588235294118,
      "recall": 0.22,
      "f1-score": 0.3283582089552239,
      "support": 100.0
    },
    "accuracy": 0.55,
    "macro avg": {
      "precision": 0.5885896527285613,
      "recall": 0.55,
      "f1-score": 0.4950061721467849,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5885896527285613,
      "recall": 0.55,
      "f1-score": 0.49500617214678483,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.74,
  "f1": 0.6904761904761905,
  "f1_macro": 0.7331691297208538,
  "f1_weighted": 0.7331691297208539,
  "precision": 0.8529411764705882,
  "recall": 0.58,
  "balanced_accuracy": 0.74,
  "roc_auc": 0.8348,
  "mcc": 0.5066403971048988,
  "cohen_kappa": 0.48,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.6818181818181818,
      "recall": 0.9,
      "f1-score": 0.7758620689655172,
      "support": 100.0
    },
    "1": {
      "precision": 0.8529411764705882,
      "recall": 0.58,
      "f1-score": 0.6904761904761905,
      "support": 100.0
    },
    "accuracy": 0.74,
    "macro avg": {
      "precision": 0.767379679144385,
      "recall": 0.74,
      "f1-score": 0.7331691297208538,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.767379679144385,
      "recall": 0.74,
      "f1-score": 0.7331691297208539,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.006418547793728898,
  "batch_size": 32,
  "weight_decay": 0.0004419492251353177,
  "optimizer": "adam",
  "scheduler": "none",
  "epochs": 25
}
```

**Metrics:**
```json
{
  "accuracy": 0.5,
  "f1": 0.6666666666666666,
  "f1_macro": 0.3333333333333333,
  "f1_weighted": 0.33333333333333326,
  "precision": 0.5,
  "recall": 1.0,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.498,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 100.0
    },
    "1": {
      "precision": 0.5,
      "recall": 1.0,
      "f1-score": 0.6666666666666666,
      "support": 100.0
    },
    "accuracy": 0.5,
    "macro avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.3333333333333333,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.33333333333333326,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.004119589779064705,
  "batch_size": 128,
  "weight_decay": 2.7765066894552695e-05,
  "optimizer": "adam",
  "scheduler": "step",
  "epochs": 40,
  "scheduler_step_size": 6,
  "scheduler_gamma": 0.7913632482655487
}
```

**Metrics:**
```json
{
  "accuracy": 0.495,
  "f1": 0.15126050420168066,
  "f1_macro": 0.39591494960973717,
  "f1_weighted": 0.39591494960973717,
  "precision": 0.47368421052631576,
  "recall": 0.09,
  "balanced_accuracy": 0.495,
  "roc_auc": 0.49545000000000006,
  "mcc": -0.01705233720429863,
  "cohen_kappa": -0.010000000000000009,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.4972375690607735,
      "recall": 0.9,
      "f1-score": 0.6405693950177936,
      "support": 100.0
    },
    "1": {
      "precision": 0.47368421052631576,
      "recall": 0.09,
      "f1-score": 0.15126050420168066,
      "support": 100.0
    },
    "accuracy": 0.495,
    "macro avg": {
      "precision": 0.4854608897935446,
      "recall": 0.495,
      "f1-score": 0.39591494960973717,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.4854608897935446,
      "recall": 0.495,
      "f1-score": 0.39591494960973717,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen37_769`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.6,
  "f1": 0.5555555555555556,
  "f1_macro": 0.595959595959596,
  "f1_weighted": 0.595959595959596,
  "precision": 0.625,
  "recall": 0.5,
  "balanced_accuracy": 0.6,
  "roc_auc": 0.6486,
  "mcc": 0.20412414523193154,
  "cohen_kappa": 0.19999999999999996,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5833333333333334,
      "recall": 0.7,
      "f1-score": 0.6363636363636364,
      "support": 100.0
    },
    "1": {
      "precision": 0.625,
      "recall": 0.5,
      "f1-score": 0.5555555555555556,
      "support": 100.0
    },
    "accuracy": 0.6,
    "macro avg": {
      "precision": 0.6041666666666667,
      "recall": 0.6,
      "f1-score": 0.595959595959596,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.6041666666666667,
      "recall": 0.6,
      "f1-score": 0.595959595959596,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.785,
  "f1": 0.7700534759358288,
  "f1_macro": 0.7840877708317642,
  "f1_weighted": 0.7840877708317642,
  "precision": 0.8275862068965517,
  "recall": 0.72,
  "balanced_accuracy": 0.7849999999999999,
  "roc_auc": 0.888,
  "mcc": 0.5748784218232036,
  "cohen_kappa": 0.5700000000000001,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.7522123893805309,
      "recall": 0.85,
      "f1-score": 0.7981220657276995,
      "support": 100.0
    },
    "1": {
      "precision": 0.8275862068965517,
      "recall": 0.72,
      "f1-score": 0.7700534759358288,
      "support": 100.0
    },
    "accuracy": 0.785,
    "macro avg": {
      "precision": 0.7898992981385413,
      "recall": 0.7849999999999999,
      "f1-score": 0.7840877708317642,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.7898992981385413,
      "recall": 0.785,
      "f1-score": 0.7840877708317642,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.008623748235509068,
  "batch_size": 64,
  "weight_decay": 0.0009203008485053073,
  "optimizer": "sgd",
  "scheduler": "exponential",
  "epochs": 29,
  "momentum": 0.9755675168826639,
  "scheduler_gamma": 0.9715301264590614
}
```

**Metrics:**
```json
{
  "accuracy": 0.565,
  "f1": 0.5396825396825397,
  "f1_macro": 0.5636801324005115,
  "f1_weighted": 0.5636801324005115,
  "precision": 0.5730337078651685,
  "recall": 0.51,
  "balanced_accuracy": 0.565,
  "roc_auc": 0.5665000000000001,
  "mcc": 0.1307937102275355,
  "cohen_kappa": 0.13,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5585585585585585,
      "recall": 0.62,
      "f1-score": 0.5876777251184834,
      "support": 100.0
    },
    "1": {
      "precision": 0.5730337078651685,
      "recall": 0.51,
      "f1-score": 0.5396825396825397,
      "support": 100.0
    },
    "accuracy": 0.565,
    "macro avg": {
      "precision": 0.5657961332118635,
      "recall": 0.565,
      "f1-score": 0.5636801324005115,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5657961332118635,
      "recall": 0.565,
      "f1-score": 0.5636801324005115,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.0075486316522085225,
  "batch_size": 128,
  "weight_decay": 1.010819995234636e-05,
  "optimizer": "adam",
  "scheduler": "cosine",
  "epochs": 47,
  "scheduler_t_max": 38
}
```

**Metrics:**
```json
{
  "accuracy": 0.475,
  "f1": 0.6391752577319587,
  "f1_macro": 0.33793625271919037,
  "f1_weighted": 0.3379362527191903,
  "precision": 0.4869109947643979,
  "recall": 0.93,
  "balanced_accuracy": 0.47500000000000003,
  "roc_auc": 0.4751,
  "mcc": -0.12059576754873692,
  "cohen_kappa": -0.050000000000000044,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.2222222222222222,
      "recall": 0.02,
      "f1-score": 0.03669724770642202,
      "support": 100.0
    },
    "1": {
      "precision": 0.4869109947643979,
      "recall": 0.93,
      "f1-score": 0.6391752577319587,
      "support": 100.0
    },
    "accuracy": 0.475,
    "macro avg": {
      "precision": 0.35456660849331006,
      "recall": 0.47500000000000003,
      "f1-score": 0.33793625271919037,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.35456660849331,
      "recall": 0.475,
      "f1-score": 0.3379362527191903,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen38_4426`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.565,
  "f1": 0.6390041493775933,
  "f1_macro": 0.5459171690284192,
  "f1_weighted": 0.5459171690284194,
  "precision": 0.5460992907801419,
  "recall": 0.77,
  "balanced_accuracy": 0.565,
  "roc_auc": 0.5671,
  "mcc": 0.1425304939292765,
  "cohen_kappa": 0.13,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.6101694915254238,
      "recall": 0.36,
      "f1-score": 0.4528301886792453,
      "support": 100.0
    },
    "1": {
      "precision": 0.5460992907801419,
      "recall": 0.77,
      "f1-score": 0.6390041493775933,
      "support": 100.0
    },
    "accuracy": 0.565,
    "macro avg": {
      "precision": 0.5781343911527828,
      "recall": 0.565,
      "f1-score": 0.5459171690284192,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5781343911527828,
      "recall": 0.565,
      "f1-score": 0.5459171690284194,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.635,
  "f1": 0.6439024390243903,
  "f1_macro": 0.634771732332708,
  "f1_weighted": 0.6347717323327079,
  "precision": 0.6285714285714286,
  "recall": 0.66,
  "balanced_accuracy": 0.635,
  "roc_auc": 0.6583000000000001,
  "mcc": 0.27033813413374974,
  "cohen_kappa": 0.27,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.6421052631578947,
      "recall": 0.61,
      "f1-score": 0.6256410256410256,
      "support": 100.0
    },
    "1": {
      "precision": 0.6285714285714286,
      "recall": 0.66,
      "f1-score": 0.6439024390243903,
      "support": 100.0
    },
    "accuracy": 0.635,
    "macro avg": {
      "precision": 0.6353383458646616,
      "recall": 0.635,
      "f1-score": 0.634771732332708,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.6353383458646618,
      "recall": 0.635,
      "f1-score": 0.6347717323327079,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.005804920133185138,
  "batch_size": 64,
  "weight_decay": 0.0006883963077311655,
  "optimizer": "adam",
  "scheduler": "cosine",
  "epochs": 35,
  "scheduler_t_max": 30
}
```

**Metrics:**
```json
{
  "accuracy": 0.475,
  "f1": 0.6236559139784946,
  "f1_macro": 0.37794365946858616,
  "f1_weighted": 0.3779436594685862,
  "precision": 0.4860335195530726,
  "recall": 0.87,
  "balanced_accuracy": 0.475,
  "roc_auc": 0.47434999999999994,
  "mcc": -0.08155185451433536,
  "cohen_kappa": -0.050000000000000044,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.38095238095238093,
      "recall": 0.08,
      "f1-score": 0.1322314049586777,
      "support": 100.0
    },
    "1": {
      "precision": 0.4860335195530726,
      "recall": 0.87,
      "f1-score": 0.6236559139784946,
      "support": 100.0
    },
    "accuracy": 0.475,
    "macro avg": {
      "precision": 0.43349295025272677,
      "recall": 0.475,
      "f1-score": 0.37794365946858616,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.4334929502527268,
      "recall": 0.475,
      "f1-score": 0.3779436594685862,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.0012706640964494547,
  "batch_size": 128,
  "weight_decay": 2.963492040547409e-05,
  "optimizer": "sgd",
  "scheduler": "cosine",
  "epochs": 43,
  "momentum": 0.8606654808962139,
  "scheduler_t_max": 18
}
```

**Metrics:**
```json
{
  "accuracy": 0.51,
  "f1": 0.6711409395973155,
  "f1_macro": 0.3551783129359126,
  "f1_weighted": 0.3551783129359126,
  "precision": 0.5050505050505051,
  "recall": 1.0,
  "balanced_accuracy": 0.51,
  "roc_auc": 0.51,
  "mcc": 0.10050378152592121,
  "cohen_kappa": 0.020000000000000018,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 1.0,
      "recall": 0.02,
      "f1-score": 0.0392156862745098,
      "support": 100.0
    },
    "1": {
      "precision": 0.5050505050505051,
      "recall": 1.0,
      "f1-score": 0.6711409395973155,
      "support": 100.0
    },
    "accuracy": 0.51,
    "macro avg": {
      "precision": 0.7525252525252526,
      "recall": 0.51,
      "f1-score": 0.3551783129359126,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.7525252525252526,
      "recall": 0.51,
      "f1-score": 0.3551783129359126,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen39_5578`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.62,
  "f1": 0.5421686746987951,
  "f1_macro": 0.6086911749562351,
  "f1_weighted": 0.6086911749562351,
  "precision": 0.6818181818181818,
  "recall": 0.45,
  "balanced_accuracy": 0.62,
  "roc_auc": 0.6777,
  "mcc": 0.255203646035468,
  "cohen_kappa": 0.24,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5895522388059702,
      "recall": 0.79,
      "f1-score": 0.6752136752136753,
      "support": 100.0
    },
    "1": {
      "precision": 0.6818181818181818,
      "recall": 0.45,
      "f1-score": 0.5421686746987951,
      "support": 100.0
    },
    "accuracy": 0.62,
    "macro avg": {
      "precision": 0.635685210312076,
      "recall": 0.62,
      "f1-score": 0.6086911749562351,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.6356852103120759,
      "recall": 0.62,
      "f1-score": 0.6086911749562351,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.88,
  "f1": 0.8811881188118812,
  "f1_macro": 0.87998799879988,
  "f1_weighted": 0.87998799879988,
  "precision": 0.8725490196078431,
  "recall": 0.89,
  "balanced_accuracy": 0.88,
  "roc_auc": 0.955,
  "mcc": 0.7601520456152054,
  "cohen_kappa": 0.76,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.8877551020408163,
      "recall": 0.87,
      "f1-score": 0.8787878787878788,
      "support": 100.0
    },
    "1": {
      "precision": 0.8725490196078431,
      "recall": 0.89,
      "f1-score": 0.8811881188118812,
      "support": 100.0
    },
    "accuracy": 0.88,
    "macro avg": {
      "precision": 0.8801520608243297,
      "recall": 0.88,
      "f1-score": 0.87998799879988,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.8801520608243297,
      "recall": 0.88,
      "f1-score": 0.87998799879988,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.00445034084354637,
  "batch_size": 32,
  "weight_decay": 0.0017707880392787319,
  "optimizer": "adam",
  "scheduler": "exponential",
  "epochs": 50,
  "scheduler_gamma": 0.9865052940214932
}
```

**Metrics:**
```json
{
  "accuracy": 0.5,
  "f1": 0.0,
  "f1_macro": 0.3333333333333333,
  "f1_weighted": 0.33333333333333326,
  "precision": 0.0,
  "recall": 0.0,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.5,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5,
      "recall": 1.0,
      "f1-score": 0.6666666666666666,
      "support": 100.0
    },
    "1": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 100.0
    },
    "accuracy": 0.5,
    "macro avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.3333333333333333,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.33333333333333326,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.0038492253332661507,
  "batch_size": 128,
  "weight_decay": 0.009977367348609632,
  "optimizer": "sgd",
  "scheduler": "exponential",
  "epochs": 35,
  "momentum": 0.9730719213409851,
  "scheduler_gamma": 0.9899616335896173
}
```

**Metrics:**
```json
{
  "accuracy": 0.505,
  "f1": 0.5787234042553191,
  "f1_macro": 0.48936170212765956,
  "f1_weighted": 0.48936170212765956,
  "precision": 0.5037037037037037,
  "recall": 0.68,
  "balanced_accuracy": 0.505,
  "roc_auc": 0.50375,
  "mcc": 0.010675210253672476,
  "cohen_kappa": 0.010000000000000009,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5076923076923077,
      "recall": 0.33,
      "f1-score": 0.4,
      "support": 100.0
    },
    "1": {
      "precision": 0.5037037037037037,
      "recall": 0.68,
      "f1-score": 0.5787234042553191,
      "support": 100.0
    },
    "accuracy": 0.505,
    "macro avg": {
      "precision": 0.5056980056980056,
      "recall": 0.505,
      "f1-score": 0.48936170212765956,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5056980056980056,
      "recall": 0.505,
      "f1-score": 0.48936170212765956,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen3_769`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.96,
  "f1": 0.9595959595959596,
  "f1_macro": 0.95999599959996,
  "f1_weighted": 0.9599959995999601,
  "precision": 0.9693877551020408,
  "recall": 0.95,
  "balanced_accuracy": 0.96,
  "roc_auc": 0.9875999999999999,
  "mcc": 0.9201840552184065,
  "cohen_kappa": 0.92,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.9509803921568627,
      "recall": 0.97,
      "f1-score": 0.9603960396039604,
      "support": 100.0
    },
    "1": {
      "precision": 0.9693877551020408,
      "recall": 0.95,
      "f1-score": 0.9595959595959596,
      "support": 100.0
    },
    "accuracy": 0.96,
    "macro avg": {
      "precision": 0.9601840736294518,
      "recall": 0.96,
      "f1-score": 0.95999599959996,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.9601840736294517,
      "recall": 0.96,
      "f1-score": 0.9599959995999601,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.96,
  "f1": 0.96,
  "f1_macro": 0.96,
  "f1_weighted": 0.96,
  "precision": 0.96,
  "recall": 0.96,
  "balanced_accuracy": 0.96,
  "roc_auc": 0.9827,
  "mcc": 0.92,
  "cohen_kappa": 0.92,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.96,
      "recall": 0.96,
      "f1-score": 0.96,
      "support": 100.0
    },
    "1": {
      "precision": 0.96,
      "recall": 0.96,
      "f1-score": 0.96,
      "support": 100.0
    },
    "accuracy": 0.96,
    "macro avg": {
      "precision": 0.96,
      "recall": 0.96,
      "f1-score": 0.96,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.96,
      "recall": 0.96,
      "f1-score": 0.96,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.0058820640701654206,
  "batch_size": 64,
  "weight_decay": 6.165479782674587e-05,
  "optimizer": "sgd",
  "scheduler": "none",
  "epochs": 41,
  "momentum": 0.8111337091294457
}
```

**Metrics:**
```json
{
  "accuracy": 0.58,
  "f1": 0.6216216216216216,
  "f1_macro": 0.5748557546310356,
  "f1_weighted": 0.5748557546310356,
  "precision": 0.5655737704918032,
  "recall": 0.69,
  "balanced_accuracy": 0.58,
  "roc_auc": 0.5823499999999999,
  "mcc": 0.16401847362094593,
  "cohen_kappa": 0.16000000000000003,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.6025641025641025,
      "recall": 0.47,
      "f1-score": 0.5280898876404494,
      "support": 100.0
    },
    "1": {
      "precision": 0.5655737704918032,
      "recall": 0.69,
      "f1-score": 0.6216216216216216,
      "support": 100.0
    },
    "accuracy": 0.58,
    "macro avg": {
      "precision": 0.5840689365279529,
      "recall": 0.58,
      "f1-score": 0.5748557546310356,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5840689365279529,
      "recall": 0.58,
      "f1-score": 0.5748557546310356,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.009789565889024484,
  "batch_size": 32,
  "weight_decay": 0.00033147188561096614,
  "optimizer": "adam",
  "scheduler": "exponential",
  "epochs": 32,
  "scheduler_gamma": 0.9696987427924894
}
```

**Metrics:**
```json
{
  "accuracy": 0.5,
  "f1": 0.0,
  "f1_macro": 0.3333333333333333,
  "f1_weighted": 0.33333333333333326,
  "precision": 0.0,
  "recall": 0.0,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.5,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5,
      "recall": 1.0,
      "f1-score": 0.6666666666666666,
      "support": 100.0
    },
    "1": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 100.0
    },
    "accuracy": 0.5,
    "macro avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.3333333333333333,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.33333333333333326,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen40_5390`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.515,
  "f1": 0.6689419795221843,
  "f1_macro": 0.38119996172370896,
  "f1_weighted": 0.3811999617237089,
  "precision": 0.5077720207253886,
  "recall": 0.98,
  "balanced_accuracy": 0.515,
  "roc_auc": 0.5463,
  "mcc": 0.08161943426864147,
  "cohen_kappa": 0.030000000000000027,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.7142857142857143,
      "recall": 0.05,
      "f1-score": 0.09345794392523364,
      "support": 100.0
    },
    "1": {
      "precision": 0.5077720207253886,
      "recall": 0.98,
      "f1-score": 0.6689419795221843,
      "support": 100.0
    },
    "accuracy": 0.515,
    "macro avg": {
      "precision": 0.6110288675055515,
      "recall": 0.515,
      "f1-score": 0.38119996172370896,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.6110288675055514,
      "recall": 0.515,
      "f1-score": 0.3811999617237089,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.525,
  "f1": 0.3262411347517731,
  "f1_macro": 0.4797228839782031,
  "f1_weighted": 0.4797228839782031,
  "precision": 0.5609756097560976,
  "recall": 0.23,
  "balanced_accuracy": 0.525,
  "roc_auc": 0.4979,
  "mcc": 0.06192692475666345,
  "cohen_kappa": 0.050000000000000044,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5157232704402516,
      "recall": 0.82,
      "f1-score": 0.6332046332046332,
      "support": 100.0
    },
    "1": {
      "precision": 0.5609756097560976,
      "recall": 0.23,
      "f1-score": 0.3262411347517731,
      "support": 100.0
    },
    "accuracy": 0.525,
    "macro avg": {
      "precision": 0.5383494400981745,
      "recall": 0.525,
      "f1-score": 0.4797228839782031,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5383494400981746,
      "recall": 0.525,
      "f1-score": 0.4797228839782031,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.009955636745625019,
  "batch_size": 32,
  "weight_decay": 0.000597698605629842,
  "optimizer": "adam",
  "scheduler": "step",
  "epochs": 44,
  "scheduler_step_size": 10,
  "scheduler_gamma": 0.7311022338868206
}
```

**Metrics:**
```json
{
  "accuracy": 0.5,
  "f1": 0.6666666666666666,
  "f1_macro": 0.3333333333333333,
  "f1_weighted": 0.33333333333333326,
  "precision": 0.5,
  "recall": 1.0,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.4903,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 100.0
    },
    "1": {
      "precision": 0.5,
      "recall": 1.0,
      "f1-score": 0.6666666666666666,
      "support": 100.0
    },
    "accuracy": 0.5,
    "macro avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.3333333333333333,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.33333333333333326,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.008299171637238622,
  "batch_size": 32,
  "weight_decay": 0.0019510503323493132,
  "optimizer": "adam",
  "scheduler": "exponential",
  "epochs": 21,
  "scheduler_gamma": 0.9869815748010201
}
```

**Metrics:**
```json
{
  "accuracy": 0.51,
  "f1": 0.6474820143884892,
  "f1_macro": 0.4221016629319495,
  "f1_weighted": 0.4221016629319495,
  "precision": 0.5056179775280899,
  "recall": 0.9,
  "balanced_accuracy": 0.51,
  "roc_auc": 0.5385,
  "mcc": 0.03196013860502966,
  "cohen_kappa": 0.020000000000000018,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5454545454545454,
      "recall": 0.12,
      "f1-score": 0.19672131147540983,
      "support": 100.0
    },
    "1": {
      "precision": 0.5056179775280899,
      "recall": 0.9,
      "f1-score": 0.6474820143884892,
      "support": 100.0
    },
    "accuracy": 0.51,
    "macro avg": {
      "precision": 0.5255362614913177,
      "recall": 0.51,
      "f1-score": 0.4221016629319495,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5255362614913177,
      "recall": 0.51,
      "f1-score": 0.4221016629319495,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen4_860`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.575,
  "f1": 0.5728643216080402,
  "f1_macro": 0.5749893747343684,
  "f1_weighted": 0.5749893747343684,
  "precision": 0.5757575757575758,
  "recall": 0.57,
  "balanced_accuracy": 0.575,
  "roc_auc": 0.6095,
  "mcc": 0.15000750056254686,
  "cohen_kappa": 0.15000000000000002,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5742574257425742,
      "recall": 0.58,
      "f1-score": 0.5771144278606966,
      "support": 100.0
    },
    "1": {
      "precision": 0.5757575757575758,
      "recall": 0.57,
      "f1-score": 0.5728643216080402,
      "support": 100.0
    },
    "accuracy": 0.575,
    "macro avg": {
      "precision": 0.575007500750075,
      "recall": 0.575,
      "f1-score": 0.5749893747343684,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.575007500750075,
      "recall": 0.575,
      "f1-score": 0.5749893747343684,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.56,
  "f1": 0.6589147286821705,
  "f1_macro": 0.5195982094115078,
  "f1_weighted": 0.5195982094115078,
  "precision": 0.5379746835443038,
  "recall": 0.85,
  "balanced_accuracy": 0.56,
  "roc_auc": 0.6686,
  "mcc": 0.14730858484207088,
  "cohen_kappa": 0.12,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.6428571428571429,
      "recall": 0.27,
      "f1-score": 0.38028169014084506,
      "support": 100.0
    },
    "1": {
      "precision": 0.5379746835443038,
      "recall": 0.85,
      "f1-score": 0.6589147286821705,
      "support": 100.0
    },
    "accuracy": 0.56,
    "macro avg": {
      "precision": 0.5904159132007234,
      "recall": 0.56,
      "f1-score": 0.5195982094115078,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5904159132007234,
      "recall": 0.56,
      "f1-score": 0.5195982094115078,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 1.6455415912947516e-05,
  "batch_size": 128,
  "weight_decay": 0.000476962937556555,
  "optimizer": "adam",
  "scheduler": "cosine",
  "epochs": 23,
  "scheduler_t_max": 39
}
```

**Metrics:**
```json
{
  "accuracy": 0.49,
  "f1": 0.5446428571428571,
  "f1_macro": 0.4825487012987013,
  "f1_weighted": 0.48254870129870125,
  "precision": 0.49193548387096775,
  "recall": 0.61,
  "balanced_accuracy": 0.49,
  "roc_auc": 0.48615,
  "mcc": -0.020602141085758228,
  "cohen_kappa": -0.020000000000000018,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.4868421052631579,
      "recall": 0.37,
      "f1-score": 0.42045454545454547,
      "support": 100.0
    },
    "1": {
      "precision": 0.49193548387096775,
      "recall": 0.61,
      "f1-score": 0.5446428571428571,
      "support": 100.0
    },
    "accuracy": 0.49,
    "macro avg": {
      "precision": 0.48938879456706286,
      "recall": 0.49,
      "f1-score": 0.4825487012987013,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.4893887945670628,
      "recall": 0.49,
      "f1-score": 0.48254870129870125,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.0034583727019413943,
  "batch_size": 128,
  "weight_decay": 0.00012116880104182466,
  "optimizer": "adam",
  "scheduler": "cosine",
  "epochs": 41,
  "scheduler_t_max": 11
}
```

**Metrics:**
```json
{
  "accuracy": 0.56,
  "f1": 0.5056179775280899,
  "f1_macro": 0.5546107905658468,
  "f1_weighted": 0.5546107905658468,
  "precision": 0.5769230769230769,
  "recall": 0.45,
  "balanced_accuracy": 0.56,
  "roc_auc": 0.56485,
  "mcc": 0.12301385521570944,
  "cohen_kappa": 0.12,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5491803278688525,
      "recall": 0.67,
      "f1-score": 0.6036036036036037,
      "support": 100.0
    },
    "1": {
      "precision": 0.5769230769230769,
      "recall": 0.45,
      "f1-score": 0.5056179775280899,
      "support": 100.0
    },
    "accuracy": 0.56,
    "macro avg": {
      "precision": 0.5630517023959647,
      "recall": 0.56,
      "f1-score": 0.5546107905658468,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5630517023959647,
      "recall": 0.56,
      "f1-score": 0.5546107905658468,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen5_6949`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.575,
  "f1": 0.6255506607929515,
  "f1_macro": 0.5671105905120827,
  "f1_weighted": 0.5671105905120827,
  "precision": 0.5590551181102362,
  "recall": 0.71,
  "balanced_accuracy": 0.575,
  "roc_auc": 0.5432,
  "mcc": 0.155785835751024,
  "cohen_kappa": 0.15000000000000002,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.6027397260273972,
      "recall": 0.44,
      "f1-score": 0.5086705202312138,
      "support": 100.0
    },
    "1": {
      "precision": 0.5590551181102362,
      "recall": 0.71,
      "f1-score": 0.6255506607929515,
      "support": 100.0
    },
    "accuracy": 0.575,
    "macro avg": {
      "precision": 0.5808974220688168,
      "recall": 0.575,
      "f1-score": 0.5671105905120827,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5808974220688168,
      "recall": 0.575,
      "f1-score": 0.5671105905120827,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.805,
  "f1": 0.8202764976958525,
  "f1_macro": 0.8035808717987459,
  "f1_weighted": 0.8035808717987458,
  "precision": 0.7606837606837606,
  "recall": 0.89,
  "balanced_accuracy": 0.8049999999999999,
  "roc_auc": 0.8602,
  "mcc": 0.6190102749737828,
  "cohen_kappa": 0.61,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.8674698795180723,
      "recall": 0.72,
      "f1-score": 0.7868852459016393,
      "support": 100.0
    },
    "1": {
      "precision": 0.7606837606837606,
      "recall": 0.89,
      "f1-score": 0.8202764976958525,
      "support": 100.0
    },
    "accuracy": 0.805,
    "macro avg": {
      "precision": 0.8140768201009165,
      "recall": 0.8049999999999999,
      "f1-score": 0.8035808717987459,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.8140768201009165,
      "recall": 0.805,
      "f1-score": 0.8035808717987458,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.0018833729187531815,
  "batch_size": 128,
  "weight_decay": 0.00036415066816079014,
  "optimizer": "adam",
  "scheduler": "none",
  "epochs": 23
}
```

**Metrics:**
```json
{
  "accuracy": 0.47,
  "f1": 0.5859375,
  "f1_macro": 0.4249131944444444,
  "f1_weighted": 0.4249131944444444,
  "precision": 0.4807692307692308,
  "recall": 0.75,
  "balanced_accuracy": 0.47,
  "roc_auc": 0.47,
  "mcc": -0.07242068243779014,
  "cohen_kappa": -0.06000000000000005,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.4318181818181818,
      "recall": 0.19,
      "f1-score": 0.2638888888888889,
      "support": 100.0
    },
    "1": {
      "precision": 0.4807692307692308,
      "recall": 0.75,
      "f1-score": 0.5859375,
      "support": 100.0
    },
    "accuracy": 0.47,
    "macro avg": {
      "precision": 0.4562937062937063,
      "recall": 0.47,
      "f1-score": 0.4249131944444444,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.45629370629370625,
      "recall": 0.47,
      "f1-score": 0.4249131944444444,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 3.488709696917098e-05,
  "batch_size": 64,
  "weight_decay": 0.0003667105224974157,
  "optimizer": "adam",
  "scheduler": "cosine",
  "epochs": 47,
  "scheduler_t_max": 21
}
```

**Metrics:**
```json
{
  "accuracy": 0.53,
  "f1": 0.4268292682926829,
  "f1_macro": 0.51426209177346,
  "f1_weighted": 0.51426209177346,
  "precision": 0.546875,
  "recall": 0.35,
  "balanced_accuracy": 0.53,
  "roc_auc": 0.5237499999999999,
  "mcc": 0.06431196942844082,
  "cohen_kappa": 0.06000000000000005,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5220588235294118,
      "recall": 0.71,
      "f1-score": 0.6016949152542372,
      "support": 100.0
    },
    "1": {
      "precision": 0.546875,
      "recall": 0.35,
      "f1-score": 0.4268292682926829,
      "support": 100.0
    },
    "accuracy": 0.53,
    "macro avg": {
      "precision": 0.5344669117647058,
      "recall": 0.53,
      "f1-score": 0.51426209177346,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.534466911764706,
      "recall": 0.53,
      "f1-score": 0.51426209177346,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen6_466`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.52,
  "f1": 0.45454545454545453,
  "f1_macro": 0.512987012987013,
  "f1_weighted": 0.512987012987013,
  "precision": 0.5263157894736842,
  "recall": 0.4,
  "balanced_accuracy": 0.52,
  "roc_auc": 0.5517,
  "mcc": 0.041204282171516456,
  "cohen_kappa": 0.040000000000000036,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5161290322580645,
      "recall": 0.64,
      "f1-score": 0.5714285714285714,
      "support": 100.0
    },
    "1": {
      "precision": 0.5263157894736842,
      "recall": 0.4,
      "f1-score": 0.45454545454545453,
      "support": 100.0
    },
    "accuracy": 0.52,
    "macro avg": {
      "precision": 0.5212224108658743,
      "recall": 0.52,
      "f1-score": 0.512987012987013,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5212224108658744,
      "recall": 0.52,
      "f1-score": 0.512987012987013,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.495,
  "f1": 0.5702127659574469,
  "f1_macro": 0.47904577691811734,
  "f1_weighted": 0.4790457769181174,
  "precision": 0.4962962962962963,
  "recall": 0.67,
  "balanced_accuracy": 0.495,
  "roc_auc": 0.5286,
  "mcc": -0.010675210253672476,
  "cohen_kappa": -0.010000000000000009,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.49230769230769234,
      "recall": 0.32,
      "f1-score": 0.3878787878787879,
      "support": 100.0
    },
    "1": {
      "precision": 0.4962962962962963,
      "recall": 0.67,
      "f1-score": 0.5702127659574469,
      "support": 100.0
    },
    "accuracy": 0.495,
    "macro avg": {
      "precision": 0.4943019943019943,
      "recall": 0.495,
      "f1-score": 0.47904577691811734,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.4943019943019943,
      "recall": 0.495,
      "f1-score": 0.4790457769181174,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.0008681462614092432,
  "batch_size": 64,
  "weight_decay": 0.0034655401487856626,
  "optimizer": "adam",
  "scheduler": "exponential",
  "epochs": 29,
  "scheduler_gamma": 0.9816228949129914
}
```

**Metrics:**
```json
{
  "accuracy": 0.555,
  "f1": 0.6482213438735178,
  "f1_macro": 0.5213895835013848,
  "f1_weighted": 0.5213895835013848,
  "precision": 0.5359477124183006,
  "recall": 0.82,
  "balanced_accuracy": 0.5549999999999999,
  "roc_auc": 0.5457000000000001,
  "mcc": 0.12971734190749126,
  "cohen_kappa": 0.10999999999999999,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.6170212765957447,
      "recall": 0.29,
      "f1-score": 0.3945578231292517,
      "support": 100.0
    },
    "1": {
      "precision": 0.5359477124183006,
      "recall": 0.82,
      "f1-score": 0.6482213438735178,
      "support": 100.0
    },
    "accuracy": 0.555,
    "macro avg": {
      "precision": 0.5764844945070227,
      "recall": 0.5549999999999999,
      "f1-score": 0.5213895835013848,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5764844945070227,
      "recall": 0.555,
      "f1-score": 0.5213895835013848,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.0003775375880810032,
  "batch_size": 64,
  "weight_decay": 0.0015310377461009263,
  "optimizer": "adam",
  "scheduler": "none",
  "epochs": 44
}
```

**Metrics:**
```json
{
  "accuracy": 0.505,
  "f1": 0.4648648648648649,
  "f1_macro": 0.5021998742928976,
  "f1_weighted": 0.5021998742928976,
  "precision": 0.5058823529411764,
  "recall": 0.43,
  "balanced_accuracy": 0.505,
  "roc_auc": 0.505,
  "mcc": 0.010114434748483472,
  "cohen_kappa": 0.010000000000000009,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5043478260869565,
      "recall": 0.58,
      "f1-score": 0.5395348837209303,
      "support": 100.0
    },
    "1": {
      "precision": 0.5058823529411764,
      "recall": 0.43,
      "f1-score": 0.4648648648648649,
      "support": 100.0
    },
    "accuracy": 0.505,
    "macro avg": {
      "precision": 0.5051150895140665,
      "recall": 0.505,
      "f1-score": 0.5021998742928976,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5051150895140665,
      "recall": 0.505,
      "f1-score": 0.5021998742928976,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen7_6949`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.555,
  "f1": 0.5240641711229946,
  "f1_macro": 0.55311189776807,
  "f1_weighted": 0.5531118977680701,
  "precision": 0.5632183908045977,
  "recall": 0.49,
  "balanced_accuracy": 0.5549999999999999,
  "roc_auc": 0.5507,
  "mcc": 0.11094144982553052,
  "cohen_kappa": 0.10999999999999999,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5486725663716814,
      "recall": 0.62,
      "f1-score": 0.5821596244131455,
      "support": 100.0
    },
    "1": {
      "precision": 0.5632183908045977,
      "recall": 0.49,
      "f1-score": 0.5240641711229946,
      "support": 100.0
    },
    "accuracy": 0.555,
    "macro avg": {
      "precision": 0.5559454785881395,
      "recall": 0.5549999999999999,
      "f1-score": 0.55311189776807,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5559454785881396,
      "recall": 0.555,
      "f1-score": 0.5531118977680701,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.61,
  "f1": 0.5301204819277109,
  "f1_macro": 0.5983935742971888,
  "f1_weighted": 0.5983935742971888,
  "precision": 0.6666666666666666,
  "recall": 0.44,
  "balanced_accuracy": 0.61,
  "roc_auc": 0.6436999999999999,
  "mcc": 0.23393667553251238,
  "cohen_kappa": 0.21999999999999997,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.582089552238806,
      "recall": 0.78,
      "f1-score": 0.6666666666666666,
      "support": 100.0
    },
    "1": {
      "precision": 0.6666666666666666,
      "recall": 0.44,
      "f1-score": 0.5301204819277109,
      "support": 100.0
    },
    "accuracy": 0.61,
    "macro avg": {
      "precision": 0.6243781094527363,
      "recall": 0.61,
      "f1-score": 0.5983935742971888,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.6243781094527363,
      "recall": 0.61,
      "f1-score": 0.5983935742971888,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.006590370694638561,
  "batch_size": 32,
  "weight_decay": 0.0033050917453554506,
  "optimizer": "adam",
  "scheduler": "none",
  "epochs": 21
}
```

**Metrics:**
```json
{
  "accuracy": 0.5,
  "f1": 0.6666666666666666,
  "f1_macro": 0.3333333333333333,
  "f1_weighted": 0.33333333333333326,
  "precision": 0.5,
  "recall": 1.0,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.5,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 100.0
    },
    "1": {
      "precision": 0.5,
      "recall": 1.0,
      "f1-score": 0.6666666666666666,
      "support": 100.0
    },
    "accuracy": 0.5,
    "macro avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.3333333333333333,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.33333333333333326,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.009299036122867223,
  "batch_size": 64,
  "weight_decay": 0.007221394308611984,
  "optimizer": "adam",
  "scheduler": "step",
  "epochs": 28,
  "scheduler_step_size": 7,
  "scheduler_gamma": 0.6125905908483897
}
```

**Metrics:**
```json
{
  "accuracy": 0.5,
  "f1": 0.6666666666666666,
  "f1_macro": 0.3333333333333333,
  "f1_weighted": 0.33333333333333326,
  "precision": 0.5,
  "recall": 1.0,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.5165500000000001,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 100.0
    },
    "1": {
      "precision": 0.5,
      "recall": 1.0,
      "f1-score": 0.6666666666666666,
      "support": 100.0
    },
    "accuracy": 0.5,
    "macro avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.3333333333333333,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.33333333333333326,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen8_4426`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.5,
  "f1": 0.46236559139784944,
  "f1_macro": 0.4975379358858406,
  "f1_weighted": 0.4975379358858406,
  "precision": 0.5,
  "recall": 0.43,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.5512000000000001,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5,
      "recall": 0.57,
      "f1-score": 0.5327102803738317,
      "support": 100.0
    },
    "1": {
      "precision": 0.5,
      "recall": 0.43,
      "f1-score": 0.46236559139784944,
      "support": 100.0
    },
    "accuracy": 0.5,
    "macro avg": {
      "precision": 0.5,
      "recall": 0.5,
      "f1-score": 0.4975379358858406,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5,
      "recall": 0.5,
      "f1-score": 0.4975379358858406,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.67,
  "f1": 0.6071428571428571,
  "f1_macro": 0.6613300492610837,
  "f1_weighted": 0.6613300492610837,
  "precision": 0.75,
  "recall": 0.51,
  "balanced_accuracy": 0.6699999999999999,
  "roc_auc": 0.6882,
  "mcc": 0.3588702812826367,
  "cohen_kappa": 0.33999999999999997,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.6287878787878788,
      "recall": 0.83,
      "f1-score": 0.7155172413793104,
      "support": 100.0
    },
    "1": {
      "precision": 0.75,
      "recall": 0.51,
      "f1-score": 0.6071428571428571,
      "support": 100.0
    },
    "accuracy": 0.67,
    "macro avg": {
      "precision": 0.6893939393939394,
      "recall": 0.6699999999999999,
      "f1-score": 0.6613300492610837,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.6893939393939393,
      "recall": 0.67,
      "f1-score": 0.6613300492610837,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.009560906604065545,
  "batch_size": 32,
  "weight_decay": 0.003225130844317398,
  "optimizer": "adam",
  "scheduler": "none",
  "epochs": 27
}
```

**Metrics:**
```json
{
  "accuracy": 0.5,
  "f1": 0.6666666666666666,
  "f1_macro": 0.3333333333333333,
  "f1_weighted": 0.33333333333333326,
  "precision": 0.5,
  "recall": 1.0,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.4742,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 100.0
    },
    "1": {
      "precision": 0.5,
      "recall": 1.0,
      "f1-score": 0.6666666666666666,
      "support": 100.0
    },
    "accuracy": 0.5,
    "macro avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.3333333333333333,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.33333333333333326,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.008684107350224644,
  "batch_size": 32,
  "weight_decay": 0.00042570962161135904,
  "optimizer": "adam",
  "scheduler": "exponential",
  "epochs": 47,
  "scheduler_gamma": 0.9743246292618605
}
```

**Metrics:**
```json
{
  "accuracy": 0.5,
  "f1": 0.6666666666666666,
  "f1_macro": 0.3333333333333333,
  "f1_weighted": 0.33333333333333326,
  "precision": 0.5,
  "recall": 1.0,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.5,
  "mcc": 0.0,
  "cohen_kappa": 0.0,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 100.0
    },
    "1": {
      "precision": 0.5,
      "recall": 1.0,
      "f1-score": 0.6666666666666666,
      "support": 100.0
    },
    "accuracy": 0.5,
    "macro avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.3333333333333333,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.25,
      "recall": 0.5,
      "f1-score": 0.33333333333333326,
      "support": 200.0
    }
  }
}
```

---

## Dataset: `digen9_7270`

### CNN + NCTD

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.61,
  "f1": 0.6138613861386139,
  "f1_macro": 0.60996099609961,
  "f1_weighted": 0.60996099609961,
  "precision": 0.6078431372549019,
  "recall": 0.62,
  "balanced_accuracy": 0.61,
  "roc_auc": 0.6586,
  "mcc": 0.22004401320440156,
  "cohen_kappa": 0.21999999999999997,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.6122448979591837,
      "recall": 0.6,
      "f1-score": 0.6060606060606061,
      "support": 100.0
    },
    "1": {
      "precision": 0.6078431372549019,
      "recall": 0.62,
      "f1-score": 0.6138613861386139,
      "support": 100.0
    },
    "accuracy": 0.61,
    "macro avg": {
      "precision": 0.6100440176070427,
      "recall": 0.61,
      "f1-score": 0.60996099609961,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.6100440176070429,
      "recall": 0.61,
      "f1-score": 0.60996099609961,
      "support": 200.0
    }
  }
}
```

### CNN + Ours

**Hyperparameters:**
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

**Metrics:**
```json
{
  "accuracy": 0.64,
  "f1": 0.625,
  "f1_macro": 0.6394230769230769,
  "f1_weighted": 0.639423076923077,
  "precision": 0.6521739130434783,
  "recall": 0.6,
  "balanced_accuracy": 0.64,
  "roc_auc": 0.7047000000000001,
  "mcc": 0.28090032386679475,
  "cohen_kappa": 0.28,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.6296296296296297,
      "recall": 0.68,
      "f1-score": 0.6538461538461539,
      "support": 100.0
    },
    "1": {
      "precision": 0.6521739130434783,
      "recall": 0.6,
      "f1-score": 0.625,
      "support": 100.0
    },
    "accuracy": 0.64,
    "macro avg": {
      "precision": 0.6409017713365539,
      "recall": 0.64,
      "f1-score": 0.6394230769230769,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.640901771336554,
      "recall": 0.64,
      "f1-score": 0.639423076923077,
      "support": 200.0
    }
  }
}
```

### EfficientNet + NCTD

**Hyperparameters:**
```json
{
  "learning_rate": 0.0018705059904981038,
  "batch_size": 64,
  "weight_decay": 0.0035157652667013968,
  "optimizer": "adam",
  "scheduler": "exponential",
  "epochs": 34,
  "scheduler_gamma": 0.9720238296877716
}
```

**Metrics:**
```json
{
  "accuracy": 0.545,
  "f1": 0.5603864734299517,
  "f1_macro": 0.5444419413781882,
  "f1_weighted": 0.5444419413781882,
  "precision": 0.5420560747663551,
  "recall": 0.58,
  "balanced_accuracy": 0.5449999999999999,
  "roc_auc": 0.55005,
  "mcc": 0.09022131366062779,
  "cohen_kappa": 0.08999999999999997,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.5483870967741935,
      "recall": 0.51,
      "f1-score": 0.5284974093264249,
      "support": 100.0
    },
    "1": {
      "precision": 0.5420560747663551,
      "recall": 0.58,
      "f1-score": 0.5603864734299517,
      "support": 100.0
    },
    "accuracy": 0.545,
    "macro avg": {
      "precision": 0.5452215857702742,
      "recall": 0.5449999999999999,
      "f1-score": 0.5444419413781882,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.5452215857702742,
      "recall": 0.545,
      "f1-score": 0.5444419413781882,
      "support": 200.0
    }
  }
}
```

### EfficientNet + Ours

**Hyperparameters:**
```json
{
  "learning_rate": 0.0006197268049542244,
  "batch_size": 64,
  "weight_decay": 0.00015790868934478927,
  "optimizer": "sgd",
  "scheduler": "step",
  "epochs": 43,
  "momentum": 0.8920043925381331,
  "scheduler_step_size": 12,
  "scheduler_gamma": 0.4710962634575563
}
```

**Metrics:**
```json
{
  "accuracy": 0.48,
  "f1": 0.10344827586206896,
  "f1_macro": 0.3686255463817387,
  "f1_weighted": 0.3686255463817388,
  "precision": 0.375,
  "recall": 0.06,
  "balanced_accuracy": 0.48,
  "roc_auc": 0.4753,
  "mcc": -0.07372097807744857,
  "cohen_kappa": -0.040000000000000036,
  "num_classes": 2,
  "num_samples": 200,
  "per_class_report": {
    "0": {
      "precision": 0.4891304347826087,
      "recall": 0.9,
      "f1-score": 0.6338028169014085,
      "support": 100.0
    },
    "1": {
      "precision": 0.375,
      "recall": 0.06,
      "f1-score": 0.10344827586206896,
      "support": 100.0
    },
    "accuracy": 0.48,
    "macro avg": {
      "precision": 0.4320652173913043,
      "recall": 0.48,
      "f1-score": 0.3686255463817387,
      "support": 200.0
    },
    "weighted avg": {
      "precision": 0.4320652173913044,
      "recall": 0.48,
      "f1-score": 0.3686255463817388,
      "support": 200.0
    }
  }
}
```

---

