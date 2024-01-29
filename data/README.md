# Dataset

The dataset in F-Eval. The dataset is in the format of [jsonlines](https://jsonlines.org/).

```tree
data
  ├─language_quality  // The overall dataset
  │  ├─commonsense
  │  ├─expression
  │  └─logic
  ├─select_data  // The selected dataset for correlation and distinction
  │   ├─commonsense
  │   ├─expression
  │   ├─logic_human
  │   └─reference  // The reference dataset for correlation on reference-free subjective tasks
  └─param.json  // The parameters for the normalization of each subdataset
```