# VBench metadata

This directory vendors the two VBench text-to-video metadata files required by
the MeanFlowNFT evaluator:

- `VBench_aug_full_info.json`: augmented generation prompts.
- `VBench_full_info.json`: official prompts and metric dimensions.

They are distributed with the VBench/AnyFlow evaluation protocol. The evaluator
uses only repository-relative paths; no external AnyFlow checkout is required.
