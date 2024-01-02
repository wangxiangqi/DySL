# DySL

## Dynamic graph datasets

| Dataset       | Node    | Edge       | Attribute | Heter | Time-span        | Label | Source                                                                          |
| ------------- | ------- | ---------- | --------- | ----- | ---------------- | ----- | ------------------------------------------------------------------------------- |
| bitcoin-Alpha | 3,777   | 24,173     | -         | -     | 136              | -     | [EvolveGCN AAAI20](https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html)     |
| MovieLens-10M | 20,537  | 43,760     | -         | -     | 13               | -     | [DySAT WSDM20](https://drive.google.com/open?id=1TAWipN2y6uYf5BRtlKp-NY2BT3znH1YB) |
| Higgs Twitter | 456,626 | 14,855,842 | -         | edge  | 7 Days           | -     | [SNAP standford](http://snap.stanford.edu/data/higgs-twitter.html)                 |
| PPI           | 16,458  | 144,033    | -         | -     | 37               | -     | [tNodeEmbedding IJCAI19](https://github.com/urielsinger/tNodeEmbed)                |
| yelp          |         |            |           |       | Will check later |       |                                                                                 |
| Enron         |         |            |           |       | Will check later |       |                                                                                 |

2023.12.30 update: Finished preprocessing raw data into same format, npz format with multigraph, initially dividing the multigraphs into 10 respective ones due to the datetime. Can be read with readnpz.py

## Voting Ensembleing:
