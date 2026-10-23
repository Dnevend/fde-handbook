| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | BLAS,MTL   |       4 |           pp512 |        209.42 ± 0.16 |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | BLAS,MTL   |       4 |           tg128 |         17.23 ± 0.05 |

build: 620d2225c (11669)
