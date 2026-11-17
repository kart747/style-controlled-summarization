# Results

## Average over styles

| System | R-1 | R-2 | R-Lsum | Style acc. |
|---|---:|---:|---:|---:|
| Reference (human) | 100.0 | 100.0 | 100.0 | 88% |
| Lead-3 (extractive) | 25.0 | 6.8 | 21.9 | 72% |
| TextRank (extractive) | 27.9 | 8.0 | 24.4 | 75% |
| FLAN-T5-small zero-shot + prompt | 24.3 | 6.8 | 21.9 | 68% |
| FLAN-T5-small fine-tuned + style prefix | 34.2 | 10.2 | 31.2 | 72% |

## Per-style quality (each test example in its own style)

| System | Style | R-1 | R-2 | R-Lsum | Style acc. | Words | FK grade |
|---|---|---:|---:|---:|---:|---:|---:|
| Reference (human) | academic | 100.0 | 100.0 | 100.0 | 86% | 240 | 16.3 |
| Reference (human) | simple | 100.0 | 100.0 | 100.0 | 76% | 177 | 15.9 |
| Reference (human) | formal | 100.0 | 100.0 | 100.0 | 100% | 41 | 8.1 |
| Reference (human) | casual | 100.0 | 100.0 | 100.0 | 92% | 20 | 6.3 |
| Lead-3 (extractive) | academic | 23.8 | 5.7 | 21.6 | 12% | 74 | 16.4 |
| Lead-3 (extractive) | simple | 26.5 | 6.8 | 23.2 | 86% | 71 | 16.5 |
| Lead-3 (extractive) | formal | 32.6 | 12.4 | 30.5 | 92% | 62 | 11.6 |
| Lead-3 (extractive) | casual | 17.2 | 2.3 | 12.4 | 100% | 59 | 7.2 |
| TextRank (extractive) | academic | 30.0 | 8.9 | 27.0 | 56% | 89 | 17.1 |
| TextRank (extractive) | simple | 33.7 | 9.0 | 29.6 | 52% | 89 | 17.0 |
| TextRank (extractive) | formal | 30.4 | 10.4 | 27.8 | 92% | 77 | 11.6 |
| TextRank (extractive) | casual | 17.4 | 3.7 | 13.2 | 100% | 72 | 8.2 |
| FLAN-T5-small zero-shot + prompt | academic | 25.4 | 7.1 | 23.0 | 30% | 62 | 16.7 |
| FLAN-T5-small zero-shot + prompt | simple | 26.5 | 6.8 | 23.5 | 60% | 55 | 16.1 |
| FLAN-T5-small zero-shot + prompt | formal | 31.6 | 10.7 | 29.1 | 94% | 42 | 10.0 |
| FLAN-T5-small zero-shot + prompt | casual | 13.8 | 2.9 | 12.0 | 90% | 11 | 4.3 |
| FLAN-T5-small fine-tuned + style prefix | academic | 41.0 | 11.7 | 37.9 | 26% | 159 | 15.6 |
| FLAN-T5-small fine-tuned + style prefix | simple | 40.2 | 10.3 | 36.4 | 68% | 141 | 15.3 |
| FLAN-T5-small fine-tuned + style prefix | formal | 34.4 | 13.0 | 32.4 | 94% | 45 | 8.7 |
| FLAN-T5-small fine-tuned + style prefix | casual | 21.2 | 5.8 | 18.0 | 98% | 15 | 4.8 |

## Style controllability

**FLAN-T5-small zero-shot + prompt** - same documents, every style requested

| Requested style | Classified as requested | Words | FK grade | Contractions % | 1st-person % |
|---|---:|---:|---:|---:|---:|
| academic | 10% | 66 | 14.1 | 1.10 | 2.17 |
| simple | 30% | 62 | 12.0 | 1.10 | 2.23 |
| formal | 40% | 36 | 11.4 | 1.97 | 2.90 |
| casual | 37% | 16 | 8.2 | 1.55 | 3.05 |

**FLAN-T5-small fine-tuned + style prefix** - same documents, every style requested

| Requested style | Classified as requested | Words | FK grade | Contractions % | 1st-person % |
|---|---:|---:|---:|---:|---:|
| academic | 10% | 106 | 11.4 | 1.80 | 3.21 |
| simple | 27% | 93 | 10.8 | 1.94 | 3.41 |
| formal | 40% | 45 | 9.9 | 1.72 | 3.32 |
| casual | 27% | 32 | 10.1 | 0.96 | 4.10 |

