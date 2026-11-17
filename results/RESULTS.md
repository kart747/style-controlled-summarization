# Results

## Average over styles

| System | R-1 | R-2 | R-Lsum | Style acc. (lexical) | Style acc. (markers) |
|---|---:|---:|---:|---:|---:|
| Reference (human) | 100.0 | 100.0 | 100.0 | 88% | 91% |
| Lead-3 (extractive) | 25.0 | 6.8 | 21.9 | 72% | 51% |
| TextRank (extractive) | 27.9 | 8.0 | 24.4 | 75% | 48% |
| FLAN-T5-small zero-shot + prompt | 24.3 | 6.8 | 21.9 | 68% | 60% |
| FLAN-T5-small fine-tuned + style prefix (2 epochs) | 34.2 | 10.2 | 31.2 | 72% | 75% |
| FLAN-T5-small fine-tuned + style prefix (4 epochs) | 33.6 | 9.8 | 30.5 | 77% | 75% |

## Per-style quality (each test example in its own style)

| System | Style | R-1 | R-2 | R-Lsum | Style acc. (lexical) | Style acc. (markers) | Words | FK grade |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Reference (human) | academic | 100.0 | 100.0 | 100.0 | 86% | 74% | 240 | 16.3 |
| Reference (human) | simple | 100.0 | 100.0 | 100.0 | 76% | 90% | 177 | 15.9 |
| Reference (human) | formal | 100.0 | 100.0 | 100.0 | 100% | 100% | 41 | 8.1 |
| Reference (human) | casual | 100.0 | 100.0 | 100.0 | 92% | 100% | 20 | 6.3 |
| Lead-3 (extractive) | academic | 23.8 | 5.7 | 21.6 | 12% | 0% | 74 | 16.4 |
| Lead-3 (extractive) | simple | 26.5 | 6.8 | 23.2 | 86% | 76% | 71 | 16.5 |
| Lead-3 (extractive) | formal | 32.6 | 12.4 | 30.5 | 92% | 48% | 62 | 11.6 |
| Lead-3 (extractive) | casual | 17.2 | 2.3 | 12.4 | 100% | 80% | 59 | 7.2 |
| TextRank (extractive) | academic | 30.0 | 8.9 | 27.0 | 56% | 0% | 89 | 17.1 |
| TextRank (extractive) | simple | 33.7 | 9.0 | 29.6 | 52% | 96% | 89 | 17.0 |
| TextRank (extractive) | formal | 30.4 | 10.4 | 27.8 | 92% | 28% | 77 | 11.6 |
| TextRank (extractive) | casual | 17.4 | 3.7 | 13.2 | 100% | 66% | 72 | 8.2 |
| FLAN-T5-small zero-shot + prompt | academic | 25.4 | 7.1 | 23.0 | 30% | 0% | 62 | 16.7 |
| FLAN-T5-small zero-shot + prompt | simple | 26.5 | 6.8 | 23.5 | 60% | 64% | 55 | 16.1 |
| FLAN-T5-small zero-shot + prompt | formal | 31.6 | 10.7 | 29.1 | 94% | 78% | 42 | 10.0 |
| FLAN-T5-small zero-shot + prompt | casual | 13.8 | 2.9 | 12.0 | 90% | 98% | 11 | 4.3 |
| FLAN-T5-small fine-tuned + style prefix (2 epochs) | academic | 41.0 | 11.7 | 37.9 | 26% | 0% | 159 | 15.6 |
| FLAN-T5-small fine-tuned + style prefix (2 epochs) | simple | 40.2 | 10.3 | 36.4 | 68% | 100% | 141 | 15.3 |
| FLAN-T5-small fine-tuned + style prefix (2 epochs) | formal | 34.4 | 13.0 | 32.4 | 94% | 100% | 45 | 8.7 |
| FLAN-T5-small fine-tuned + style prefix (2 epochs) | casual | 21.2 | 5.8 | 18.0 | 98% | 100% | 15 | 4.8 |
| FLAN-T5-small fine-tuned + style prefix (4 epochs) | academic | 40.5 | 11.6 | 37.4 | 36% | 0% | 162 | 15.1 |
| FLAN-T5-small fine-tuned + style prefix (4 epochs) | simple | 40.0 | 10.2 | 35.9 | 80% | 100% | 139 | 15.4 |
| FLAN-T5-small fine-tuned + style prefix (4 epochs) | formal | 33.0 | 11.8 | 30.8 | 96% | 100% | 47 | 8.7 |
| FLAN-T5-small fine-tuned + style prefix (4 epochs) | casual | 20.8 | 5.6 | 17.9 | 96% | 100% | 15 | 4.8 |

## Style controllability

**FLAN-T5-small zero-shot + prompt** - same documents, every style requested

| Requested style | Style acc. (lexical) | Style acc. (markers) | Words | FK grade | Contractions % | 1st-person % |
|---|---:|---:|---:|---:|---:|---:|
| academic | 10% | 0% | 66 | 14.1 | 1.10 | 2.17 |
| simple | 30% | 40% | 62 | 12.0 | 1.10 | 2.23 |
| formal | 40% | 47% | 36 | 11.4 | 1.97 | 2.90 |
| casual | 37% | 93% | 16 | 8.2 | 1.55 | 3.05 |

**FLAN-T5-small fine-tuned + style prefix (2 epochs)** - same documents, every style requested

| Requested style | Style acc. (lexical) | Style acc. (markers) | Words | FK grade | Contractions % | 1st-person % |
|---|---:|---:|---:|---:|---:|---:|
| academic | 10% | 0% | 106 | 11.4 | 1.80 | 3.21 |
| simple | 27% | 47% | 93 | 10.8 | 1.94 | 3.41 |
| formal | 40% | 53% | 45 | 9.9 | 1.72 | 3.32 |
| casual | 27% | 33% | 32 | 10.1 | 0.96 | 4.10 |

**FLAN-T5-small fine-tuned + style prefix (4 epochs)** - same documents, every style requested

| Requested style | Style acc. (lexical) | Style acc. (markers) | Words | FK grade | Contractions % | 1st-person % |
|---|---:|---:|---:|---:|---:|---:|
| academic | 20% | 0% | 105 | 11.1 | 1.49 | 3.37 |
| simple | 33% | 57% | 94 | 11.2 | 1.95 | 3.01 |
| formal | 40% | 63% | 45 | 9.3 | 1.42 | 3.42 |
| casual | 27% | 33% | 32 | 10.0 | 0.72 | 4.21 |

