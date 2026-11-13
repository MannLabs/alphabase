# Fix export regressions without new tests

## Summary

Change production code and documentation only. Fix stream handling, compression, empty-input behavior, and dataframe memory use. Adopt exact numeric round-trip preservation instead of pandas-compatible number spelling.

**Do not add or modify tests, fixtures, snapshots, or test scripts.**

## Implementation

In `alphabase/spectral_library/translate.py`:

- Open path destinations through pandas’ compression-aware binary handle API, preserving suffix inference such as `.tsv.gz`.
- Recognize text-mode wrappers and duck-typed text sinks; decode Arrow buffers only for text destinations. Preserve binary writes, caller-owned stream lifetimes, and error propagation.
- Return correctly typed empty fragment arrays before reshaping empty precursor input, restoring empty dataframe merge results.
- Batch dataframe fragment selection internally at 10,000 precursors. Reuse source arrays, preserve absolute fragment offsets and precursor indices, and concatenate only selected fragments.
- Simplify floating-point serialization to Arrow conversion with explicit NaN-to-empty mapping. Remove `.0` restoration and magnitude-dependent formatting. Preserve exact source-dtype round trips, signed zero, and infinity signs without downcasting or arbitrary rounding.

Keep public signatures, fragment selection, tie ordering, and threaded output ordering unchanged.

Update the export documentation to describe numerical preservation, permitted spelling differences, missing values, and supported output destinations. Prepare replacement text for the PR’s byte-equivalence claim.

## Validation

- Run existing spectral-library and fragment tests unchanged.
- Reuse existing review probes and benchmarks to verify the reported regressions, numeric preservation, peak memory reduction, and export performance.
- Report measured results and the remaining regression-coverage gap. Do not create additional validation infrastructure.

## Delivery

Make local code and documentation changes only. No new dependencies or public configuration. Do not commit, push, or edit the GitHub PR unless separately requested.

## Implementation results

Implemented locally on 2026-11-13. No tests, fixtures, snapshots, or scripts were added or changed.

- Existing `tests/unit/spectral_library` and `tests/test_fragment.py`: 43 passed, with 13 existing warnings.
- Repository-pinned Ruff lint and formatting checks passed for the changed Python file.
- Reused runtime probes passed for plain, gzip, bz2, xz, zip, tar, and tar.gz paths in serial and threaded modes; binary, text, wrapped, and duck-typed streams; caller-owned stream lifetimes; propagated write errors; and typed empty merges.
- One million seeded float32/float64 values round-tripped exactly in their source dtype. Separate edge probes preserved signed zero, infinity signs, smallest subnormals, and maximum finite values; NaN produced an empty field.
- Dataframe output matched the preceding implementation with reordered precursors, gaps in fragment offsets, and non-default indices.

The existing 50,000-precursor memory benchmark produced the same 600,000 selected rows, indices, and column dtypes. Peak RSS fell from 1,273.49 MiB to 585.66 MiB (54.0%). Merge time was 1.991 s before and 1.689 s after, measured once per version in separate processes.

The existing synthetic 100,000-precursor export benchmark was repeated five times per version and mode, after warm-up, alternating versions. Output went to `/dev/null`, so these timings cover rendering and writes but exclude storage throughput. Times are median seconds, with population standard deviation in parentheses:

| Export mode | Before | After |
| --- | ---: | ---: |
| Serial | 2.315 (0.012) | 2.170 (0.021) |
| Four threads | 1.385 (0.045) | 1.359 (0.051) |

Validation used Python 3.11, NumPy 2.4.6, pandas 3.1.1, and PyArrow 26.0.1. The existing suite still lacks direct regression coverage for these TSV cases. Runtime probes do not establish downstream DIA-NN compatibility.

## Prepared PR wording

Replace the byte-equivalence claim with:

> TSV serialization preserves exported floating-point values exactly when parsed back into their original dtype, including signed zero and infinity signs, and writes NaN as an empty field. Numeric spelling may differ from pandas output, so byte equivalence is no longer guaranteed. Fragment selection, normalization, tie ordering, and ordered batch output remain unchanged.
