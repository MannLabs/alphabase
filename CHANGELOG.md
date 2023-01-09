# Changelog

Follow the changelog format from https://keepachangelog.com/en/1.0.0/.

## 1.0.0 - TODO Next Release

### Added

- `Percolator` and `SupervisedPercolator` (`alphabase.scoring.ml_scoring.py`).
- `SpecLibFat` for flat fragment dataframes from `SpecLibBase` (`alphabase.spectral_library.flat.py`).
- `SpecLibBase` from TSV library (`alphabase.spectral_library.reader.py`)
- `LOESSRegression` for recalibration stuffs (`alphabase.statistics.regression.py`)
- Auto include contaminants.fasta with `include_contaminants` in `alphabase.protein.fasta.SpecLibFasta`.

## 0.4.0 - 2022.12.28

### Changed

- `frag_end_idx` -> `frag_stop_idx`

## 0.3.0 - 2022.12.27

### Added

- `alphabase.protein.fasta.SpecLibFasta.add_peptide_labeling()` supports mDIA.
- Move `alphabase.spectral_library.translate` module (.py) from alphapeptdeep to alphabase.
- `alphabase.constants.modification.add_new_modifications()` to add user-defined modifications that are not in UniMod.
- `alphabase.psm_reader.psm_reader.PSMReaderBase.add_modification_mapping()` to support arbitrary modifications of different search engines.

### Changed

- Use sphinx and readthedocs for documentation. nbdev is no longer for documentation, it is only used for unit testing (`nbdev_test`).
- FastaLib to SpecLibFasta.
- Protease: `trypsin`==`trypsin/p`, add `trypsin_not_p` will not cleave peptides before N-term of P.

## 0.2.0

First official release.

## 0.0.1

FEAT: Initial creation of AlphaBase.
