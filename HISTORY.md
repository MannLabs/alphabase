# Changelog

Follow the changelog format from https://keepachangelog.com/en/1.0.0/.

## 1.0.0 - TODO Next Release

## 0.3.0 - 2022.12.27

### Added

- `alphabase.protein.fasta.SpecLibFasta.add_peptide_labeling()` supports mDIA.
- `alphabase.spectral_library.flat.SpecLibFlat` class for flat spec_lib.
- Move `alphabase.spectral_library.translate` module (.py) from alphapeptdeep to alphabase.
- `alphabase.constants.modification.add_new_modifications()` to add user-defined modifications that are not in UniMod.
- `alphabase.psm_reader.psm_reader.PSMReaderBase.add_modification_mapping()` to support arbitrary modifications of different search engines.
- `alphabase.statistics.regression.LOESSRegression`
- `alphabase.scoring` module added but not finished yet.

### Changed

- Use sphinx and readthedocs for documentation. nbdev is no longer for documentation, it is only used for unit testing (`nbdev_test`).
- FastaLib to SpecLibFasta.
- Protease: `trypsin`==`trypsin/p`, add `trypsin_not_p` will not cleave peptides before N-term of P.

## 0.2.0

First official release.

## 0.0.1

FEAT: Initial creation of AlphaBase.
