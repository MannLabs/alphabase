# Changelog

Follow the changelog format from https://keepachangelog.com/en/1.0.0/.

## 1.1.0 - 2023.05.03

### Added

- Separate `library_reader_base` in `psm_reader.yaml` config for `LibraryReaderBase`.

### Changed

- Enable customizing dtypes of peak mz and intensty values.
- `SWATHLibraryReader` to `LibraryBaseReader` in `alphabase.spectral_library.reader`.
- New `LibraryReaderBase._get_fragment_intensity` implementation which is called at the end of the parsing process in `PSMReaderBase._post_process`. This allows it to operate only on the translated column names. By default, all non-fragment columns will be grouped and part of the final output.
- `SpecLibBase.copy()` for copying spectral libraries including all attributes.
- `SpecLibBase.append()` for appending spectral libraries while maintaining the fragment index mapping.

## 1.0.2 - 2023.02.10

### Changed

- Use `flat_frag_start/stop_idxes` for  `SpecLibFlat`.
- Check if group name is `*_df` for DataFrame in hdf files; remove `is_pd_dataframe` attr.

## 1.0.1 - 2023.01.15

### Added

- 'protein_reverse' decoy added in `alphabase.protein.protein_level_decoy.py`.

## 1.0.0 - 2023.01.10

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
