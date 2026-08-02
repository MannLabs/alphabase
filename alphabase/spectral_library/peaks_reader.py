"""Reader for PEAKS Studio DIA spectral library exports.

PEAKS stores one row per precursor with all fragment peaks packed into a single
``Peaks List`` column; :class:`PeaksLibraryReader` reads this into a standard
alphabase spectral library.
"""

import re
import warnings
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from alphabase.constants.modification import MOD_MASS, ModificationKeys
from alphabase.psm_reader.keys import PsmDfCols
from alphabase.psm_reader.psm_reader import psm_reader_yaml
from alphabase.spectral_library.reader import LibraryReaderBase

# Column names in the raw PEAKS export.
_PEAKS_SEQUENCE_COLUMN = "Sequence (backbone)"
_PEAKS_MOD_COLUMN = "Modifications"
_PEAKS_PEAKS_COLUMN = "Peaks List"

# Internal (synthesized) long-format fragment column names, mapped in psm_reader.yaml.
_FRAGMENT_MZ = "FragmentMz"
_FRAGMENT_INTENSITY = "FragmentIntensity"
_FRAGMENT_TYPE = "FragmentType"
_FRAGMENT_CHARGE = "FragmentCharge"
_FRAGMENT_NUMBER = "FragmentNumber"
_FRAGMENT_LOSS_TYPE = "FragmentLossType"

# A fragment annotation, e.g. 'b6', 'y13[2+]', 'b12-H2O', 'y5-NH3[3+]'.
_FRAGMENT_ANNOTATION_PATTERN = re.compile(
    r"^([abcxyz])(\d+)(?:-(H2O|NH3))?(?:\[(\d+)\+\])?$"
)

# A modification entry, e.g. '9-Oxidation (M)-(15.99)', '0-Acetylation (Protein N-term)-(42.01)'.
_MOD_ENTRY_PATTERN = re.compile(r"^(\d+)-(.+)-\(([0-9.]+)\)$")

# The fragment types present in PEAKS DIA libraries: b/y ions up to charge 3 with
# H2O and NH3 neutral losses. Used as the default so none of the file's peaks are
# silently dropped (the LibraryReaderBase default omits H2O/NH3 losses).
DEFAULT_PEAKS_CHARGED_FRAG_TYPES = [
    "b_z1",
    "b_z2",
    "b_z3",
    "y_z1",
    "y_z2",
    "y_z3",
    "b_H2O_z1",
    "b_H2O_z2",
    "b_H2O_z3",
    "y_H2O_z1",
    "y_H2O_z2",
    "y_H2O_z3",
    "b_NH3_z1",
    "b_NH3_z2",
    "b_NH3_z3",
    "y_NH3_z1",
    "y_NH3_z2",
    "y_NH3_z3",
]


def _parse_fragment_annotation(
    annotation: str,
) -> Optional[Tuple[str, int, int, str]]:
    """Parse a PEAKS fragment annotation into its components.

    Parameters
    ----------
    annotation : str
        Fragment annotation like 'b6', 'y13[2+]', 'b12-H2O', 'y5-NH3[3+]'.

    Returns
    -------
    tuple or None
        (fragment_type, fragment_number, fragment_charge, loss_type), e.g.
        ('y', 13, 2, ''). Returns None if the annotation cannot be parsed.

    """
    match = _FRAGMENT_ANNOTATION_PATTERN.match(annotation)
    if match is None:
        return None
    frag_type, number, loss, charge = match.groups()
    return frag_type, int(number), int(charge) if charge else 1, loss or ""


def _match_mod_by_mass(
    mass: float,
    aa_or_term: str,
    mass_mapped_mods: List[str],
    mod_mass_tol: float,
) -> Optional[str]:
    """Resolve a PEAKS modification to an alphabase name by mass and residue/terminus.

    Parameters
    ----------
    mass : float
        Mass shift from the PEAKS entry (e.g. 15.99).
    aa_or_term : str
        The amino acid at the modification site, or ``Any_N-term`` / ``Any_C-term``.
    mass_mapped_mods : List[str]
        Candidate alphabase mod names (e.g. ['Oxidation@M', 'Carboxymethyl@C']).
    mod_mass_tol : float
        Mass tolerance in Daltons.

    Returns
    -------
    str or None
        The matched alphabase modification name (e.g. 'Oxidation@M'), or ``None``
        if no candidate matches the mass and residue/terminus within tolerance.

    """
    is_terminal = aa_or_term in (
        ModificationKeys.ANY_N_TERM,
        ModificationKeys.ANY_C_TERM,
    )
    term_keyword = "N-term" if aa_or_term == ModificationKeys.ANY_N_TERM else "C-term"

    best_match = None
    best_mass_diff = float("inf")
    for mod_name in mass_mapped_mods:
        if mod_name not in MOD_MASS:
            continue
        mass_diff = abs(mass - MOD_MASS[mod_name])
        if mass_diff >= mod_mass_tol or mass_diff >= best_mass_diff:
            continue

        mod_site = mod_name.split(ModificationKeys.SITE_SEPARATOR)[1]
        matches = (
            mod_site.endswith(term_keyword) if is_terminal else mod_site == aa_or_term
        )
        if matches:
            best_match = mod_name
            best_mass_diff = mass_diff

    return best_match


class PeaksLibraryReader(LibraryReaderBase):
    """Read a PEAKS Studio DIA spectral library into an alphabase spectral library.

    Examples
    --------
    >>> reader = PeaksLibraryReader()
    >>> reader.import_file("peak_dia_db_library.tsv")
    >>> reader.precursor_df           # precursors: sequence, mods, charge, rt, ...
    >>> reader.fragment_intensity_df  # per-fragment intensities
    >>> reader.fragment_mz_df         # per-fragment m/z

    """

    _reader_type = "peaks_library"

    def __init__(  # noqa: PLR0913 many arguments
        self,
        charged_frag_types: Optional[List[str]] = None,
        column_mapping: Optional[dict] = None,
        modification_mapping: Optional[dict] = None,
        fdr: float = 0.01,
        fixed_C57: bool = False,  # noqa: N803, FBT001, FBT002
        mod_seq_columns: Optional[List[str]] = None,
        rt_unit: Optional[str] = None,
        precursor_mz_min: float = 400,
        precursor_mz_max: float = 2000,
        decoy: Optional[str] = None,
        **kwargs,
    ):
        """Create a PEAKS library reader.

        Parameters are those of :class:`LibraryReaderBase`. By default
        ``charged_frag_types`` covers b/y ions up to charge 3 with H2O and NH3
        neutral losses, matching the fragment types PEAKS reports.
        """
        super().__init__(
            charged_frag_types=(
                DEFAULT_PEAKS_CHARGED_FRAG_TYPES
                if charged_frag_types is None
                else charged_frag_types
            ),
            column_mapping=column_mapping,
            modification_mapping=modification_mapping,
            fdr=fdr,
            fixed_C57=fixed_C57,
            mod_seq_columns=mod_seq_columns,
            rt_unit=rt_unit,
            precursor_mz_min=precursor_mz_min,
            precursor_mz_max=precursor_mz_max,
            decoy=decoy,
            **kwargs,
        )
        reader_config = psm_reader_yaml[self._reader_type]
        self._mass_mapped_mods = reader_config["mass_mapped_mods"]
        self._mod_mass_tol = reader_config.get("mod_mass_tol", 0.1)

    def _load_file(self, filename: str) -> pd.DataFrame:
        """Load the wide PEAKS export and explode it into long (per-fragment) format."""
        # Explicit tab separator: the 'Activation Mode' value contains a comma
        # (e.g. "CID, CAD(y and b ions)") which would fool delimiter sniffing.
        wide_df = pd.read_csv(filename, sep="\t", keep_default_na=False)
        return self._explode_peaks_list(wide_df)

    def _explode_peaks_list(self, wide_df: pd.DataFrame) -> pd.DataFrame:
        """Explode the packed ``Peaks List`` column into one row per fragment."""
        sequences = wide_df[_PEAKS_SEQUENCE_COLUMN].to_numpy()
        charges = wide_df["z"].to_numpy()
        precursor_mzs = wide_df["m/z"].to_numpy()
        rts = wide_df["rt (seconds)"].to_numpy()
        modifications = wide_df[_PEAKS_MOD_COLUMN].to_numpy()
        peaks_lists = wide_df[_PEAKS_PEAKS_COLUMN].to_numpy()

        records = []
        for sequence, charge, precursor_mz, rt, mod_str, peaks_list in zip(
            sequences, charges, precursor_mzs, rts, modifications, peaks_lists
        ):
            if not peaks_list:
                continue
            for peak in peaks_list.split(";"):
                fields = peak.split(":")
                if len(fields) != 3:  # noqa: PLR2004 mz:intensity:annotation
                    continue
                mz_str, intensity_str, annotation = fields
                parsed = _parse_fragment_annotation(annotation)
                if parsed is None:
                    continue
                frag_type, frag_number, frag_charge, loss_type = parsed
                records.append(
                    (
                        sequence,
                        charge,
                        precursor_mz,
                        rt,
                        mod_str,
                        float(mz_str),
                        float(intensity_str),
                        frag_type,
                        frag_charge,
                        frag_number,
                        loss_type,
                    )
                )

        return pd.DataFrame(
            records,
            columns=[
                _PEAKS_SEQUENCE_COLUMN,
                "z",
                "m/z",
                "rt (seconds)",
                _PEAKS_MOD_COLUMN,
                _FRAGMENT_MZ,
                _FRAGMENT_INTENSITY,
                _FRAGMENT_TYPE,
                _FRAGMENT_CHARGE,
                _FRAGMENT_NUMBER,
                _FRAGMENT_LOSS_TYPE,
            ],
        )

    def _load_modifications(self, origin_df: pd.DataFrame) -> None:
        """Parse the PEAKS ``Modifications`` column into ``mods``/``mod_sites``.

        Precursors whose modifications are not in ``mass_mapped_mods`` are dropped
        and reported together in a single warning.
        """
        cache = {}
        unknown_mods: Set[str] = set()
        mods_list = []
        sites_list = []
        for sequence, mod_str in zip(
            origin_df[_PEAKS_SEQUENCE_COLUMN].to_numpy(),
            origin_df[_PEAKS_MOD_COLUMN].to_numpy(),
        ):
            key = (sequence, mod_str)
            if key not in cache:
                cache[key] = self._parse_peaks_modifications(sequence, mod_str)
            mods, sites, unknown = cache[key]
            unknown_mods.update(unknown)
            # NaN mods flag the precursor for removal by the inherited _post_process
            mods_list.append(mods if mods is not None else np.nan)
            sites_list.append(sites if sites is not None else np.nan)

        self._psm_df[PsmDfCols.MODS] = mods_list
        self._psm_df[PsmDfCols.MOD_SITES] = sites_list

        if unknown_mods:
            warnings.warn(
                f"Unknown PEAKS modifications: {sorted(unknown_mods)}. Precursors "
                f"with these modifications will be removed. Add them to "
                f"'mass_mapped_mods' for 'peaks_library' in psm_reader.yaml to keep them."
            )

    def _parse_peaks_modifications(
        self, sequence: str, mod_str: str
    ) -> Tuple[Optional[str], Optional[str], Set[str]]:
        """Parse a single PEAKS ``Modifications`` cell.

        Positions are 0-based indices into the backbone sequence; a mod whose name
        contains 'N-term'/'C-term' is terminal regardless of position.

        Returns
        -------
        tuple
            ``(mods, mod_sites, unknown)``. If every modification resolves, ``mods``
            and ``mod_sites`` are the ``;``-joined strings and ``unknown`` is empty.
            If any modification is unresolved, ``mods``/``mod_sites`` are ``None``
            (signalling the precursor should be dropped) and ``unknown`` holds the
            unresolved names for aggregated warning.

        """
        if not mod_str:
            return "", "", set()

        mods = []
        sites = []
        unknown: Set[str] = set()
        for raw_entry in mod_str.split(";"):
            entry = raw_entry.strip()
            if not entry:
                continue
            match = _MOD_ENTRY_PATTERN.match(entry)
            if match is None:
                raise ValueError(
                    f"Invalid PEAKS modification entry '{entry}': expected "
                    f"'<position><Name>-(<mass>)' (e.g. '9-Oxidation (M)-(15.99)')."
                )
            position, name, mass_str = match.groups()
            mass = float(mass_str)

            if "N-term" in name:
                site = "0"
                aa_or_term = ModificationKeys.ANY_N_TERM
            elif "C-term" in name:
                site = "-1"
                aa_or_term = ModificationKeys.ANY_C_TERM
            else:
                site = str(int(position) + 1)  # 0-based -> 1-based
                aa_or_term = sequence[int(position)]

            mod = _match_mod_by_mass(
                mass, aa_or_term, self._mass_mapped_mods, self._mod_mass_tol
            )
            if mod is None:
                unknown.add(f"{name} ({mass_str}) @ {aa_or_term}")
            else:
                mods.append(mod)
                sites.append(site)

        if unknown:
            return None, None, unknown

        return (
            ModificationKeys.SEPARATOR.join(mods),
            ModificationKeys.SEPARATOR.join(sites),
            unknown,
        )

    def _translate_modifications(self) -> None:
        """No-op: modifications are already resolved in :meth:`_load_modifications`."""
