import re
from collections import defaultdict

import pytest

pytest.importorskip("rdkit", reason="rdkit not installed, skipping smiles tests")

from rdkit import Chem  # noqa: E402
from rdkit.Chem import rdCIPLabeler  # noqa: E402

from alphabase.constants.atom import ChemicalCompositonFormula  # noqa: E402
from alphabase.smiles.smiles import (  # noqa: E402
    C_TERM_ANCHOR,
    N_TERM_ANCHOR,
    AminoAcidModifier,
    merge_at_anchors,
    parse_smiles,
)

aa_modifier = AminoAcidModifier()
modify_amino_acid = aa_modifier.modify_amino_acid
aa_smiles = aa_modifier.aa_smiles
n_term_modifications = aa_modifier.n_term_modifications
c_term_modifications = aa_modifier.c_term_modifications
ptm_dict = aa_modifier.ptm_dict

ALANINE = "[NH2:1][C@@H](C)C(=O)[OH:2]"
PROLINE = "[NH:1]1CCC[C@H]1C(=O)[OH:2]"
S_PHENYLETHYLAMINE = "[NH2:1][C@@H](C)c1ccccc1"

# Residues whose alpha-carbon stereo is not written in modification.tsv.
UNDEFINED_ALPHA_CARBON_STEREO = {
    "Carbamidomethyl@C",
    "Carbamidomethyl@M",
    "Phospho@Y",
    "Phospho@T",
    "Pyridylethyl@C",
    "Oxidation@M",
    "Propionyl@K",
    "Succinyl@K",
    "Butyryl@K",
    "Crotonyl@K",
    "hydroxyisobutyryl@K",
    "Pro->3,4-Dehydro-L-proline@P",
}


def canonical(smiles: str) -> str:
    """Canonicalize, so that different writings of one molecule compare equal."""
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def cip_labels(mol: Chem.Mol) -> list[str]:
    """Return the sorted R/S labels of all stereocentres, independent of atom order."""
    mol = Chem.Mol(mol)
    rdCIPLabeler.AssignCIPLabels(mol)
    return sorted(
        atom.GetProp("_CIPCode") for atom in mol.GetAtoms() if atom.HasProp("_CIPCode")
    )


def random_writings(smiles: str, n: int = 8) -> list[str]:
    """The same molecule written with its atoms, and hence bonds, in different orders."""
    return list(
        Chem.MolToRandomSmilesVect(Chem.MolFromSmiles(smiles), n, randomSeed=42)
    )


def alpha_carbon_idx(residue: Chem.Mol) -> int:
    """Return the index of the alpha carbon, the atom linking the amine and carboxyl anchors."""
    amine = next(a for a in residue.GetAtoms() if a.GetAtomMapNum() == N_TERM_ANCHOR)
    return next(
        carbon.GetIdx()
        for carbon in amine.GetNeighbors()
        for carboxyl in carbon.GetNeighbors()
        if any(o.GetAtomMapNum() == C_TERM_ANCHOR for o in carboxyl.GetNeighbors())
    )


@pytest.mark.parametrize(
    "residue, n_term_mod, c_term_mod, expected",
    [
        ("A", None, None, "N[C@@H](C)C(=O)O"),
        ("A", "Acetyl@Any_N-term", None, "CC(=O)N[C@@H](C)C(=O)O"),
        ("A", None, "Methyl@Any_C-term", "N[C@@H](C)C(=O)OC"),
        ("A", None, "Amidated@Any_C-term", "N[C@@H](C)C(N)=O"),
        ("A", "Acetyl@Any_N-term", "Methyl@Any_C-term", "CC(=O)N[C@@H](C)C(=O)OC"),
        ("A", "Dimethyl@Any_N-term", None, "CN(C)[C@@H](C)C(=O)O"),
        ("A", "Propionyl@Any_N-term", None, "CCC(=O)N[C@@H](C)C(=O)O"),
        ("P", "Acetyl@Any_N-term", None, "CC(=O)N1CCC[C@H]1C(=O)O"),
        ("Acetyl@K", None, None, "CC(=O)NCCCC[C@H](N)C(=O)O"),
        (
            "Acetyl@K",
            "mTRAQ@Any_N-term",
            None,
            "CC(=O)NCCCC[C@H](NC(=O)CN1CCN(C)CC1)C(=O)O",
        ),
    ],
)
def test_modify_amino_acid(residue, n_term_mod, c_term_mod, expected):
    # Given: an amino acid or PTM residue from the tables
    residue_smiles = aa_smiles.get(residue) or ptm_dict[residue]

    # When
    result = modify_amino_acid(
        residue_smiles, n_term_mod=n_term_mod, c_term_mod=c_term_mod
    )

    # Then: atom maps are gone and the modification sits on the right atom
    assert canonical(result) == canonical(expected)


@pytest.mark.parametrize(
    "n_term_mod, c_term_mod, match",
    [
        ("InvalidMod", None, "Unrecognized N-terminal modification"),
        (None, "InvalidMod", "Unrecognized C-terminal modification"),
    ],
)
def test_modify_amino_acid_rejects_unknown_modification(n_term_mod, c_term_mod, match):
    with pytest.raises(ValueError, match=match):
        modify_amino_acid(ALANINE, n_term_mod=n_term_mod, c_term_mod=c_term_mod)


@pytest.mark.parametrize(
    "n_term_smiles, present, absent",
    [
        # double bond to the anchor: an aryl aldimine consumes both amine hydrogens
        ("[N:1]=Cc1ccc(C)cn1", "[NX2]=[CX3]", "[NX3;H1][CX4]"),
        # charged aromatic anchor: the amine becomes the ring nitrogen of an isoquinolinium
        ("c1ccc2c[nH+:1]ccc2c1", "[n+]1ccc2ccccc2c1", "[NX3]"),
        # isotope on the anchor carries over to the amine
        ("C[15N:1]C", "C[15N](C)[CX4]", "[14N]"),
    ],
)
def test_merge_takes_bond_order_charge_and_isotope_from_fragment_anchor(
    n_term_smiles, present, absent
):
    # Given
    residue = Chem.MolFromSmiles(ALANINE)
    fragment = Chem.MolFromSmiles(n_term_smiles)

    # When
    merged = merge_at_anchors(residue, N_TERM_ANCHOR, fragment, N_TERM_ANCHOR)

    # Then
    assert merged.HasSubstructMatch(Chem.MolFromSmarts(present))
    assert not merged.HasSubstructMatch(Chem.MolFromSmarts(absent))


@pytest.mark.parametrize("fragment_smiles", random_writings(S_PHENYLETHYLAMINE))
def test_merge_keeps_fragment_stereo_next_to_anchor(fragment_smiles):
    # Given: (S)-1-phenylethylamine, its stereocentre bonded to the anchor
    residue = Chem.MolFromSmiles(ALANINE)
    fragment = Chem.MolFromSmiles(fragment_smiles)

    # When
    merged = merge_at_anchors(residue, N_TERM_ANCHOR, fragment, N_TERM_ANCHOR)

    # Then: the fragment's stereocentre and L-alanine's alpha carbon both stay S
    assert cip_labels(merged) == ["S", "S"]


@pytest.mark.parametrize("next_residue_smiles", random_writings(ALANINE))
def test_merge_keeps_residue_stereo_across_peptide_bond(next_residue_smiles):
    # Given: the incoming residue's alpha carbon is bonded to its N anchor
    chain = Chem.MolFromSmiles(ALANINE)
    next_residue = Chem.MolFromSmiles(next_residue_smiles)

    # When
    dipeptide = merge_at_anchors(chain, C_TERM_ANCHOR, next_residue, N_TERM_ANCHOR)

    # Then: L-Ala-L-Ala, and only the outer anchors remain
    assert Chem.MolToSmiles(dipeptide) == canonical(
        "C[C@H]([NH2:1])C(=O)N[C@@H](C)C(=O)[OH:2]"
    )


@pytest.mark.parametrize(
    "smiles, kind, match",
    [
        ("InvalidSMILES", "residue", "Invalid SMILES"),
        ("CC(=O)N", "n_term", "atom maps [1], found []"),
        ("[NH2:1]CC(=O)[NH2:1]", "n_term", "atom maps [1], found [1, 1]"),
        ("[NH2:1][C@@H](C)C(=O)O", "residue", "atom maps [1, 2], found [1]"),
        ("C[OH:2]", "n_term", "atom maps [1], found [2]"),
    ],
)
def test_parse_smiles_rejects_invalid_anchors(smiles, kind, match):
    with pytest.raises(ValueError, match=re.escape(match)):
        parse_smiles(smiles, kind)


@pytest.mark.parametrize(
    "target, fragment, match",
    [
        # proline's secondary amine has room for one more substituent only
        (PROLINE, "C[NH:1]C", "Cannot fuse anchor :1"),
        (ALANINE, "CC(=O)N", "Expected exactly one atom with atom map :1"),
    ],
)
def test_merge_rejects_impossible_fusion(target, fragment, match):
    with pytest.raises(ValueError, match=match):
        merge_at_anchors(
            Chem.MolFromSmiles(target),
            N_TERM_ANCHOR,
            Chem.MolFromSmiles(fragment),
            N_TERM_ANCHOR,
        )


TABLE_ENTRIES = (
    [pytest.param(s, "residue", id=f"aa {aa}") for aa, s in aa_smiles.items()]
    + [pytest.param(s, "residue", id=name) for name, s in ptm_dict.items()]
    + [pytest.param(s, "n_term", id=name) for name, s in n_term_modifications.items()]
    + [pytest.param(s, "c_term", id=name) for name, s in c_term_modifications.items()]
)


@pytest.mark.parametrize("smiles, kind", TABLE_ENTRIES)
def test_table_entry_carries_anchors_of_its_kind(smiles, kind):
    parse_smiles(smiles, kind)


@pytest.mark.parametrize(
    "n_term_mod, c_term_mod",
    [(name, None) for name in n_term_modifications]
    + [(None, name) for name in c_term_modifications],
)
def test_table_terminal_modification_applies_to_alanine(n_term_mod, c_term_mod):
    result = modify_amino_acid(ALANINE, n_term_mod=n_term_mod, c_term_mod=c_term_mod)

    assert Chem.MolFromSmiles(result) is not None


@pytest.mark.parametrize(
    "name, smiles",
    [
        pytest.param(
            name,
            smiles,
            marks=pytest.mark.xfail(
                name in UNDEFINED_ALPHA_CARBON_STEREO,
                reason="alpha-carbon stereo missing in modification.tsv",
                strict=True,
            ),
        )
        for name, smiles in {**aa_smiles, **ptm_dict}.items()
        # explicitly racemic
        if "(DL)" not in name
    ],
)
def test_table_residue_defines_alpha_carbon_stereo(name, smiles):
    # Given
    residue = parse_smiles(smiles, "residue")

    # When
    centers = dict(
        Chem.FindMolChiralCenters(
            residue, includeUnassigned=True, useLegacyImplementation=False
        )
    )

    # Then: glycine's alpha carbon is no stereocentre, every other one is assigned
    assert centers.get(alpha_carbon_idx(residue), "not a stereocentre") != "?"


@pytest.fixture
def water_formula():
    return ChemicalCompositonFormula("H(2)O(1)")


@pytest.fixture
def methane_formula():
    return ChemicalCompositonFormula("C(1)H(4)")


def test_init():
    formula = ChemicalCompositonFormula("C(6)H(12)O(6)")
    expected = defaultdict(int, {"C": 6, "H": 12, "O": 6})
    assert formula.elements == expected


def test_init_with_isotopes():
    formula = ChemicalCompositonFormula("13C(1)C(5)H(12)O(6)")
    expected = defaultdict(int, {"13C": 1, "C": 5, "H": 12, "O": 6})
    assert formula.elements == expected


def test_from_smiles():
    formula = ChemicalCompositonFormula.from_smiles("CCO")
    expected = defaultdict(int, {"C": 2, "H": 6, "O": 1})
    assert formula.elements == expected


def test_str_representation(water_formula):
    assert str(water_formula) == "H(2)O(1)"


def test_repr_representation(water_formula):
    assert repr(water_formula) == "ChemicalCompositonFormula('H(2)O(1)')"


def test_addition(water_formula, methane_formula):
    result = water_formula + methane_formula
    expected = ChemicalCompositonFormula("C(1)H(6)O(1)")
    assert result.elements == expected.elements


def test_subtraction(water_formula, methane_formula):
    result = methane_formula - water_formula
    expected = ChemicalCompositonFormula("C(1)H(2)O(-1)")
    assert result.elements == expected.elements


def test_parse_formula_with_parentheses():
    formula = ChemicalCompositonFormula("C(6)H(12)O(6)")
    expected = defaultdict(int, {"C": 6, "H": 12, "O": 6})
    assert formula.elements == expected


def test_parse_rdkit_formula():
    formula = ChemicalCompositonFormula._from_rdkit_formula("[13C]CH3OH")
    expected = defaultdict(int, {"13C": 1, "C": 1, "H": 4, "O": 1})
    assert formula.elements == expected


def test_zero_count_elements():
    formula = ChemicalCompositonFormula("C(1)H(0)O(2)")
    assert str(formula) == "C(1)O(2)"


def test_error_handling_invalid_rdkit_mol():
    with pytest.raises(ValueError):
        ChemicalCompositonFormula.from_smiles("InvalidSMILES")


def test_error_unknown_atom():
    with pytest.raises(ValueError):
        ChemicalCompositonFormula("C(6)H(12)Atom(6)")
