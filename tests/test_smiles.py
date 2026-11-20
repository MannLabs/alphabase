import re
from collections import defaultdict

import pytest

pytest.importorskip("rdkit", reason="rdkit not installed, skipping smiles tests")

from rdkit import Chem  # noqa: E402
from rdkit.Chem import rdCIPLabeler  # noqa: E402

from alphabase.constants.atom import ChemicalCompositonFormula  # noqa: E402
from alphabase.smiles.constants import Anchor  # noqa: E402
from alphabase.smiles.smiles import (  # noqa: E402
    AminoAcidModifier,
    merge_at_anchors,
    parse_anchored_smiles,
    process_modifications,
)

aa_modifier = AminoAcidModifier()
modify_amino_acid = aa_modifier.modify_amino_acid
aa_smiles = aa_modifier.aa_smiles
n_term_modifications = aa_modifier.n_term_modifications
c_term_modifications = aa_modifier.c_term_modifications
ptm_dict = aa_modifier.ptm_dict


@pytest.fixture
def alanine_smiles():
    return aa_smiles["A"]


@pytest.fixture
def lysine_smiles():
    return aa_smiles["K"]


def test_modify_amino_acid_no_modification(alanine_smiles):
    """Without modifications the placeholders become the free amine and carboxylic acid."""
    result = modify_amino_acid(alanine_smiles)
    expected = "N[C@@H](C)C(=O)O"
    assert Chem.MolToSmiles(Chem.MolFromSmiles(result)) == Chem.MolToSmiles(
        Chem.MolFromSmiles(expected)
    )


def test_modify_amino_acid_n_term_modification(alanine_smiles):
    """An N-terminal modification is attached to the alpha-amine."""
    result = modify_amino_acid(alanine_smiles, n_term_mod="Acetyl@Any_N-term")
    expected = "CC(=O)N[C@@H](C)C(=O)O"
    assert Chem.MolToSmiles(Chem.MolFromSmiles(result)) == Chem.MolToSmiles(
        Chem.MolFromSmiles(expected)
    )


def test_modify_amino_acid_c_term_modification(alanine_smiles):
    """A C-terminal modification replaces the hydroxyl of the carboxylic acid."""
    result = modify_amino_acid(alanine_smiles, c_term_mod="Methyl@Any_C-term")
    expected = "N[C@@H](C)C(=O)OC"
    assert Chem.MolToSmiles(Chem.MolFromSmiles(result)) == Chem.MolToSmiles(
        Chem.MolFromSmiles(expected)
    )


def test_modify_amino_acid_both_modifications(alanine_smiles):
    """N- and C-terminal modifications can be applied to the same amino acid."""
    result = modify_amino_acid(
        alanine_smiles, n_term_mod="Acetyl@Any_N-term", c_term_mod="Methyl@Any_C-term"
    )
    expected = "CC(=O)N[C@@H](C)C(=O)OC"
    assert Chem.MolToSmiles(Chem.MolFromSmiles(result)) == Chem.MolToSmiles(
        Chem.MolFromSmiles(expected)
    )


def test_modify_amino_acid_with_ptm(lysine_smiles):
    """A residue-level modification is given as the whole modified amino acid."""
    lysine_with_ptm = ptm_dict["Acetyl@K"]
    result = modify_amino_acid(lysine_with_ptm)
    expected = "CC(=O)NCCCC[C@H](N)C(=O)O"
    assert Chem.MolToSmiles(Chem.MolFromSmiles(result)) == Chem.MolToSmiles(
        Chem.MolFromSmiles(expected)
    )


def test_modify_amino_acid_with_ptm_and_n_term_mod(lysine_smiles):
    """A terminal modification can be applied to an already modified amino acid."""
    lysine_with_ptm = ptm_dict["Acetyl@K"]
    result = modify_amino_acid(lysine_with_ptm, n_term_mod="mTRAQ@Any_N-term")
    expected = "CC(=O)NCCCC[C@H](NC(=O)CN1CCN(C)CC1)C(=O)O"
    assert Chem.MolToSmiles(Chem.MolFromSmiles(result)) == Chem.MolToSmiles(
        Chem.MolFromSmiles(expected)
    )


@pytest.mark.parametrize("n_term_mod", n_term_modifications.keys())
def test_all_n_term_modifications(alanine_smiles, n_term_mod):
    """Every N-terminal modification of the table yields a valid molecule on alanine."""
    result = modify_amino_acid(alanine_smiles, n_term_mod=n_term_mod)
    assert Chem.MolFromSmiles(result) is not None


@pytest.mark.parametrize("c_term_mod", c_term_modifications.keys())
def test_all_c_term_modifications(alanine_smiles, c_term_mod):
    """Every C-terminal modification of the table yields a valid molecule on alanine."""
    result = modify_amino_acid(alanine_smiles, c_term_mod=c_term_mod)
    assert Chem.MolFromSmiles(result) is not None


def test_invalid_n_term_modification(alanine_smiles):
    """An unknown N-terminal modification name is rejected."""
    with pytest.raises(ValueError, match="Unrecognized N-terminal modification"):
        modify_amino_acid(alanine_smiles, n_term_mod="InvalidMod")


def test_invalid_c_term_modification(alanine_smiles):
    """An unknown C-terminal modification name is rejected."""
    with pytest.raises(ValueError, match="Unrecognized C-terminal modification"):
        modify_amino_acid(alanine_smiles, c_term_mod="InvalidMod")


def test_invalid_amino_acid_smiles():
    """An invalid amino acid SMILES is rejected."""
    with pytest.raises(ValueError):
        modify_amino_acid("InvalidSMILES")


class TestNTermModWithPlaceholder:
    """Tests for N-terminal modification using [Lv] placeholder representing the amine."""

    def test_single_neighbor_placeholder_acetyl(self, alanine_smiles):
        """Acetyl has one neighbor to [Lv], creating one bond to N."""
        # Given: alanine with [Fl] placeholders and Acetyl modification with [Lv]
        acetyl_smiles = n_term_modifications["Acetyl@Any_N-term"]
        assert aa_modifier._has_n_term_mod_placeholder(acetyl_smiles)

        # When: applying Acetyl modification
        result = modify_amino_acid(alanine_smiles, n_term_mod="Acetyl@Any_N-term")

        # Then: carbonyl C bonds to N, remaining [Fl] becomes H
        expected = "CC(=O)N[C@@H](C)C(=O)O"
        assert Chem.MolToSmiles(Chem.MolFromSmiles(result)) == Chem.MolToSmiles(
            Chem.MolFromSmiles(expected)
        )

    def test_multi_neighbor_placeholder_dimethyl(self, alanine_smiles):
        """Dimethyl has two neighbors to [Lv], creating two bonds to N."""
        # Given: alanine with [Fl] placeholders and Dimethyl modification with [Lv]
        dimethyl_smiles = n_term_modifications["Dimethyl@Any_N-term"]
        assert aa_modifier._has_n_term_mod_placeholder(dimethyl_smiles)
        assert dimethyl_smiles == "C[Lv]C"

        # When: applying Dimethyl modification
        result = modify_amino_acid(alanine_smiles, n_term_mod="Dimethyl@Any_N-term")

        # Then: both methyl carbons bond to N
        expected = "CN(C)[C@@H](C)C(=O)O"
        assert Chem.MolToSmiles(Chem.MolFromSmiles(result)) == Chem.MolToSmiles(
            Chem.MolFromSmiles(expected)
        )

    def test_placeholder_detection(self):
        """_has_n_term_mod_placeholder correctly identifies placeholder presence."""
        # Given: SMILES with and without placeholder
        with_placeholder = "CC(=O)[Lv]"
        without_placeholder = "C(=O)C"

        # When/Then: detection returns correct result
        assert aa_modifier._has_n_term_mod_placeholder(with_placeholder) is True
        assert aa_modifier._has_n_term_mod_placeholder(without_placeholder) is False

    def test_error_when_mod_missing_placeholder(self):
        """Raises error when modification mol lacks [Lv] but placeholder method is called."""
        # Given: amino acid mol and modification mol WITHOUT [Lv] placeholder
        aa_mol = Chem.MolFromSmiles(aa_smiles["A"])
        mod_without_placeholder = Chem.MolFromSmiles("C(=O)C")

        # When/Then: calling placeholder method raises ValueError
        with pytest.raises(ValueError, match="does not contain placeholder"):
            aa_modifier._apply_n_term_mod_with_placeholder(
                aa_mol, mod_without_placeholder
            )

    def test_error_when_mod_has_multiple_placeholders(self):
        """Raises error when modification mol has multiple [Lv] placeholders."""
        # Given: amino acid mol and modification mol with multiple [Lv] placeholders
        aa_mol = Chem.MolFromSmiles(aa_smiles["A"])
        mod_with_multiple_placeholders = Chem.MolFromSmiles("[Lv]CC(=O)[Lv]")

        # When/Then: calling placeholder method raises ValueError
        with pytest.raises(ValueError, match="expected exactly 1"):
            aa_modifier._apply_n_term_mod_with_placeholder(
                aa_mol, mod_with_multiple_placeholders
            )

    def test_error_when_aa_missing_fl_placeholder(self):
        """Raises error when amino acid lacks [Fl] placeholder."""
        # Given: molecule WITHOUT [Fl] placeholder and modification WITH [Lv]
        mol_without_fl = Chem.MolFromSmiles("NCC(=O)O")  # Glycine without placeholders
        mod_with_placeholder = Chem.MolFromSmiles("CC(=O)[Lv]")

        # When/Then: calling placeholder method raises ValueError
        with pytest.raises(ValueError, match="does not contain N-terminal placeholder"):
            aa_modifier._apply_n_term_mod_with_placeholder(
                mol_without_fl, mod_with_placeholder
            )

    def test_error_when_fl_has_wrong_neighbor_count(self):
        """Raises error when [Fl] has more than one neighbor."""
        # Given: programmatically create molecule where [Fl] has two neighbors
        # (can't do this via SMILES as RDKit enforces valence)
        from rdkit.Chem import Atom, RWMol

        rw_mol = RWMol()
        c_idx = rw_mol.AddAtom(Atom(6))  # Carbon
        fl_idx = rw_mol.AddAtom(Atom(114))  # Flerovium [Fl]
        n_idx = rw_mol.AddAtom(Atom(7))  # Nitrogen
        rw_mol.AddBond(c_idx, fl_idx, Chem.rdchem.BondType.SINGLE)
        rw_mol.AddBond(
            n_idx, fl_idx, Chem.rdchem.BondType.SINGLE
        )  # [Fl] now has 2 neighbors
        malformed_mol = rw_mol.GetMol()

        mod_mol = Chem.MolFromSmiles("CC(=O)[Lv]")

        # When/Then: raises error about wrong neighbor count
        with pytest.raises(ValueError, match="should have exactly one neighbor"):
            aa_modifier._apply_n_term_mod_with_placeholder(malformed_mol, mod_mol)


@pytest.mark.parametrize("aa, smiles", aa_smiles.items())
def test_all_amino_acids(aa, smiles):
    """Every amino acid SMILES of the table yields a valid molecule."""
    result = modify_amino_acid(smiles)
    assert Chem.MolFromSmiles(result) is not None


@pytest.mark.parametrize("ptm, smiles", ptm_dict.items())
def test_all_ptms(ptm, smiles):
    """Every residue-level modification SMILES of the table yields a valid molecule."""
    result = modify_amino_acid(smiles)
    assert Chem.MolFromSmiles(result) is not None


@pytest.fixture
def water_formula():
    return ChemicalCompositonFormula("H(2)O(1)")


@pytest.fixture
def methane_formula():
    return ChemicalCompositonFormula("C(1)H(4)")


def test_init():
    """A composition string is parsed into element counts."""
    formula = ChemicalCompositonFormula("C(6)H(12)O(6)")
    expected = defaultdict(int, {"C": 6, "H": 12, "O": 6})
    assert formula.elements == expected


def test_init_with_isotopes():
    """Isotopes are counted separately from their element."""
    formula = ChemicalCompositonFormula("13C(1)C(5)H(12)O(6)")
    expected = defaultdict(int, {"13C": 1, "C": 5, "H": 12, "O": 6})
    assert formula.elements == expected


def test_from_smiles():
    """The composition of a SMILES includes its implicit hydrogens."""
    formula = ChemicalCompositonFormula.from_smiles("CCO")
    expected = defaultdict(int, {"C": 2, "H": 6, "O": 1})
    assert formula.elements == expected


def test_str_representation(water_formula):
    """The string form is the composition string."""
    assert str(water_formula) == "H(2)O(1)"


def test_repr_representation(water_formula):
    """The repr can be evaluated to recreate the formula."""
    assert repr(water_formula) == "ChemicalCompositonFormula('H(2)O(1)')"


def test_addition(water_formula, methane_formula):
    """Adding two formulas sums their element counts."""
    result = water_formula + methane_formula
    expected = ChemicalCompositonFormula("C(1)H(6)O(1)")
    assert result.elements == expected.elements


def test_subtraction(water_formula, methane_formula):
    """Subtracting formulas can yield negative element counts."""
    result = methane_formula - water_formula
    expected = ChemicalCompositonFormula("C(1)H(2)O(-1)")
    assert result.elements == expected.elements


def test_parse_formula_with_parentheses():
    """Counts are read from the parentheses after each element."""
    formula = ChemicalCompositonFormula("C(6)H(12)O(6)")
    expected = defaultdict(int, {"C": 6, "H": 12, "O": 6})
    assert formula.elements == expected


def test_parse_rdkit_formula():
    """A formula in RDKit notation, including isotopes, is parsed into element counts."""
    formula = ChemicalCompositonFormula._from_rdkit_formula("[13C]CH3OH")
    expected = defaultdict(int, {"13C": 1, "C": 1, "H": 4, "O": 1})
    assert formula.elements == expected


def test_zero_count_elements():
    """Elements with a count of zero are left out of the string form."""
    formula = ChemicalCompositonFormula("C(1)H(0)O(2)")
    assert str(formula) == "C(1)O(2)"


def test_error_handling_invalid_rdkit_mol():
    """An invalid SMILES is rejected."""
    with pytest.raises(ValueError):
        ChemicalCompositonFormula.from_smiles("InvalidSMILES")


def test_error_unknown_atom():
    """An unknown element is rejected."""
    with pytest.raises(ValueError):
        ChemicalCompositonFormula("C(6)H(12)Atom(6)")


@pytest.mark.parametrize(
    ("mod_name", "smiles", "mod_type", "expected_composition"),
    [
        (
            "Phospho@S",
            "O=P(O)(O)OC[C@@H](C(=O)[Ts])N([Fl])([Fl])",
            "ptm",
            "H(1)O(3)P(1)",
        ),
        ("Acetyl@Any_N-term", "CC(=O)[Lv]", "n_term", "C(2)H(2)O(1)"),
        ("Methyl@Any_C-term", "OC", "c_term", "C(1)H(2)"),
        # Terminal modifications that are not part of the modification table
        ("Unknown@Any_N-term", "CC(=O)[Lv]", "n_term", "C(2)H(2)O(1)"),
        ("Unknown@Any_C-term", "OC", "c_term", "C(1)H(2)"),
    ],
)
def test_process_modifications(mod_name, smiles, mod_type, expected_composition):
    """The composition of a modification is the difference its SMILES makes to the unmodified amino acid."""
    processed = process_modifications({mod_name: smiles}, mod_type=mod_type)

    assert processed == {
        mod_name: {"composition": expected_composition, "smiles": smiles}
    }


def test_process_modifications_rejects_unknown_mod_type():
    """A modification type other than ptm, n_term or c_term is rejected."""
    with pytest.raises(ValueError, match="Invalid modification type"):
        process_modifications({"Phospho@S": "OC"}, mod_type="side_chain")


@pytest.fixture
def anchored_alanine() -> str:
    return "[NH2:1][C@@H](C)C(=O)[OH:2]"


@pytest.fixture
def anchored_proline() -> str:
    return "[NH:1]1CCC[C@H]1C(=O)[OH:2]"


@pytest.fixture
def anchored_s_phenylethylamine() -> str:
    """N-terminal modification whose stereocentre is bonded to the anchor."""
    return "[NH2:1][C@@H](C)c1ccccc1"


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


@pytest.fixture
def random_writings():
    """Return a function that writes a molecule with its atoms, and hence bonds, in different orders."""

    def write(smiles: str, n_writings: int = 8) -> list[str]:
        return list(
            Chem.MolToRandomSmilesVect(
                Chem.MolFromSmiles(smiles), n_writings, randomSeed=42
            )
        )

    return write


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
    anchored_alanine, n_term_smiles, present, absent
):
    """The fused atom takes bond order, formal charge and isotope from the anchor of the fragment."""
    residue = Chem.MolFromSmiles(anchored_alanine)
    fragment = Chem.MolFromSmiles(n_term_smiles)

    merged = merge_at_anchors(residue, Anchor.N_TERM, fragment, Anchor.N_TERM)

    assert merged.HasSubstructMatch(Chem.MolFromSmarts(present))
    assert not merged.HasSubstructMatch(Chem.MolFromSmarts(absent))


def test_merge_keeps_fragment_stereo_next_to_anchor(
    anchored_alanine, anchored_s_phenylethylamine, random_writings
):
    """A stereocentre of the fragment bonded to its anchor keeps its configuration, however the fragment is written."""
    # Given: (S)-1-phenylethylamine, its stereocentre bonded to the anchor
    residue = Chem.MolFromSmiles(anchored_alanine)
    fragments = [
        Chem.MolFromSmiles(writing)
        for writing in random_writings(anchored_s_phenylethylamine)
    ]

    # When
    merged = [
        merge_at_anchors(residue, Anchor.N_TERM, fragment, Anchor.N_TERM)
        for fragment in fragments
    ]

    # Then: the fragment's stereocentre and L-alanine's alpha carbon both stay S
    assert [cip_labels(mol) for mol in merged] == [["S", "S"]] * len(fragments)


def test_merge_keeps_residue_stereo_across_peptide_bond(
    anchored_alanine, random_writings
):
    """The alpha carbon keeps its configuration in a peptide bond, however the residue is written."""
    # Given: the incoming residue's alpha carbon is bonded to its N anchor
    chain = Chem.MolFromSmiles(anchored_alanine)
    next_residues = [
        Chem.MolFromSmiles(writing) for writing in random_writings(anchored_alanine)
    ]

    dipeptides = [
        merge_at_anchors(chain, Anchor.C_TERM, next_residue, Anchor.N_TERM)
        for next_residue in next_residues
    ]

    # Then: L-Ala-L-Ala, and only the outer anchors remain
    expected = canonical("C[C@H]([NH2:1])C(=O)N[C@@H](C)C(=O)[OH:2]")
    assert [Chem.MolToSmiles(mol) for mol in dipeptides] == [expected] * len(
        next_residues
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
def test_parse_anchored_smiles_rejects_invalid_anchors(smiles, kind, match):
    """A SMILES that is invalid or does not carry exactly the anchors of its kind is rejected."""
    with pytest.raises(ValueError, match=re.escape(match)):
        parse_anchored_smiles(smiles, kind)


@pytest.mark.parametrize(
    "target_fixture, fragment_smiles, match",
    [
        # proline's secondary amine has room for one more substituent only
        ("anchored_proline", "C[NH:1]C", "Cannot fuse anchor :1"),
        ("anchored_alanine", "CC(=O)N", "Expected exactly one atom with atom map :1"),
    ],
)
def test_merge_rejects_impossible_fusion(
    request, target_fixture, fragment_smiles, match
):
    """A fusion that exceeds a valence or lacks an anchor is rejected."""
    target = Chem.MolFromSmiles(request.getfixturevalue(target_fixture))
    fragment = Chem.MolFromSmiles(fragment_smiles)

    with pytest.raises(ValueError, match=match):
        merge_at_anchors(target, Anchor.N_TERM, fragment, Anchor.N_TERM)
