Adding SMILES modifications
===========================

This guide shows how to add new amino-acid or peptide-level modifications to AlphaBase that are fully defined by their SMILES representation.

Why SMILES?
-----------
AlphaBase uses SMILES strings to allow users to work with peptides as graphs.
Defining your modifications in this way guarantees that the elemental composition is always consistent with the underlying chemistry.
See `Wikipedia <https://en.wikipedia.org/wiki/Simplified_Molecular_Input_Line_Entry_System>`_ for more information.

Overview of the workflow
------------------------
1. Fill three dictionaries (``n_term_modifications``, ``c_term_modifications``, and ``ptm_dict``) with the modification names and the corresponding SMILES strings.
2. Run the helper script ::

       python -m alphabase.smiles.adding_smiles

   The script will:
   • validate that the generated composition of every modified amino-acid matches the unmodified one
     (after subtracting H\ :sup:`2`\ O to compensate for peptide bond formation)
   • compute the elemental composition difference of the modification
   • append the new rows to the global modification table (``alphabase/constants/const_files/modification.tsv``)

3. Use the newly added modifications in the rest of AlphaBase exactly as the built-in ones.

Where are the dictionaries defined?
-----------------------------------
All three dictionaries live inside ``alphabase.smiles.adding_smiles``.
Feel free to extend them directly or generate them programmatically.

Tip: if you only need to *apply* the modifications and not to permanently store them, you can import and modify the
``AminoAcidModifier`` instance at run-time instead of running the script above.

Underlying API
--------------
Internally the script leverages three public classes / functions:

* ``alphabase.smiles.smiles.AminoAcidModifier`` - applies SMILES modifications to an amino-acid backbone
* ``alphabase.constants.atom.ChemicalCompositonFormula`` - converts SMILES ⇒ elemental composition
* ``alphabase.constants.modification.add_new_modifications`` - inserts calculated compositions into the global table

The full source code of the script is available at
`alphabase/smiles/adding_smiles.py <https://github.com/MannLabs/alphabase/blob/main/alphabase/smiles/adding_smiles.py>`_

Usage example
--------------
Below is a minimal snippet that adds a hypothetical modification called *Foo@K* ::

    from alphabase.smiles.adding_smiles import process_modifications, ptm_dict
    from alphabase.constants.modification import add_new_modifications

    # 1) add your SMILES
    ptm_dict["Foo@K"] = "CC(O)CN"

    # 2) calculate the composition and store it
    new_mods = process_modifications(ptm_dict, mod_type="ptm")
    add_new_modifications(new_mods)

    # At this point Foo@K is available in alphabase.constants.modification.MOD_DF

After running the code you will be able to use ``Foo@K`` in any AlphaBase pipeline, for example when generating in-silico
digests or when reading PSM tables.

----

If you encounter problems or would like to contribute additional modifications, please open an issue or a pull-request
on GitHub.
