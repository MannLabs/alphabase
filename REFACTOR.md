# Refactor TODO

## PSM Reader Modification Translators

✅ **IMPLEMENTED** - Replace `__call__` with explicit `translate` method in modification translator classes:

- `alphabase/psm_reader/msfragger_reader.py`: `MSFraggerModificationTranslation.__call__` → `translate`
- `alphabase/psm_reader/sage_reader.py`: `SageModificationTranslation.__call__` → `translate`

Reference: https://github.com/MannLabs/alphabase/pull/XXX (PR review comment by @mschwoer)

## MSFragger Custom Modification Support

✅ **IMPLEMENTED** - Custom modifications via inherited `modification_mapping` parameter:

```python
reader = MSFraggerPsmTsvReader(
    modification_mapping={
        'Phospho@S': 'S(79.9663)',
        'Phospho@T': 'T(79.9663)',
        'Oxidation@M': 'M(15.9949)',
        'TMTpro@Any_N-term': 'N-term(304.2071)',
        'Amidated@Any_C-term': 'C-term(17.0265)',
    }
)
```

Keys use alphabase format (`Mod@AA`), values use MSFragger's native format (`AA(mass)` or `N-term(mass)`).
Uses the inherited `modification_mapping` parameter from `PSMReaderBase`, which creates a `rev_mod_mapping`
that is passed to `MSFraggerModificationTranslation`.

Implementation:
1. ✅ Use inherited `modification_mapping` parameter (same as other readers)
2. ✅ Check `rev_mod_mapping` first before falling back to `mass_mapped_mods` from yaml
3. ✅ Validation handled by base `ModificationMapper` class

Reference: PR review comment by @lucas-diedrich
