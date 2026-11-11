alphabase.spectral_library.translate
=====================================

Export a spectral library to tsv (:func:`translate_to_tsv`). For DIA-NN ``.parquet``
export see :doc:`translate_diann <translate_diann>`.

TSV output
----------

``translate_to_tsv`` accepts a string or path-like destination, or an open binary
or text stream. For paths, compression is inferred from the filename using
pandas' compression support; for example, ``library.tsv.gz`` produces gzip
output. Text streams receive decoded UTF-8 text, and binary streams receive
bytes. Text-mode wrappers are supported. Custom sinks without a recognized binary
stream type or a binary ``mode`` are treated as text sinks. Caller-owned streams
remain open after export, and write errors propagate to the caller.

Floating-point formatting preserves the exported numeric values exactly when
parsed back into their original dtype, including signed zero and the signs of
infinities. This applies to fragment intensities after normalization. Values are
neither downcast nor rounded to a fixed number of decimal places. NaN is written
as an empty field.

Numeric spelling may differ from pandas CSV output: an integral float may be
written as ``1`` instead of ``1.0``, and scientific notation may use different
exponent formatting. TSV bytes are therefore not guaranteed to match pandas
output. These formatting differences do not change fragment selection,
normalization, or row order. Batches are written in precursor order, including
when export uses multiple threads.

.. automodule:: alphabase.spectral_library.translate
   :members:
   :undoc-members:
   :show-inheritance:
