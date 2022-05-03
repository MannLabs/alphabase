# to register all readers into .psm_reader.psm_reader_provider
import alphabase.io.psm_reader.psm_reader 
import alphabase.io.psm_reader.alphapept_reader
import alphabase.io.psm_reader.dia_search_reader
import alphabase.io.psm_reader.maxquant_reader
import alphabase.io.psm_reader.pfind_reader

try:
    import alphabase.io.psm_reader.msfragger_reader
except ImportError:
    pass

from alphabase.io.psm_reader.psm_reader import (
    psm_reader_provider
)