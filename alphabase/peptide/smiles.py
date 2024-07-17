import functools

def requires_rdkit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            from rdkit import Chem
        except ImportError:
            raise ImportError(
                f"RDKit is required for the function '{func.__name__}'. "
                "Please install it with 'pip install alphabase[smiles]'"
            )
        return func(*args, **kwargs)
    return wrapper

@requires_rdkit
def rdkit_dependent_function(smiles):
    from rdkit import Chem
    # Your implementation using RDKit
    mol = Chem.MolFromSmiles(smiles)
    return mol