import ahocorasick

def build_AC(peptides:list):
    AC = ahocorasick.Automaton()

    for seq in peptides:
        AC.add_word(seq, seq)
    AC.make_automaton()

def match_AC(AC:ahocorasick.Automaton, protein_seq):
    start_last_list = []
    for last_index, seq in AC.iter(protein_seq):
        start_index = last_index - len(seq) + 1
        start_last_list.append((start_index, last_index))
    return start_last_list