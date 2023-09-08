def filter_input(filter_dict, input):
    if filter_dict == None:
        return input
    for filtname,filterconf in filter_dict.items():
        param = filterconf.get('param')
        comparator = filterconf.get('comparator')
        value = filterconf.get('value')

        if comparator not in [">",">=", "<", "<=", "==", "!="]:
            raise TypeError(f"cannot identify the filter comparator of {filtname} given in the longtable config yaml!")

        if comparator=="==":
            input = input[input[param] ==value]
            continue
        try:
            input = input.astype({f"{param}" : "float"})
        except:
            pass

        if comparator==">":
            input = input[input[param].astype(type(value)) >value]

        if comparator==">=":
            input = input[input[param].astype(type(value)) >=value]

        if comparator=="<":
            input = input[input[param].astype(type(value)) <value]

        if comparator=="<=":
            input = input[input[param].astype(type(value)) <=value]

        if comparator=="!=":
            input = input[input[param].astype(type(value)) !=value]

    return input