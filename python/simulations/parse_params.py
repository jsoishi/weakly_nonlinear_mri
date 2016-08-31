def parse_params(dirname,basename):
    parstr = dirname.split(basename, 1)[1].lstrip("_")
    parstr = parstr.split("_")

    params = {}
    for p in parstr:
        m = re.match("([a-zA-Z]+)([\d.+-e]+)",p)
        k, v = m.groups()
        params[k] = v

    return params


