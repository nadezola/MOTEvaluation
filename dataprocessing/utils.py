def mkdir(dir):
    if not dir.exists():
        dir.mkdir(parents=True)