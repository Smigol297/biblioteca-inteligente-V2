import importlib.metadata as m

lineas = sorted("%s==%s" % (d.metadata["Name"], d.version) for d in m.distributions())
with open(r"Z:\src\requirements-windows.lock", "w") as f:
    f.write("\n".join(lineas) + "\n")
