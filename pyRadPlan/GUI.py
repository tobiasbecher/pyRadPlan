import pyRadPlan.matRad._engine as eng


def matRadGUI(ct, cst, pln, resultGUI):
    eng.currentEngine.engine.callFromPython("matRadGUI", "GUI", ct, cst, pln, resultGUI, nargout=0)
