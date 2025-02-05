import os


def readHLUT(hlutFileName):
    hlut = [[], []]

    filePath = os.getcwd() + "\\pyRadPlan\\stf\\hlutLibrary\\" + hlutFileName

    with open(filePath, "r") as fl:
        for ln in fl:
            if not ln.startswith("#"):
                splitLine = ln.split()
                hlut[0].append(float(splitLine[0]))
                hlut[1].append(float(splitLine[1]))

    return hlut
