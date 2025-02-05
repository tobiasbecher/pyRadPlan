import regex as re
import os
from pyRadPlan.io.readHLUT import readHLUT


def loadHLUT(ct, pln):
    hlutDir = os.path.dirname(os.path.realpath(__file__)) + "/hlutLibrary/"

    # If possible -> file standard out of dicom tags
    try:
        hlutFileName = ""
        particle = pln["radiationMode"]
        manufacturer = ct["dicomInfo"]["Manufacturer"]
        model = ct["dicomInfo"]["ManufacturerModelName"]
        convKernel = ct["dicomInfo"]["ConvolutionKernel"]

        hlutFileName = (
            manufacturer
            + "-"
            + model
            + "-ConvolutionKernel"
            + convKernel
            + "_"
            + particle
            + ".hlut"
        )

        # Check whether fileNames used blanks instead of '_' or '-'
        hlutFilePaths = [0, 0, 0]
        hlutFilePaths[0] = hlutFileName
        hlutFilePaths[1] = re.sub("-", " ", hlutFileName)
        hlutFilePaths[2] = re.sub("_", " ", hlutFileName)

        # Add pathName
        hlutFilePaths = [hlutDir + hlutFilePaths[i] for i in range(len(hlutFilePaths))]

        # Check if files exist
        exist = -1
        for i, filePath in enumerate(hlutFilePaths):
            if os.path.exists(filePath):
                exist = i

        if exist == -1:
            warnText = (
                "Could not find HLUT "
                + hlutFileName
                + " in hlutLibrary folder. matRad default HLUT loaded"
            )
            print(warnText)
        else:
            hlutFileName = hlutFilePaths[exist]

    except FileNotFoundError:
        # Load default HLUT
        hlutFileName = hlutDir + "matRad_default.hlut"

    # name override for testing
    hlutFileName = "matRad_default.hlut"

    hlut = readHLUT(hlutFileName)

    return hlut
