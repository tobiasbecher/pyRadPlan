import os
import pyRadPlan.io.matRad as matRadIO


class MatLabFileHandler:
    def __init__(self, tempPath):
        self.tempPath = tempPath

    def load(self, *args):
        mat_file_contents = {}
        for arg in args:
            mat_file_contents[arg] = matRadIO.load(os.path.join(self.tempPath, f"{arg}.mat"))
        return mat_file_contents

    def save(self, **kwargs):
        for key, value in kwargs.items():
            if not isinstance(value, dict):
                value = {key: value}
            matRadIO.save(os.path.join(self.tempPath, f"{key}.mat"), value)

    def delete(self, *args):
        for arg in args:
            file_path = os.path.join(self.tempPath, f"{arg}.mat")
            os.remove(file_path)
            setattr(self, arg, {})
