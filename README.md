# pyRadPlan
pyRadPlan is an open-source radiotherapy treatment planning toolkit designed for interoperability with [matRad](http://www.matRad.org).

Development is lead by the [Radiotherapy Optimization group](https://www.dkfz.de/radopt) at the [German Cancer Research Center (DKFZ)](https://www.dkfz.de)

## Concept and Goals
pyRadPlan uses a similar datastructure and workflow concept as in matRad, while trying to ensure that the corresponding datastructures can be easily imported and exported from/to matRad. This facilitates the application of either algortihms from matRad or native pyRadPlan at any staget of the treatment planning workflow.

To enforce valid datastructures, we perform validation and serialization with pydantic.
Datastructures and algorithms rely mostly on SimpleITK, numpy, and scipy for internal data representation and processing.

If you are interested in helping with development, get in touch.

## Note for Developers
pyRadPlan development uses unit-testing and code formatting via pre-commit hooks to ensure clean code.
If you are a developer or want to contribute, make sure to clone the latest state via git. Then, we strongly suggest to create a virtual python environment with a suitable version and do an editable installation of pyRadPlan in dev mode:  `pip install -e .[dev]`

> **Note**
> If you are using venv to create a virtual environment in the project's root folder, we suggest to name it `.venv` as this folder will be automatically excluded by all formatters and linters

This will install an editable pyRadPlan module including `pytest` and `pre-commit` in addition to the standard modules.
- **pytest** is used to run unit tests before publishing the code. Run on the test folder via `pytest test`, or choose any test file following the pytest syntax. We encourage writing at least fundamental unit tests for new code. Will also install `coverage` and the pytest extension to monitor coverage.
- **pre-commit** allows for automatic code formatting to ensure following of PEPs (mainly PEP8 and PEP257).

After successfully running `pip install -e .[dev]` check in the console that `pre-commit --version` provides a correct response.
Afterwards run `pre-commit install`. This reads the `.pre-commit_config.yaml` and adds a hook to your git repository, which whenever a commit is made the changed files are reformatted to ensure the PEP standards.

### Matlab / Octave Engine
If you want to interface via matRad, you can use install the matlab engine by appending `[matlab]` to the pip install command. However, as there is some compatibility matrix for what matlabengine version is suitable for your Matlab installation, it might be better to not install the latest version but instead install the correct matlabengine manually before.
To use Octave via oct2py, analogously append `[octave]`.
