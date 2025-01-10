# Rules for Contribution

## Branch Name Rules
In the future, branch name rules will be enforced.

There's two predefined branch names for main development
- **main:**       Protected main branch. ~~Only latest release version + hotfixes~~ (more relaxed development usage until the first package version)
- ~~**develop:**  Development branch. Used to prepare patches & minor versions.~~ (will be used in the future)

For individual development work, we will use branch groups. The following groups are allowed
- **bugfix/*:**      Fixing a bug
- **hotfix/*:**      Fixing a breaking change on the main branch
- **feature/*:**     Individual larger novel feature contributions, e.g., feature/myCoolNewDoseCalculation
- **refactor/*:**    An overhaul / refactor of code
- **interface/*:**   Interface to other software
- **devops/*:**      Work on DevOps like the CI pipeline or formatting hooks (e.g., devops/addPython311)
- **dev/*:**         Individual development efforts that are not part of a single group above, or don't have a clear purpose or timeline regarding merging
