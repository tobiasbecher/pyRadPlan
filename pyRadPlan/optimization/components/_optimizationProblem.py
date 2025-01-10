#%% External package import

from collections.abc import Iterable

# from jax import jacfwd, jacrev
from numpy import array

#%% Class definition


class OptimizationProblem:
    def __init__(self, back_projection, cst):

        self.backprojection = back_projection
        self.cst = cst

        def flatten(iterable):

            for it in iterable:
                if isinstance(it, Iterable) and not isinstance(it, (str, bytes)):
                    yield from flatten(it)
                else:
                    yield it

        # Aggregating objective and constraint functions
        def get_components():

            objs = tuple(
                zip(
                    flatten(
                        [struct] * len(self.cst[struct]["dose_objective"])
                        if isinstance(self.cst[struct]["dose_objective"], list)
                        else [struct]
                        for struct in self.cst
                        if self.cst[struct]["dose_objective"] is not None
                    ),
                    flatten(
                        self.cst[struct]["dose_objective"]
                        for struct in self.cst
                        if self.cst[struct]["dose_objective"] is not None
                    ),
                )
            )
            cons = tuple(
                zip(
                    flatten(
                        [struct] * len(self.cst[struct]["doseConstraint"])
                        if isinstance(self.cst[struct]["doseConstraint"], list)
                        else [struct]
                        for struct in self.cst
                        if self.cst[struct]["doseConstraint"] != None
                    ),
                    flatten(
                        self.cst[struct]["doseConstraint"]
                        for struct in self.cst
                        if self.cst[struct]["doseConstraint"] != None
                    ),
                )
            )

            return objs, cons

        components = get_components()

        # Generate lists of objectives and constraints with linked sets
        self.objs = tuple(
            ([obj[0]] + obj[1].link, obj[1]) if obj[1].link != None else ([obj[0]], obj[1])
            for obj in components[0]
        )
        self.cons = tuple(
            ([con[0]] + con[1].link, con[1]) if con[1].link != None else ([con[0]], con[1])
            for con in components[1]
        )

        # Initialize tracker dictionary to monitor objective function values
        self.tracker = {"Total": []}
        self.tracker.update(
            {comp: [] for comp in ("-".join((obj[0], obj[1].name)) for obj in components[0])}
        )

        # Precompute and store jacobian and hessian computation functions
        # self.jac = jacfwd(self.objective)
        # self.hess = jacfwd(jacrev(self.objective))

    def __str__(self):
        return "\n".join(("Problem class attributes:", "------------", str((*self.__dict__,))))

    def objective(self, w):

        self.backprojection.compute_dose(w)
        dose = self.backprojection.get_dose_result()

        def value(obj):

            structs = obj[0]
            obj_class = obj[1]

            if all(x in str(type(obj_class).__bases__[0]) for x in ("ObjectiveClass")):

                fval = obj_class.compute_objective(
                    dose=tuple(dose[self.cst[struct]["resized_indices"]] for struct in structs),
                    structs=structs,
                )
                self.tracker["-".join((structs[0], obj_class.name))] += (fval,)

                return fval

        dose_objective = sum(value(obj) for obj in self.objs)
        self.tracker["Total"] += (dose_objective,)

        return dose_objective

    def gradient(self, w):

        self.backprojection.compute_dose(w)
        dose = self.backprojection.get_dose_result()

        def value(obj):

            structs = obj[0]
            obj_class = obj[1]

            if all(x in str(type(obj_class).__bases__[0]) for x in ("ObjectiveClass")):

                return obj_class.compute_gradient(
                    dose=tuple(dose[self.cst[struct]["resized_indices"]] for struct in structs),
                    structs=structs,
                )

        dose_gradient = sum(value(obj) for obj in self.objs)

        self.backprojection.compute_weight_gradient(dose_gradient, w)
        weightGradient = self.backprojection.get_weight_gradient()

        return weightGradient

    def constraints(self, w):

        self.backprojection.compute_dose(w)
        dose = self.backprojection.get_dose_result()

        def value(con):

            structs = con[0]
            con_class = con[1]

            if all(x in str(type(con_class).__bases__[0]) for x in ("ConstraintClass")):

                return con_class.compute_constraint(
                    dose=tuple(dose[self.cst[struct]["res_indices"]] for struct in structs),
                    structs=structs,
                )

        dose_constraints = array(value(con) for con in self.cons)

        return dose_constraints

    # def jacobian(self, w):
    #     return self.jac(w)

    # def hessian(self, w):
    #     return self.hess(w)
