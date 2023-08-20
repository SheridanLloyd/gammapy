from modeling import Fit
def get_fit(fit_datasets,models):
    fit_datasets.models=models
    fit_joint = Fit(store_trace=True)
    minuit_opts = {"tol": 0.1, "strategy": 2} # 0=quick optimize 2=Best optimize
    fit_joint.backend = "minuit"
    fit_joint.optimize_opts = minuit_opts

    results_joint = fit_joint.run(datasets=fit_datasets)
    print (fit_joint.minuit)
    print(results_joint.models.to_parameters_table())

    return results_joint
