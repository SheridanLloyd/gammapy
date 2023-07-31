from modeling import Fit
def get_fit(fit_datasets,models):
    fit_datasets.models=models
    fit_joint = Fit()
    results_joint = fit_joint.run(datasets=fit_datasets)
    return results_joint
