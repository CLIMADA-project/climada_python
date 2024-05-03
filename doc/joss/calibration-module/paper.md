---
title: 'A Module for Calibrating Impact Functions in the Climate Risk Modeling Platform CLIMADA'
tags:
  - Python
  - climate risk
  - impact function
  - vulnerability
  - optimization
authors:
  - name: Lukas Riedel
    orcid: 0000-0002-4667-3652
    affiliation: "1, 2"
    corresponding: true
  - name: Chahan M. Kropf
    orcid: 0000-0002-3761-2292
    affiliation: "1, 2"
  - name: Timo Schmid
    orcid: 0000-0002-6788-2154
    affiliation: "1, 2"
affiliations:
 - name: Institute for Environmental Decisions, ETH Zürich, Zürich, Switzerland
   index: 1
 - name: Federal Office of Meteorology and Climatology MeteoSwiss, Zürich-Airport, Switzerland
   index: 2
date: 01 May 2024
bibliography: paper.bib

---

# Summary

Impact functions model the vulnerability of people and assets exposed to weather and climate hazards.
Given probabilistic hazard event sets or weather forecasts, they enable the computation of associated risks or impacts, respectively.
Because impact functions are difficult to determine on larger spatial and temporal scales of interest, they are often calibrated using hazard, exposure, and impact data from past events.
We present a module for calibrating impact functions based on such data using established calibration techniques like Bayesian optimization.
It is implemented as Python submodule `climada.util.calibrate` of the climate risk modeling platform CLIMADA [@gabriela_aznar_siguan_2023_8383171], and fully integrates into its workflow.

# Statement of Need

Natural hazards like storms, floods, droughts, and extreme temperatures will be exacerbated by climate change.
In 2022 alone, weather- and climate-related disasters affected 185 million people and caused economic losses of more than US$ 200 billion [@cred_2022_2023].
Forecasting the impacts of imminent events as well as determining climate risk according to future socio-economic pathways is crucial for decision-making in societal, humanitarian, political, socio-economic, and ecological issues [@smith_human_2014].
One major source of uncertainty in such computations is the vulnerability [@rougier_risk_2013].
Typically modeled as impact function[^1] that yields the percentage of affected exposure depending on hazard intensity, vulnerability is difficult to determine *a priori*.
Using hazard footprints, exposure, and recorded impacts from past events, researchers therefore employ calibration techniques to estimate unknown impact functions and use these functions for future risk projections or impact forecasts [@eberenz_regional_2021; @luthi_globally_2021; @welker_comparing_2021; @roosli_towards_2021; @kam_impact-based_2023; @Schmid2023a; @riedel_fluvial_2024].

CLIMADA is a widely used, open-source framework for calculating weather- and climate-related impacts and risks [@aznar-siguan_climada_2019], and for appraising the benefits of adaptation options [@bresch_climada_2021].
The aforementioned studies calibrate impact functions with different optimization algorithms within the CLIMADA ecosystem, but lack a consistent implementation of these algorithms.
OASIS LMF, another well-adopted, open-source loss modeling framework, also does not feature tools for impact function calibration, and regards vulnerability solely as model input [@oasis].
With the proposed calibration module, we aim at unifying the approaches to impact function calibration, while providing an extensible structure that can be easily adjusted to particular applications.

# Module Structure

Calibrating the parameters of impact functions can be a complex task involving highly non-linear objective functions, constrained and ambiguous parameters, and parameter spaces with multiple local optima.
To support different iterative optimization algorithms, we implemented an `Optimizer` base class from which concrete optimizers employing different algorithms are derived.
All optimizers receive data through a common `Input` object, which provides all necessary information for the calibration task.
This enables users to quickly switch between optimizers without disrupting their workflow.
Pre-defined optimizers are based on the `scipy.optimize` module [@virtanen_scipy_2020] and the `BayesianOptimization` package [@nogueira_bayesian_2014].
The latter is especially useful for calibration tasks because it supports constraints and only requires bounds of the parameter space as prior information.
We provide a `BayesianOptimizerController` that iteratively explores and "exploits" the parameter space to find the global optimum and stop the optimization process then.
This only requires minimal user input for indicating the sampling density.

# Example Code

Given a hazard event set, and exposure, and associated impact data in the appropriate CLIMADA data structures, the calibration input can quickly be defined.
Users have to supply a cost function, the parameter space bounds, a function that creates an impact function from the estimated parameter set, and a function that transforms a CLIMADA `Impact` object into the same structure as the input impact data.

```python
import pandas as pd
from sklearn.metrics import mean_squared_log_error

from climada.hazard import Hazard
from climada.entity import Exposure, ImpactFuncSet, ImpfTropCyclone
from climada.engine import Impact

from climada.util.calibrate import (
    Input,
    BayesianOptimizer,
    BayesianOptimizerController,
    BayesianOptimizerOutputEvaluator,
    select_best,
)

def calibrate(hazard: Hazard, exposure: Exposure, impact_data: pd.DataFrame):
    """Calibrate an impact function with BayesianOptimizer"""

    def impact_function_tropical_cyclone(v_half: float, scale: float) -> ImpactFuncSet:
        """Return an impact function set for tropical cyclones given two parameters

        This assumes that the input data relates to tropical cyclone asset damages.
        """
        return ImpactFuncSet(
            [ImpfTropCyclone.from_emanuel_usa(v_half=v_half, scale=scale)]
        )

    def aggregate_impact(impact: Impact) -> pd.DataFrame:
        """Aggregate modeled impacts per region ID and return as DataFrame"""
        return impact.impact_at_reg()

    # Prepare the calibration input
    calibration_input = Input(
        hazard=hazard
        exposure=exposure,
        data=impact_data,
        bounds={"v_half": (25.8, 150), "scale": (0.01, 1)},
        cost_func=mean_squared_log_error,
        impact_func_creator=impact_function_tropical_cyclone,
        impact_to_dataframe=aggregate_impact,
    )

    # Execute the calibration
    controller = BayesianOptimizerController.from_input(calibration_input)
    optimizer = BayesianOptimizer(calibration_input)
    calibration_output = optimizer.run(controller)

    # Store calibration results
    calibration_output.to_hdf5("calibration.h5")

    # Plot the parameter space
    calibration_output.plot_p_space(x="v_half", y="scale")

    # Plot best 3% of calibrated impact functions in terms of cost function value
    output_evaluator = BayesianOptimizerOutputEvaluator(
      calibration_input, calibration_output
    )
    output_evaluator.plot_impf_variability(select_best(p_space_df, 0.03))
```

Within the CLIMADA documentation, we provide a tutorial Jupyter script[^2] demonstrating the setup of all calibration data for executing code like the one above.
In this tutorial, we use data on tropical cyclone (TC) damages in the North Atlantic basin between 2010 and 2017 from the Emergency Events Database EM-DAT [@delforge_em-dat_2023], which lists total damages per cyclone and country.
As hazard event set we use the associated TC tracks from IBTrACS [@knapp_international_2010] and the windfield model by @holland_revised_2008.
As exposure, we use asset values estimated from gross domestic product, population distribution, and nightlight intensity [@eberenz_asset_2020].
For an easier setup and visualization, we chose to only calibrate two parameters of the impact function.
However, parameter spaces of any dimension may be sampled the same way.
Executing the calibration with these data and the above code yields plots for the sampled parameter space (see \autoref{fig:pspace}) and impact functions (see \autoref{fig:impfs}).

![Parameter space sampling with Bayesian optimization. The 'x' marks the optimal parameter set found by the algorithm. Colors indicate the respective values of the cost function (here: mean squared log error between recorded and modeled impacts).\label{fig:pspace}](pspace.png){ width=80% }

![Impact functions for tropical cyclone asset damages in the North Atlantic basin found with Bayesian optimization. The dark blue line shows the optimal function given by the parameter set noted with 'x' in \autoref{fig:pspace}. Light blue lines give the functions whose cost function value is not greater than 103% of the estimated optimum. The orange histogram denotes the hazard intensities observed.\label{fig:impfs}](impfs.png){ width=80% }

# References

[^1]: Other common names are "vulnerability function" or "damage function".
[^2]: See \url{https://climada-python--692.org.readthedocs.build/en/692/tutorial/climada_util_calibrate.html}.
