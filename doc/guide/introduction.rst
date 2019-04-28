Introduction
============

CLIMADA implements a fully probabilistic risk assessment model. According to the [IPCC2014]_, natural risks emerge through the interplay of climate and weather-related hazards, the exposure of goods or people to this hazard, and the specific vulnerability of exposed people, infrastructure and environment. The unit chosen to measure risk has to be the most relevant one in a specific decision problem, not necessarily monetary units. Wildfire hazard might be measured by burned area, exposure by population or replacement value of homes and hence risk might be expressed as number of affected people in the context of evacuation, or repair cost of buildings in the context of property insurance.

Risk has been defined by the International Organization for Standardization as the “effect of uncertainty on objectives” as the potential for consequences when something of value is at stake and the outcome is uncertain, recognizing the diversity of values. Risk can then be quantified as the combination of the probability of a consequence and its magnitude:

   :math:`risk = probability \times severity`

In the simplest case, :math:`\times` stands for a multiplication, but more generally, it represents a convolution of the respective distributions of probability and severity. We approximate the :math:`severity` as follows:

   :math:`severity = F(hazard \text{  } intensity, exposure, vulnerability) = exposure * f_{imp}(hazard \text{  } intensity)`

where :math:`f_{imp}` is the impact function which parametrizes to what extent an exposure will be affected by a specific hazard. While ‘vulnerability function’ is broadly used in the modelers community, we refer to it as ‘impact function’ to explicitly include the option of opportunities (i.e. negative damages). Using this approach, CLIMADA constitutes a platform to analyse risks of different hazard types in a globally consistent fashion at different resolution levels, at scales from multiple kilometres down to meters, depending on the purpose.

.. [IPCC2014] IPCC: Climate Change 2014: Impacts, Adaptation and Vulnerability. Part A: Global and Sectoral Aspects. Contribution of Working Group II to the Fifth Assessment Report of the Intergovernmental Panel on Climate Change, edited by C. B. Field, V. R. Barros, D. J. Dokken, K. J. Mach, M. D. Mastrandrea, T. E. Bilir, M. Chatterjee, K. L. Ebi, Y. O. Estrada, R. C. Genova, B. Girma, E. S. Kissel, A. N. Levy, S. MacCracken, P. R. Mastrandrea, and L. L. White, Cambridge University Press, United Kingdom and New York, NY, USA., 2014.

