# CLIMADA-BR

Climada-BR is a project of LABIC (Laboratory of Computational Intelligence from USP São Carlos), that seeks to improve hazard assesment in Brazil using LLM's (Large Language Models) for extracting event data from auternative sources. This repository is a fork from the original [CLIMADA](https://github.com/CLIMADA-project/climada_python) which stands for CLIMate ADAptation and is a probabilistic natural catastrophe impact model, that also calculates averted damage (benefit) thanks to adaptation measures of any kind (from grey to green infrastructure, behavioural, etc.). For the installation follow [Climada's Install Guide Advanced Instructions](https://climada-python.readthedocs.io/en/latest/guide/install.html#advanced-instructions) but remember to clone this repository instead of the original.

## Introduction

Our proposal addresses the challenge of improving hazard assessment by extracting event data from alternative sources to enhance risk analysis and climate adaptation in Brazil. Hazards, dynamic events, or circumstances that can cause harm, play a significant role as crucial input parameters in modeling the impact of climatic events. For example, changes in weather patterns or the detection of invasive vector species exemplify hazards that significantly influence these models.

Hazards are fundamental for calibrating models, acting as key parameters to better understand the risks associated with climate change impacts. In this context, [CLIMADA](https://github.com/CLIMADA-project/) model, short for CLIMate ADAptation, serves as a probabilistic natural catastrophe impact model, encompassing the evaluation of averted damage (benefit) resulting from diverse adaptation measures, ranging from grey to green infrastructure and behavioral adaptations [1]. With these models, public managers and researchers can conduct risk analyses, taking into account the effects of climatic events based on local data. This enables the formulation of preventive plans and awareness strategies. Presently, CLIMADA offers historical and probabilistic event sets for various time horizons, spanning past, present, and future climates. However, CLIMADA faces limitations when directly applied to the Brazilian context. These limitations stem from a scarcity of local data and the intricate regional complexities present in Brazil.

The overarching idea of the CLIMADA-BR proposal is to develop an open-source software tailored for Brazilian decision-makers. This software aims to act as a facilitating tool for community participation and crowd-sourcing data collection. By enabling the extraction of hazards from diverse local sources, such as news, bulletins, and reports from public managers, CLIMADA-BR strives to facilitate comprehensive climate risk modeling. This, in turn, enhances decision-making processes and fosters community involvement in tracking climate change impacts at the local level.

## Proposal Information

Currently, CLIMADA is the first global platform for probabilistic multi-hazard risk modeling, incorporating uncertainty and sensitivity analyses. This model enables the assessment of natural hazard risks and the evaluation of adaptation options by comparing averted impacts to implementation costs. Hazard, in CLIMADA, is modeled as a probabilistic set of events, each representing intensity at geographical locations with an associated probability of occurrence. The risk of a single event is defined as its impact multiplied by its probability of occurrence. CLIMADA allows for globally consistent risk assessment from city to continental scale, considering historical data, future projections, and various adaptation options.

Our proposal is straightforward and has a direct impact on public managers, researchers, and stakeholders interested in monitoring and understanding the impact of climate change in impoverished regions. We propose extending the model to CLIMADA-BR, integrating Large Language Models (LLMs) for real-time hazard extraction from news, bulletins, and reports. This adaptation will significantly enhance the model's sensitivity to localized climate events, resulting in improved risk assessments for various societal sectors. The idea is to leverage significant advancements in LLMs to enrich the model's hazard database.

Our hypothesis is that the integration of Large Language Models (LLMs) into the CLIMADA-BR framework enhances the capacity to collect, classify, and extract hazards. This integration aims to provide a more accurate representation of climate impacts on a local scale, considering diverse regions in Brazil.

The LLMs employed will extract hazard events in the 5W1H format (what, where, when, who, why, and how), an area where the project coordinator already has expertise, particularly in climate change events [2,3]. To assess the software's performance, we will conduct experiments on events modeling variables related to the detection of invasive vector species and disease transmission dynamics, providing insights for public officials to make informed decisions regarding public health impacts.

This project's development involves two Ph.D. students under the coordinator's supervision, who are currently engaged in the creation of artificial intelligence models for event extraction and sensing. These students bring valuable expertise to the initiative, contributing to the advancement of the project's objectives.

## Path to impact

First, we will conduct rigorous testing and validation of the CLIMADA-BR model, refining its functionalities based on feedback and real-world data. Simultaneously, we will collaborate with key stakeholders, including government agencies, public health institutions, and environmental organizations, to ensure the model's alignment with their needs. We will leverage existing collaborations with stakeholders such as the Center for Artificial Intelligence in Brazil ([C4AI](https://c4ai.inova.usp.br/research_2/#Climate_B_eng)), IBM, and FAPESP to ensure widespread implementation and impactful utilization of CLIMADA-BR in addressing climate-related hazards.

Once validated, the CLIMADA-BR framework will be disseminated through workshops, training sessions, and online platforms, targeting decision-makers, researchers, and the public. An open-source release of the model will be pivotal to encourage broader adoption and continuous improvement. To maximize the project's impact, we will establish partnerships with local communities, leveraging their knowledge and contributing to the model's enrichment.

## References

[1] KROPF, Chahan M. et al. Uncertainty and sensitivity analysis for probabilistic weather and climate-risk modelling: an implementation in CLIMADA v. 3.1. 0. Geoscientific Model Development, v. 15, n. 18, p. 7177-7201, 2022.

[2] GÔLO, Marcos Paulo Silva; ROSSI, Rafael Geraldeli; MARCACINI, Ricardo Marcondes. Learning to sense from events via semantic variational autoencoder. Plos one, v. 16, n. 12, p. e0260701, 2021.

[3] MATTOS, Joao Pedro Rodrigues; MARCACINI, Ricardo M. Semi-supervised graph attention networks for event representation learning. In: 2021 IEEE International Conference on Data Mining (ICDM). IEEE, 2021. p. 1234-1239.


