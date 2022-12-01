# Persistence and transmission dynamics of emerging tick-borne pathogens: Extending a 2-pathogen, 1-host, 1-vector SIR metapopulation model

This Github Project contains the code that accompanies a CSE8803-EPI project completed in Fall 2022.

## Project Authors:

* Hubert E. Pan (@hepaces89)
* Dorian J. Feistel (@dfeistel3)
* Matthew H. Seabolt (@hseabolt)

## Project Abstract

Novel extensions to a published SIR compartment model describing the dynamics of vector-borne disease in a 2-pathogen/1-vector/1-host system are described and implemented.  The goals of these model extensions are to analyze the resulting changes to the transmission dynamics of two established pathogens induced by the introduction of novel diversity in the form of either a newly emerged (i.e. competing) pathogen or the expansion of the pathogens’ ecological niche to new vectors and new hosts.  While our proposed model extensions are theoretical by design, we obtained plausible initial values gleaned from real-world case studies of applicable vector-borne model systems in an effort to demonstrate realistic parameter ranges that may underly similar biological systems as they occur in nature.  Extended epidemic models such as those included in this study have potential to become valuable tools in the public health science toolbox by helping scientists and decision-makers evaluate threats posed by -for example- the emergence of new vector-borne pathogens or by the expansion of  the host or geographic range of established pathogens due to climate change.


## Project Keywords

* SIR models
* transmission dynamics
* vector-borne diseases
* metapopulation models
* mathematical biology

## Summary of Project Outcomes

The  goals of this study were to simulate and examine the effects on transmission dynamics between competing Rickettsia pathogens by extending the 2-1-1 SIR compartment model outlined in White et al. (2019) to three additional compartment models that introduced new biologic diversity in the form of a new pathogen (3-1-1 model), a new vector species capable of transmitting pathogens (2-2-1 model), and lastly, a new host that acts as a reservoir species for vector-borne pathogens (2-1-2 model).   In all three of our model simulations, our results revealed that host coinfection with multiple pathogens became the most prevalent outcome despite co-infected ticks themselves being comparatively quite rare in the vector populations.  We can surmise from the model curves that indeed, in all cases, all pathogens persist over time, eventually coinfecting the same host.  Future work on this study should include completion of the analysis to compare reproductive numbers between these model extensions, as well as to published vector-borne SIR models like the 2-1-1 model described in White et al (2009).  These models should be additionally calibrated using (a) real-world data to the extent possible and/or (b) in the absence of suitable real-world data, calibrated using a range of plausible values to determine a range of expected values for the model curves under different conditions.  Further, additional analysis of these model extensions should be undertaken, for example, using phase plane analysis to determine potential equilibria points and stability.
