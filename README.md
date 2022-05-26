# Landscape-Aware Automated Algorithm Configuration

Includes several experimental automated algorithm configiruation procedures designed
to exploit landscape structure. In particular, these configurators were designed
to target AutoML hyper-parameter optimization scenarios.

The original search procedures in this repository were designed to work with
ray-tune. However, the version of ray that was available at the time did not
appear to be able to make adequate usage of our available parallel resources.
For this reason, these search procedures have been re-purposed to be used
as alternative search procedures to the Bayesian optimization used by
BOHB (Bayesian optimization with Hyperband).

This version of the code can be used in exactly the same way as BOHB, 
(https://automl.github.io/HpBandSter/build/html/index.html) except
instead of using their optimizer, import and use the one from this package.
In particular:

    from laaac.bohb_optimizers import CQAHB 

By default, this will use a convex quadratic approximation (CQA) model as the
surrogate model that is fitted to the landscape. However, this can be changed
to a spline approximation by passing the argument 
`cqa_kwargs={'surrogate': 'spline'}`. Documentation for the additional 
CQA-supported key-word arguments can be found in `laaac/cqa_searcher.py`.

More information about how to use the ray-based version of the code can
be found in the `ray` branch.

This repository is a work in progress and it builds on a line of research 
(see https://www.cs.ubc.ca/labs/algorithms/Projects/ACLandscapes/index.html) 
that seeks to analyze and exploit algorithm configuration landscape structure.

- \[Pushak & Hoos, 2022a\] Yasha Pushak and Holger H. Hoos.  
**AutoML Loss Landscapes.**  
Under review at *Transactions on Evolutionary Optimization and Learning (TELO)*.
- \[Pushak & Hoos, 2022b\] Yasha Pushak and Holger H. Hoos.  
**Experimental Procedures for Exploiting AutoML Loss Landscape Structure.**  
Preprint.  
- \[Pushak, 2022\] Yasha Pushak.  
**Algorithm Configuration Landscapes: Analysis & Exploitation.**  
*PhD Thesis, The University of British Columbia.*  
 - \[Pushak & Hoos, 2020\] Yasha Pushak and Holger H. Hoos.  
**Golden Parameter Search: Exploiting Structure to Quickly Configure Parameters
In Parallel.**  
*In Proceedings of the Twenty-Second Interntional Genetic and Evolutionary 
Computation Conference (GECCO 2020)*. pp 245-253 (2020).  
**Won the 2020 GECCO ECOM Track best paper award.**
 - \[Pushak & Hoos, 2018\] Yasha Pushak and Holger H. Hoos.  
**Algorithm Configuration Landscapes: More Benign than Expected?**  
*In Proceedings of the Fifteenth Internationl Conference on Parallel Problem 
Solving from Nature (PPSN 2018)*. pp 271-283 (2018).  
**Won the 2018 PPSN best paper award.**


# Installing LAAAC

 1. Create a python virtual environment
 2. Download the latest version of LAAAC from https://github.com/YashaPushak/LAAAC
 3. While in the main LAAAC directory, install LAAAC with 
`pip install .`


# Contact

Yasha Pushak  
ypushak@cs.ubc.ca  

PhD Student & Vanier Scholar  
Department of Computer Science  
The University of British Columbia  
