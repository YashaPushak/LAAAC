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

This version of the code can be used in exactly the same way as BOHB, except
instead of using their optimizer, import and use the one from this package.
In particular:

    from laaac.bohb_optimizers import CQAHB 

By default, this will use a convex quadratic approximation (CQA) model as the
surrogate model that is fitted to the landscape. However, this can be changed
to a spline approximation by passing the argument 
`cqa_kwargs={'surrogate': 'spline'}`.

This repository is a work in progress and it builds upon the line of research:

 - Yasha Pushak and Holger H. Hoos.  
**Golden Parameter Search: Exploiting Structure to Quickly Configure Parameters
In Parallel.**  
*In Proceedings of the Twenty-Second Interntional Genetic and Evolutionary 
Computation Conference (GECCO 2020)*. pp 245-253 (2020).  
**Won the 2020 GECCO ECOM Track best paper award.**
 - Yasha Pushak and Holger H. Hoos.  
**Algorithm Configuration Landscapes: More Benign than Expected?**  
*In Proceedings of the Fifteenth Internationl Conference on Parallel Problem 
Solving from Nature (PPSN 2018)*. pp 271-283 (2018).  
**Won the 2018 PPSN best paper award.**

# Table of Contents

   * [Landscape-Aware Automated Algorithm Configuration](#landscape-aware-automated-algorithm-configuration)
   * [Table of Contents](#table-of-contents)
   * [Installing LAAAC](#installing-laaac)
   * [Quick Start Guide](#quick-start-guide)
   * [LAAAC Arguments](#laaac-arguments)
      * [Setup Arguments](#setup-arguments)
      * [Scenario Arguments](#scenario-arguments)
   * [Contact](#contact)

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
