.. _Installation_EULER:

Installation on a server / the Euler Cluster (ETH internal)
***********************************************************

Please execute the instructions of the following text boxes in a Terminal.
For ETH WCR group members, there are two directories that can be used for the installation:

1. "Work": /cluster/work/climate/USERNAME

2. "Home": /cluster/home/USERNAME/


The advantage of using "Home" is that it is the location with the fastest access for the use of python and CLIMADA.
The disadvantage of "Home" is the limited quota of 20 GB and 100'000 files, since both miniconda and CLIMADA come with many single files.
Therefore, (1) is the recommended option.

For all steps listed below, first enter the Cluster via SSH.

On the server, go to either your "Home" or your "Work" environment. In the following we will go for option 1 and install everything in the "Work" path.
If you are using a different server or option, please customise the paths in each step::

    cd /cluster/work/climate/USERNAME

Download
========
**Version in development**: Download or clone the GitHub repository `climada_python <https://github.com/CLIMADA-project/climada_python.git>`_.

To clone the repository, you need to first `install git <https://www.linode.com/docs/development/version-control/how-to-install-git-on-linux-mac-and-windows/>`_.
Afterwards you can use the following command line with climada_python's URL::

    git clone https://github.com/CLIMADA-project/climada_python.git


Unix Operating System
=====================


Install environment with Miniconda
----------------------------------
1. **Miniconda**: Download or update to the latest version of `Miniconda <https://conda.io/miniconda.html>`_. You can do so by following these steps
(using wget to download miniconda installation file: https://conda.io/en/latest/miniconda.html, Python 3.7, LINUX, 64bit)::

    cd /cluster/work/climate/USERNAME
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    rm Miniconda3-latest-Linux-x86_64.sh
    


During the installation process of Miniconda, you are prompted to set the working directory according to your choice.

2. **Install dependencies**: Create the virtual environment *climada_env* with climada's dependencies::

    cd /cluster/work/climate/USERNAME/climada_python
    conda env create -f requirements/env_climada.yml --name climada_env 

(You might need to restart the terminal for the command "conda" to work after installation of miniconda. Alternatively, execute the command *source activate base* before executing the above comments.)

   To include *climada_python* in the environment's path, do the following. In your environments folder, for example /cluster/work/climate/USERNAME/miniconda3/*::
   
    cd envs/climada_env/lib/pythonX.X/site-packages
    echo '/your/path/to/climada_python/' > climada_env_path.pth

! Replace pythonX.X with the correct version of Python, i.e. python3.7 !

3. **Test installation**: Activate the environment, execute the unit tests and deactivate the environment when finished using climada::

    source activate climada_env
    python3 tests_runner.py
    source deactivate
  

   If the installation has been successful, an OK will appear at the end of the execution.

   Warning: Executing the whole test_runner takes pretty long and downloads some files to climada_python/data that you might not need.
   Consider aborting the test once you see first "OK"s in the output.

4. **Submit a test BJOB to the queue of the cluster**: Write a shell script that initiates the Python environment climada_env and submit the job::

    touch run_climada_python.sh
    echo "#!/bin/sh" > run_climada_python.sh
    echo "cd /cluster/work/climate/USERNAME/climada_python" >> run_climada_python.sh
    echo "source activate climada_env" >> run_climada_python.sh
    echo "python3 script/tutorial/climada_hazard_drought.py" >> run_climada_python.sh
    echo "conda deactivate" >> run_climada_python.sh
    echo "echo script completed" >> run_climada_python.sh
    chmod 755 run_climada_python.sh
    bsub -J "test01" -W 1:00 -R "rusage[mem=5120]" -oo logs/test01.txt -eo logs/e_test01.txt < run_climada_python.sh
    bjobs


   Notes:

   - Change the script after echo "python3" to the path of the Python script you want to execute

   - Change the working path to the path you have cloned the climada_python repository to

   - Customise the bsub options accordingly

   - Check out https://scicomp.ethz.ch/wiki/Using_the_batch_system#Basic_job_submission for more info on job submission.



Update climada's environment
----------------------------
Before using climada's code in development, remember to update your code as well as climada's environment. The requirements in ``requirements/env_developer.yml`` contain all the packages which are necessary to execute the continuous integration of climada. These can be therefore useful for climada's contributors. 

If you use conda, you might use the following commands to update the environments::

    cd climada_python
    git pull
    source activate climada_env
    conda env update --file requirements/env_climada.yml
    conda env update --file requirements/env_developer.yml
    
If any problem occurs during this process, consider reinstalling everything from scratch following the `Installation`_ instructions. 
You can find more information about virtual environments with conda `here <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_.

