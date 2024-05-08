# dzhanibekovEffect 
Support material for ["A Deep Dive into the Tennis Racket Paradox: Analysis and Numerical Simulations of the Intermediate Axis Theorem".](https://www.google.com/)

DOI.............

## Creating a Python virtual enviroment to run the scripts

> Python2.7 already installed on your system

In case the `virtualenv` package is not installed, and/or the required version of `python2.7` is also not available to ensure compatibility, follow the steps outlined in [Installing Python and virtualenv.](https://github.com/ntrivisonno/dzhanibekovEffect/tree/main?tab=readme-ov-file#installing-python-and-virtualenv-libraries)

If you have `virtualenv` installed, you can configure an isoleted Python enviroment to run the scripts of the repository. 

Also the repository provides a `requirements.txt` file to facilitate the installation of the dependencies.

First, you need to create a directory to contain your virtual environments and then create one for this project:

```
~$ mkdir virtualenvs
~$ cd virtualenvs
~/virtualenvs$ virtualenv --python=python2.7 dzhanibekovEffect
```

Note that the virtual enviroment is created using `python2.7` ensuaring the compatibility with the scripts.

Once the virtual enviroment is created, it needs to be activated:

`~/virtualenvs$ source dzhanibekovEffect/bin/activate`

The name of the current virtual environment will now appear on the left of the prompt to let you know that it’s active. From now on, any package that you install using pip will be placed in the `dzhanibekovEffect` folder, isolated from the global Python installation.

We need to make sure that the pip version contained in the virtual environment is up to date

`$ pip install -U pip`

Now download the file `requirements.txt` and the all the `*.py` scripts along with their respectivefolders, and place them in the current folder. Then, we can install the required dependencies:

`~/virtualenvs$ pip install -r requirements.txt`

The main script is `test.py` and to execute it:

`$python test.py`

Once the program has been executed, several variables will be printed on the console, and the following message indicating that the script has run successfully will appear on the console:

```
#--------------------------------------------'

 FINISHED, OK!
```

Also, several figures will pop up showing the evolution of all the variables involved. Note that some plots may appear incorrect at first glance, but it's only a matter of plotting scales.

Once you are done working in the virtual environment for the moment, you can deactivate it:

`$ deactivate`

Now running python will just use the system’s default Python interpreter, which is not modified by anything done while being inside the virtual environment.

To delete a virtual environment, just delete the corresponding folder. (In this case, it would be `~$rm -r dzhanibekovEffect`).

## Installing Python and virtualenv libraries

> Python2.7 not installed in your system

This instructions are if you don't have `python2.7` installed on the system, so first you have to download `python2.7` package, followed by the `virtualenv` library and then continue with the step show in [Creating a Python virtual enviroment.](https://github.com/ntrivisonno/dzhanibekovEffect/tree/main?tab=readme-ov-file#creating-a-python-virtual-enviroment-to-run-the-scripts).

### Installing Python 2.7 on Linux/Unix/macOS:

Open terminal and run the following command to install Python 2.7 using the package manager of your system.

`sudo apt install python2.7  # For Debian/Ubuntu-based systems`

or

`sudo yum install python2.7  # For CentOS/RHEL-based systems`

For macOS, you can also install Python 2.7 using Homebrew:

`brew install python@2  # For macOS systems with Homebrew installed`

Installing Python in the previous ways requires administrative privileges (superuser). If you don't have them, you can install it locally from [Python Download.](https://www.python.org/downloads/)

### Installing Python 2.7 on Windows:

Download the Python 2.7 installer from the official Python website: [Python2.7 Downloads](https://www.python.org/downloads/release/python-2718/)

Run the downloaded installer and follow the on-screen instructions to install Python 2.7 on your system.

Once you have installed `Python 2.7` on your system, you need to download the `virtualenv` 

`sudo pip install virtualenv`

or 

`pip install virtualenv`

Once these packages are installed, you are ready to create the virtual environment, [Creating a Python virtual enviroment.](https://github.com/ntrivisonno/dzhanibekovEffect/tree/main?tab=readme-ov-file#creating-a-python-virtual-enviroment-to-run-the-scripts)

