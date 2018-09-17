# perceptron - naive implementation

this is school work, the goal is to do a naive implementation of a perceptron

## how to run the program ?

you need to run the install script, it will create a virtual environment for python and install all dependencies.

The program was developed with python 3.5

#### Linux

check if your python version is 3.5 and the registered path (how you call python) is python35.

if not, change install_linux.sh with the appropriate python version

```shell
# don't forget to activate the script
chmod +x install_linux.sh
source install_linux.sh
```

to run the program, you need to activate the virtual environment and run main.py:

```shell
source boxpython/bin/activate

cd src/

python main.py
```


#### Windows
on windows, just double click the bat file to install.

to run the program, open a cmd console and type:

```batch
boxpython\Script\activate.bat

cd src\

python main.py
```