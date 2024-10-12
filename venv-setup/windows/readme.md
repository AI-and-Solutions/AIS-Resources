# This is the readme for the venv fast setup feature

### abstract

The venv fast setup feature is to help standardize our coding practices among all machines.
For windows users, you should have two files; the requirements.txt and a runme.bat
the requirements.txt lists all the python libraries and their respective version that you need for your venv

### setup

download the files in this directory
the runme.bat executes commands inside your terminal to build the venv with the name *ais*
this is the name of your virtual environment. your venv is named *ais*
to activate your venv, you should be in the same directory as it

execute the command `.\runme` in your terminal. this installs the venv
it will take a bit, but a confirmation message will print once the process has finished


### bug testing

When you are, run `ais/Scripts/activate` in your terminal. this activates the terminal
you know this happens because the left side (where your current directory is) has changed a bit.

try the command `python` in your terminal. the left side should change again. this verifies you have python
now you have activated a python script inside your terminal. you can write python code in here, and it will execute
try `import numpy` and press enter. this verifies the packages from the requirements.txt were installed

if both of these work, thats great! if not, contact me.

exit the python editor with `quit()`
exit the venv with `deativate`


### integrating the venv with vscode

open vscode. inside vscode, open the folder you're using for this project (should be the same/parent of venv directory)
click on the search bar top center of the screen. search `>Python: Select Interpreter`
select the enter interpreter path option, and use the find feature to navigate to its location.

enter the folder called ais, enter the Scripts subfolder, and select the file named python.
now you have the python interpreter identified, try writing to a .py file some test code

`import numpy`

if vscode says this module isn't found, then something went wrong.

else, you are good to start coding! go through the tutorials, and write out the code they do
you'll learn fastest by **not copy pasting code from the tutorial to your screen**
its a little tedious, but have both windows open and follow along
I personally haven't verified *when* the tutorial starts developing the image classification network, but hopefully the whole thing takes less than two hours. this is my best guess

if you run into any questions or errors, send a message in the #help-me section of the discord

