from cx_Freeze import setup, Executable
import sys, os
import idna

#fileName = input("What's the name of the py file to be converted to .exe?\n")
sys.argv.append('build')

os.environ['TCL_LIBRARY'] = r'C:\Program Files\Python36\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Program Files\Python36\tcl\tk8.6'

base = None
if (sys.platform == "win32"):
    base = "Win32GUI"    # Tells the build script to hide the console.
elif (sys.platform == "win64"):
    base = "Win64GUI"    # Tells the build script to hide the console.



setup(
    name='SVM-RCE',
    version='0.1',              #Further information about its version
    description='Feature Selection',  #It's description
    executables=[Executable("TrainedClassifier" , base=base)])



