import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install("opencv-python")
install("scikit-learn==0.20.3")
install("imutils")
install("numpy")
install("eel")