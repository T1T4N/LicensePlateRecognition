LicensePlateRecognition
=======================
### Summary
An automated car license plate detection and recognition application

### Credits
The application was developed by:
* [Robert Armenski](https://github.com/T1T4N)
* [Martin Boncanoski](https://github.com/makedon4e)
* [Nikola Furnadjiski](https://github.com/nikola3103)
* [eevee](https://github.com/eevee/pyhq2x) for the hq2x module

Under the mentorship of:
* [Dr. Igor Trajkovski](http://www.time.mk/trajkovski/)

#### Setup instructions: ####
1. Install Python v2.7.* 32-bit
2. Install OpenCV 2.x along with the Python extensions (OS X guide [here](https://jjyap.wordpress.com/2014/05/24/installing-opencv-2-4-9-on-mac-osx-with-python-support/))
  * You need the latest version of libpng on Mac OS X: brew install libpng
3. Install Tesseract
  * Mac OS X: brew install tesseract
  * Ubuntu: apt-get install tesseract-ocr
4. Install the following Python packages:
  * six
  * numpy
  * Pillow (fork of PIL)
  * python-dateutil
  * matplotlib
  * pyparsing
  * [python-tesseract](https://bitbucket.org/3togo/python-tesseract/downloads) (You might need Python setuptools to install the .egg file)
5. Extract [tessdata](https://drive.google.com/file/d/0B61RgxZKvD2JT011cWluN3JMMUk/view?usp=sharing) in the main folder
6. Place pictures on which to perform detection in the main/images folder, or you can download [my test pictures](https://drive.google.com/file/d/0B61RgxZKvD2JOVZTZjY1OHVMTmc/view?usp=sharing)
7. Enter the project directory and run the following commands:
  * export PYTHONPATH=$PYTHONPATH:.
  * python main/main.py
