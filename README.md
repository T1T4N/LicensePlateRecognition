LicensePlateRecognition
=======================
### Summary
An automated system for car license plate detection and recognition

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
2. Install OpenCV 2.x
3. Install [Tesseract 3.02](https://github.com/tesseract-ocr/tesseract/releases/tag/3.02.02) (higher versions don't work with python-tesseract yet)
4. Install the following Python packages:
  * six
  * numpy
  * Pillow (fork of PIL)
  * python-dateutil
  * matplotlib
  * pyparsing
  * [python-tesseract](https://bitbucket.org/3togo/python-tesseract/downloads)
5. Extract [tessdata](https://drive.google.com/file/d/0B61RgxZKvD2JT011cWluN3JMMUk/view?usp=sharing) in the main folder
6. Place pictures on which to perform detection in the main/images folder, or you can download [my test pictures](https://drive.google.com/file/d/0B61RgxZKvD2JOVZTZjY1OHVMTmc/view?usp=sharing)
7. Enter the project directory and run the following commands:
  * export PYTHONPATH=$PYTHONPATH:.
  * python main/main.py
