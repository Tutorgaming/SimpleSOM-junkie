# SimpleSOM-junkie
High level Implementation Of Simple Self-Organzing Map (Kohonen Map)

follow by the instruction on the ai-junkie website
: http://www.ai-junkie.com/ann/som/som1.html

This is the second task on ICCS internship program
at TU-Chemnitz , Germany

by Theppasith Nisitsukcharoen

Third-year Computer Engineering student , Chulalongkorn University

### Requirement
**Desktop Side**

1. Windows 7 (According to serial transmission Library)
2. SFML Library - Draw Visualization 
  - [SFML](http://www.sfml-dev.org/download.php) -Please Select 32-Bit - code:block set path to = C:\SFML-2.3
3. CODE::BLOCK (IDE for C++/C)  - Project files formatted for code:Block
  - [CODE:BLOCK](http://www.codeblocks.org/downloads)

**Microcontroller Side**

1. Source-Code here [SimpleSOM-MSP430](https://github.com/Tutorgaming/SimpleSOM-MSP430) 
2. MSP430F5529 Board (Texas instrument)
3. Code Composer Studio (Texas instument modded version of Eclipse) 


### Setting it up

* First, you have to clone this repo into your computer.
* You can use SOURCETREE or any git clients you want to.
```sh 
   $ cd path/to/your/workspace
   $ git clone https://github.com/Tutorgaming/SimpleSOM-junkie.git
```
* Add "C:\SFML-2.3\bin" to WINDOWS PATH

* Then Open the project file [Tutor-SOM.cbp]
* Here you go :D 

### Workflow

![alt tag](https://raw.github.com/username/projectname/branch/path/to/img.png)

* Normally , the test dataset is set to "iris.data" which is located on root directory of the project file.
* There are some flag to set before compiling 
```
/*================================
   @FLAG
==================================*/
    int                 serial_flag     = 1; //Enable Serial
    int                 finished        = 0;
    int                 classy          = 0;
    int                 num             = 0;
    int                 plotty          = 0;
    
```
* Also the Parameters for self organizing map network
```
/*================================
   @PARAMETERS
==================================*/
    unsigned const int  ROW             = 15;
    unsigned const int  COL             = 15;
    unsigned const int  MAX_ITERATION   = 250;
    double              MAX_radius      = max(ROW, COL)/2;
    const double        LEARNING_CONST  = 0.1;
    
```




1. Program will count lines and elements of the dataset.
2. After that, it imports dataset onto those vectors
  -  vector <<a>double> data
  -  vector <<a>double> real
3. Next , create the self-organizing network with randomized weight
4. User will prompt to press "ENTER" in order to start program
5. Traning Process  - Bring the dataset from vector <<a>double> data   one by one to train the network until the end of dataset
6. Shuffle Dataset (vector <<a>double> data)
7. Do Training Process until reach the iteration round ( according to MAX_ITERATION )
8. After training , Plot the dataset on the trained map ( to specify which region are the data )
  - Use plotter[Row][Col] 
9. Sending the trained Self-Organizing network to Microcontroller ( according to serial_flag )
  - Running Microcontroller is required 
  - function " sentMapToMSP(); "
10. Finally, the program will ask the input test vector ( its size depends on elements of dataset ) 
and send it to microcontroller 
11. The Result 
  - Shown on the console (Desktop Side)
  - Blinking LED on microcontroller ( numbers depends on the class tag )
