#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <random>
#include <fstream>
#include <math.h>

//Node Data Structure
#include "Node.h"

//Windows Serial Stuff
#include "Serial.h"
//Plot Stuff
#include <SFML/System.hpp>
#include <SFML/Graphics.hpp>

using namespace std;

/*================================
   @DEBUG
==================================*/
    int                 debug           = 0;
    // Mouse Coordinate in detail window
    int                 mouse_x         = 0;
    int                 mouse_y         = 0;
    // Untrained Input Location
    int                 new_i           = 0;
    int                 new_j           = 0;
    int                 plot_match_x    = 0;
    int                 plot_match_y    = 0;
/*================================
   @PARAMETERS
==================================*/
    unsigned const int  ROW             = 15;
    unsigned const int  COL             = 15;
    unsigned const int  MAX_ITERATION   = 250;
    double              MAX_radius      = max(ROW, COL)/2;
    const double        LEARNING_CONST  = 0.1;
    unsigned int        iteration_count = 0;
    int                 element_count   = 4;
    int                 class_count     = 3; // IRIS DATASET = 3
    int                 line_count      = ROW*COL;
    int                 sizeMonitor     = 400;
    int                 margin          = ceil((double)sizeMonitor/(double)ROW);
    CSerial             serial;         //Serial interface
/*================================
   @Data Structure
==================================*/
    // Self-Organzing Map
    Node som_map[ROW][COL];
    double U_Matrix[ROW][COL];
    // Plotter for Classification on Training data (R,G,B)
    int plotter[ROW][COL];
    // Dataset Vectors - will be random shuffle
    vector<vector<double> > data;
    // Dataset Vectors - Original Sequence
    vector<vector<double> > real;
    // Dataset Classes - index by index
    vector<string> real_class;
    // Class TAG
    set<string> class_tag;
    map<string,int> class_number;
/*================================
   @FLAG
==================================*/
    int                 serial_flag     = 1; //Enable Serial
    int                 finished        = 0; //Training Process Finished ? (Enable Mouse Click on visualizer)
    int                 classy          = 0; //U-Matrix is ready to plot trained data
   // int                 num             = 0;
    int                 plotty          = 0;
/*==============================================================
   @FUNCTION
================================================================*/
// Display Vector<double> (Debugging Purpose)
void displayVector(vector<double> input){
    cout << "<" ;
    for ( vector<double>::iterator i = input.begin() ; i < input.end() ; i ++){
        cout << *i << ((i==input.end()-1)? "":",");
    }
    cout << ">";
}

void computeClassTag(){
    class_number["NULL"] = 0;
    int i = 1;
    for(std::set<string>::iterator ita = class_tag.begin() ; ita != class_tag.end() ; ita++){
        class_number[*ita] = i;
        i++;
    }
}

// RANDOM RGB DATASET
void randomdata(){
    line_count = 150;
    element_count = 3;
    for(int i = 0 ; i < line_count ; i++){
            vector<double> temp;
            int r = ((double) rand() / (RAND_MAX))*3;
            if(r == 0 ){
              temp.push_back(255);
              temp.push_back(0);
              temp.push_back(0);
            }else if(r == 1 ){
              temp.push_back(0);
              temp.push_back(255);
              temp.push_back(0);
            }else if(r == 2 ){
              temp.push_back(0);
              temp.push_back(0);
              temp.push_back(255);
            }
        displayVector(temp);
        cout <<endl;
        data.push_back(temp);
        real.push_back(temp);
    }
}

// READFILE FUNCTION
// INPUT FORMAT    3
void readfile_ucl(string filename){
    line_count = -1;
    element_count = 0;
    cout << "READING FILE = "<< filename <<endl;
    cout << "-------------------"<<endl;
    string line,value;
    ifstream inputfile(filename);

    //Counting The Inputs and Elements (Last is Class Tag)
    if (inputfile.is_open()){
        //Get First Line To Count Element
        string temp;
        getline ( inputfile, temp );
        for(unsigned int i = 0 ; i < temp.size() ; i++){
            if(temp[i] == ',')element_count++;
        }
        cout << "   ELEMENT COUNT = " << element_count <<endl;
        //count the first line
        line_count++;
        while(getline ( inputfile, line )){
            line_count++;
        }
        cout << "   LINE COUNT    = " << line_count <<endl;
    }
    inputfile.close();
    inputfile.clear();
    //Input Gathering From File
    inputfile.open(filename);
    if (inputfile.is_open()){
    int counter = line_count;
        while ( --counter !=-1  ){
            getline ( inputfile, value , ',');
            vector<double> one_input;
            double d = strtod(value.c_str(), NULL);
            one_input.push_back(d);
            for( int i = 1 ; i< element_count ;i++){
                getline ( inputfile, value , ',');
                d = strtod(value.c_str(), NULL);
                one_input.push_back(d);
            }
            //TAG(Last Attribute)
                getline ( inputfile, value , '\n');
                string class_name = value;
                real_class.push_back(class_name);
                class_tag.insert(class_name);
            //Push the data to vector
            data.push_back(one_input);
            real.push_back(one_input);
        }
    inputfile.close();
    }
    cout << "==================="<<endl;
}

// Find Min and Max For Normalization
vector<pair<double,double> > findMinMax (){
    vector<pair<double,double> > results;
    double minimum = 10000;
    double maximum = -10000;
    for(int i = 0 ; i < element_count ; i++ ){
        for(int j = 0 ; j < line_count-1 ; j++){
            if(data[j][i] < minimum) minimum = data[j][i];
            if(data[j][i] > maximum) maximum = data[j][i];
        }
        results.push_back(make_pair(minimum,maximum));
    }
    return results;
}
// Normalization data to range (0,1)
void normalization(){
    vector<pair<double,double> > max_min = findMinMax();
    double data_min[element_count];
    double data_max[element_count];
    for(int i = 0 ; i < element_count ; i++){
        data_max[i] = max_min[i].second;
        data_min[i] = max_min[i].first;
    }
    for(int attribute = 0 ; attribute < element_count ; attribute++){
        for(int index =0 ; index < line_count ; index++){
            data[index][attribute] = (data[index][attribute] - data_min[attribute]) / (data_max[attribute] - data_min[attribute]);
            real[index][attribute] = data[index][attribute];
        }
    }
}

//Find Euclidean Distance for Vector
double euclidean_distance(vector<double> input1 , vector<double> input2){
    double distance=0;
    for(int i = 0 ; i < element_count ; i++ ){
        distance  += (input1[i]-input2[i]) *  (input1[i]-input2[i]) ;
    }
    return sqrt(distance);
}

//Find Euclidean distance for two points
double euclidean_distance(int x1,int y1 , int x2, int y2){
    double distance=0;
    distance = ((x1-x2)*(x1-x2)) + ((y1-y2)*(y1-y2));
    return sqrt(distance);
}

//Training the SOM
void training(vector<double> data_it , Node som_map[][COL]){
     //Variables
    double min_dist = 100000;
    int min_x = 0;
    int min_y = 0;
    double m_dLearningRate = LEARNING_CONST;

    //Find the Winner Node according to the minimum euclidean distance
    for(unsigned int i = 0 ; i < ROW ; i++){
        for(unsigned int j = 0 ; j < COL ; j++){
             double distance = euclidean_distance(data_it , som_map[i][j].weights);
             if(distance < min_dist ){ // MINIMUM - Euclidean Distance
                min_x = i;
                min_y = j;
                min_dist = distance;
             }
        }
    }

    //After we got the winner node ( Minimum Distance )
    //Update the weight at Winner Node
    double m_dTimeConstant = MAX_ITERATION/log(MAX_radius);
    //calculate the width of the neighborhood for this time step
    double m_dNeighbourhoodRadius = MAX_radius * exp(-(double)iteration_count/m_dTimeConstant);

        //ITERATE THROUGH EVERYNODE TO FIND CORRESPONDENT NEIGHBOR
        for(unsigned int i = 0 ; i < ROW ; i++){
            for(unsigned int j = 0 ; j < COL ; j++){
             //Distance from Winner Node
             double DistToNodeSq = (min_x-i) *
                                   (min_x-i) +
                                   (min_y-j) *
                                   (min_y-j);

            // Radius from Center of Winning Node
             double WidthSq = m_dNeighbourhoodRadius * m_dNeighbourhoodRadius;
            // If within the neighborhood Radius adjust its weights
             if (DistToNodeSq < (m_dNeighbourhoodRadius * m_dNeighbourhoodRadius)){
                //calculate by how much its weights are adjusted
                    double m_dInfluence = exp(-(DistToNodeSq) / (2*WidthSq));
                    som_map[i][j].AdjustWeights(data_it,m_dLearningRate,m_dInfluence);
             }
            }
        }
       //REDUCE THE LEARNING RATE
         m_dLearningRate = LEARNING_CONST * exp(-(double)iteration_count/MAX_ITERATION);
}

void drawWeightWindow(){
    sf::RenderWindow anotherWindow(sf::VideoMode(400 , sizeMonitor), "Weight Plot Window",sf::Style::Titlebar);
    while (anotherWindow.isOpen()){
        sf::Event event;
        while (anotherWindow.pollEvent(event)){ //CLOSE BUTTON POLL
            if (event.type == sf::Event::Closed)
                anotherWindow.close();
        }

        sf::Vector2i position = sf::Mouse::getPosition(anotherWindow);
        if (sf::Mouse::isButtonPressed(sf::Mouse::Left) && finished && position.x >=0 && position.x <=anotherWindow.getSize().x && position.x >=0 && position.y <=anotherWindow.getSize().y ){
             if((position.x/margin >= 0 || position.x/margin < COL-1 )&& (position.y/margin >= 0 || position.y/margin < ROW-1)){
                    mouse_x = position.x / margin ;
                    mouse_y = position.y / margin ;
             }else{
                mouse_x = 0;
                mouse_y = 0;
             }
        }
        for(unsigned int i = 0 ; i < ROW ; i++){
            for(unsigned int j = 0 ;j < COL ; j++){
                sf::RectangleShape temp(sf::Vector2f(margin, margin));
                temp.setPosition( j * margin , i * margin);
                //RGBA // CMYK
                int r = 255*som_map[i][j].weights[0]; //int r = 255*(1-som_map[i][j].weights[0])*(1-som_map[i][j].weights[3]);
                int g = 255*som_map[i][j].weights[1]; //int g = 255*(1-som_map[i][j].weights[1])*(1-som_map[i][j].weights[3]);
                int b = 255*som_map[i][j].weights[2]; //int b = 255*(1-som_map[i][j].weights[2])*(1-som_map[i][j].weights[3]);

                sf::Color attribute_color(r,g,b);
                temp.setFillColor(attribute_color);
                anotherWindow.draw(temp);
            }
        }
        anotherWindow.display();
    }
}

void drawUmatrix(){
    sf::RenderWindow UWindow(sf::VideoMode(400 , sizeMonitor), "U-Matrix Thread Window");
    while (UWindow.isOpen()){
        sf::Event event;
        while (UWindow.pollEvent(event)){ //CLOSE BUTTON POLL
            if (event.type == sf::Event::Closed)
                UWindow.close();
        }

        double average=0;
        double TOP=0,BOTTOM=0,LEFT=0,RIGHT=0,LEFTTOP=0,RIGHTTOP=0,LEFTBOTTOM=0,RIGHTBOTTOM=0;
        for(unsigned int i = 0 ; i < ROW ; i++){
                    for(unsigned int j = 0 ;j < COL ; j++){
                    //CALCULATE Average Distance Between Neighborhood ( CITY TOPOLOGY )
                    double divider = 0;
                    TOP=0;BOTTOM=0;LEFT=0;RIGHT=0;

                    if(i!=0){       // TOP CAN BE CALCULATED
                        TOP     = euclidean_distance( som_map[i][j].weights , som_map[i-1][j].weights  );
                        divider++;
                    }
                    if(i!=ROW-1){   // BOTTOM CAN BE CALCULATED
                        BOTTOM  = euclidean_distance( som_map[i][j].weights , som_map[i+1][j].weights  );
                        divider++;
                    }
                    if(j!=0){       // LEFT CAN BE CALCULATED
                        LEFT    = euclidean_distance( som_map[i][j].weights , som_map[i][j-1].weights  );
                        divider++;
                    }
                    if(j!=COL-1){   // RIGHT CAN BE CALCULATED
                        RIGHT   = euclidean_distance( som_map[i][j].weights , som_map[i][j+1].weights  );
                        divider++;
                    }
                    if(i-1 < ROW - 1 && i-1 > 0 && j-1 < COL-1 && j-1 >0){ // LEFTTOP CAN BE CALCULATED
                        LEFTTOP     = euclidean_distance( som_map[i][j].weights , som_map[i-1][j-1].weights  );
                        divider++;
                    }
                    if(i+1 < ROW - 1 && i+1 > 0 && j-1 < COL-1 && j-1 >0){ // LEFTBOTTOM CAN BE CALCULATED
                         LEFTBOTTOM  = euclidean_distance( som_map[i][j].weights , som_map[i+1][j-1].weights  );
                        divider++;
                    }
                    if(i-1 < ROW - 1 && i-1 > 0 && j+1 < COL-1 && j+1 >0){ // RIGHTTOP CAN BE CALCULATED
                       RIGHTTOP    = euclidean_distance( som_map[i][j].weights , som_map[i-1][j+1].weights  );
                        divider++;
                    }
                    if(i+1 < ROW - 1 && i+1 > 0 && j+1 < COL-1 && j+1 >0){ // RIGHTBOTTOM CAN BE CALCULATED
                        RIGHTBOTTOM = euclidean_distance( som_map[i][j].weights , som_map[i+1][j+1].weights  );
                        divider++;
                    }

                        average = TOP + BOTTOM + LEFT + RIGHT + RIGHTTOP + RIGHTBOTTOM + LEFTTOP + LEFTBOTTOM ;
                        average = average / divider;

                        sf::RectangleShape temp(sf::Vector2f(margin, margin));
                        temp.setPosition( j * margin , i * margin);
                        int r = 255-255*average;
                        U_Matrix[i][j] = r;
                        sf::Color attribute_color(r,r,r);
                        temp.setFillColor(attribute_color);
                        UWindow.draw(temp);
                        }
                    }

            //AFTER TRAINING IS COMPLETE classy will be set to 1
            //in order to plot trained data
            if(classy){
                for(unsigned int i = 0 ; i <ROW ; i++){
                    for(unsigned int j = 0 ; j < COL ; j++){
                        sf::RectangleShape tempa(sf::Vector2f(margin, margin));
                        tempa.setPosition( j * margin , i * margin);
                        sf::Color attribute_color(((plotter[i][j]==1)? 255:0),((plotter[i][j]==2)? 255:0),((plotter[i][j]==3)? 255:0),((plotter[i][j]==0)? 0:255));
                        tempa.setFillColor(attribute_color);
                        UWindow.draw(tempa);
                    }
                }
            }

            //PLOT INPUT VALUE with ITS CLASS RELATE TO ACTUAL TRAINING DATA
            if(plotty){
                sf::CircleShape circle;
                sf::CircleShape selected;
                selected.setRadius(margin/3);
                circle.setRadius(margin/2);
                circle.setOutlineColor(sf::Color::Black);
                circle.setOutlineThickness(2);
                 sf::Color attribute_color(((plotter[plot_match_y][plot_match_x]==1)? 255:0),((plotter[plot_match_y][plot_match_x]==2)? 255:0)
                                           ,((plotter[plot_match_y][plot_match_x]==3)? 255:0));
                circle.setFillColor(attribute_color);
                circle.setPosition(new_j*margin,new_i*margin);
                selected.setFillColor(sf::Color::Yellow);
                selected.setPosition(plot_match_x*margin+margin/4, plot_match_y*margin+margin/4);
                UWindow.draw(circle);
                UWindow.draw(selected);
            }
        UWindow.display();
    }
}


pair<int,int> findWinner(vector<double> input_weight){
    double min_dist=100000;
    int x;
    int y;
    for(unsigned int i = 0 ; i < ROW ; i++){
        for(unsigned int j = 0 ; j < COL ; j++){
            double dist = euclidean_distance(input_weight,som_map[i][j].weights);
            if(dist < min_dist){
                x = i;
                y = j;
                min_dist = dist;
            }
        }
    }
    return make_pair(x,y);
}


string number_tostring(double x){
    std::stringstream ss;
    ss << x;
    std::string str = ss.str();
    return str;
}

void detail(){
    sf::RenderWindow detailWindow(sf::VideoMode(400 , sizeMonitor), "DETAIL");
    while (detailWindow.isOpen()){
        sf::Event event;
        while (detailWindow.pollEvent(event)){
            if (event.type == sf::Event::Closed)
                detailWindow.close();
        }
        //Clear Window before drawing
        detailWindow.clear();

        //Text Drawing
        sf::Font font;
        if (!font.loadFromFile("alef.ttf"))cout<< "Font File Not Found" <<endl;
        sf::Text text;
        text.setFont(font);
        //Select Pixel According to mouse position and cell boundary
        if(mouse_y >=0 && mouse_y <=ROW -1 && mouse_x >=0 && mouse_x <= COL-1){
            string a0 = number_tostring(som_map[mouse_y][mouse_x].weights[0]);
            string a1 = number_tostring(som_map[mouse_y][mouse_x].weights[1]);
            string a2 = number_tostring(som_map[mouse_y][mouse_x].weights[2]);
            string a3 = number_tostring(som_map[mouse_y][mouse_x].weights[3]);

            string display = "MOUSE POSITION \n X = "+ number_tostring(mouse_x) +"\n Y = "+ number_tostring(mouse_y)+"\n"+
            " WEIGHT  : " + "\n" + a0 + "\n" + a1 + "\n"+ a2+"\n" + a3 + "\n";

            text.setString(display);
            text.setCharacterSize(24); // in pixels, not points!
            text.setColor(sf::Color::White);

            //Plot the Color Box
            sf::RectangleShape temp(sf::Vector2f(120, 120));
            temp.setPosition( 200 , 200);
            int r = 255*som_map[mouse_y][mouse_x].weights[0];
            int g = 255*som_map[mouse_y][mouse_x].weights[1];
            int b = 255*som_map[mouse_y][mouse_x].weights[2];
            //int a = 255-255*average;
            sf::Color attribute_color(r,g,b);
            temp.setFillColor(attribute_color);
            text.setStyle(sf::Text::Underlined);
            detailWindow.draw(text);
            detailWindow.draw(temp);
        }
        detailWindow.display();
    }
}

//Show Distance between selected node and all the others
void showDistance(){
    sf::RenderWindow showdistance(sf::VideoMode(400 , sizeMonitor), "Distance Difference between selected node and Networks");
    while (showdistance.isOpen()){
        sf::Event event;
        while (showdistance.pollEvent(event)){
            if (event.type == sf::Event::Closed)
                showdistance.close();
        }
        //Clear Window before drawing
        showdistance.clear();
        if(mouse_y >=0 && mouse_y <=ROW -1 && mouse_x >=0 && mouse_x <= COL-1){
            for(unsigned int i = 0 ; i < ROW ; i++){
                for(unsigned int j = 0 ;j < COL ; j++){
                    sf::RectangleShape temp(sf::Vector2f(margin, margin));
                    temp.setPosition( j * margin , i * margin);
                    double distance = euclidean_distance(som_map[mouse_y][mouse_x].weights,som_map[i][j].weights);
                    //GreyScale Plot
                    int r = 255-255*distance;
                    sf::Color attribute_color(r,r,r);
                    temp.setFillColor(attribute_color);
                    showdistance.draw(temp);
                }
            }
        }
         showdistance.display();
    }
}

//Find the nearest class in the trained data
// in order to verify input's class
pair<int,int> findClass(){
    int win_i=0,win_j=0;
    double distance = 10000;
    for(int i = 0 ; i < ROW ; i++ ){
        for( int j = 0 ; j < COL ; j ++){
            if(plotter[i][j] != 0 ){
                double dist = euclidean_distance(i,j,new_i,new_j);
                if( dist < distance ){
                   distance = dist;
                   win_j = j;
                   win_i = i;
                }
            }
        }
    }
    return make_pair(win_i,win_j);
}

//Serial Function
int numDigits(int number){
    int digits = 0;
    if (number < 0) digits = 1;
    while (number) {
        number /= 10;
        digits++;
    }
    return digits;
}
void serial_sent_int(int input){
    cout << "                                                     " << "\r";
    //Find The Amount Of Digits
        int digits = numDigits(input);
        if(input == 0) digits = 1;
    //Convert to CONST CHAR * For transfer
        stringstream temp_str;
        temp_str << (input);
        string str = temp_str.str();
        const char * tempChar = str.c_str();
    //Transfer via Serial
        cout << "[Serial] Sending (integer) : " << tempChar << "\r";
        serial.SendData(tempChar,digits);
    //Ending Seperator
        serial.SendData(",",1);
        Sleep(40);
}
string convertDouble(double value) {
  std::ostringstream o;
  if (!(o << value))
    return "";
  return o.str();
}


void serial_sent_double(double input){
    cout << "                                                     " << "\r";
    //Find the Amount of Integer
    int int_digits = numDigits((int)input); //Cast to int then find
    //convert to CONST CHAR * with fixing 6 precision of decimal
    char tempChar[50];
    int digits = int_digits + 1 + 6;
    snprintf(tempChar,50,"%f",input);
    //Transfer via Serial
        cout << "[Serial] Sending (double ) : " << tempChar << "\r";
        serial.SendData(tempChar,digits);
    //Ending Seperator
        serial.SendData(",",1);
        Sleep(40);
}

void sentMapToMSP(){
    cout << "[Serial] Opening Serial Port Comm. . . . " <<endl;
        if (serial.Open(4, 9600)){
            cout << "[Serial] Port opened successfully" << endl;
        }else{
            cout << "[Serial] Failed to open port!" << endl;
            return ;
        }
        // send SOM_MAP to MSP 430
        for(int i = 0 ; i < ROW ; i++ ){
            for(int j = 0 ; j < COL ; j++ ){
                for(int e = 0 ; e < element_count ; e++){
                    cout << "["<<i<<"]["<<j<<"]["<<e<<"]";
                    serial_sent_double(som_map[i][j].weights[e]);
                }
            }
        }

        for(int i = 0 ; i < ROW ; i++ ){
            for(int j = 0 ; j < COL ; j++ ){
                    cout << "["<<i<<"]["<<j<<"]";
                    serial_sent_int(plotter[i][j]);
            }
        }


}

/*==============================================================
   @MAIN
================================================================*/
int main(){
    cout << "==================="<<endl;
    cout << "Simple-SOM (Junkie-ai's solution) V1.0 Initialized ! "<<endl;
    cout << "==================="<<endl;
    // Set display precision format on console
    cout << setprecision(5);
    cout << fixed;
/*================================
   @INPUT DATA
==================================*/
    //READ INPUT DATA to vector<vector<double>> data
    string filename = "iris.data";
    readfile_ucl(filename); //IRIS DATA
    //random by current time (for rand(); )
    srand(time(0));

    normalization(); //randomdata(); //RGB 3 Element Data
    //Compute the class name to numerical format ( Visualization Purpose )
    computeClassTag();
    //U-Matrix
    sf::Thread u_matrix(&drawUmatrix);
    u_matrix.launch();
/*================================
   @SELF ORGANIZING MAP INITIALIZATION
==================================*/

    //Initialize Weight by Random Number (0,1)
        for (unsigned int i = 0 ; i < ROW ; i++){
            for(unsigned int j = 0 ; j < COL ; j++){
                vector<double> weights;
                if(debug)cout << "For Node = ("<<i << "," << j << ")" <<endl;
                for(int e = 0 ; e < element_count ;e++){
                    double r = ((double) rand() / (RAND_MAX));
                    weights.push_back(r);
                    if(debug)cout << r <<endl;
                }
                som_map[i][j] = Node(weights);
                if(debug)cout <<endl;
            }
        }

    // Consumer Drawing Thread   SHARED DATA = SOM_MAP Weight
    // Drawing Weight
        sf::Thread thread(&drawWeightWindow);
        thread.launch();

    cout << "PRESS ENTER TO START ! ";
    cin.ignore(); //Wait for Any key
/*================================
   @TRAINING PROCESS
   Start
   - Find the Node which have minimum distance according to SINGLE DATA
   - Update that "Winner" Node
   - Update neighborhood related to the "Winner" Node
   End Round ( 1 data from dataset )

   @VISUALIZATION
   - Class Visualization And U-Matrix are running on another thread.
==================================*/
    cout << "==================="<<endl;
    cout << "TRAINING PROCESS " <<endl;
    cout << "==================="<<endl;

        vector<vector<double>>::iterator data_it;
        // FOR EVERY EPOCH ( Training round )
        for(;iteration_count < MAX_ITERATION; ++iteration_count){
            // FOR EACH EVERY INPUT DATA
            for(data_it = data.begin() ; data_it != data.end() ; data_it++){
                cout << "Iteration Count = " << iteration_count << "/" << MAX_ITERATION<< "\r";
                // Train dataset and produce som_map shared data for drawing thread
                training(*data_it,som_map);
            }
            // SHUFFLE DATA for next EPOCH
            std::random_shuffle ( data.begin(), data.end() );
        }
    cout <<endl;
    finished = 1; //Finished Flag to draw detail window

/*================================
   Verification Process
   - Input 4 Data and find the minimum euclidian distance
   between input and weight vector
==================================*/
    cout << "==================="<<endl;
    cout <<endl;
    // Cursor Detail Window
    sf::Thread detailer(&detail);
    detailer.launch();
    // Drawing Weight
    sf::Thread showdistance(&showDistance);
    showdistance.launch();

    //Classification on Training data [Using vector<vector<double>> real]
    // which is the data not shuffled
     for(int i = 0 ; i < line_count ; i++){
            pair<int,int> winner = findWinner(real[i]);
                plotter[winner.first][winner.second] = class_number[real_class[i]];
     }

    //SENDING THINGS TO SERIAL
    if(serial_flag){
    cout << "Send Data to serial : Press Enter to continue."<<endl;
    cout << "==================="<<endl;
        cin.ignore();
        sentMapToMSP();
    }else{
    cout << "==================="<<endl;
    }

     // Set Class Drawing Flag (Draw on U-Matrix Window)
     // will draw the input data with the color R,G,B
     // drawing according to "plotter" array
     classy = 1;
     // Receive New Input test
     double buff;
     if(classy){
        while(1){
            vector<double> test_input;
            cout << "Enter Data For Test input : "<<endl;
            for(int i = 0 ; i < element_count ; i++){
                cout << "test_input["<<i<<"] = " ;
                cin >> buff;
                test_input.push_back(buff);
                serial_sent_double(buff); //SEND VALUE TO MICROCONTROLLER
                cout <<endl;
            }

            pair<int,int> result;
            result = findWinner(test_input);
            new_i = result.first;
            new_j = result.second;
            cout << new_j << ","<<new_i << endl;
            result = findClass(); // Find Reference From Training Data
            plot_match_x = result.second; //J
            plot_match_y = result.first; //I
            plotty = 1;
        }
    }
    cout << "==================="<<endl;
    cout <<endl << "CLOSE THREAD WINDOW TO EXIT" << "\r";
    return 0;
}
