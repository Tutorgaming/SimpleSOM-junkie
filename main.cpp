#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <random>
#include <fstream>

//Node Data Structure
#include "Node.h"

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
    unsigned const int  ROW             = 20;
    unsigned const int  COL             = 20;
    unsigned const int  MAX_ITERATION   = 80;
    double              MAX_radius      = max(ROW, COL)/2;
    const double        LEARNING_CONST  = 0.5;
    unsigned int        iteration_count = 0;
    int                 element_count   = 4;
    int                 class_count     = 3; // IRIS DATASET = 3
    int                 line_count      = ROW*COL;
    int                 sizeMonitor     = 400;
    int                 margin          = sizeMonitor/ROW;
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
    int                 finished        = 0;
    int                 classy          = 0;
    int                 num             = 0;
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
        //for(int j = 0 ; j < element_count ; j++ ){
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
// INPUT FORMAT    [data1,data2,data3,...,class]
void readfile_ucl(string filename){
    line_count = -1;
    element_count = 0;
    cout << "==================="<<endl;
    cout << "READING FILE = "<< filename <<endl;
    cout << "==================="<<endl;
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
        cout << "   LINE COUNT = " << line_count <<endl;
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
                min_x = som_map[i][j].x_pos;
                min_y = som_map[i][j].y_pos;
                min_dist = distance;
             }
        }
    }

    //After we got the winner node ( Minimum Distance )
    //Update the weight at Winner Node
        double m_dTimeConstant = MAX_ITERATION/log(MAX_radius);
        //calculate the width of the neighborhood for this timestep
        double m_dNeighbourhoodRadius = MAX_radius * exp(-(double)iteration_count/m_dTimeConstant);

        //ITERATE THROUGH EVERYNODE TO FIND CORRESPONDENT NEIGHBOR
        for(unsigned int i = 0 ; i < ROW ; i++){
            for(unsigned int j = 0 ; j < COL ; j++){
             //Distance from Winner Node
             double DistToNodeSq = (min_x-som_map[i][j].x_pos) *
                                   (min_x-som_map[i][j].x_pos) +
                                   (min_y-som_map[i][j].y_pos) *
                                   (min_y-som_map[i][j].y_pos);

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

void drawthis(){
    sf::RenderWindow anotherWindow(sf::VideoMode(400 , sizeMonitor), "Weight Plot Window");
    while (anotherWindow.isOpen()){
        sf::Event event;
        while (anotherWindow.pollEvent(event)){ //CLOSE BUTTON POLL
            if (event.type == sf::Event::Closed)
                anotherWindow.close();
        }
        if (sf::Mouse::isButtonPressed(sf::Mouse::Left) && finished){
             sf::Vector2i position = sf::Mouse::getPosition(anotherWindow);
             if((position.x/margin >= 0 || position.x/margin < COL-1 )&& (position.y/margin >= 0 || position.y/margin < ROW-1)){
                if(mouse_x != 40 || mouse_y != 40){
                    mouse_x = position.x / margin ;
                    mouse_y = position.y / margin ;
                }else{
                    mouse_x = 0;
                    mouse_y = 0;
                }
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
            //in order to plot training data
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


string inttostring(double x){
    std::stringstream ss;
    ss << x;
    std::string str = ss.str();
    return str;
}

void detail(){
    sf::RenderWindow detailWindow(sf::VideoMode(400 , sizeMonitor), "DETAIL");
    while (detailWindow.isOpen()){
        sf::Event event;
        while (detailWindow.pollEvent(event))
        {
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
            string a0 = inttostring(som_map[mouse_y][mouse_x].weights[0]);
            string a1 = inttostring(som_map[mouse_y][mouse_x].weights[1]);
            string a2 = inttostring(som_map[mouse_y][mouse_x].weights[2]);
            string a3 = inttostring(som_map[mouse_y][mouse_x].weights[3]);

            string display = "MOUSE POSITION \n X = "+ inttostring(mouse_x) +"\n Y = "+ inttostring(mouse_y)+"\n"+
            " WEIGHT  : " + "\n" + a0 + "\n" + a1 + "\n"+ a2+"\n" + a3 + "\n";

            //cout << "Distance between input and selected node = " << euclidean_distance(real[num],som_map[mouse_y][mouse_x].weights) << "\r";
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

void showDistance(){
    sf::RenderWindow showdistance(sf::VideoMode(400 , sizeMonitor), "DETAIL");
    while (showdistance.isOpen()){
        sf::Event event;
        while (showdistance.pollEvent(event)){
            if (event.type == sf::Event::Closed)
                showdistance.close();
        }
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
        showdistance.display();
    }
}


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


/*==============================================================
   @MAIN
================================================================*/
int main(){
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
    //randomdata(); //RGB 3 Element Data
    normalization();
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
                som_map[i][j] = Node(weights,i,j);
                if(debug)cout <<endl;
            }
        }
    // Consumer Drawing Thread   SHARED DATA = SOM_MAP Weight
    // Drawing Weight
        sf::Thread thread(&drawthis);
        thread.launch();

    cout << "PRESS TO START ! ";
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

    cout <<endl <<endl;
    // Cursor Detail Window
    sf::Thread detailer(&detail);
    detailer.launch();

    //Classification on Training data [Using vector<vector<double>> real]
    // which is the data not shuffled
     for(int i = 0 ; i < line_count ; i++){
            pair<int,int> winner = findWinner(real[i]);
                plotter[winner.first][winner.second] = class_number[real_class[i]];
     }
     // Set Class Drawing Flag (Draw on U-Matrix Window)
     // will draw the input data with the color R,G,B
     classy = 1;

     // Receive New Input test
     double buff;
     // Drawing Weight
        sf::Thread showdistance(&showDistance);
        showdistance.launch();

     while(1){
        vector<double> test_input;
        cout << "Enter Data For Test input : "<<endl;
         for(int i = 0 ; i < element_count ; i++){
            cout << "test_input["<<i<<"] = " ;
            cin >> buff;
            test_input.push_back(buff);
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
    cout << "==================="<<endl;
    cout <<endl << "CLOSE THREAD WINDOW TO EXIT" << "\r";
    return 0;
}
