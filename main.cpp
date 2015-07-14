#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <random>
#include <fstream>

//Node Data Structure
#include "Node.h"

//Plot Stuff
#include <SFML/System.hpp>
#include <SFML/Graphics.hpp>

using namespace std;

int debug = 0;
const double LEARNING_CONST = 0.2;
unsigned const int ROW = 40;
unsigned const int COL = 40;
unsigned const int MAX_ITERATION = 250;
double MAX_radius = max(ROW, COL)/2;
unsigned int iteration_count = 0;
Node som_map[ROW][COL];
int line_count = ROW*COL;
int element_count =4;
  int sizeMonitor = 400;
    int margin = sizeMonitor/ROW;

//CREATE DATASET
    vector<vector<double> > data;
    vector<vector<double> > real;
//Random DATASET
void randomdata(){

    for(int i = 0 ; i < line_count ; i++){
            vector<double> temp;
        for(int j = 0 ; j < element_count ; j++ ){
            double r = ((double) rand() / (RAND_MAX))*255;
            temp.push_back((int)r);
        }
        data.push_back(temp);
        real.push_back(temp);
    }
}

//READFILE FUNCTION
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
    //cout << "==================="<<endl;
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
            //DROP TAG(Last Attribute)
            getline ( inputfile, value , '\n');
            /*cout << "   VECTOR CONTENT = <" ;
            for ( vector<double>::iterator i = one_input.begin() ; i < one_input.end() ; i ++){
                cout << *i << ((i==one_input.end()-1)? "":",");
            }
            cout << ">" <<endl;*/
            data.push_back(one_input);
            real.push_back(one_input);
        }
    inputfile.close();
    }

    cout << "==================="<<endl;
}

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


void displayVector(vector<double> input){
    cout << "<" ;
    for ( vector<double>::iterator i = input.begin() ; i < input.end() ; i ++){
        cout << *i << ((i==input.end()-1)? "":",");
    }
    cout << ">";
}

double euclidian_distance(vector<double> input1 , vector<double> input2){
    double distance=0;
    for(int i = 0 ; i < element_count ; i++ ){
        distance  += (input1[i]-input2[i]) *  (input1[i]-input2[i]) ;
    }

    return sqrt(distance);

}

void training(vector<double> data_it , Node som_map[][COL]){
     //Variables
    double min_dist = 100000;
    int min_x = 0;
    int min_y = 0;
    double m_dLearningRate = LEARNING_CONST;

    for(int i = 0 ; i < ROW ; i++){
        for(int j = 0 ; j < COL ; j++){
             double distance = euclidian_distance(data_it , som_map[i][j].weights);
             if(distance < min_dist ){
                min_x = som_map[i][j].x_pos;
                min_y = som_map[i][j].y_pos;
                min_dist = distance;
             }
        }
    }

    if(debug){
        cout << "DATA " << endl;
        displayVector(data_it);
        cout << endl <<"Match the node = (" << min_x << "," << min_y << ")" <<endl;
        cout << "with distance = " << min_dist <<endl <<endl;
    }
        //Update the weight at Winner Node
        double m_dTimeConstant = MAX_ITERATION/log(MAX_radius);
        //calculate the width of the neighbourhood for this timestep
        double m_dNeighbourhoodRadius = MAX_radius * exp(-(double)iteration_count/m_dTimeConstant);

        //ITERATE THROUGH EVERYNODE TO FIND CORRESPONDENT NEIGHBOR
        for(int i = 0 ; i < ROW ; i++){
            for(int j = 0 ; j < COL ; j++){
             //Distance from Winner Node
             double DistToNodeSq = (min_x-som_map[i][j].x_pos) *
                                   (min_x-som_map[i][j].x_pos) +
                                   (min_y-som_map[i][j].y_pos) *
                                   (min_y-som_map[i][j].y_pos);

            //Radius from Center of Winning Node
             double WidthSq = m_dNeighbourhoodRadius * m_dNeighbourhoodRadius;

             //if within the neighbourhood adjust its weights
             if (DistToNodeSq < (m_dNeighbourhoodRadius * m_dNeighbourhoodRadius)){
                //calculate by how much its weights are adjusted
                    double m_dInfluence = exp(-(DistToNodeSq) / (2*WidthSq));
                    som_map[i][j].AdjustWeights(data_it,m_dLearningRate,m_dInfluence);
             }
            }
        }

       //REDUCE THE LEARNING RATE
         m_dLearningRate = LEARNING_CONST * exp(-(double)iteration_count/MAX_ITERATION);

        //Reset minimum data after finishing finding minimal node
        min_dist = 100000;
        min_x = 0;
        min_y = 0;
}

void drawthis(){
    sf::RenderWindow anotherWindow(sf::VideoMode(400 , sizeMonitor), "Threaded Window");
    while (anotherWindow.isOpen()){
        sf::Event event;
        while (anotherWindow.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                anotherWindow.close();
        }
        for(unsigned int i = 0 ; i < ROW ; i++){
                    for(unsigned int j = 0 ;j < COL ; j++){
                        sf::RectangleShape temp(sf::Vector2f(margin, margin));
                        temp.setPosition( j * margin , i * margin);
                        int r = 255*som_map[i][j].weights[0];
                        int g = 255*som_map[i][j].weights[1];
                        int b = 255*som_map[i][j].weights[2];
                        int a = 255*som_map[i][j].weights[3];
                        sf::Color attribute_color(r,g,b);
                        temp.setFillColor(attribute_color);
                        anotherWindow.draw(temp);
                        }
                    }
        anotherWindow.display();
    }


}


void findTheResult(){
    vector<double> temp;
    cout << "INPUT VALUES = ";
    double dump;
    for(int z = 0 ; z < element_count ; z++){
        cin>>dump;
        temp.push_back(dump);
    }
    double min_dist=100000;
    int x;
    int y;
    for(int i = 0 ; i < ROW ; i++){
        for(int j = 0 ; j < COL ; j++){
            double dist = euclidian_distance(temp,som_map[i][j].weights);
            if(dist < min_dist){
                x = i;
                y = j;
                min_dist = dist;
            }
        }
    }
    cout << "MIN IS = (" << x << "," << y << ") =" << min_dist<<endl;
}

int main(){
/*================================
   PARAMETERS
==================================*/
    cout << setprecision(5);
    cout << fixed;
/*================================
   INPUT DATA
==================================*/
    //READ INPUT DATA to vector<vector<double>> data
    string filename = "iris.data";
    readfile_ucl(filename);
    srand(time(0));
    //randomdata();
    normalization();
/*================================
   SELF ORGANIZING MAP INITIALIZATION
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
    //Consumer Drawing Thread   SHARED DATA = SOM_MAP
    sf::Thread thread(&drawthis);
    //Start Thread
    thread.launch();
    cout << "PRESS TO START ! ";
    cin.ignore();
/*================================
   TRAINING PROCESS
   - Find the Node which have minimum distance according to SINGLE DATA
   - Update that "Winner" Node
   - Update neighborhood related to the "Winner" Node
   END ROUND ( 1 data from dataset )
==================================*/
    cout << "TRAINING PROCESS " <<endl;
    cout << "==================="<<endl;
    unsigned int counter = line_count;

        //FOR EACH EVERY INPUT DATA
        vector<vector<double>>::iterator data_it;
        for(;iteration_count < MAX_ITERATION; ++iteration_count){
            for(data_it = data.begin() ; data_it != data.end() ; data_it++){
                //FOR EVERY NODE IN MAP
                cout << "Iteration Count = " << iteration_count << "/" << MAX_ITERATION<< "\r";
                //PRODUCER for SOM_MAP
                training(*data_it,som_map);
            }
            //SHUFFLE
            std::random_shuffle ( data.begin(), data.end() );
        }
    cout <<endl;

    findTheResult();

    cout << "==================="<<endl;
    cout <<endl << "CLOSE THREAD WINDOW TO EXIT" << "\r";
    return 0;
}
