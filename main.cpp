#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <random>
#include <fstream>
#include "Node.h"
using namespace std;

int debug = 1;
const double LEARNING_CONST = 0.1;
unsigned int ROW = 2;
unsigned int COL = 2;
int line_count = 0;
int element_count =0;

//CREATE DATASET
    vector<vector<double> > data;

void randomdata(){

    for(int i = 0 ; i < 5 ; i++){
            vector<double> temp;
        for(int j = 0 ; j < 4 ; j++ ){
            double r = ((double) rand() / (RAND_MAX));
            temp.push_back(r);
        }
        data.push_back(temp);
    }

}

//READFILE FUNCTION
void readfile_ucl(string filename){
    cout << "==================="<<endl;
    cout << "READING FILE = test.data" <<endl;
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
    cout << "==================="<<endl;
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
            //DROP TAG
            getline ( inputfile, value , '\n');
            cout << "   VECTOR CONTENT = <" ;
            for ( vector<double>::iterator i = one_input.begin() ; i < one_input.end() ; i ++){
                cout << *i << ((i==one_input.end()-1)? "":",");
            }
            cout << ">" <<endl;
            data.push_back(one_input);
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
    string filename = "test.data";
    readfile_ucl(filename);
    //randomdata();
    normalization();
/*================================
   SELF ORGANIZING MAP INITIALIZATION
==================================*/
    //Data Structure
    Node som_map[ROW][COL];//vector<Node> som_map;
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

/*================================
   TRAINING PROCESS
   - Find the Node which have minimum distance according to SINGLE DATA
   - Update that "Winner" Node
   - Update neighborhood related to the "Winner" Node
   END ROUND ( 1 data from dataset )
==================================*/
    //Variables
    vector<vector<double>>::iterator data_it;
    vector<Node>::iterator node_it;
    double min_dist = 100000;
    int min_x = 0;
    int min_y = 0;
    unsigned int iteration_count=0;
    double m_dLearningRate = LEARNING_CONST;

    //FOR EACH EVERY INPUT DATA
    for(data_it = data.begin() ; data_it != data.end() ; data_it++){
        //FOR EVERY NODE IN MAP
        for(int i = 0 ; i < ROW ; i++){
            for(int j = 0 ; j < COL ; j++){
             double distance = euclidian_distance(*data_it , som_map[i][j].weights);
             if(distance < min_dist ){
                min_x = som_map[i][j].x_pos;
                min_y = som_map[i][j].y_pos;
                min_dist = distance;
            }
            }
        }


        cout << "DATA " << endl;
        displayVector(*data_it);
        cout << endl <<"Match the node = (" << min_x << "," << min_y << ")" <<endl;
        cout << "with distance = " << min_dist <<endl <<endl;

        //Update the weight at Winner Node
        double MAX_radius = max(ROW, COL)/2;
        //calculate the width of the neighbourhood for this timestep
        double m_dNeighbourhoodRadius = MAX_radius * exp(-(double)iteration_count/0.1);


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
                    som_map[i][j].AdjustWeights(*data_it,m_dLearningRate,m_dInfluence);
             }
            }
           }

       //REDUCE THE LEARNING RATE
         m_dLearningRate = LEARNING_CONST * exp(-(double)iteration_count/line_count);

        ++iteration_count;

        //Reset minimum data after finishing finding minimal node
        min_dist = 100000;
        min_x = 0;
        min_y = 0;

    //UPDATESCREEN();
    }//End 1 Data Iteration

    return 0;
}
