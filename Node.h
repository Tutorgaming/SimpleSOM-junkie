#include <vector>
class Node{

public:
    std::vector<double> weights;
    int x_pos;
    int y_pos;
    Node(std::vector<double> weight,int x,int y){
        weights = weight;
        x_pos = x;
        y_pos = y;
    }

    Node(){
        x_pos = 0;
        y_pos = 0;
    }

//    *Node getnode(int x , int y){
//        return
//    }
    void AdjustWeights(const std::vector<double> &target,
                          const double LearningRate,
                          const double Influence){
      for (int w=0; w<target.size(); ++w)
      {
        weights[w] += LearningRate * Influence * (target[w] - weights[w]);
      }
    }

};


