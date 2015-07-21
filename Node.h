#include <vector>
class Node{

public:
    std::vector<double> weights;
    Node(std::vector<double> weight){
        weights = weight;
    }

    Node(){
    }

    void AdjustWeights(const std::vector<double> &target,
                          const double LearningRate,
                          const double Influence){
      for (unsigned int w=0; w<target.size(); ++w)
      {
        weights[w] += LearningRate * Influence * (target[w] - weights[w]);
      }
    }

};


