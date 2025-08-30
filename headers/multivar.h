//Sebastian Gallardo --- Septiembre 2020
//This header defines the classes mv_domain, and mv_iterator.
//mv_domain represents a cartesian product of evenly spaced points (an orthogonal lattice) in VAR^nVar space (the spacing can be different for different variables).
//mv_iterator includes the function go() which performs the calculation of of chosen member functions in funcList for all points in the mv_iterator's mv_domain and saves the data to an ofstream.
//mv_iterator's constructor asks the user for input via console.
//I use this header to analyze N-dimensional functions.

//Libraries
#include <iostream>
#include <fstream>
#include <functional>
#include <iomanip>  // Add this line for setprecision

#define OBJVAR_FUNC function<VAR(OBJECT*,VAR[])>
#define FILENAME_BUFFER_SIZE 256

using namespace std;

//Class definitions
template <class VAR> struct mv_domain{
    int nVar; //Number of variables (dimension) of the mv_domain
    int nIter; //how many variables to iterate over
    int *iterVar; //indexes of the variables to iterate over
    VAR *minIter; //starting values for variables to iterate
    VAR *maxIter; //ending values for variables to iterate
    VAR *stepIter; //step values for variables to iterate
    VAR *fixedValue; //starting value for fixed variables, in order least-greatest
    const string *varNames; //names of the variables in question 
    const string *funcNames; // names of the functions
    mv_domain(){};
    ~mv_domain();
    int readFromUserInput();
};

template <class OBJECT, class VAR> class mv_iterator{
    OBJECT *obj;    //object whose functions we wish to iterate
    OBJVAR_FUNC *funcList;   //list of functions (dependent variables)
    int nFuncList;    //length of funcList
    int *iterFunc;  //which funcitons of funcList to iterate
    int nIterFunc;  //length of iterFunc
    mv_domain<VAR> dom;   //domain of dependent variables to iterate over
    ofstream out;   //stream to which to save output
    VAR *currentVar;  //stores current value of dependent variables

    public:
    mv_iterator(OBJECT&, int nVarI, const OBJVAR_FUNC funcListI[], int nFuncListI, const string varNamesI[], const string funcNamesI[], const string folderNameI);
    ~mv_iterator();
    void go();
    void go_aux(int currentStep=0);
};

//Class Functions

template <class VAR> mv_domain<VAR>::~mv_domain(){
    delete[] iterVar;
    delete[] fixedValue;
    delete[] minIter;
    delete[] maxIter;
    delete[] stepIter;
}

template <class VAR> int mv_domain<VAR>::readFromUserInput(void){
    cout << "Total number of independent variables: " << nVar << '\n';
    cout << "Number of independent variables to iterate over: ";
    cin >> nIter;

    iterVar = new int[nIter];
    minIter = new VAR[nIter];
    maxIter = new VAR[nIter];
    stepIter = new VAR[nIter];
    fixedValue = new VAR[nVar-nIter];

    for(int i=0; i<nIter; i++){
        cout << "Iteration bounds of x";
        cin >> iterVar[i];
        cout << "\t(" << varNames[iterVar[i]] << ")\n";
        cout << "\tMin: ";
        cin >> minIter[i];
        cout << "\tMax: ";
        cin >> maxIter[i];
        cout << "\tStep: ";
        cin >> stepIter[i];
    }

    {
        int i=0; 
        bool isIter;
        int fixedNumber = nVar-nIter;
        for(int fixed=0; fixed<fixedNumber; i++){
            if(i>=nVar) throw "Couldn't find all expected fixed independent variables!";
            //for every variable check if it is iterating, when not ask for the fixed value
            isIter=false;
            for(int j=0; j<nIter; j++){
                if(i==iterVar[j]){
                    isIter=true;
                    break;
                }
            }
            if(!isIter){
                cout << "Fixed value of " << varNames[i] << ": ";
                cin >> fixedValue[fixed];
                fixed++;
            }
        }
    }
    return 0;
}


template <class OBJECT,class VAR> mv_iterator<OBJECT,VAR>::mv_iterator(OBJECT& pobj, int nVarI, const OBJVAR_FUNC funcListI[], int nFuncListI, const string varNamesI[], const string funcNamesI[], const string folderNameI){
    int input;
    char filenameInput[FILENAME_BUFFER_SIZE];

    obj = &pobj;
    dom.varNames = varNamesI;
    dom.funcNames = funcNamesI;
    dom.nVar = nVarI;
    currentVar = new VAR[nVarI];
    funcList = new OBJVAR_FUNC[nFuncListI];
    for(int i=0;i<nFuncListI;i++) funcList[i]=funcListI[i];

    //ask user for the domain
    dom.readFromUserInput();

    
    //now ask user for the functions
    cout << "Total number of functions: " << nFuncListI << '\n';
    cout << "Number of functions to iterate over: ";
    cin >> nIterFunc;
    if(nIterFunc<1||nIterFunc>nFuncListI) throw "Invalid number of functions!";

    iterFunc = new int[nIterFunc];

    cout << "Iterate over these functions:\n";
    for(int i=0;i<nIterFunc;i++){
        cout <<"\tf";
        cin >> input;
        cout << "\t(" << dom.funcNames[input] << ")\n";
        if(input<0||input>=nFuncListI) throw "Invalid function number!";
        iterFunc[i]=input;
    }

    //finally, ask user for output filename
    cout << "Save output to filename: ";
    cin >> filenameInput;
    out.open(folderNameI+filenameInput+"_NK"+to_string(NK)+".dat");
    out << std::fixed << std::setprecision(10); // PRECISION
    if(!out.is_open()) throw "Couldn't open file!";

    cout << '\n';
}
    
template <class OBJECT,class VAR> mv_iterator<OBJECT,VAR>::~mv_iterator(){
    delete[] currentVar;
    delete[] funcList;
    delete[] iterFunc;
    out.close();
}

template <class OBJECT,class VAR> void mv_iterator<OBJECT,VAR>::go(){
    //print and set fixed variables
    out << '#';
    {
        int i=0; 
        bool isIter;
        int fixedNumber = dom.nVar-dom.nIter;
        for(int fixed=0; fixed<fixedNumber; i++){
            if(i>=dom.nVar) throw "Couldn't find all expected fixed variables!";
            //for every variable check if it is iterating, when not set the fixed value
            isIter=false;
            for(int j=0; j<dom.nIter; j++){
                if(i==dom.iterVar[j]){
                    isIter=true;
                    break;
                }
            }
            if(!isIter){
                currentVar[i]=dom.fixedValue[fixed];
                out << dom.varNames[i] << '=' << currentVar[i] << "; ";
                fixed++;
            }
        }
    }
    out << '\n';

    //print column names
    out << '#';
    for(int i=0; i<dom.nIter; i++) out << dom.varNames[dom.iterVar[i]] << '\t';
    for(int i=0; i<nIterFunc; i++) out << dom.funcNames[iterFunc[i]] << '\t';
    out << '\n';
    
    cout << '#';
    for(int i=0; i<dom.nIter; i++) cout << dom.varNames[dom.iterVar[i]] << '\t';
    for(int i=0; i<nIterFunc; i++) cout << dom.funcNames[iterFunc[i]] << '\t';
    cout << '\n';
    

    //run nested loops
    cout << "Iterando...\n";
    go_aux();
}


template <class OBJECT,class VAR> void mv_iterator<OBJECT,VAR>::go_aux(int currentStep){
    if(currentStep!=dom.nIter){
        //iterate over the current variable
        int currentVarN = dom.iterVar[currentStep];
        for(currentVar[currentVarN]=dom.minIter[currentStep]; currentVar[currentVarN]<=dom.maxIter[currentStep]; currentVar[currentVarN]+=dom.stepIter[currentStep])
            go_aux(currentStep+1);
    }else{
        //time to print a line
        double aux;
        for(int i=0; i<dom.nIter; i++){
            out << (aux=currentVar[dom.iterVar[i]]) << '\t';
            cout << aux << '\t';
        }
        for(int i=0; i<nIterFunc-1; i++){
            out << (aux=funcList[iterFunc[i]](obj,currentVar)) << '\t';
            cout << aux << '\t';
        }
        out << (aux=funcList[iterFunc[nIterFunc-1]](obj,currentVar)) << '\n';
        cout << aux << '\n';
    }
}

