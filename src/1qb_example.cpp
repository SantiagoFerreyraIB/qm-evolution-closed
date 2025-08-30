#include <iostream>
#include <fstream>
#include <complex>
#include <armadillo>
#include <math.h>
#include <functional>
#include "../include/evolucion.h"
#include "../include/multivar.h"

#define H_FUNC function<double(Qubit*, double*)>

#define C1 (cx_double){1,0}
#define CI (cx_double){0,1}
#define C0 (cx_double){0,0}
#define Csqrt2 (cx_double){1/sqrt(2),0}

using namespace std;
using namespace arma;

// Pauli Matrices
#ifndef PAULI_SIGMA_MATRICES
#define PAULI_SIGMA_MATRICES
const cx_mat ID = {{C1,C0},{C0,C1}};
const cx_mat SX = {{C0,C1},{C1,C0}};
const cx_mat SY = {{C0,-CI},{CI,C0}};
const cx_mat SZ = {{C1,C0},{C0,-C1}};
const cx_mat Ypi2 = {{Csqrt2,-Csqrt2},{Csqrt2,Csqrt2}};
#endif

cx_mat Utg = Ypi2; // Target unitary (for Fidelity example)

// Hamiltonian and driving definitions

#define DIM 2 // Dimension of the quantum system
#define H0_PARS 1 // Parameters of H0 that could be iterated


double w_q; // H0 pars
double A, w; // Driving pars

cx_mat H0(Qubit *Q, double *par){
	w_q = par[0]; // Qubit frequency
    Q->wq = w_q;

    // Definition of H0 and V in some initial basis
    cx_mat H0 = - .5 *w_q * SZ;
    cx_mat V = SX;
    
	vec driving_evals(DIM); cx_mat driving_evecs(DIM, DIM);
	eig_sym(driving_evals, driving_evecs, symmatu(V));
	Q->V = driving_evals; // Set driving (in diagonal basis)

	return driving_evecs.t() * H0 * driving_evecs; // Return H0 in the basis where V is diagonal
}

// DRIVING FUNCTION f_drive(params, t)
std::function<double(double)> f_drive(
    double A_, // driving amplitude
    double w_ // driving frequency
) {
    auto f = [=](double t) {
        return - .5 * A_ * sin(w_ * t);
    };

    return f;
}

// Setter function (after main)
void setPars(Qubit *Q, double *x);

// Name of destination folder
string folderName = "../examples/";

int main() {

    char input;

    Qubit single_qubit(H0, DIM, H0_PARS);
    cout << "NK = " << NK << '\n';
    cout << "x0: w_q\n";
    cout << "x1: A/w_q\n";
    cout << "x2: w/w_q\n";

    // Name of variables and functions to iterate.

	string varNames[] = {"w_q", "A/w_q", "w/w_q"};
	string funcNames[] = {"P01_T", "Fidelidad"};

    // Here I initialize functions from <functional> with lambdas.
    // The functions take the Qubit class and a set of parameters given by the user.

    // Iterable functions
    #define NFUNCS 2
    H_FUNC funcList[NFUNCS];

    // Transition probability after a period
    funcList[0] = [](Qubit *Q, double *x){
        setPars(Q,x);
        cx_vec psi_i = Q->eigst.col(0);
        cx_vec psi_f = Q->eigst.col(1);
        return Q->transProb(psi_i, psi_f);
    };
    cout << "f0: P01_T\n";

    // Fidelity between the numerical time evolution and a unitary target
    funcList[1] = [](Qubit *Q, double *x){
        setPars(Q,x);
        return Q->fidelidad(Utg);
    };
    cout << "f1: Fidelidad\n";

    mv_iterator<Qubit,double> iter(single_qubit, H0_PARS+2, funcList, NFUNCS, varNames, funcNames, folderName);
    iter.go();

}

// Setter function for iteration
void setPars(Qubit *Q, double *x){
    int npar = Q->npar;
    bool H0_changed = false;
    bool V_changed = false;

    // Update H0 vars
	for(int i=0;i<npar;i++) if(Q->par[i]!=x[i]){
		H0_changed = true;	
		Q->par[i] = x[i];
	}

    // If H0 vars changed, recalculate H0
    if (H0_changed){
        cout << "H0 changed\n";
		Q->calcH0();
	}

    // Update driving vars
    if(A!=(x[npar]*Q->wq) || w!=(x[npar+1]*Q->wq)){
        V_changed = true;
        A = x[npar]*Q->wq;
        w = x[npar+1]*Q->wq;

        Q->f = f_drive(A, w);
        Q->T = 2*M_PI/w;
    }

    // If H0 or V changed, recalculate the time evolution operator
    if(H0_changed || V_changed){
        Q->calcTimeEvolution();
    }

}