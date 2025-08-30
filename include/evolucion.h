// Santiago Ferreyra -- September 2025
// This library defines the Qubit class, which lets define and solve a time dependent quantum mechanics problem with Hamiltonian of the form H(t)=H0+V*f(t),
// where V is an arbitrary hermitian matrix, and f(t) is a protocol defined from 0 to T. It is optimized for when we are not interested in intermediate steps, just the final time T.
// H0 and f(t) is allowed to depend in a set of parameters which can be set via a setter function.
// The problem is described in a basis where V is diagonal w/o loss of generality.

#define NK 100 // Discretization of time period

//Libraries
#include <cstdlib>
#include <math.h>
#include <complex>
#include <armadillo>

#define C1 (cx_double){1,0}
#define CI (cx_double){0,1}
#define C0 (cx_double){0,0}

using namespace std;
using namespace arma;

//Pauli Matrices
#ifndef PAULI_SIGMA_MATRICES
#define PAULI_SIGMA_MATRICES
const cx_mat ID = {{C1,C0},{C0,C1}};
const cx_mat SX = {{C0,C1},{C1,C0}};
const cx_mat SY = {{C0,-CI},{CI,C0}};
const cx_mat SZ = {{C1,C0},{C0,-C1}};
#endif

class Qubit {
    public:
    int dim, npar; //dim: Hamiltonian dimension, npar: number of H0 parameters
    cx_mat (*H0_func)(Qubit* ,double*); //gives H0 from par
    vec V; //driving operator
    std::function<double(double)> f;  //driving function f(t)

    // Hamiltonian parameters
    double *par;   // vector of H0 parameters
    double wq; // qubit frequency wq = E1 - E0
    double T; // Total time of the protocol

    //Solution Output (H0)
    cx_mat H0;
    cx_mat eigst;
    vec energy; //of H0, in ascending energy order

    //Solution Output (Time evolution operator at final time T)
    cx_mat U_T; // (dim, dim)

    // function to construct H0
    void calcH0();
    // function for calculating time evolution
    void calcTimeEvolution();

    // Output functions
    cx_vec propagate(cx_vec psi0);
    double transProb(cx_vec psi_i, cx_vec psi_f);
    double fidelidad(cx_mat U_tg);

    //constructors/destructors
    Qubit(cx_mat(*H0_funcI)(Qubit*, double*), int dimI, int nparI):
        H0_func(H0_funcI),
        V(zeros<vec>(dimI)),
        f(nullptr),
        dim(dimI),
        npar(nparI),
        T(0),
        eigst(zeros<cx_mat>(dimI,dimI)),
        energy(zeros<vec>(dimI)),
        U_T(zeros<cx_mat>(dimI,dimI))
        {
        par = new double[nparI];
        for(int i=0;i<nparI;i++) par[i]=0;
        }
    ~Qubit()
        {
        delete[] par;
        }
};

// Class functions

void Qubit::calcH0(){ //Constructs H0 and calculates its eigensystem with setted params
    H0 = H0_func(this, par);
    eig_sym(energy,eigst,symmatu(H0)); //Symmatu is to avoid non-hermiticity caused by rounding errors.
}

void Qubit::calcTimeEvolution() { //Calculates time evolution operator at time T
    #define s (1./(4.-pow(4.,1./3.)))
    #define z (1-4*s)

    int i, it;
    double t;
    double dt = T/NK;

    // Identity at first step
    U_T = cx_mat(dim, dim, fill::eye);

    for(it=0; it<NK; it++){
        //cout << "t = " << it*dt << endl;
        //cout << "f(t) = " << f(it*dt) << endl;
        double v1=f((it*dt+s*dt/2));
        double v2=f((it*dt+3*s*dt/2));
        double v3=f((it*dt+2*s*dt+z*dt/2));
        double v4=f((it*dt+2*s*dt+z*dt+s*dt/2));
        double v5=f(((it+1)*dt-s*dt/2));
        cx_mat uv1_(dim,dim,fill::zeros);
        cx_mat uv2_(dim,dim,fill::zeros);
        cx_mat uv3_(dim,dim,fill::zeros);
        cx_mat uv4_(dim,dim,fill::zeros);
        cx_mat uv5_(dim,dim,fill::zeros);
        cx_mat uv6_(dim,dim,fill::zeros);
        for(i=0;i<dim;i++){
            uv1_(i,i)=exp(-CI*V[i]*(v1*dt*s/2));
            uv2_(i,i)=exp(-CI*V[i]*(v1*dt*s/2+v2*dt*s/2));
            uv3_(i,i)=exp(-CI*V[i]*(v2*dt*s/2+v3*dt*z/2));
            uv4_(i,i)=exp(-CI*V[i]*(v3*dt*z/2+v4*dt*s/2));
            uv5_(i,i)=exp(-CI*V[i]*(v4*dt*s/2+v5*dt*s/2));
            uv6_(i,i)=exp(-CI*V[i]*(v5*dt*s/2));
        }

        cx_mat uh1_(dim,dim,fill::zeros);
        cx_mat uh2_(dim,dim,fill::zeros);
        for(i=0;i<dim;i++){
            cx_mat tensorprod = kron(eigst.col(i).t(),eigst.col(i));
            uh1_+=exp(-CI*energy[i]*dt*s)*tensorprod;
            uh2_+=exp(-CI*energy[i]*dt*z)*tensorprod;
        }

        // 4th-order Trotter-Susuki step
        U_T = uv6_*uh1_*uv5_*uh1_*uv4_*uh2_*uv3_*uh1_*uv2_*uh1_*uv1_*U_T;
    }
}

cx_vec Qubit::propagate(cx_vec psi_i){ // psi(t) = U(t)*psi(0)
    cx_vec psi_f(dim,fill::zeros);
    psi_f = U_T * psi_i;
    return normalise(psi_f);
}

double Qubit::transProb(cx_vec psi_i, cx_vec psi_f) { // Transition probability P(i->f) = |<psi_f|U_T|psi_i>|^2
    cx_vec psi_T = propagate(psi_i);
    return norm(cdot(psi_f, psi_T));
}

double Qubit::fidelidad(cx_mat U_tg) { // Fidelity (d=2) between a target unitary and U_T
    cx_mat U(2,2);

    cx_vec psi0 = eigst.col(0);
    cx_vec psi1 = eigst.col(1);

    cx_vec psi0_f = propagate(psi0);
    cx_vec psi1_f = propagate(psi1);

    U(0,0) = cdot(psi0, psi0_f); U(0,1) = cdot(psi1, psi0_f);
    U(1,0) = cdot(psi0, psi1_f); U(1,1) = cdot(psi1, psi1_f);

    // F = 1/6 * (Tr(U^dagger * U) + | Tr(U_tg^dagger * U) | ^2)

    double u = real(trace(U.t() * U));
    double v = abs(trace(U_tg.t() * U));

    return (u + v*v)/6.0;
}
