        data {
        int P;
        int N;
        matrix[N,P*P] XA;
        int y[N];
        }

        parameters {
            vector [P*P] lambda;
            cov_matrix[m] Omega;
            cov_matrix[p] Psi;            
        }

        model {
        matrix[m*p,m*p] Sigma;

        Omega ~ wishart(P+1.1,eye(P)+1.);
        Psi ~ wishart(P+1.1,eye(P)+1.);

        for (i in 1:m)
            for (j in 1:m)
            for (k in 1:p)
                for (l in 1:p)
                Sigma[p*(i-1)+k,p*(j-1)+l] = Omega[i,j]*Psi[k,l];

            lambda ~ multi_normal(rep_vector(-6,P*P), Sigma);

            y ~ bernoulli_logit(XA * lambda);

        } generated quantities {
        }
