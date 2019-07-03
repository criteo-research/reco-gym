        data {
        int P;
        int N;
        matrix[N,P*P] XA;
        int y[N];

        matrix[P*P,P*P] Sigma;
        }

        parameters {
            vector [P*P] lambda;
        }

        model {

            lambda ~ multi_normal(rep_vector(-6,P*P), Sigma);

            y ~ bernoulli_logit(XA * lambda);

        } generated quantities {
        }
